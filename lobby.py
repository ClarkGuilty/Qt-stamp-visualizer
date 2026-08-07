# This Python file uses the following encoding: utf-8
"""Launcher GUI: pick a data path, pick/configure a tool (mosaic, 1-by-1, or a
mosaic-then-1-by-1 chain), configure the 1-by-1 classification scheme, and
extract "positive" classifications into a fresh directory for the next stage.
"""

import glob
import json
import os
import sys
from os.path import join

from PySide6 import QtWidgets
from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import QCheckBox, QComboBox

import extraction

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PATH_TO_LOBBY_CONFIG = join(REPO_ROOT, ".config_lobby.json")

MODE_MOSAIC_ONLY = 0
MODE_SINGLE_ONLY = 1
MODE_CHAINED = 2

TYPE_MAJOR = "Major"
TYPE_SUBCLASS = "Subclass"

COL_TYPE, COL_MAJOR, COL_SUB, COL_KEY, COL_POSITIVE = range(5)

DEFAULT_SCHEME_ROWS = [
    {'type': 'major', 'major': 'A', 'sub': '', 'key': '1', 'positive': False},
    {'type': 'major', 'major': 'B', 'sub': '', 'key': '2', 'positive': False},
    {'type': 'major', 'major': 'C', 'sub': '', 'key': '3', 'positive': False},
    {'type': 'major', 'major': 'X', 'sub': '', 'key': '4', 'positive': False},
    {'type': 'major', 'major': 'I', 'sub': '', 'key': '5', 'positive': False},
]


def count_vis_images(source_path):
    return len({os.path.basename(f) for f in glob.glob(join(source_path, "VIS", "*.fits"))})


def predict_mosaic_csv_path(source_path, name, ncols, nrows, seed):
    """Best-effort prediction of the CSV the mosaic tool just wrote.

    Replicates mosaic_viewer_ERO_edition.py's own obtain_df() base_filename
    formula exactly -- including its no-seed-branch bug where nrows is
    silently dropped and replaced by a literal '99'. Rather than also
    replicating obtain_df()'s pre-run file-selection quirks (natural-sort +
    picking the second-to-last match, its file_iteration suffix logic for
    mismatched datasets), this just globs for anything matching the base
    filename pattern and returns the most recently modified match -- mosaic
    rewrites its CSV to disk on every single click, so "most recently
    modified" reliably identifies the file that session just wrote.
    Returns None if nothing matches, so the caller can fall back to asking
    the user.
    """
    name = name or ''
    n_images = count_vis_images(source_path)
    if seed is None:
        base_filename = f'classification_mosaic_autosave_{name}_{n_images}_{ncols}_99'
    else:
        base_filename = f'classification_mosaic_autosave_{name}_{n_images}_{ncols}_{nrows}_{seed}'
    matches = glob.glob(join(REPO_ROOT, "Classifications", f"{base_filename}*.csv"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def build_mosaic_argv(path, name, seed, ncols, nrows):
    argv = [join(REPO_ROOT, "mosaic_viewer_ERO_edition.py"),
            "-p", path, "-l", str(ncols), "-m", str(nrows)]
    if name:
        argv += ["-N", name]
    if seed is not None:
        argv += ["-s", str(seed)]
    return argv


def build_single_argv(path, name, seed, classifications_string):
    argv = [join(REPO_ROOT, "single_viewer_multiband_ERO_edition.py"),
            "-p", path, "--classifications", classifications_string]
    if name:
        argv += ["-N", name]
    if seed is not None:
        argv += ["-s", str(seed)]
    return argv


class LobbyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt-stamp-visualizer Lobby")

        self.defaults = {
            'data_path': '',
            'output_path': '',
            'session_name': '',
            'seed_enabled': False,
            'seed_value': 0,
            'run_mode_index': MODE_CHAINED,
            'mosaic_ncols': 5,
            'mosaic_nrows': 8,
            'mosaic_lens_positive': False,
            'mosaic_interesting_positive': False,
            'copy_instead_of_symlink': False,
            'scheme_rows': DEFAULT_SCHEME_ROWS,
        }
        self.config_dict = self.load_dict()

        self._launched_process = None
        self.stage1_proc = None
        self._stage1_extraction_context = None

        self._build_ui()
        self._apply_config_to_widgets()
        self.on_mode_changed()

    # ---------------------------------------------------------- UI building

    def _build_ui(self):
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        layout.addWidget(self._build_mode_group())
        layout.addWidget(self._build_data_group())
        self.mosaic_group = self._build_mosaic_group()
        layout.addWidget(self.mosaic_group)
        self.single_group = self._build_single_group()
        layout.addWidget(self.single_group)
        layout.addWidget(self._build_extraction_group())

        action_bar = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run")
        self.run_btn.clicked.connect(self.on_run_clicked)
        self.stage_status_label = QtWidgets.QLabel("")
        action_bar.addWidget(self.run_btn)
        action_bar.addWidget(self.stage_status_label, stretch=1)
        layout.addLayout(action_bar)

        log_group = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        log_layout.addWidget(self.log_view)
        layout.addWidget(log_group)

        self.setCentralWidget(central)

    def _build_mode_group(self):
        group = QtWidgets.QGroupBox("Run mode")
        vbox = QtWidgets.QVBoxLayout(group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Mosaic only",
            "1-by-1 only",
            "Mosaic → 1-by-1 (chained)",
        ])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        vbox.addWidget(self.mode_combo)
        return group

    def _build_data_group(self):
        group = QtWidgets.QGroupBox("Data")
        form = QtWidgets.QFormLayout(group)

        path_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_browse_btn = QtWidgets.QPushButton("Browse...")
        self.path_browse_btn.clicked.connect(self.on_browse_path)
        path_row.addWidget(self.path_edit)
        path_row.addWidget(self.path_browse_btn)
        form.addRow("Path to images:", path_row)

        self.name_edit = QtWidgets.QLineEdit()
        form.addRow("Session name:", self.name_edit)

        seed_row = QtWidgets.QHBoxLayout()
        self.seed_enabled_cb = QCheckBox("Use seed")
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 1_000_000)
        self.seed_enabled_cb.toggled.connect(self.seed_spin.setEnabled)
        seed_row.addWidget(self.seed_enabled_cb)
        seed_row.addWidget(self.seed_spin)
        form.addRow("Seed:", seed_row)

        return group

    def _build_mosaic_group(self):
        group = QtWidgets.QGroupBox("Mosaic options")
        form = QtWidgets.QFormLayout(group)

        self.ncols_spin = QtWidgets.QSpinBox()
        self.ncols_spin.setRange(1, 100)
        self.ncols_spin.setValue(5)
        form.addRow("Columns per page:", self.ncols_spin)

        self.nrows_spin = QtWidgets.QSpinBox()
        self.nrows_spin.setRange(1, 100)
        self.nrows_spin.setValue(8)
        form.addRow("Rows per page:", self.nrows_spin)

        self.mosaic_lens_positive_cb = QCheckBox("Treat 'Lens' (code 1) as positive")
        self.mosaic_interesting_positive_cb = QCheckBox("Treat 'Interesting' (code 2) as positive")
        form.addRow(self.mosaic_lens_positive_cb)
        form.addRow(self.mosaic_interesting_positive_cb)

        return group

    def _build_single_group(self):
        group = QtWidgets.QGroupBox("1-by-1 classification scheme")
        vbox = QtWidgets.QVBoxLayout(group)

        self.scheme_table = QtWidgets.QTableWidget(0, 5)
        self.scheme_table.setHorizontalHeaderLabels(
            ["Type", "Major", "Sub", "Key", "Positive"])
        self.scheme_table.horizontalHeader().setStretchLastSection(True)
        vbox.addWidget(self.scheme_table)

        btn_row = QtWidgets.QHBoxLayout()
        add_major_btn = QtWidgets.QPushButton("Add major")
        add_major_btn.clicked.connect(lambda: self.add_scheme_row(TYPE_MAJOR))
        add_sub_btn = QtWidgets.QPushButton("Add subclass")
        add_sub_btn.clicked.connect(lambda: self.add_scheme_row(TYPE_SUBCLASS))
        remove_btn = QtWidgets.QPushButton("Remove selected row")
        remove_btn.clicked.connect(self.remove_selected_scheme_row)
        btn_row.addWidget(add_major_btn)
        btn_row.addWidget(add_sub_btn)
        btn_row.addWidget(remove_btn)
        vbox.addLayout(btn_row)

        self.classifications_preview_edit = QtWidgets.QLineEdit()
        self.classifications_preview_edit.setReadOnly(True)
        vbox.addWidget(self.classifications_preview_edit)

        self.scheme_table.itemChanged.connect(self.update_classifications_preview)

        return group

    def _build_extraction_group(self):
        group = QtWidgets.QGroupBox("Extraction options")
        form = QtWidgets.QFormLayout(group)

        out_row = QtWidgets.QHBoxLayout()
        self.output_path_edit = QtWidgets.QLineEdit()
        self.output_browse_btn = QtWidgets.QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self.on_browse_output)
        out_row.addWidget(self.output_path_edit)
        out_row.addWidget(self.output_browse_btn)
        form.addRow("Output path:", out_row)

        self.copy_instead_cb = QCheckBox("Copy files instead of symlinking")
        form.addRow(self.copy_instead_cb)

        self.extract_only_btn = QtWidgets.QPushButton("Extract only...")
        self.extract_only_btn.clicked.connect(self.on_extract_only_clicked)
        form.addRow(self.extract_only_btn)

        return group

    # ---------------------------------------------------------- scheme table

    def add_scheme_row(self, row_type, major='', sub='', key='', positive=False):
        row = self.scheme_table.rowCount()
        self.scheme_table.insertRow(row)

        type_combo = QComboBox()
        type_combo.addItems([TYPE_MAJOR, TYPE_SUBCLASS])
        type_combo.setCurrentText(row_type)
        type_combo.setProperty("row", row)
        type_combo.currentTextChanged.connect(self._on_row_type_changed)
        self.scheme_table.setCellWidget(row, COL_TYPE, type_combo)

        # Set the Positive checkbox cell widget before the text items below --
        # setItem() fires itemChanged, which rescans every row (including this
        # one) to rebuild the preview, so the checkbox must already exist.
        positive_cb = QCheckBox()
        positive_cb.setChecked(positive)
        positive_cb.setProperty("row", row)
        positive_cb.toggled.connect(lambda _: self.update_classifications_preview())
        container = QtWidgets.QWidget()
        cbox_layout = QtWidgets.QHBoxLayout(container)
        cbox_layout.addWidget(positive_cb)
        cbox_layout.setAlignment(positive_cb, Qt.AlignCenter)
        cbox_layout.setContentsMargins(0, 0, 0, 0)
        self.scheme_table.setCellWidget(row, COL_POSITIVE, container)

        self.scheme_table.setItem(row, COL_MAJOR, QtWidgets.QTableWidgetItem(major))
        self.scheme_table.setItem(row, COL_SUB, QtWidgets.QTableWidgetItem(sub))
        self.scheme_table.setItem(row, COL_KEY, QtWidgets.QTableWidgetItem(key))

        self.update_classifications_preview()

    def _on_row_type_changed(self, _text):
        self.update_classifications_preview()

    def remove_selected_scheme_row(self):
        row = self.scheme_table.currentRow()
        if row >= 0:
            self.scheme_table.removeRow(row)
            self.update_classifications_preview()

    def _row_positive_checkbox(self, row):
        container = self.scheme_table.cellWidget(row, COL_POSITIVE)
        return container.findChild(QCheckBox)

    def _scheme_table_to_rows(self):
        rows = []
        for row in range(self.scheme_table.rowCount()):
            type_combo = self.scheme_table.cellWidget(row, COL_TYPE)
            row_type = type_combo.currentText() if type_combo else TYPE_MAJOR
            major_item = self.scheme_table.item(row, COL_MAJOR)
            sub_item = self.scheme_table.item(row, COL_SUB)
            key_item = self.scheme_table.item(row, COL_KEY)
            positive_cb = self._row_positive_checkbox(row)
            rows.append({
                'type': 'major' if row_type == TYPE_MAJOR else 'subclass',
                'major': major_item.text().strip() if major_item else '',
                'sub': sub_item.text().strip() if sub_item else '',
                'key': key_item.text().strip() if key_item else '',
                'positive': positive_cb.isChecked() if positive_cb else False,
            })
        return rows

    def _populate_scheme_table(self, rows):
        self.scheme_table.setRowCount(0)
        for row_dict in rows:
            row_type = TYPE_MAJOR if row_dict.get('type') == 'major' else TYPE_SUBCLASS
            self.add_scheme_row(
                row_type,
                major=row_dict.get('major', ''),
                sub=row_dict.get('sub', ''),
                key=row_dict.get('key', ''),
                positive=row_dict.get('positive', False),
            )

    def build_classifications_string(self):
        """Returns (classifications_string, positive_majors)."""
        tokens = []
        positive_majors = set()
        seen_keys = {}
        known_majors = set()

        rows = self._scheme_table_to_rows()
        for row_dict in rows:
            if row_dict['type'] == 'major':
                known_majors.add(row_dict['major'])

        for row_dict in rows:
            major, sub, key = row_dict['major'], row_dict['sub'], row_dict['key']
            if not major or not key:
                continue
            if row_dict['type'] == 'major':
                tokens.append(f"{major}={key}")
                if row_dict['positive']:
                    positive_majors.add(major)
            else:
                if major not in known_majors:
                    self.log(f"Warning: subclass '{sub}' refers to unknown major '{major}'")
                tokens.append(f"{major}:{sub}={key}")
            label = f"{major}:{sub}" if row_dict['type'] == 'subclass' and sub else major
            seen_keys.setdefault(key, []).append(label)

        for key, labels in seen_keys.items():
            if len(labels) > 1:
                self.log(f"Warning: keyboard shortcut '{key}' is assigned to more than "
                          f"one button: {', '.join(labels)}")

        return ";".join(tokens), positive_majors

    def update_classifications_preview(self, *_args):
        classifications_string, _ = self.build_classifications_string()
        self.classifications_preview_edit.setText(classifications_string)

    # ---------------------------------------------------------- misc UI glue

    def log(self, message):
        self.log_view.appendPlainText(message)

    def on_mode_changed(self):
        mode = self.mode_combo.currentIndex()
        self.mosaic_group.setVisible(mode in (MODE_MOSAIC_ONLY, MODE_CHAINED))
        self.single_group.setVisible(mode in (MODE_SINGLE_ONLY, MODE_CHAINED))

    def on_browse_path(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select data path")
        if path:
            self.path_edit.setText(path)

    def on_browse_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output path")
        if path:
            self.output_path_edit.setText(path)

    def _current_seed(self):
        return self.seed_spin.value() if self.seed_enabled_cb.isChecked() else None

    def _set_controls_enabled(self, enabled):
        for widget in (self.run_btn, self.mode_combo, self.path_edit,
                       self.path_browse_btn, self.mosaic_group, self.single_group):
            widget.setEnabled(enabled)

    # ---------------------------------------------------------- launching

    def on_run_clicked(self):
        mode = self.mode_combo.currentIndex()
        path = self.path_edit.text().strip()
        name = self.name_edit.text().strip()
        seed = self._current_seed()

        if not path:
            self.log("Please select a data path first.")
            return

        if mode == MODE_MOSAIC_ONLY:
            argv = build_mosaic_argv(path, name, seed,
                                      self.ncols_spin.value(), self.nrows_spin.value())
            self._launch_fire_and_forget(argv)
        elif mode == MODE_SINGLE_ONLY:
            classifications_string, _ = self.build_classifications_string()
            argv = build_single_argv(path, name, seed, classifications_string)
            self._launch_fire_and_forget(argv)
        elif mode == MODE_CHAINED:
            self._launch_stage1_chained(path, name, seed)

    def _launch_fire_and_forget(self, argv):
        proc = QProcess(self)
        proc.setWorkingDirectory(REPO_ROOT)
        proc.setProgram(sys.executable)
        proc.setArguments(argv)
        self.log(f"Launching: {sys.executable} {' '.join(argv)}")
        proc.start()
        self._launched_process = proc

    def _launch_stage1_chained(self, path, name, seed):
        ncols, nrows = self.ncols_spin.value(), self.nrows_spin.value()
        argv = build_mosaic_argv(path, name, seed, ncols, nrows)
        self._stage1_extraction_context = {
            'source_path': path,
            'name': name,
            'seed': seed,
            'ncols': ncols,
            'nrows': nrows,
        }
        self.stage1_proc = QProcess(self)
        self.stage1_proc.setWorkingDirectory(REPO_ROOT)
        self.stage1_proc.setProgram(sys.executable)
        self.stage1_proc.setArguments(argv)
        self.stage1_proc.finished.connect(self.on_stage1_finished)
        self._set_controls_enabled(False)
        self.stage_status_label.setText("Stage 1 (mosaic) running...")
        self.log(f"Launching: {sys.executable} {' '.join(argv)}")
        self.stage1_proc.start()

    def on_stage1_finished(self, exit_code, exit_status):
        self._set_controls_enabled(True)
        self.stage_status_label.setText(f"Stage 1 finished (exit code {exit_code}).")
        self.log(f"Stage 1 (mosaic) finished with exit code {exit_code}.")

        context = self._stage1_extraction_context or {}
        source_path = context.get('source_path', self.path_edit.text().strip())

        csv_path = predict_mosaic_csv_path(
            source_path, context.get('name'), context.get('ncols'),
            context.get('nrows'), context.get('seed'))
        if csv_path:
            self.log(f"Auto-detected classification CSV: {csv_path}")
        else:
            self.log("Could not auto-detect the classification CSV -- please select it.")
            csv_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select the classification CSV produced by the mosaic tool",
                join(REPO_ROOT, "Classifications"), "CSV files (*.csv)")
            if not csv_path:
                self.log("Extraction cancelled -- no CSV selected.")
                return

        output_path = self.output_path_edit.text().strip()
        if not output_path:
            self.log("Please set an output path before running the chained workflow.")
            return

        positive_values = set()
        if self.mosaic_lens_positive_cb.isChecked():
            positive_values.add(1)
        if self.mosaic_interesting_positive_cb.isChecked():
            positive_values.add(2)

        result = extraction.extract(
            csv_path, source_path, output_path, positive_values,
            use_symlink=not self.copy_instead_cb.isChecked(),
            on_progress=self.log,
        )
        self.log(f"Extraction complete. {result.summary()}")

        classifications_string, _ = self.build_classifications_string()
        argv = build_single_argv(output_path, context.get('name'), context.get('seed'),
                                  classifications_string)
        self._launch_fire_and_forget(argv)

    def on_extract_only_clicked(self):
        csv_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select classification CSV",
            join(REPO_ROOT, "Classifications"), "CSV files (*.csv)")
        if not csv_path:
            return

        source_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select source images path", self.path_edit.text().strip())
        if not source_path:
            return

        output_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output path", self.output_path_edit.text().strip())
        if not output_path:
            return

        positive_values = set()
        if self.mosaic_lens_positive_cb.isChecked():
            positive_values.add(1)
        if self.mosaic_interesting_positive_cb.isChecked():
            positive_values.add(2)
        _, scheme_positive_majors = self.build_classifications_string()
        positive_values |= scheme_positive_majors

        result = extraction.extract(
            csv_path, source_path, output_path, positive_values,
            use_symlink=not self.copy_instead_cb.isChecked(),
            on_progress=self.log,
        )
        self.log(f"Extraction complete. {result.summary()}")

    # ---------------------------------------------------------- persistence

    def _apply_config_to_widgets(self):
        c = self.config_dict
        self.path_edit.setText(c['data_path'])
        self.output_path_edit.setText(c['output_path'])
        self.name_edit.setText(c['session_name'])
        self.seed_enabled_cb.setChecked(c['seed_enabled'])
        self.seed_spin.setValue(c['seed_value'])
        self.seed_spin.setEnabled(c['seed_enabled'])
        self.mode_combo.setCurrentIndex(c['run_mode_index'])
        self.ncols_spin.setValue(c['mosaic_ncols'])
        self.nrows_spin.setValue(c['mosaic_nrows'])
        self.mosaic_lens_positive_cb.setChecked(c['mosaic_lens_positive'])
        self.mosaic_interesting_positive_cb.setChecked(c['mosaic_interesting_positive'])
        self.copy_instead_cb.setChecked(c['copy_instead_of_symlink'])
        self._populate_scheme_table(c['scheme_rows'])

    def _sync_widgets_to_config(self):
        c = self.config_dict
        c['data_path'] = self.path_edit.text()
        c['output_path'] = self.output_path_edit.text()
        c['session_name'] = self.name_edit.text()
        c['seed_enabled'] = self.seed_enabled_cb.isChecked()
        c['seed_value'] = self.seed_spin.value()
        c['run_mode_index'] = self.mode_combo.currentIndex()
        c['mosaic_ncols'] = self.ncols_spin.value()
        c['mosaic_nrows'] = self.nrows_spin.value()
        c['mosaic_lens_positive'] = self.mosaic_lens_positive_cb.isChecked()
        c['mosaic_interesting_positive'] = self.mosaic_interesting_positive_cb.isChecked()
        c['copy_instead_of_symlink'] = self.copy_instead_cb.isChecked()
        c['scheme_rows'] = self._scheme_table_to_rows()

    def save_dict(self):
        self._sync_widgets_to_config()
        with open(PATH_TO_LOBBY_CONFIG, 'w') as f:
            json.dump(self.config_dict, f, ensure_ascii=False, indent=4)

    def load_dict(self):
        try:
            with open(PATH_TO_LOBBY_CONFIG) as f:
                temp_dict = json.load(f)
                for key in self.defaults.keys():
                    if key not in temp_dict.keys():
                        temp_dict[key] = self.defaults[key]
                return temp_dict
        except FileNotFoundError:
            return dict(self.defaults)

    def closeEvent(self, event):
        self.save_dict()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = LobbyWindow()
    win.show()
    sys.exit(app.exec())
