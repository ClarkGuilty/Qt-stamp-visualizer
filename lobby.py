# This Python file uses the following encoding: utf-8
"""Launcher GUI: pick a data path, pick/configure a tool (mosaic, 1-by-1, or a
mosaic-then-1-by-1 chain), configure the 1-by-1 classification scheme, and
extract "positive" classifications into a fresh directory for the next stage.

The same launcher can also run headless -- ``python lobby.py --no-gui ...`` (or
``--print-command`` to just print the viewer command line) drives the whole
workflow, chained extraction included, without ever showing the lobby window.
Run ``python lobby.py --help`` for the CLI.
"""

import argparse
import copy
import glob
import json
import os
import shlex
import subprocess
import sys
from os.path import join

from PySide6 import QtWidgets
from PySide6.QtCore import QByteArray, QProcess, QProcessEnvironment, Qt
from PySide6.QtWidgets import QCheckBox, QComboBox

import extraction
from imaging import detect_band_filetype
from widgets import PredefinedConfigBar

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PATH_TO_LOBBY_CONFIG = join(REPO_ROOT, ".config_lobby.json")
PATH_TO_PREDEFINED_CONFIGS = join(REPO_ROOT, ".predefined_configs")

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

DEFAULT_CONFIG = {
    'data_path': '',
    'output_path': '',
    'session_name': '',
    'seed_enabled': False,
    'seed_value': 0,
    'run_mode_index': MODE_CHAINED,
    'mosaic_ncols': 5,
    'mosaic_nrows': 8,
    'mosaic_printname': True,
    'mosaic_uninteresting_positive': False,
    'mosaic_lens_positive': True,
    'mosaic_interesting_positive': False,
    'copy_instead_of_symlink': False,
    'scheme_rows': DEFAULT_SCHEME_ROWS,
    'main_band': 'VIS',
    'color_bands': ['Y', 'J', 'H'],
    'rgb_composites': [['H', 'Y', 'I'], ['H', 'J', 'Y']],
    'dock_state': None,
}


def load_config_dict(path=PATH_TO_LOBBY_CONFIG):
    """Config file contents layered on top of DEFAULT_CONFIG (missing keys filled
    from the defaults). Returns a fresh, fully-owned dict -- safe to mutate."""
    merged = copy.deepcopy(DEFAULT_CONFIG)
    try:
        with open(path) as f:
            merged.update(json.load(f))
    except FileNotFoundError:
        pass
    return merged


def count_vis_images(source_path):
    vis_path = join(source_path, "VIS")
    count = len({os.path.basename(f) for f in glob.glob(join(vis_path, "*.fits"))})
    if count:
        return count
    return len({os.path.basename(f) for f in (glob.glob(join(vis_path, "*.png")) +
                                                glob.glob(join(vis_path, "*.jpg")) +
                                                glob.glob(join(vis_path, "*.jpeg")))})


def discover_bands(path):
    "Subdirectories directly under path -- the same notion of 'band' the viewer tools use."
    if not path or not os.path.isdir(path):
        return []
    return sorted(d for d in os.listdir(path) if os.path.isdir(join(path, d)))


def predict_mosaic_csv_path(source_path, name, ncols, nrows, seed):
    """Best-effort prediction of the CSV the mosaic tool just wrote.

    Replicates mosaic.py's own obtain_df() base_filename
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


def build_band_argv(main_band, color_bands, composites):
    """Shared -b/-B/--rgb-composites fragment, appended to both tools' argv.

    Empty -B/--rgb-composites values are passed explicitly rather than omitted:
    "no color bands" and "no composites" are states the lobby can be in
    deliberately (and the latter is forced on any dataset with fewer than 3 FITS
    bands), so falling back to the tool's Y,J,H / H,Y,I;H,J,Y defaults would
    launch a viewer that exits with "band directory not found" before its window
    opens. Only an unset main band is omitted -- there "unset" means no path has
    been scanned yet, and -b '' would name a band directory that cannot exist.
    """
    argv = []
    main_band = (main_band or '').strip()
    if main_band:
        argv += ["-b", main_band]
    color_bands = [b.strip() for b in color_bands if b.strip()]
    argv += ["-B", ",".join(color_bands)]
    composite_terms = [",".join(b.strip() for b in triple)
                        for triple in composites if all(b.strip() for b in triple)]
    argv += ["--rgb-composites", ";".join(composite_terms)]
    return argv


def build_mosaic_argv(path, name, seed, ncols, nrows, printname=False, band_argv=None):
    argv = [join(REPO_ROOT, "mosaic.py"),
            "-p", path, "-l", str(ncols), "-m", str(nrows)]
    if name:
        argv += ["-N", name]
    if seed is not None:
        argv += ["-s", str(seed)]
    # Passed explicitly either way: mosaic prints names by default, so leaving the
    # flag out would not turn the printing off.
    argv += ["--printname" if printname else "--no-printname"]
    argv += band_argv or []
    return argv


def build_single_argv(path, name, seed, classifications_string, band_argv=None):
    argv = [join(REPO_ROOT, "single_viewer.py"),
            "-p", path, "--classifications", classifications_string]
    if name:
        argv += ["-N", name]
    if seed is not None:
        argv += ["-s", str(seed)]
    argv += band_argv or []
    return argv


def classifications_string_from_rows(rows, log=None):
    """Turn scheme rows (as produced by the scheme table / stored in the config)
    into the single viewer's --classifications string.

    Returns (classifications_string, positive_majors). `log`, if given, is
    called with warning messages for unknown-major subclasses and duplicate
    keyboard shortcuts.
    """
    def warn(message):
        if log is not None:
            log(message)

    tokens = []
    positive_majors = set()
    seen_keys = {}
    known_majors = {r['major'] for r in rows if r.get('type') == 'major'}

    for row_dict in rows:
        major, sub, key = row_dict.get('major', ''), row_dict.get('sub', ''), row_dict.get('key', '')
        if not major or not key:
            continue
        if row_dict.get('type') == 'major':
            tokens.append(f"{major}={key}")
            if row_dict.get('positive'):
                positive_majors.add(major)
        else:
            if major not in known_majors:
                warn(f"Warning: subclass '{sub}' refers to unknown major '{major}'")
            tokens.append(f"{major}:{sub}={key}")
        label = f"{major}:{sub}" if row_dict.get('type') == 'subclass' and sub else major
        seen_keys.setdefault(key, []).append(label)

    for key, labels in seen_keys.items():
        if len(labels) > 1:
            warn(f"Warning: keyboard shortcut '{key}' is assigned to more than "
                 f"one button: {', '.join(labels)}")

    return ";".join(tokens), positive_majors


class LobbyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt-stamp-visualizer Lobby")

        self.defaults = DEFAULT_CONFIG
        self.config_dict = self.load_dict()

        self._launched_process = None
        self.stage1_proc = None
        self._stage1_extraction_context = None
        self.available_bands = []
        self.fits_bands = []

        self._build_ui()
        self._apply_config_to_widgets()
        self.on_mode_changed()
        self._rescan_bands()

    # ---------------------------------------------------------- UI building

    def _build_ui(self):
        # Every setup group lives in its own dock widget rather than a fixed
        # stack, so the window doesn't have to be tall enough to show
        # everything at once: users can drag panels to rearrange them
        # (side-by-side, stacked, tabbed together, or floated into their own
        # window) and drag their borders to stretch whichever one they're
        # using. There's no central widget -- with one, the leftover sliver
        # between the left/right dock columns turns into a dead strip of
        # empty space, so the dock areas are left free to fill the whole
        # window instead. The Run button lives in a plain top toolbar so it
        # stays put, full-width, regardless of how panels get rearranged.
        self.setDockNestingEnabled(True)
        self.setDockOptions(QtWidgets.QMainWindow.AnimatedDocks |
                             QtWidgets.QMainWindow.AllowNestedDocks |
                             QtWidgets.QMainWindow.AllowTabbedDocks)

        toolbar = QtWidgets.QToolBar("Actions", self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        action_bar_widget = QtWidgets.QWidget()
        action_bar = QtWidgets.QHBoxLayout(action_bar_widget)
        action_bar.setContentsMargins(4, 2, 4, 2)
        self.run_btn = QtWidgets.QPushButton("Run")
        self.run_btn.clicked.connect(self.on_run_clicked)
        self.stage_status_label = QtWidgets.QLabel("")
        action_bar.addWidget(self.run_btn)
        action_bar.addWidget(self.stage_status_label, stretch=1)
        self.preset_bar = PredefinedConfigBar(
            PATH_TO_PREDEFINED_CONFIGS, self._preset_snapshot, self._apply_preset)
        action_bar.addWidget(self.preset_bar)
        toolbar.addWidget(action_bar_widget)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        self.bands_group = self._build_bands_group()
        self.mosaic_group = self._build_mosaic_group()
        self.single_group = self._build_single_group()

        session_dock = self._make_dock("Session", self._build_session_group(), "dock_session")
        self.bands_dock = self._make_dock("Bands", self.bands_group, "dock_bands")
        self.mosaic_dock = self._make_dock("Mosaic options", self.mosaic_group, "dock_mosaic")
        extraction_dock = self._make_dock("Extraction options", self._build_extraction_group(),
                                           "dock_extraction")
        self.single_dock = self._make_dock("1-by-1 classification scheme", self.single_group,
                                            "dock_single")
        log_dock = self._make_dock("Log", self._build_log_widget(), "dock_log")

        # Explicit two-column grid (rather than repeated addDockWidget calls
        # to the same area, whose automatic placement tends to leave an
        # orphaned empty cell): left column stacks Bands/Mosaic/Extraction,
        # right column stacks Session/1-by-1, Log spans the bottom.
        self.addDockWidget(Qt.LeftDockWidgetArea, self.bands_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, session_dock)
        self.splitDockWidget(self.bands_dock, self.mosaic_dock, Qt.Vertical)
        self.splitDockWidget(self.mosaic_dock, extraction_dock, Qt.Vertical)
        self.splitDockWidget(session_dock, self.single_dock, Qt.Vertical)
        self.addDockWidget(Qt.BottomDockWidgetArea, log_dock)

    def _make_dock(self, title, widget, object_name):
        "Wraps a plain content widget in a movable/floatable/resizable dock."
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setWidget(widget)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable |
                          QtWidgets.QDockWidget.DockWidgetFloatable)
        return dock

    def _build_log_widget(self):
        widget = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(widget)
        vbox.setContentsMargins(4, 4, 4, 4)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        vbox.addWidget(self.log_view)
        return widget

    def _build_session_group(self):
        group = QtWidgets.QGroupBox()
        form = QtWidgets.QFormLayout(group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Mosaic only",
            "1-by-1 only",
            "Mosaic → 1-by-1 (chained)",
        ])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        form.addRow("Run mode:", self.mode_combo)

        path_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.editingFinished.connect(self._rescan_bands)
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

    def _build_bands_group(self):
        group = QtWidgets.QGroupBox()
        form = QtWidgets.QFormLayout(group)

        self.bands_status_label = QtWidgets.QLabel(
            "Set a path above, then Rescan bands -- subdirectories of the path become available bands.")
        self.bands_status_label.setWordWrap(True)
        form.addRow(self.bands_status_label)

        self.main_band_combo = QComboBox()
        self.main_band_combo.setEditable(True)
        form.addRow("Main band:", self.main_band_combo)

        self.color_bands_list = QtWidgets.QListWidget()
        self.color_bands_list.setMinimumHeight(50)
        form.addRow("Color bands:", self.color_bands_list)

        self.composites_table = QtWidgets.QTableWidget(0, 3)
        self.composites_table.setHorizontalHeaderLabels(["R", "G", "B"])
        self.composites_table.horizontalHeader().setStretchLastSection(True)
        self.composites_table.verticalHeader().setVisible(False)
        self.composites_table.setMinimumHeight(60)
        form.addRow("RGB composites:", self.composites_table)

        self.composites_hint_label = QtWidgets.QLabel(
            "RGB composites need at least 3 FITS-format bands -- rescan a path to check.")
        self.composites_hint_label.setWordWrap(True)
        form.addRow(self.composites_hint_label)

        composite_btn_row = QtWidgets.QHBoxLayout()
        self.add_composite_btn = QtWidgets.QPushButton("Add composite")
        self.add_composite_btn.clicked.connect(lambda: self._add_composite_row())
        self.remove_composite_btn = QtWidgets.QPushButton("Remove selected")
        self.remove_composite_btn.clicked.connect(self._remove_selected_composite_row)
        rescan_btn = QtWidgets.QPushButton("Rescan bands")
        rescan_btn.clicked.connect(self._rescan_bands)
        composite_btn_row.addWidget(self.add_composite_btn)
        composite_btn_row.addWidget(self.remove_composite_btn)
        composite_btn_row.addWidget(rescan_btn)
        form.addRow(composite_btn_row)

        return group

    def _build_mosaic_group(self):
        group = QtWidgets.QGroupBox()
        form = QtWidgets.QFormLayout(group)

        self.ncols_spin = QtWidgets.QSpinBox()
        self.ncols_spin.setRange(1, 100)
        self.ncols_spin.setValue(5)
        form.addRow("Columns per page:", self.ncols_spin)

        self.nrows_spin = QtWidgets.QSpinBox()
        self.nrows_spin.setRange(1, 100)
        self.nrows_spin.setValue(8)
        form.addRow("Rows per page:", self.nrows_spin)

        self.printname_cb = QCheckBox("Print name on click (printname)")
        form.addRow(self.printname_cb)

        self.mosaic_uninteresting_positive_cb = QCheckBox("Treat 'Uninteresting' (code 0) as positive")
        self.mosaic_lens_positive_cb = QCheckBox("Treat 'Lens' (code 1) as positive")
        self.mosaic_interesting_positive_cb = QCheckBox("Treat 'Interesting' (code 2) as positive")
        form.addRow(self.mosaic_uninteresting_positive_cb)
        form.addRow(self.mosaic_lens_positive_cb)
        form.addRow(self.mosaic_interesting_positive_cb)

        return group

    def _build_single_group(self):
        group = QtWidgets.QGroupBox()
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
        group = QtWidgets.QGroupBox()
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

    # ---------------------------------------------------------------- bands

    def _rescan_bands(self):
        path = self.path_edit.text().strip()
        self.available_bands = discover_bands(path)
        self.fits_bands = [b for b in self.available_bands
                            if detect_band_filetype(join(path, b)) == 'FITS']
        self._reconcile_main_band()
        self._refresh_band_widgets()
        self._prune_missing_composite_bands()
        self._update_composites_availability()
        if not path:
            self.bands_status_label.setText(
                "Set a path above, then Rescan bands -- subdirectories of the path become available bands.")
        elif self.available_bands:
            self.bands_status_label.setText("Available bands: " + ", ".join(self.available_bands))
        else:
            self.bands_status_label.setText(f"No subdirectories found under {path}.")

    def _reconcile_main_band(self):
        """If the current main band no longer exists under the (re)scanned path, fall back to
        the first available band -- otherwise the launched tool exits with an error (band
        directory not found) before its window even opens."""
        current = self.main_band_combo.currentText().strip()
        if not self.available_bands or current in self.available_bands:
            return
        fallback = self.available_bands[0]
        self.main_band_combo.setCurrentText(fallback)
        self.log(f"Main band '{current}' not found under the new path -- switched to '{fallback}'.")

    def _prune_missing_composite_bands(self):
        """Clears any composite-table cell referencing a band no longer found under the path --
        same crash risk as a stale main band, since build_band_argv only skips a composite row
        when a cell is empty, not when it names a nonexistent directory."""
        pruned = set()
        for row in range(self.composites_table.rowCount()):
            for col in range(3):
                combo = self.composites_table.cellWidget(row, col)
                if combo is None:
                    continue
                value = combo.currentText().strip()
                if value and value not in self.available_bands:
                    pruned.add(value)
                    combo.setCurrentText('')
        if pruned:
            self.log(f"Composite band(s) not found under the new path -- cleared: {', '.join(sorted(pruned))}")

    def _update_composites_availability(self):
        "RGB composites need >=3 FITS bands -- PNG/JPG bands can never be composite members."
        enough_fits = len(self.fits_bands) >= 3
        self.composites_table.setEnabled(enough_fits)
        self.add_composite_btn.setEnabled(enough_fits)
        if enough_fits:
            self.composites_hint_label.setText(
                "FITS bands available for composites: " + ", ".join(self.fits_bands))
        else:
            self.composites_table.setRowCount(0)
            found = f" (found: {', '.join(self.fits_bands)})" if self.fits_bands else " (found none)"
            self.composites_hint_label.setText(
                f"RGB composites need at least 3 FITS-format bands{found}.")

    def _refresh_band_widgets(self):
        "Repopulate every band combo/list with self.available_bands, preserving current selections."
        bands = self.available_bands

        current_main = self.main_band_combo.currentText().strip()
        self.main_band_combo.blockSignals(True)
        self.main_band_combo.clear()
        self.main_band_combo.addItems(bands)
        self.main_band_combo.setCurrentText(current_main)
        self.main_band_combo.blockSignals(False)

        checked = set(self._checked_color_bands()) if self.color_bands_list.count() \
            else set(self.config_dict.get('color_bands', []))
        self._populate_color_bands_list(bands, checked)

        for row in range(self.composites_table.rowCount()):
            for col in range(3):
                combo = self.composites_table.cellWidget(row, col)
                if combo is None:
                    continue
                current = combo.currentText().strip()
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(self.fits_bands)
                combo.setCurrentText(current)
                combo.blockSignals(False)

    def _populate_color_bands_list(self, bands, checked):
        self.color_bands_list.clear()
        for band in bands:
            item = QtWidgets.QListWidgetItem(band)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if band in checked else Qt.Unchecked)
            self.color_bands_list.addItem(item)

    def _checked_color_bands(self):
        checked = []
        for i in range(self.color_bands_list.count()):
            item = self.color_bands_list.item(i)
            if item.checkState() == Qt.Checked:
                checked.append(item.text())
        return checked

    def _add_composite_row(self, r='', g='', b=''):
        row = self.composites_table.rowCount()
        self.composites_table.insertRow(row)
        for col, value in enumerate((r, g, b)):
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(self.fits_bands)
            combo.setCurrentText(value)
            self.composites_table.setCellWidget(row, col, combo)

    def _remove_selected_composite_row(self):
        row = self.composites_table.currentRow()
        if row >= 0:
            self.composites_table.removeRow(row)

    def _composite_rows(self):
        rows = []
        for row in range(self.composites_table.rowCount()):
            triple = []
            for col in range(3):
                combo = self.composites_table.cellWidget(row, col)
                triple.append(combo.currentText().strip() if combo else '')
            rows.append(tuple(triple))
        return rows

    def _band_argv(self):
        return build_band_argv(self.main_band_combo.currentText(),
                                self._checked_color_bands(),
                                self._composite_rows())

    def _log_band_warnings(self):
        if not self.available_bands:
            return
        referenced = set()
        main_band = self.main_band_combo.currentText().strip()
        if main_band:
            referenced.add(main_band)
        referenced.update(self._checked_color_bands())
        for triple in self._composite_rows():
            referenced.update(b for b in triple if b)
        missing = sorted(b for b in referenced if b not in self.available_bands)
        if missing:
            self.log(f"Warning: band(s) not found as subdirectories of the data path: {', '.join(missing)}")

        composite_bands_referenced = {b for triple in self._composite_rows() if all(triple) for b in triple}
        non_fits = sorted(b for b in composite_bands_referenced
                           if b in self.available_bands and b not in self.fits_bands)
        if non_fits:
            self.log(f"Warning: RGB composite band(s) are not FITS and will be skipped by the viewer: "
                      f"{', '.join(non_fits)}")

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
        return classifications_string_from_rows(self._scheme_table_to_rows(), log=self.log)

    def update_classifications_preview(self, *_args):
        classifications_string, _ = self.build_classifications_string()
        self.classifications_preview_edit.setText(classifications_string)

    # ---------------------------------------------------------- misc UI glue

    def log(self, message):
        self.log_view.appendPlainText(message)

    def on_mode_changed(self):
        mode = self.mode_combo.currentIndex()
        self.mosaic_dock.setVisible(mode in (MODE_MOSAIC_ONLY, MODE_CHAINED))
        self.single_dock.setVisible(mode in (MODE_SINGLE_ONLY, MODE_CHAINED))

    def on_browse_path(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select data path")
        if path:
            self.path_edit.setText(path)
            self._rescan_bands()

    def on_browse_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output path")
        if path:
            self.output_path_edit.setText(path)

    def _current_seed(self):
        return self.seed_spin.value() if self.seed_enabled_cb.isChecked() else None

    def _mosaic_positive_values(self):
        positive_values = set()
        if self.mosaic_uninteresting_positive_cb.isChecked():
            positive_values.add(0)
        if self.mosaic_lens_positive_cb.isChecked():
            positive_values.add(1)
        if self.mosaic_interesting_positive_cb.isChecked():
            positive_values.add(2)
        return positive_values

    def _set_controls_enabled(self, enabled):
        for widget in (self.run_btn, self.mode_combo, self.path_edit,
                       self.path_browse_btn, self.bands_group,
                       self.mosaic_group, self.single_group):
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

        self._log_band_warnings()
        band_argv = self._band_argv()

        if mode == MODE_MOSAIC_ONLY:
            argv = build_mosaic_argv(path, name, seed,
                                      self.ncols_spin.value(), self.nrows_spin.value(),
                                      printname=self.printname_cb.isChecked(),
                                      band_argv=band_argv)
            self._launch_fire_and_forget(argv)
        elif mode == MODE_SINGLE_ONLY:
            classifications_string, _ = self.build_classifications_string()
            argv = build_single_argv(path, name, seed, classifications_string, band_argv=band_argv)
            self._launch_fire_and_forget(argv)
        elif mode == MODE_CHAINED:
            self._launch_stage1_chained(path, name, seed)

    def _wire_process_output_logging(self, proc):
        """Stream a launched tool's stdout/stderr into the lobby's own log
        pane -- QProcess pipes a child's output by default rather than
        inheriting the parent's terminal, so without this, printed output
        (e.g. --printname) would go nowhere visible at all."""
        proc.setProcessChannelMode(QProcess.MergedChannels)
        # Python block-buffers a piped stdout, so a child's prints would only show up
        # here when it exits; unbuffered output makes the log pane live.
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(
            lambda: self.log(bytes(proc.readAllStandardOutput())
                              .decode(errors='replace').rstrip('\n')))

    def _launch_fire_and_forget(self, argv):
        proc = QProcess(self)
        proc.setWorkingDirectory(REPO_ROOT)
        proc.setProgram(sys.executable)
        proc.setArguments(argv)
        self._wire_process_output_logging(proc)
        proc.finished.connect(lambda code, status: self.log(f"Process exited (code {code})."))
        self.log(f"Launching: {sys.executable} {' '.join(argv)}")
        proc.start()
        self._launched_process = proc

    def _launch_stage1_chained(self, path, name, seed):
        ncols, nrows = self.ncols_spin.value(), self.nrows_spin.value()
        argv = build_mosaic_argv(path, name, seed, ncols, nrows,
                                  printname=self.printname_cb.isChecked(),
                                  band_argv=self._band_argv())
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
        self._wire_process_output_logging(self.stage1_proc)
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

        positive_values = self._mosaic_positive_values()

        result = extraction.extract(
            csv_path, source_path, output_path, positive_values,
            use_symlink=not self.copy_instead_cb.isChecked(),
            on_progress=self.log,
        )
        self.log(f"Extraction complete. {result.summary()}")

        classifications_string, _ = self.build_classifications_string()
        argv = build_single_argv(output_path, context.get('name'), context.get('seed'),
                                  classifications_string, band_argv=self._band_argv())
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

        positive_values = self._mosaic_positive_values()
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
        self.printname_cb.setChecked(c['mosaic_printname'])
        self.mosaic_uninteresting_positive_cb.setChecked(c['mosaic_uninteresting_positive'])
        self.mosaic_lens_positive_cb.setChecked(c['mosaic_lens_positive'])
        self.mosaic_interesting_positive_cb.setChecked(c['mosaic_interesting_positive'])
        self.copy_instead_cb.setChecked(c['copy_instead_of_symlink'])
        self._populate_scheme_table(c['scheme_rows'])

        self.main_band_combo.setCurrentText(c['main_band'])
        self._populate_color_bands_list(self.available_bands, set(c['color_bands']))
        self.composites_table.setRowCount(0)
        for triple in c['rgb_composites']:
            r, g, b = (list(triple) + ['', '', ''])[:3]
            self._add_composite_row(r, g, b)

        if c.get('dock_state'):
            self.restoreState(QByteArray.fromBase64(c['dock_state'].encode('ascii')))

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
        c['mosaic_printname'] = self.printname_cb.isChecked()
        c['mosaic_uninteresting_positive'] = self.mosaic_uninteresting_positive_cb.isChecked()
        c['mosaic_lens_positive'] = self.mosaic_lens_positive_cb.isChecked()
        c['mosaic_interesting_positive'] = self.mosaic_interesting_positive_cb.isChecked()
        c['copy_instead_of_symlink'] = self.copy_instead_cb.isChecked()
        c['scheme_rows'] = self._scheme_table_to_rows()
        c['main_band'] = self.main_band_combo.currentText().strip()
        c['color_bands'] = self._checked_color_bands()
        c['rgb_composites'] = [list(triple) for triple in self._composite_rows()]
        c['dock_state'] = bytes(self.saveState().toBase64()).decode('ascii')

    def _preset_snapshot(self):
        "Current widget state as a plain dict, suitable for saving/comparing as a preset."
        self._sync_widgets_to_config()
        return {k: v for k, v in self.config_dict.items() if k != 'dock_state'}

    def _apply_preset(self, preset):
        "Applies a preset (or a snapshot from _preset_snapshot) on top of the current config."
        merged = dict(self.defaults)
        merged.update(self.config_dict)
        merged.update({k: v for k, v in preset.items() if k != 'dock_state'})
        self.config_dict = merged
        self._apply_config_to_widgets()
        self.on_mode_changed()
        self._rescan_bands()
        self._log_missing_preset_bands(preset)

    def _log_missing_preset_bands(self, preset):
        """Warns about bands a preset asks for that aren't subdirectories of the current data
        path. The preset still loads in full -- rescanning has already dropped those bands from
        the color-bands checklist (and the composite/main-band combos won't offer them either)
        since only real subdirectories are listed there, so this is purely a heads-up."""
        if not self.available_bands:
            return
        referenced = set()
        main_band = (preset.get('main_band') or '').strip()
        if main_band:
            referenced.add(main_band)
        referenced.update(b.strip() for b in preset.get('color_bands', []) if b.strip())
        for triple in preset.get('rgb_composites', []):
            referenced.update(b.strip() for b in triple if b and b.strip())
        missing = sorted(b for b in referenced if b not in self.available_bands)
        if missing:
            self.log(f"Preset references band(s) not found under the current data path -- "
                      f"not offered in the band configuration: {', '.join(missing)}")

    def save_dict(self):
        self._sync_widgets_to_config()
        with open(PATH_TO_LOBBY_CONFIG, 'w') as f:
            json.dump(self.config_dict, f, ensure_ascii=False, indent=4)

    def load_dict(self):
        return load_config_dict(PATH_TO_LOBBY_CONFIG)

    def closeEvent(self, event):
        self.save_dict()
        event.accept()


# ------------------------------------------------------------------ headless CLI

MODE_NAMES = {'mosaic': MODE_MOSAIC_ONLY, 'single': MODE_SINGLE_ONLY,
              'chained': MODE_CHAINED}


def mosaic_positive_values_from_config(c):
    "The mosaic numeric codes marked 'positive' for chained-mode extraction."
    values = set()
    if c.get('mosaic_uninteresting_positive'):
        values.add(0)
    if c.get('mosaic_lens_positive'):
        values.add(1)
    if c.get('mosaic_interesting_positive'):
        values.add(2)
    return values


def _parse_rgb_composites(text):
    return [[b.strip() for b in triple.split(',')]
            for triple in text.split(';') if triple.strip()]


def config_from_cli(args):
    """Lobby config dict: the file named by --config, with whichever CLI
    overrides were actually passed layered on top."""
    c = load_config_dict(args.config)
    if args.mode is not None:
        c['run_mode_index'] = MODE_NAMES[args.mode]
    if args.path is not None:
        c['data_path'] = args.path
    if args.output is not None:
        c['output_path'] = args.output
    if args.name is not None:
        c['session_name'] = args.name
    if args.seed is not None:
        c['seed_enabled'], c['seed_value'] = True, args.seed
    if args.no_seed:
        c['seed_enabled'] = False
    if args.main_band is not None:
        c['main_band'] = args.main_band
    if args.color_bands is not None:
        c['color_bands'] = [b.strip() for b in args.color_bands.split(',') if b.strip()]
    if args.rgb_composites is not None:
        c['rgb_composites'] = _parse_rgb_composites(args.rgb_composites)
    if args.ncols is not None:
        c['mosaic_ncols'] = args.ncols
    if args.nrows is not None:
        c['mosaic_nrows'] = args.nrows
    if args.printname is not None:
        c['mosaic_printname'] = args.printname
    if args.copy_files is not None:
        c['copy_instead_of_symlink'] = args.copy_files
    return c


def run_headless(config, classifications_override=None, print_command_only=False,
                 log=print):
    """Run the configured workflow without the lobby window; returns an exit code.

    With print_command_only, prints the viewer command line(s) the lobby would
    launch -- chained mode prints both stages -- and returns 0 without running
    anything.
    """
    mode = config['run_mode_index']
    path = config['data_path']
    name = config['session_name']
    seed = config['seed_value'] if config['seed_enabled'] else None
    band_argv = build_band_argv(config['main_band'], config['color_bands'],
                                config['rgb_composites'])

    if classifications_override is not None:
        classifications_string = classifications_override
    else:
        classifications_string, _ = classifications_string_from_rows(
            config['scheme_rows'], log=log)

    def emit(argv):
        log(shlex.join([sys.executable, *argv]))

    def run(argv):
        log("Launching: " + shlex.join([sys.executable, *argv]))
        return subprocess.run([sys.executable, *argv], cwd=REPO_ROOT).returncode

    if print_command_only:
        path = path or "<path>"
    elif not path:
        log("No data path -- pass --path or set 'data_path' in the lobby config.")
        return 2

    if mode == MODE_MOSAIC_ONLY:
        argv = build_mosaic_argv(path, name, seed, config['mosaic_ncols'],
                                 config['mosaic_nrows'],
                                 printname=config['mosaic_printname'],
                                 band_argv=band_argv)
        if print_command_only:
            emit(argv)
            return 0
        return run(argv)

    if mode == MODE_SINGLE_ONLY:
        argv = build_single_argv(path, name, seed, classifications_string,
                                 band_argv=band_argv)
        if print_command_only:
            emit(argv)
            return 0
        return run(argv)

    # chained: mosaic -> extract positives -> 1-by-1
    ncols, nrows = config['mosaic_ncols'], config['mosaic_nrows']
    mosaic_argv = build_mosaic_argv(path, name, seed, ncols, nrows,
                                    printname=config['mosaic_printname'],
                                    band_argv=band_argv)
    output_path = config['output_path']

    if print_command_only:
        emit(mosaic_argv)
        emit(build_single_argv(output_path or "<output_path>", name, seed,
                               classifications_string, band_argv=band_argv))
        return 0

    if not output_path:
        log("Chained mode needs an output path -- pass --output or set 'output_path'.")
        return 2

    code = run(mosaic_argv)
    if code != 0:
        log(f"Mosaic stage exited with code {code} -- stopping before extraction.")
        return code

    csv_path = predict_mosaic_csv_path(path, name, ncols, nrows, seed)
    if not csv_path:
        log("Could not auto-detect the mosaic classification CSV -- run the chained "
            "workflow from the lobby GUI instead, or extract manually.")
        return 3
    log(f"Auto-detected classification CSV: {csv_path}")

    result = extraction.extract(
        csv_path, path, output_path, mosaic_positive_values_from_config(config),
        use_symlink=not config['copy_instead_of_symlink'], on_progress=log)
    log(f"Extraction complete. {result.summary()}")

    return run(build_single_argv(output_path, name, seed, classifications_string,
                                 band_argv=band_argv))


def _build_arg_parser():
    p = argparse.ArgumentParser(
        prog="lobby.py",
        description="Launcher for the mosaic and 1-by-1 stamp viewers. With no "
                    "arguments it opens the lobby window; --no-gui / --print-command "
                    "run the configured workflow straight from the command line, "
                    "reading unspecified values from the lobby config file.")
    p.add_argument("--no-gui", action="store_true",
                   help="Run the configured workflow without opening the lobby window.")
    p.add_argument("--print-command", action="store_true",
                   help="Print the viewer command line(s) that would run, then exit "
                        "(implies --no-gui; chained mode prints both stages).")
    p.add_argument("--config", default=PATH_TO_LOBBY_CONFIG, metavar="PATH",
                   help="Lobby config JSON to read defaults from (default: %(default)s).")

    g = p.add_argument_group("workflow overrides (with --no-gui / --print-command)")
    g.add_argument("-m", "--mode", choices=sorted(MODE_NAMES),
                   help="mosaic only, 1-by-1 (single) only, or mosaic->1-by-1 chained.")
    g.add_argument("-p", "--path", help="Path to the images to inspect.")
    g.add_argument("-o", "--output", help="Output path for chained-mode extraction.")
    g.add_argument("-N", "--name", help="Session name.")
    g.add_argument("-s", "--seed", type=int, help="Shuffle seed.")
    g.add_argument("--no-seed", action="store_true",
                   help="Ignore any seed set in the config.")
    g.add_argument("-b", "--main-band", help='Main / high-resolution band (e.g. "VIS").')
    g.add_argument("-B", "--color-bands", metavar="A,B,C",
                   help="Comma-separated individually-selectable bands.")
    g.add_argument("--rgb-composites", metavar="R,G,B;R,G,B",
                   help="Semicolon-separated R,G,B band-name triples.")
    g.add_argument("--classifications", metavar="SPEC",
                   help='1-by-1 scheme string (e.g. "A=1;B=2;C=3;X=4;I=5"); '
                        "overrides the scheme rows from the config.")
    g.add_argument("--ncols", type=int, help="Mosaic columns per page.")
    g.add_argument("--nrows", type=int, help="Mosaic rows per page.")
    g.add_argument("--printname", action=argparse.BooleanOptionalAction,
                   help="Mosaic: print the filename on click.")
    copy_group = g.add_mutually_exclusive_group()
    copy_group.add_argument("--copy", dest="copy_files", action="store_true", default=None,
                            help="Chained extraction copies files.")
    copy_group.add_argument("--symlink", dest="copy_files", action="store_false",
                            default=None, help="Chained extraction symlinks files (default).")
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    headless = args.no_gui or args.print_command
    gave_override = any([
        args.mode is not None, args.path is not None, args.output is not None,
        args.name is not None, args.seed is not None, args.no_seed,
        args.main_band is not None, args.color_bands is not None,
        args.rgb_composites is not None, args.classifications is not None,
        args.ncols is not None, args.nrows is not None,
        args.printname is not None, args.copy_files is not None,
    ])
    if gave_override and not headless:
        parser.error("workflow overrides only apply with --no-gui or --print-command")

    if not headless:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        win = LobbyWindow()
        win.show()
        sys.exit(app.exec())

    sys.exit(run_headless(config_from_cli(args),
                          classifications_override=args.classifications,
                          print_command_only=args.print_command))


if __name__ == "__main__":
    main()
