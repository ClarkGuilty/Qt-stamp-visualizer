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

import pandas as pd
from PySide6 import QtWidgets
from PySide6.QtCore import QByteArray, QProcess, QProcessEnvironment, Qt
from PySide6.QtWidgets import QCheckBox, QComboBox

import extraction
import fits_io
import paths
import state
from paths import add_state_dir_args, resolve_classifications_dir, resolve_state_dir_override
from widgets import PredefinedConfigBar

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

LOBBY_CONFIG_NAME = "config_lobby.json"
# 2b-iii's spelling. Never migrated out of the per-user config dir (other
# workspaces read it), so it stays findable there forever; inside a state dir it
# is migrated to the modern name on first run. See paths.find_config.
LEGACY_LOBBY_CONFIG_NAME = ".config_lobby.json"

# BUGS.md item 14: the lobby's own tool name in the shared sessions.json, and
# the subset of DEFAULT_CONFIG/config_lobby.json that is dataset-scoped rather
# than workspace-scoped -- see state.py's module docstring and STEP.md for the
# split. These six keys stay in DEFAULT_CONFIG too, now meaning "template for
# a session that has no record of its own yet".
SESSION_TOOL = 'lobby'
SESSION_CONFIG_KEYS = ('scheme_rows', 'main_band', 'color_bands', 'rgb_composites',
                       'mosaic_ncols', 'mosaic_nrows')


def lobby_state(state_dir_override=None):
    """Resolve where the lobby's own config and presets live, and migrate once.

    Local-first (PLAN.md 2b-v): `<CWD>/.qtstamp` unless `--state-dir`/`--global`
    says otherwise. Returns `(config_path, preset_dirs, preset_write_dir)` --
    `config_path` is where the config is *written*; reading goes through
    `paths.find_config`, which also sees the per-user dir and the presets
    shipped in `qtstamp_defaults/`.

    Called at startup rather than resolved at import, because the override
    comes from parsed args and because the CWD is only meaningful once the
    process is actually running (`--print-command` resolves paths without ever
    touching disk).
    """
    state = paths.state_dir(override=state_dir_override)
    presets = join(state, paths.PRESETS_SUBDIR)
    # Within the state dir: 2b-iii's dotted spellings, renamed in place.
    paths.migrate_once(join(state, ".config_lobby.json"), join(state, LOBBY_CONFIG_NAME))
    paths.migrate_once(join(state, ".predefined_configs"), presets)
    # From REPO_ROOT (where the old hardcoded constants always pointed): into the
    # *per-user* dir, not this workspace's. There is only one REPO_ROOT copy but
    # any number of workspaces, so moving it into whichever one happened to start
    # the lobby first would take it away from all the others. The per-user dir is
    # the one destination that stays on every workspace's search path, which
    # makes this the single case where 2b-v writes outside the state dir -- it is
    # a one-time rescue of a file that would otherwise become unreachable, not a
    # settings write. Never migrated *out of* the per-user dir for the same
    # reason: other workspaces are reading it.
    user_dir = paths.user_config_dir()
    paths.migrate_once(join(REPO_ROOT, ".config_lobby.json"),
                       join(user_dir, LEGACY_LOBBY_CONFIG_NAME))
    paths.migrate_once(join(REPO_ROOT, ".predefined_configs"),
                       join(user_dir, ".predefined_configs"))
    # Both spellings on every entry: the per-user dir still holds a 2b-iii
    # `.predefined_configs/` that is never migrated out (other workspaces read
    # it), so it has to stay listed. `presets/` first, so a modern preset
    # shadows a same-named legacy one rather than the other way round.
    preset_dirs = []
    for d in paths.config_search_path(override=state_dir_override):
        preset_dirs += [join(d, paths.PRESETS_SUBDIR), join(d, ".predefined_configs")]
    return join(state, LOBBY_CONFIG_NAME), preset_dirs, presets

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
    'classifications_path': '',
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


def load_config_dict(path=None, state_dir_override=None):
    """Config file contents layered on top of DEFAULT_CONFIG (missing keys filled
    from the defaults). Returns a fresh, fully-owned dict -- safe to mutate.

    `path=None` means "wherever the search path finds one" -- the workspace's
    state dir, then the per-user dir, then `qtstamp_defaults/`. Finding nothing
    is normal for a fresh workspace and yields DEFAULT_CONFIG untouched.
    """
    if path is None:
        path = paths.find_config(LOBBY_CONFIG_NAME, override=state_dir_override,
                                 legacy=LEGACY_LOBBY_CONFIG_NAME)
    merged = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return merged
    try:
        with open(path) as f:
            merged.update(json.load(f))
    except (OSError, json.JSONDecodeError):
        pass
    return merged


def count_band_images(source_path, band, mef=False):
    if mef:
        # Every band is an extension of the same per-object file, so any band's count
        # is the dataset's object count -- there's no single per-band directory to look in.
        return len(fits_io.list_fits_files(source_path))
    band_path = join(source_path, band)
    count = len({os.path.basename(f) for f in glob.glob(join(band_path, "*.fits"))})
    if count:
        return count
    return len({os.path.basename(f) for f in (glob.glob(join(band_path, "*.png")) +
                                                glob.glob(join(band_path, "*.jpg")) +
                                                glob.glob(join(band_path, "*.jpeg")))})


def count_vis_images(source_path):
    return count_band_images(source_path, "VIS", mef=fits_io.is_mef_dataset(source_path))


def discover_bands(path):
    """Available band names under path.

    For an MEF dataset (path holds *.fits files directly, one multi-extension file
    per object) these are the first file's HDU extension names. Otherwise they're
    subdirectories directly under path -- the same notion of 'band' the
    directory-per-band viewer tools use.
    """
    if not path or not os.path.isdir(path):
        return []
    if fits_io.is_mef_dataset(path):
        sample = join(path, fits_io.list_fits_files(path)[0])
        return sorted(fits_io.discover_mef_bands(sample))
    return sorted(d for d in os.listdir(path) if os.path.isdir(join(path, d)))


def predict_mosaic_csv_path(source_path, name, seed, classifications_dir):
    """Best-effort prediction of the CSV the mosaic tool just wrote.

    Replicates mosaic.py's own obtain_df() base_filename formula: dataset
    identity only -- name, image count, seed. The grid shape is not part of it.

    Rather than also replicating obtain_df()'s file-selection logic, this globs
    for anything matching the base filename and returns the most recently
    modified match -- mosaic rewrites its CSV on every click, so "most recently
    modified" reliably identifies the file that session just wrote. Returns None
    if nothing matches, so the caller can fall back to asking the user.
    """
    name = name or ''
    n_images = count_vis_images(source_path)
    base_filename = f'classification_mosaic_autosave_{name}_{n_images}'
    if seed is None:
        # `_*` is the seeded layout, which this unseeded session must not claim.
        matches = ((set(glob.glob(join(classifications_dir, f'{base_filename}-*.csv')))
                    - set(glob.glob(join(classifications_dir, f'{base_filename}_*.csv'))))
                   | set(glob.glob(join(classifications_dir, f'{base_filename}.csv'))))
    else:
        matches = set(glob.glob(join(classifications_dir, f'{base_filename}_{seed}*.csv')))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def predict_single_csv_path(source_path, name, seed, main_band, mef, classifications_dir):
    """Best-effort prediction of the CSV a previous 1-by-1 session for this
    dataset wrote, so a pre-launch scheme check (BUGS.md #15/#14) can look at
    it before the viewer does. Replicates single_viewer.py's own obtain_df()
    base_filename formula and glob patterns; see predict_mosaic_csv_path for
    why "most recently modified match" stands in for obtain_df()'s own
    file-selection logic.
    """
    name = name or ''
    n_images = count_band_images(source_path, main_band, mef=mef)
    base_filename = f'classification_single_{name}_{n_images}'
    if seed is None:
        matches = ((set(glob.glob(join(classifications_dir, f'{base_filename}-*.csv')))
                    - set(glob.glob(join(classifications_dir, f'{base_filename}_*.csv'))))
                   | set(glob.glob(join(classifications_dir, f'{base_filename}.csv'))))
    else:
        base_filename = f'{base_filename}_{seed}'
        matches = (set(glob.glob(join(classifications_dir, f'{base_filename}-*.csv'))) |
                   set(glob.glob(join(classifications_dir, f'{base_filename}.csv'))))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def build_band_argv(main_band, color_bands, composites, mef=False):
    """Shared --mef/-b/-B/--rgb-composites fragment, appended to both tools' argv.

    Empty -B/--rgb-composites values are passed explicitly rather than omitted:
    "no color bands" and "no composites" are states the lobby can be in
    deliberately (and the latter is forced on any dataset with fewer than 3 FITS
    bands), so falling back to the tool's Y,J,H / H,Y,I;H,J,Y defaults would
    launch a viewer that exits with "band directory not found" before its window
    opens. Only an unset main band is omitted -- there "unset" means no path has
    been scanned yet, and -b '' would name a band directory that cannot exist.
    """
    argv = ['--mef'] if mef else []
    main_band = (main_band or '').strip()
    if main_band:
        argv += ["-b", main_band]
    color_bands = [b.strip() for b in color_bands if b.strip()]
    argv += ["-B", ",".join(color_bands)]
    composite_terms = [",".join(b.strip() for b in triple)
                        for triple in composites if all(b.strip() for b in triple)]
    argv += ["--rgb-composites", ";".join(composite_terms)]
    return argv


def build_mosaic_argv(path, name, seed, ncols, nrows, classifications_dir, printname=False, band_argv=None):
    argv = [join(REPO_ROOT, "mosaic.py"),
            "-p", path, "-l", str(ncols), "-m", str(nrows)]
    if name:
        argv += ["-N", name]
    if seed is not None:
        argv += ["-s", str(seed)]
    # Passed explicitly either way: mosaic prints names by default, so leaving the
    # flag out would not turn the printing off.
    argv += ["--printname" if printname else "--no-printname"]
    # Passed explicitly either way, already resolved to an absolute path by the
    # caller: the child inherits the lobby's own cwd (no cwd override any more),
    # so leaving this to the viewer's own default would resolve ./Classifications
    # against whatever directory happens to be current when the child starts,
    # not necessarily what the lobby resolved it against.
    argv += ["--classifications-dir", classifications_dir]
    argv += band_argv or []
    return argv


def build_single_argv(path, name, seed, classifications_string, classifications_dir, band_argv=None):
    argv = [join(REPO_ROOT, "single_viewer.py"),
            "-p", path, "--classifications", classifications_string]
    if name:
        argv += ["-N", name]
    if seed is not None:
        argv += ["-s", str(seed)]
    # Passed explicitly either way: see build_mosaic_argv's --classifications-dir comment.
    argv += ["--classifications-dir", classifications_dir]
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


class AddUnknownClassificationDialog(QtWidgets.QDialog):
    """Details needed to add a classification label found in a resumed CSV --
    but missing from the current scheme (BUGS.md #15/#14) -- as a new scheme
    row. `label` and `kind` ('major' or 'subclass') name what was found; the
    label itself is fixed (it must keep matching the CSV), only the shortcut
    key, and for a major an optional subclass, are asked for here.
    """

    def __init__(self, label, kind, known_majors, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setWindowTitle(f"Add '{label}' to the classification scheme")
        form = QtWidgets.QFormLayout(self)

        form.addRow("Label:", QtWidgets.QLabel(label))

        self.major_combo = None
        if kind == 'subclass':
            self.major_combo = QComboBox()
            self.major_combo.setEditable(True)
            self.major_combo.addItems(known_majors)
            form.addRow("Parent major:", self.major_combo)

        self.key_edit = QtWidgets.QLineEdit()
        form.addRow("Keyboard shortcut:", self.key_edit)

        self.sub_cb = None
        self.sub_name_edit = None
        self.sub_key_edit = None
        if kind == 'major':
            self.sub_cb = QCheckBox("Also add a subclass under this major")
            form.addRow(self.sub_cb)
            self.sub_name_edit = QtWidgets.QLineEdit()
            self.sub_name_edit.setEnabled(False)
            form.addRow("Subclass name:", self.sub_name_edit)
            self.sub_key_edit = QtWidgets.QLineEdit()
            self.sub_key_edit.setEnabled(False)
            form.addRow("Subclass shortcut:", self.sub_key_edit)
            self.sub_cb.toggled.connect(self.sub_name_edit.setEnabled)
            self.sub_cb.toggled.connect(self.sub_key_edit.setEnabled)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def key(self):
        return self.key_edit.text().strip()

    def parent_major(self):
        return self.major_combo.currentText().strip() if self.major_combo else ''

    def wants_subclass(self):
        return bool(self.sub_cb and self.sub_cb.isChecked() and self.sub_name_edit.text().strip())

    def subclass_name(self):
        return self.sub_name_edit.text().strip() if self.sub_name_edit else ''

    def subclass_key(self):
        return self.sub_key_edit.text().strip() if self.sub_key_edit else ''


class RecentSessionsDialog(QtWidgets.QDialog):
    """Picker over the lobby's own saved sessions (BUGS.md item 14).

    Lists `state.list_sessions(classifications_dir, tool=SESSION_TOOL)`,
    most-recently-opened first. Read-only, single-row-select: this is a
    picker, not an editor. `Open` (also double-click) leaves the chosen entry
    on `self.chosen_entry` for the caller to apply to its own widgets; `Forget`
    removes a row's saved record via `state.forget_session` and refreshes.
    """

    COL_NAME, COL_PATH, COL_SEED, COL_LAST_OPENED = range(4)

    def __init__(self, classifications_dir, parent=None):
        super().__init__(parent)
        self.classifications_dir = classifications_dir
        self.chosen_entry = None
        self._entries = []
        self.setWindowTitle("Recent sessions")
        self.resize(720, 420)

        vbox = QtWidgets.QVBoxLayout(self)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Session name", "Data path", "Seed", "Last opened"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.itemDoubleClicked.connect(self._on_open)
        vbox.addWidget(self.table)

        btn_row = QtWidgets.QHBoxLayout()
        self.open_btn = QtWidgets.QPushButton("Open")
        self.open_btn.clicked.connect(self._on_open)
        self.forget_btn = QtWidgets.QPushButton("Forget")
        self.forget_btn.clicked.connect(self._on_forget)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.open_btn)
        btn_row.addWidget(self.forget_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)
        vbox.addLayout(btn_row)

        self._reload()

    def _reload(self):
        self._entries = state.list_sessions(self.classifications_dir, tool=SESSION_TOOL)
        self.table.setRowCount(len(self._entries))
        for row, entry in enumerate(self._entries):
            name = entry.get('name') or '(unnamed)'
            display_path = paths.resolve_against(entry['path'], self.classifications_dir)
            seed = entry.get('seed')
            seed_text = '--' if seed is None else str(seed)
            last_opened = entry.get('last_opened') or ''
            for col, text in enumerate((name, display_path, seed_text, last_opened)):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, col, item)

    def _selected_entry(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self._entries):
            return None
        return self._entries[row]

    def _on_open(self, *_args):
        entry = self._selected_entry()
        if entry is None:
            return
        self.chosen_entry = entry
        self.accept()

    def _on_forget(self):
        entry = self._selected_entry()
        if entry is None:
            return
        session_id = state.session_id_from_entry(entry, self.classifications_dir)
        if session_id is not None:
            state.forget_session(session_id, self.classifications_dir)
        self._reload()


class LobbyWindow(QtWidgets.QMainWindow):
    def __init__(self, state_dir_override=None):
        super().__init__()
        self.setWindowTitle("Qt-stamp-visualizer Lobby")

        self.state_dir_override = state_dir_override
        (self.config_path, self.preset_dirs,
         self.preset_write_dir) = lobby_state(state_dir_override)

        self.defaults = DEFAULT_CONFIG
        self.config_dict = self.load_dict()

        self._launched_process = None
        self.stage1_proc = None
        self._stage1_extraction_context = None
        self.available_bands = []
        self.fits_bands = []
        self.mef = False

        self._build_ui()
        self._apply_config_to_widgets()
        self._applied_identity = self._identity_tuple()
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
            self.preset_dirs, self.preset_write_dir,
            self._preset_snapshot, self._apply_preset)
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
        self.path_edit.editingFinished.connect(self._on_identity_changed)
        self.path_browse_btn = QtWidgets.QPushButton("Browse...")
        self.path_browse_btn.clicked.connect(self.on_browse_path)
        path_row.addWidget(self.path_edit)
        path_row.addWidget(self.path_browse_btn)
        form.addRow("Path to images:", path_row)

        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.editingFinished.connect(self._on_identity_changed)
        form.addRow("Session name:", self.name_edit)

        classifications_row = QtWidgets.QHBoxLayout()
        self.classifications_edit = QtWidgets.QLineEdit()
        self.classifications_edit.editingFinished.connect(self._on_identity_changed)
        self.classifications_browse_btn = QtWidgets.QPushButton("Browse...")
        self.classifications_browse_btn.clicked.connect(self.on_browse_classifications)
        classifications_row.addWidget(self.classifications_edit)
        classifications_row.addWidget(self.classifications_browse_btn)
        form.addRow("Classifications dir:", classifications_row)

        seed_row = QtWidgets.QHBoxLayout()
        self.seed_enabled_cb = QCheckBox("Use seed")
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 1_000_000)
        self.seed_enabled_cb.toggled.connect(self.seed_spin.setEnabled)
        self.seed_enabled_cb.toggled.connect(self._on_identity_changed)
        self.seed_spin.valueChanged.connect(self._on_identity_changed)
        seed_row.addWidget(self.seed_enabled_cb)
        seed_row.addWidget(self.seed_spin)
        form.addRow("Seed:", seed_row)

        self.recent_sessions_btn = QtWidgets.QPushButton("Recent sessions...")
        self.recent_sessions_btn.clicked.connect(self.on_recent_sessions_clicked)
        form.addRow(self.recent_sessions_btn)

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
        self.mef = bool(path) and fits_io.is_mef_dataset(path)
        self.available_bands = discover_bands(path)
        # MEF bands are always FITS (every listed extension holds image data);
        # directory bands need the per-directory check since formats can mix.
        self.fits_bands = (list(self.available_bands) if self.mef else
                            [b for b in self.available_bands
                             if fits_io.detect_band_filetype(join(path, b)) == 'FITS'])
        self._reconcile_main_band()
        self._refresh_band_widgets()
        self._prune_missing_composite_bands()
        self._update_composites_availability()
        if not path:
            self.bands_status_label.setText(
                "Set a path above, then Rescan bands -- subdirectories of the path become available bands.")
        elif self.available_bands:
            label = "Available extensions" if self.mef else "Available bands"
            self.bands_status_label.setText(f"{label}: " + ", ".join(self.available_bands))
        elif self.mef:
            self.bands_status_label.setText(f"No image extensions found in the first FITS file under {path}.")
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
                                self._composite_rows(),
                                mef=self.mef)

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
            where = "as extensions of the sample FITS file" if self.mef else "as subdirectories of the data path"
            self.log(f"Warning: band(s) not found {where}: {', '.join(missing)}")

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

    # ------------------------------------------------ scheme/CSV reconciliation

    def resolve_single_classifications_string(self, path, name, seed, classifications_dir):
        """Classifications string for launching the 1-by-1 viewer against
        `path`/`name`/`seed`, after checking whether a CSV it would resume
        holds a classification the current scheme no longer declares
        (BUGS.md #15, caused by #14): the 1-by-1 stores the scheme's own
        major/sub names, so editing the scheme after that CSV was written
        leaves a row labelled with something no button renders any more. The
        mosaic is not affected -- it stores numeric codes, not scheme names.

        Checked here, before launch, because the lobby is the only one of the
        three tools with a scheme editor to offer adding the label to -- the
        viewer itself can only report that the label it found isn't in its
        scheme (see single_viewer.py's ApplicationWindow._button_for_grade).
        """
        main_band = (self.main_band_combo.currentText() or '').strip()
        csv_path = predict_single_csv_path(path, name, seed, main_band, self.mef, classifications_dir)
        if csv_path:
            unknown_majors, unknown_subs = self._unknown_classification_labels(csv_path)
            for label in unknown_majors:
                self._prompt_unknown_classification(label, 'major')
            for label in unknown_subs:
                self._prompt_unknown_classification(label, 'subclass')
        classifications_string, _ = self.build_classifications_string()
        return classifications_string

    def _unknown_classification_labels(self, csv_path):
        """(unknown majors, unknown subs) found in `csv_path` but absent from
        the scheme table -- sorted, sentinel values ('Empty'/'None'/blank)
        excluded. Best-effort: an unreadable or malformed CSV yields nothing
        rather than blocking the launch over it."""
        try:
            df = pd.read_csv(csv_path, index_col=0)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            return [], []

        rows = self._scheme_table_to_rows()
        known_majors = {r['major'] for r in rows if r['type'] == 'major'}
        known_subs = {r['sub'] for r in rows if r['type'] == 'subclass'}
        sentinels = {'Empty', 'None', ''}

        def unknown_values(column, known):
            if column not in df.columns:
                return []
            values = {str(v).strip() for v in df[column].dropna()}
            return sorted(values - known - sentinels)

        return unknown_values('classification', known_majors), unknown_values('subclassification', known_subs)

    def _prompt_unknown_classification(self, label, kind):
        "Asks whether to add `label` (a 'major' or 'subclass') to the scheme, or launch without it."
        noun = "major" if kind == 'major' else "subclass"
        choice = QtWidgets.QMessageBox.question(
            self, "Classification not in scheme",
            f"The saved classifications for this session include '{label}', a {noun} that "
            "isn't in the current scheme. Add it to the scheme now, or ignore it? (The row "
            f"keeps '{label}' either way, until it's reclassified -- which overwrites it.)",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Ignore,
            QtWidgets.QMessageBox.Yes)
        if choice != QtWidgets.QMessageBox.Yes:
            self.log(f"Ignored unknown {noun} '{label}' -- launching without a button for it.")
            return

        known_majors = sorted({r['major'] for r in self._scheme_table_to_rows() if r['type'] == 'major'})
        dialog = AddUnknownClassificationDialog(label, kind, known_majors, parent=self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            self.log(f"Ignored unknown {noun} '{label}' -- launching without a button for it.")
            return

        if kind == 'major':
            self.add_scheme_row(TYPE_MAJOR, major=label, sub='', key=dialog.key())
            if dialog.wants_subclass():
                self.add_scheme_row(TYPE_SUBCLASS, major=label,
                                     sub=dialog.subclass_name(), key=dialog.subclass_key())
        else:
            parent = dialog.parent_major()
            if not parent:
                self.log(f"No parent major given for subclass '{label}' -- skipped, "
                         "launching without a button for it.")
                return
            self.add_scheme_row(TYPE_SUBCLASS, major=parent, sub=label, key=dialog.key())
        self.log(f"Added {noun} '{label}' to the scheme.")

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
            self._on_identity_changed()

    def on_browse_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output path")
        if path:
            self.output_path_edit.setText(path)

    def on_browse_classifications(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select classifications directory")
        if path:
            self.classifications_edit.setText(path)

    def _current_seed(self):
        return self.seed_spin.value() if self.seed_enabled_cb.isChecked() else None

    def _classifications_dir(self):
        return resolve_classifications_dir(self.classifications_edit.text().strip())

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
        # The identity widgets are in here since BUGS.md item 14: the scheme and
        # band setup now follow the session name/seed, so letting those change
        # while stage 1 is running would swap the scheme out from under the
        # stage 2 launch that `on_stage1_finished` builds from the *captured*
        # context -- the two would then disagree about which session is open.
        for widget in (self.run_btn, self.mode_combo, self.path_edit,
                       self.path_browse_btn, self.bands_group,
                       self.mosaic_group, self.single_group,
                       self.name_edit, self.seed_enabled_cb, self.seed_spin,
                       self.classifications_edit, self.classifications_browse_btn,
                       self.recent_sessions_btn):
            widget.setEnabled(enabled)
        # The seed spin box follows its checkbox, not the blanket re-enable.
        if enabled:
            self.seed_spin.setEnabled(self.seed_enabled_cb.isChecked())

    # ---------------------------------------------------------- launching

    def on_run_clicked(self):
        mode = self.mode_combo.currentIndex()
        path = self.path_edit.text().strip()
        name = self.name_edit.text().strip()
        seed = self._current_seed()

        if not path:
            self.log("Please select a data path first.")
            return
        path = os.path.abspath(os.path.expanduser(path))

        self._save_session_record()
        self._log_band_warnings()
        band_argv = self._band_argv()
        classifications_dir = self._classifications_dir()

        if mode == MODE_MOSAIC_ONLY:
            argv = build_mosaic_argv(path, name, seed,
                                      self.ncols_spin.value(), self.nrows_spin.value(),
                                      classifications_dir,
                                      printname=self.printname_cb.isChecked(),
                                      band_argv=band_argv)
            self._launch_fire_and_forget(argv)
        elif mode == MODE_SINGLE_ONLY:
            classifications_string = self.resolve_single_classifications_string(
                path, name, seed, classifications_dir)
            argv = build_single_argv(path, name, seed, classifications_string,
                                      classifications_dir, band_argv=band_argv)
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
        proc.setProgram(sys.executable)
        proc.setArguments(argv)
        self._wire_process_output_logging(proc)
        proc.finished.connect(lambda code, status: self.log(f"Process exited (code {code})."))
        self.log(f"Launching: {sys.executable} {' '.join(argv)}")
        proc.start()
        self._launched_process = proc

    def _launch_stage1_chained(self, path, name, seed):
        ncols, nrows = self.ncols_spin.value(), self.nrows_spin.value()
        classifications_dir = self._classifications_dir()
        argv = build_mosaic_argv(path, name, seed, ncols, nrows,
                                  classifications_dir,
                                  printname=self.printname_cb.isChecked(),
                                  band_argv=self._band_argv())
        self._stage1_extraction_context = {
            'source_path': path,
            'name': name,
            'seed': seed,
            'ncols': ncols,
            'nrows': nrows,
            'classifications_dir': classifications_dir,
        }
        self.stage1_proc = QProcess(self)
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
        classifications_dir = context.get('classifications_dir', self._classifications_dir())

        csv_path = predict_mosaic_csv_path(
            source_path, context.get('name'), context.get('seed'),
            classifications_dir)
        if csv_path:
            self.log(f"Auto-detected classification CSV: {csv_path}")
        else:
            self.log("Could not auto-detect the classification CSV -- please select it.")
            csv_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select the classification CSV produced by the mosaic tool",
                classifications_dir, "CSV files (*.csv)")
            if not csv_path:
                self.log("Extraction cancelled -- no CSV selected.")
                return

        output_path = self.output_path_edit.text().strip()
        if not output_path:
            self.log("Please set an output path before running the chained workflow.")
            return
        output_path = os.path.abspath(os.path.expanduser(output_path))

        positive_values = self._mosaic_positive_values()

        result = extraction.extract(
            csv_path, source_path, output_path, positive_values,
            use_symlink=not self.copy_instead_cb.isChecked(),
            on_progress=self.log,
        )
        self.log(f"Extraction complete. {result.summary()}")

        classifications_string = self.resolve_single_classifications_string(
            output_path, context.get('name'), context.get('seed'), classifications_dir)
        self._save_session_record()
        argv = build_single_argv(output_path, context.get('name'), context.get('seed'),
                                  classifications_string, classifications_dir,
                                  band_argv=self._band_argv())
        self._launch_fire_and_forget(argv)

    def on_extract_only_clicked(self):
        csv_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select classification CSV",
            self._classifications_dir(), "CSV files (*.csv)")
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
        self.classifications_edit.setText(c['classifications_path'])
        self.name_edit.setText(c['session_name'])
        self.seed_enabled_cb.setChecked(c['seed_enabled'])
        self.seed_spin.setValue(c['seed_value'])
        self.seed_spin.setEnabled(c['seed_enabled'])
        self.mode_combo.setCurrentIndex(c['run_mode_index'])
        self.printname_cb.setChecked(c['mosaic_printname'])
        self.mosaic_uninteresting_positive_cb.setChecked(c['mosaic_uninteresting_positive'])
        self.mosaic_lens_positive_cb.setChecked(c['mosaic_lens_positive'])
        self.mosaic_interesting_positive_cb.setChecked(c['mosaic_interesting_positive'])
        self.copy_instead_cb.setChecked(c['copy_instead_of_symlink'])
        self._apply_session_config_to_widgets()

        if c.get('dock_state'):
            self.restoreState(QByteArray.fromBase64(c['dock_state'].encode('ascii')))

    def _apply_session_config_to_widgets(self):
        """Applies exactly the SESSION_CONFIG_KEYS to their widgets: the
        scheme table, band setup, and mosaic grid shape (BUGS.md item 14).

        Deliberately narrower than `_apply_config_to_widgets`, which calls
        this rather than duplicating these lines so the two cannot drift.
        Never touches path/name/seed/mode/output/extraction widgets, and never
        calls `restoreState` -- a session-record restore must not jolt the
        dock layout.
        """
        c = self.config_dict
        self._populate_scheme_table(c['scheme_rows'])
        self.main_band_combo.setCurrentText(c['main_band'])
        self._populate_color_bands_list(self.available_bands, set(c['color_bands']))
        self.composites_table.setRowCount(0)
        for triple in c['rgb_composites']:
            r, g, b = (list(triple) + ['', '', ''])[:3]
            self._add_composite_row(r, g, b)
        self.ncols_spin.setValue(c['mosaic_ncols'])
        self.nrows_spin.setValue(c['mosaic_nrows'])

    def _sync_widgets_to_config(self):
        c = self.config_dict
        c['data_path'] = self.path_edit.text()
        c['output_path'] = self.output_path_edit.text()
        c['classifications_path'] = self.classifications_edit.text()
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
        """Persist the lobby's config to the state dir. True if it was written.

        Best-effort on purpose (2b-v): the state dir is now whatever directory
        the user launched the lobby from, which may be read-only. Losing the
        remembered paths and band setup is an acceptable cost of local-first;
        refusing to close the window over it is not.
        """
        self._sync_widgets_to_config()
        try:
            if not paths.ensure_dir(os.path.dirname(self.config_path)):
                print(f"Warning: {os.path.dirname(self.config_path)} is not writable; "
                      "lobby settings will not be remembered.", file=sys.stderr)
                return False
            with open(self.config_path, 'w') as f:
                json.dump(self.config_dict, f, ensure_ascii=False, indent=4)
        except OSError as exc:
            print(f"Warning: could not save the lobby config to {self.config_path}: {exc}",
                  file=sys.stderr)
            return False
        return True

    def load_dict(self):
        return load_config_dict(state_dir_override=self.state_dir_override)

    def closeEvent(self, event):
        sid = self._session_id()
        if sid is not None and state.load_session_config(sid, self._classifications_dir()) is not None:
            self._save_session_record()
        self.save_dict()
        event.accept()

    # ------------------------------------------------- per-session record (BUGS.md 14)
    #
    # The lobby's own record in the shared sessions.json, keyed on
    # SessionId('lobby', data_path, session_name, seed) and anchored on the
    # classifications dir like everything else in state.py. It carries only
    # SESSION_CONFIG_KEYS -- the scheme and band setup describe the dataset,
    # not the user, so they travel with the session rather than living in the
    # one workspace-wide config_lobby.json every session shares. See
    # state.py's module docstring and STEP.md for the full design.

    def _identity_tuple(self):
        "The (path, name, seed, classifications_dir) the widgets currently show."
        return (self.path_edit.text().strip(), self.name_edit.text().strip(),
                self._current_seed(), self._classifications_dir())

    def _session_id(self):
        "This session's identity, or None when there is no data path to key on."
        path = self.path_edit.text().strip()
        if not path:
            return None
        path = os.path.abspath(os.path.expanduser(path))
        return state.SessionId(SESSION_TOOL, path, self.name_edit.text().strip(),
                               self._current_seed())

    def _save_session_record_for(self, session_id, classifications_dir):
        self._sync_widgets_to_config()
        state.save_session_config(session_id, classifications_dir,
                                  {k: self.config_dict[k] for k in SESSION_CONFIG_KEYS})

    def _save_session_record(self):
        "No-op without an identity -- see the module-level docstring."
        session_id = self._session_id()
        if session_id is None:
            return
        self._save_session_record_for(session_id, self._classifications_dir())

    def _restore_session_record(self):
        "Applies this identity's saved record, if any, to the six session widgets."
        session_id = self._session_id()
        if session_id is None:
            return
        classifications_dir = self._classifications_dir()
        config = state.load_session_config(session_id, classifications_dir)
        if config is None:
            return
        self.config_dict.update({k: config[k] for k in SESSION_CONFIG_KEYS if k in config})
        self._apply_session_config_to_widgets()
        self.log(f"Restored saved session config for "
                 f"'{session_id.name or '(unnamed)'}' at {session_id.path}.")

    def _on_identity_changed(self):
        """Wired to every widget that is part of the session identity.

        `editingFinished` fires on mere focus-out even with no change, and
        restoring here on every such event would silently throw away scheme
        edits the user just made -- so this does nothing at all unless the
        identity actually differs from the one currently applied. When it
        does: the outgoing identity's record is saved first (if it has one),
        then bands are rescanned for the (possibly new) path, and only then is
        the incoming identity's record restored.

        That order is load-bearing. `_rescan_bands` rebuilds the colour-band
        checklist from the boxes currently ticked in it, and
        `_apply_session_config_to_widgets` can only tick a band that the list
        already offers -- i.e. one belonging to the *scanned* path. Restoring
        before the rescan therefore ticks the incoming session's bands against
        the outgoing path's band list, which drops every band the two datasets
        do not share, and the rescan then reads that emptied list back as the
        answer. Two datasets with different bands is exactly the case this
        record exists for, so it is also exactly the case that broke.
        """
        identity = self._identity_tuple()
        if identity == self._applied_identity:
            return
        old_path, old_name, old_seed, old_classifications_dir = self._applied_identity
        if old_path:
            outgoing_id = state.SessionId(SESSION_TOOL, old_path, old_name, old_seed)
            if state.load_session_config(outgoing_id, old_classifications_dir) is not None:
                self._save_session_record_for(outgoing_id, old_classifications_dir)
        self._rescan_bands()
        self._restore_session_record()
        self._applied_identity = identity

    def on_recent_sessions_clicked(self):
        classifications_dir = self._classifications_dir()
        if not state.list_sessions(classifications_dir, tool=SESSION_TOOL):
            self.log(f"No saved sessions in {classifications_dir} yet.")
            return
        dialog = RecentSessionsDialog(classifications_dir, parent=self)
        if dialog.exec() != QtWidgets.QDialog.Accepted or dialog.chosen_entry is None:
            return
        entry = dialog.chosen_entry
        seed = entry.get('seed')
        self.path_edit.setText(paths.resolve_against(entry['path'], classifications_dir))
        self.name_edit.setText(entry.get('name') or '')
        # Signals blocked while the seed widgets are updated so
        # _on_identity_changed runs exactly once, in one place, below --
        # rather than once per widget with an intermediate (wrong) seed value.
        self.seed_enabled_cb.blockSignals(True)
        self.seed_spin.blockSignals(True)
        self.seed_enabled_cb.setChecked(seed is not None)
        self.seed_spin.setEnabled(seed is not None)
        self.seed_spin.setValue(seed if seed is not None else 0)
        self.seed_enabled_cb.blockSignals(False)
        self.seed_spin.blockSignals(False)
        self._on_identity_changed()


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
    """Lobby config dict for the headless CLI (BUGS.md item 14).

    Precedence, low to high: the config file (`--config`, or the usual
    search path) -> the identity overrides (`--path`, `--name`,
    `--seed`/`--no-seed`, `--classifications-dir`) -> this session's saved
    record, if it has one, for the six SESSION_CONFIG_KEYS it owns -> the
    remaining overrides (`--main-band`, `--color-bands`, `--rgb-composites`,
    `--ncols`, `--nrows`, and everything else).

    That ordering is what makes an explicit CLI flag beat the record while the
    record still beats the config file's own, possibly-stale copy of those six
    keys: the identity has to be resolved first (from the file, then bumped by
    `--path`/`--name`/`--seed`/`--classifications-dir`) before a record can
    even be looked up, and a later override of e.g. `--main-band` must still
    win over whatever the record says. No data path -> no identity -> the
    record lookup is skipped entirely, same as the GUI's `_session_id`.
    """
    c = load_config_dict(args.config, resolve_state_dir_override(args))

    # Identity overrides: resolved before the record lookup below, since the
    # record is keyed on the *effective* identity, not the config file's.
    if args.path is not None:
        c['data_path'] = args.path
    if args.classifications_dir is not None:
        c['classifications_path'] = args.classifications_dir
    if args.name is not None:
        c['session_name'] = args.name
    if args.seed is not None:
        c['seed_enabled'], c['seed_value'] = True, args.seed
    if args.no_seed:
        c['seed_enabled'] = False

    path = c['data_path']
    if path:
        session_id = state.SessionId(
            SESSION_TOOL, os.path.abspath(os.path.expanduser(path)), c['session_name'],
            c['seed_value'] if c['seed_enabled'] else None)
        classifications_dir = resolve_classifications_dir(c['classifications_path'])
        record = state.load_session_config(session_id, classifications_dir)
        if record is not None:
            c.update({k: record[k] for k in SESSION_CONFIG_KEYS if k in record})

    # Remaining overrides: applied last, so an explicit flag always wins even
    # over a session record that covers the same key (main band, bands, grid).
    if args.mode is not None:
        c['run_mode_index'] = MODE_NAMES[args.mode]
    if args.output is not None:
        c['output_path'] = args.output
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
                 log=print, state_dir_override=None):
    """Run the configured workflow without the lobby window; returns an exit code.

    With print_command_only, prints the viewer command line(s) the lobby would
    launch -- chained mode prints both stages -- and returns 0 without running
    anything.

    `state_dir_override` only reaches the printed command lines. A child that is
    actually launched inherits it through `QTSTAMP_STATE_DIR` in the environment
    (see `main`), but a printed command line has to stand on its own -- someone
    pasting it into a different shell would otherwise silently get the local
    default back.
    """
    mode = config['run_mode_index']
    path = config['data_path']
    path = os.path.abspath(os.path.expanduser(path)) if path else path
    name = config['session_name']
    seed = config['seed_value'] if config['seed_enabled'] else None
    band_argv = build_band_argv(config['main_band'], config['color_bands'],
                                config['rgb_composites'], mef=fits_io.is_mef_dataset(path))
    classifications_dir = resolve_classifications_dir(config['classifications_path'])

    if classifications_override is not None:
        classifications_string = classifications_override
    else:
        classifications_string, _ = classifications_string_from_rows(
            config['scheme_rows'], log=log)

    def emit(argv):
        if state_dir_override:
            argv = [*argv, "--state-dir", state_dir_override]
        log(shlex.join([sys.executable, *argv]))

    def run(argv):
        log("Launching: " + shlex.join([sys.executable, *argv]))
        return subprocess.run([sys.executable, *argv]).returncode

    if print_command_only:
        path = path or "<path>"
    elif not path:
        log("No data path -- pass --path or set 'data_path' in the lobby config.")
        return 2

    if mode == MODE_MOSAIC_ONLY:
        argv = build_mosaic_argv(path, name, seed, config['mosaic_ncols'],
                                 config['mosaic_nrows'], classifications_dir,
                                 printname=config['mosaic_printname'],
                                 band_argv=band_argv)
        if print_command_only:
            emit(argv)
            return 0
        return run(argv)

    if mode == MODE_SINGLE_ONLY:
        argv = build_single_argv(path, name, seed, classifications_string,
                                 classifications_dir, band_argv=band_argv)
        if print_command_only:
            emit(argv)
            return 0
        return run(argv)

    # chained: mosaic -> extract positives -> 1-by-1
    ncols, nrows = config['mosaic_ncols'], config['mosaic_nrows']
    mosaic_argv = build_mosaic_argv(path, name, seed, ncols, nrows, classifications_dir,
                                    printname=config['mosaic_printname'],
                                    band_argv=band_argv)
    output_path = config['output_path']
    output_path = os.path.abspath(os.path.expanduser(output_path)) if output_path else output_path

    if print_command_only:
        emit(mosaic_argv)
        emit(build_single_argv(output_path or "<output_path>", name, seed,
                               classifications_string, classifications_dir, band_argv=band_argv))
        return 0

    if not output_path:
        log("Chained mode needs an output path -- pass --output or set 'output_path'.")
        return 2

    code = run(mosaic_argv)
    if code != 0:
        log(f"Mosaic stage exited with code {code} -- stopping before extraction.")
        return code

    csv_path = predict_mosaic_csv_path(path, name, seed, classifications_dir)
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
                                 classifications_dir, band_argv=band_argv))


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
    p.add_argument("--config", default=None, metavar="PATH",
                   help="Lobby config JSON to read defaults from (default: the first "
                        f"{LOBBY_CONFIG_NAME} found in the state dir, then the per-user "
                        "config dir, then the defaults shipped with the tool).")
    add_state_dir_args(p)

    g = p.add_argument_group("workflow overrides (with --no-gui / --print-command)")
    g.add_argument("-m", "--mode", choices=sorted(MODE_NAMES),
                   help="mosaic only, 1-by-1 (single) only, or mosaic->1-by-1 chained.")
    g.add_argument("-p", "--path", help="Path to the images to inspect.")
    g.add_argument("-o", "--output", help="Output path for chained-mode extraction.")
    g.add_argument("--classifications-dir", metavar="PATH",
                   help="Directory for the autosaved classification CSVs.")
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
        args.classifications_dir is not None,
        args.name is not None, args.seed is not None, args.no_seed,
        args.main_band is not None, args.color_bands is not None,
        args.rgb_composites is not None, args.classifications is not None,
        args.ncols is not None, args.nrows is not None,
        args.printname is not None, args.copy_files is not None,
    ])
    if gave_override and not headless:
        parser.error("workflow overrides only apply with --no-gui or --print-command")

    state_dir_override = resolve_state_dir_override(args)
    if state_dir_override:
        # Every viewer the lobby launches has to agree with it about where state
        # lives. Exporting it here covers all four launch paths at once (QProcess
        # copies this process's environment; subprocess inherits it) without
        # threading the value through nine build_*_argv call sites. The default
        # -- no override -- needs nothing: since 2b-ii the child inherits the
        # lobby's CWD, so it resolves `./.qtstamp` to the same directory.
        os.environ[paths.STATE_DIR_ENV] = state_dir_override

    if not headless:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        win = LobbyWindow(state_dir_override)
        win.show()
        sys.exit(app.exec())

    sys.exit(run_headless(config_from_cli(args),
                          classifications_override=args.classifications,
                          print_command_only=args.print_command,
                          state_dir_override=state_dir_override))


if __name__ == "__main__":
    main()
