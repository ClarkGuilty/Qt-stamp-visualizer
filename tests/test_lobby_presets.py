"""Widget-level tests for lobby presets (BUGS.md item 16).

Kept apart from test_lobby.py, which touches no widget at all: these build a
real LobbyWindow, on Qt's offscreen platform so they still need no display.
A preset must never move the data path, the output paths or the session, and
applying one must show only the bands found under the current path, alerting
when its band setup and those bands differ. Written as plain pytest
functions, but also runnable directly (`python3 tests/test_lobby_presets.py`).
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Same isolation as test_lobby.py, plus a platform that needs no display.
os.environ['QTSTAMP_CONFIG_DIR'] = os.path.join(
    tempfile.gettempdir(), 'qtstamp-test-no-such-user-dir')
os.environ.pop('QTSTAMP_STATE_DIR', None)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6 import QtWidgets

import lobby
import state

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _dataset(root, name, bands):
    "A directory-per-band dataset; empty band directories are enough for discovery."
    path = os.path.join(root, name)
    for band in bands:
        os.makedirs(os.path.join(path, band))
    return path


def _lobby(tmpdir, data_path):
    w = lobby.LobbyWindow(state_dir_override=os.path.join(tmpdir, 'state'))
    w.classifications_edit.setText(os.path.join(tmpdir, 'Classifications'))
    w.path_edit.setText(data_path)
    w._on_identity_changed()
    return w


class _Alerts:
    "Records QMessageBox.warning calls instead of opening a modal dialog."

    def __enter__(self):
        self.calls = []
        self._original = QtWidgets.QMessageBox.warning
        QtWidgets.QMessageBox.warning = (
            lambda parent, title, text, *args, **kwargs: self.calls.append((title, text)))
        return self

    def __exit__(self, *exc):
        QtWidgets.QMessageBox.warning = self._original


def _pick_preset(w, name, preset):
    "Saves `preset` where the bar looks and picks it, as a user would from the dropdown."
    os.makedirs(w.preset_write_dir, exist_ok=True)
    with open(os.path.join(w.preset_write_dir, f'{name}.json'), 'w') as f:
        json.dump(preset, f)
    w.preset_bar.refresh()
    w.preset_bar._on_activated(w.preset_bar.combo.findText(name))


def test_preset_never_changes_paths_or_session(tmpdir):
    a = _dataset(tmpdir, 'A', ['I', 'J'])
    b = _dataset(tmpdir, 'B', ['J', 'Y'])
    w = _lobby(tmpdir, a)
    w.output_path_edit.setText(os.path.join(tmpdir, 'out'))
    before = (w._identity_tuple(), w.output_path_edit.text(), w.seed_enabled_cb.isChecked())

    with _Alerts():
        w._apply_preset({'data_path': b, 'output_path': os.path.join(tmpdir, 'elsewhere'),
                         'classifications_path': os.path.join(tmpdir, 'other_cls'),
                         'session_name': 'other', 'seed_enabled': True, 'seed_value': 7,
                         'mosaic_ncols': 3})

    assert (w._identity_tuple(), w.output_path_edit.text(),
            w.seed_enabled_cb.isChecked()) == before
    assert w._applied_identity == w._identity_tuple()
    assert w.ncols_spin.value() == 3  # the portable part still applies
    assert 'Preset keys ignored' in w.log_view.toPlainText()


def test_preset_snapshot_holds_no_local_keys(tmpdir):
    w = _lobby(tmpdir, _dataset(tmpdir, 'A', ['I']))
    snapshot = w._preset_snapshot()
    assert not (lobby.PRESET_LOCAL_KEYS & set(snapshot))
    assert 'scheme_rows' in snapshot and 'color_bands' in snapshot


def test_preset_shows_only_found_bands_and_alerts_on_mismatch(tmpdir):
    w = _lobby(tmpdir, _dataset(tmpdir, 'A', ['I', 'J', 'Y']))
    with _Alerts() as alerts:
        _pick_preset(w, 'mine', {'main_band': 'J', 'color_bands': ['J', 'Y', 'H'],
                                 'rgb_composites': []})
    assert w._checked_color_bands() == ['J', 'Y']
    assert len(alerts.calls) == 1
    text = alerts.calls[0][1]
    assert 'not shown: H' in text
    assert 'not used by the preset: I' in text


def test_matching_preset_raises_no_alert(tmpdir):
    w = _lobby(tmpdir, _dataset(tmpdir, 'A', ['I', 'J', 'Y']))
    with _Alerts() as alerts:
        _pick_preset(w, 'mine', {'main_band': 'J', 'color_bands': ['I', 'Y'],
                                 'rgb_composites': []})
    assert w._checked_color_bands() == ['I', 'Y']
    assert alerts.calls == []


def test_preset_rescans_before_ticking(tmpdir):
    a = _dataset(tmpdir, 'A', ['I', 'J'])
    w = _lobby(tmpdir, a)
    os.makedirs(os.path.join(a, 'Y'))  # appears after the lobby's last scan
    with _Alerts() as alerts:
        _pick_preset(w, 'mine', {'main_band': 'I', 'color_bands': ['J', 'Y'],
                                 'rgb_composites': []})
    assert w._checked_color_bands() == ['J', 'Y']
    assert alerts.calls == []


def test_restore_previous_puts_back_bands_without_alerting(tmpdir):
    w = _lobby(tmpdir, _dataset(tmpdir, 'A', ['I', 'J', 'Y']))
    w._populate_color_bands_list(w.available_bands, {'I'})
    with _Alerts() as alerts:
        _pick_preset(w, 'mine', {'main_band': 'I', 'color_bands': ['J', 'Y'],
                                 'rgb_composites': []})
        w.preset_bar._on_restore_clicked()
    assert w._checked_color_bands() == ['I']
    assert alerts.calls == []  # the restored setup leaves J and Y unused, and says nothing


def test_old_preset_with_data_path_leaves_the_sessions_record_alone(tmpdir):
    # Item 16's second symptom: a preset used to move the path without moving
    # the applied identity, so the next focus-out saved it as A's record.
    a = _dataset(tmpdir, 'A', ['I', 'J'])
    b = _dataset(tmpdir, 'B', ['J', 'Y'])
    w = _lobby(tmpdir, a)
    w._populate_color_bands_list(w.available_bands, {'I'})
    a_id = w._session_id()
    w._save_session_record()

    with _Alerts():
        _pick_preset(w, 'old', {'data_path': b, 'main_band': 'J',
                                'color_bands': ['J', 'Y'], 'rgb_composites': []})
    w._on_identity_changed()

    record = state.load_session_config(a_id, w._classifications_dir())
    assert record['color_bands'] == ['I']
    assert w.path_edit.text() == a


def test_lens_type_preset_resumes_its_csv_without_unknown_labels(tmpdir):
    # A major button writes its own name into subclassification too, so a
    # bare 'A' there is known, not an unknown subclass to prompt about.
    shipped = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'qtstamp_defaults', 'presets', 'Lens_type_AGN.json')
    with open(shipped) as f:
        preset = json.load(f)
    w = _lobby(tmpdir, _dataset(tmpdir, 'pngs', ['png']))
    w._apply_preset(preset, warn=False)
    csv_path = os.path.join(tmpdir, 'c.csv')
    with open(csv_path, 'w') as f:
        f.write(',file_name,classification,subclassification\n'
                '0,a.png,A,A\n1,b.png,C,Lensing AGN\n2,c.png,B,Group/cluster\n'
                '3,d.png,"Lens, but wrong AGN","Lens, but wrong AGN"\n'
                '4,e.png,Not a lens,Not a lens\n5,f.png,A,Lensed type 2\n')
    assert w._unknown_classification_labels(csv_path) == ([], [])


if __name__ == '__main__':
    import inspect
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    failures = 0
    for name, func in tests:
        try:
            if 'tmpdir' in inspect.signature(func).parameters:
                with tempfile.TemporaryDirectory() as d:
                    func(d)
            else:
                func()
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
