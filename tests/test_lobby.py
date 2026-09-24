"""Tests for lobby.py's Qt-free pieces: the headless CLI's config precedence
(BUGS.md item 14) and the scheme-rows -> --classifications string translation.

Deliberately does not touch any widget -- LobbyWindow needs a QApplication,
and this suite has to run without a display. Written as plain pytest
functions, but also runnable directly (`python3 tests/test_lobby.py`).
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Same reasoning as test_state.py: pin the per-user config dir somewhere that
# cannot exist and make sure no state-dir env override is inherited, so a
# developer's own config_lobby.json/sessions.json can never leak into these
# tests through the default search path.
os.environ['QTSTAMP_CONFIG_DIR'] = os.path.join(
    tempfile.gettempdir(), 'qtstamp-test-no-such-user-dir')
os.environ.pop('QTSTAMP_STATE_DIR', None)

import lobby
import state


def _parse(argv):
    return lobby._build_arg_parser().parse_args(argv)


def _write_config(path, **fields):
    with open(path, 'w') as f:
        json.dump(fields, f)


FILE_SCHEME = [{'type': 'major', 'major': 'FROM_FILE', 'sub': '', 'key': '1', 'positive': False}]
RECORD_SCHEME = [{'type': 'major', 'major': 'FROM_RECORD', 'sub': '', 'key': '1', 'positive': False}]


def _save_record(data_path, name, seed, classifications_dir, **fields):
    session_id = state.SessionId(lobby.SESSION_TOOL, os.path.abspath(data_path), name, seed)
    base = {
        'scheme_rows': RECORD_SCHEME,
        'main_band': 'VIS',
        'color_bands': [],
        'rgb_composites': [],
        'mosaic_ncols': 5,
        'mosaic_nrows': 8,
    }
    base.update(fields)
    return state.save_session_config(session_id, classifications_dir, base)


# --- config_from_cli precedence -------------------------------------------

def test_session_record_beats_config_file_scheme_rows(tmpdir):
    config_path = os.path.join(tmpdir, 'config_lobby.json')
    _write_config(config_path, scheme_rows=FILE_SCHEME)
    data_path = os.path.join(tmpdir, 'data')
    os.makedirs(data_path)
    classifications_dir = os.path.join(tmpdir, 'Classifications')

    _save_record(data_path, '', None, classifications_dir)

    args = _parse(['--config', config_path, '--path', data_path,
                   '--classifications-dir', classifications_dir])
    c = lobby.config_from_cli(args)
    assert c['scheme_rows'] == RECORD_SCHEME


def test_explicit_main_band_beats_session_record(tmpdir):
    config_path = os.path.join(tmpdir, 'config_lobby.json')
    _write_config(config_path, main_band='VIS')
    data_path = os.path.join(tmpdir, 'data')
    os.makedirs(data_path)
    classifications_dir = os.path.join(tmpdir, 'Classifications')

    _save_record(data_path, '', None, classifications_dir, main_band='VIS')

    args = _parse(['--config', config_path, '--path', data_path,
                   '--classifications-dir', classifications_dir, '--main-band', 'H'])
    c = lobby.config_from_cli(args)
    assert c['main_band'] == 'H'


def test_different_name_does_not_pick_up_other_sessions_record(tmpdir):
    config_path = os.path.join(tmpdir, 'config_lobby.json')
    _write_config(config_path, scheme_rows=FILE_SCHEME)
    data_path = os.path.join(tmpdir, 'data')
    os.makedirs(data_path)
    classifications_dir = os.path.join(tmpdir, 'Classifications')

    _save_record(data_path, 'alice', None, classifications_dir)

    args = _parse(['--config', config_path, '--path', data_path,
                   '--classifications-dir', classifications_dir, '--name', 'bob'])
    c = lobby.config_from_cli(args)
    # No record for ('lobby', data_path, 'bob', None) -- falls back to the
    # config file's own copy of scheme_rows, not alice's session record.
    assert c['scheme_rows'] == FILE_SCHEME


def test_different_seed_does_not_pick_up_other_sessions_record(tmpdir):
    data_path = os.path.join(tmpdir, 'data')
    os.makedirs(data_path)
    classifications_dir = os.path.join(tmpdir, 'Classifications')

    _save_record(data_path, '', 1, classifications_dir)

    args = _parse(['--path', data_path, '--classifications-dir', classifications_dir,
                   '--seed', '2'])
    c = lobby.config_from_cli(args)
    assert c['scheme_rows'] != RECORD_SCHEME


def test_no_data_path_skips_record_lookup_entirely(tmpdir):
    classifications_dir = os.path.join(tmpdir, 'Classifications')
    args = _parse(['--classifications-dir', classifications_dir])
    # Must not raise even though no session identity can be formed.
    c = lobby.config_from_cli(args)
    assert c['data_path'] == ''


def test_record_is_applied_when_no_explicit_overrides_given(tmpdir):
    "A bare launch against a previously-run identity picks the record back up."
    config_path = os.path.join(tmpdir, 'config_lobby.json')
    _write_config(config_path, scheme_rows=FILE_SCHEME)
    data_path = os.path.join(tmpdir, 'data')
    os.makedirs(data_path)
    classifications_dir = os.path.join(tmpdir, 'Classifications')

    _save_record(data_path, 'carol', 7, classifications_dir)

    args = _parse(['--config', config_path, '--path', data_path,
                   '--classifications-dir', classifications_dir,
                   '--name', 'carol', '--seed', '7'])
    c = lobby.config_from_cli(args)
    assert c['scheme_rows'] == RECORD_SCHEME
    assert c['mosaic_ncols'] == 5 and c['mosaic_nrows'] == 8


# --- classifications_string_from_rows --------------------------------------

def test_classifications_string_from_rows_basic():
    rows = [
        {'type': 'major', 'major': 'A', 'sub': '', 'key': '1', 'positive': True},
        {'type': 'major', 'major': 'B', 'sub': '', 'key': '2', 'positive': False},
        {'type': 'subclass', 'major': 'A', 'sub': 'x', 'key': '3', 'positive': False},
    ]
    classifications_string, positive_majors = lobby.classifications_string_from_rows(rows)
    assert classifications_string == 'A=1;B=2;A:x=3'
    assert positive_majors == {'A'}


def test_classifications_string_from_rows_warns_on_unknown_major_and_duplicate_key():
    rows = [
        {'type': 'major', 'major': 'A', 'sub': '', 'key': '1', 'positive': False},
        {'type': 'subclass', 'major': 'Z', 'sub': 'x', 'key': '1', 'positive': False},
    ]
    warnings = []
    lobby.classifications_string_from_rows(rows, log=warnings.append)
    joined = ' '.join(warnings)
    assert 'unknown major' in joined
    assert 'keyboard shortcut' in joined


def test_classifications_string_from_rows_skips_rows_missing_major_or_key():
    rows = [
        {'type': 'major', 'major': '', 'sub': '', 'key': '1', 'positive': False},
        {'type': 'major', 'major': 'A', 'sub': '', 'key': '', 'positive': False},
        {'type': 'major', 'major': 'B', 'sub': '', 'key': '2', 'positive': False},
    ]
    classifications_string, _ = lobby.classifications_string_from_rows(rows)
    assert classifications_string == 'B=2'


def test_classifications_string_from_rows_shared_subclass():
    rows = [
        {'type': 'major', 'major': 'A', 'sub': '', 'key': '1', 'positive': False},
        {'type': 'major', 'major': 'B', 'sub': '', 'key': '2', 'positive': False},
        {'type': 'subclass', 'major': ' A , B ,', 'sub': 'Merger', 'key': 'm', 'positive': False},
    ]
    warnings = []
    classifications_string, _ = lobby.classifications_string_from_rows(rows, log=warnings.append)
    assert classifications_string == 'A=1;B=2;A,B:Merger=m'
    assert warnings == []


def test_classifications_string_from_rows_shared_subclass_keeps_order_drops_repeats():
    rows = [
        {'type': 'major', 'major': 'A', 'sub': '', 'key': '1', 'positive': False},
        {'type': 'major', 'major': 'B', 'sub': '', 'key': '2', 'positive': False},
        {'type': 'subclass', 'major': 'B,A,B', 'sub': 'Merger', 'key': 'm', 'positive': False},
    ]
    classifications_string, _ = lobby.classifications_string_from_rows(rows)
    assert classifications_string == 'A=1;B=2;B,A:Merger=m'


def test_classifications_string_from_rows_shared_subclass_warnings():
    rows = [
        {'type': 'major', 'major': 'A', 'sub': '', 'key': '1', 'positive': False},
        {'type': 'subclass', 'major': 'A,Z', 'sub': 'Merger', 'key': 'm', 'positive': False},
        {'type': 'subclass', 'major': 'A', 'sub': 'Merger', 'key': 'n', 'positive': False},
    ]
    warnings = []
    lobby.classifications_string_from_rows(rows, log=warnings.append)
    joined = ' '.join(warnings)
    assert "unknown major 'Z'" in joined
    assert "unknown major 'A'" not in joined
    assert "'Merger' is in 2 rows" in joined


if __name__ == '__main__':
    import inspect
    import tempfile
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
