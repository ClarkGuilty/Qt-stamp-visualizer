"""Tests for the contract between the lobby and the two viewers: the shared
dataset arguments (cli.py), the --classifications grammar, and the names of the
classification CSVs (imaging.py) -- each written by one tool and read by
another. BUGS.md 39 and 40 were the lobby's copy of the CSV naming drifting from
the viewers'.

Qt-free apart from importing lobby.py, which needs no QApplication. Written as
plain pytest functions, but also runnable directly (`python3 tests/test_cli.py`).
"""

import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['QTSTAMP_CONFIG_DIR'] = os.path.join(
    tempfile.gettempdir(), 'qtstamp-test-no-such-user-dir')
os.environ.pop('QTSTAMP_STATE_DIR', None)

from qtstamp import cli
from qtstamp import lobby
from qtstamp.imaging import (MOSAIC_CSV_PREFIX, SINGLE_CSV_PREFIX, classification_csv_base,
                     classification_csvs, resume_csv)


def _touch(path, mtime=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').close()
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _dataset(root, bands, n=3, ext='.png'):
    "A directory-per-band dataset of `n` empty stamps per band."
    for band in bands:
        for i in range(n):
            _touch(os.path.join(root, band, f'obj{i}{ext}'))
    return str(root)    # pytest's tmpdir is a LocalPath, which argparse cannot read


def _parse(argv):
    parser = argparse.ArgumentParser()
    cli.add_dataset_args(parser)
    args = parser.parse_args(argv)
    cli.finish_dataset_args(parser, args)
    return args


def _exit_code(func):
    try:
        func()
    except SystemExit as e:
        return e.code
    return None


# --- dataset arguments --------------------------------------------------------

def test_no_main_band_promotes_first_color_band(tmpdir):
    path = _dataset(tmpdir, ['g', 'r'])
    args = _parse(['-p', path, '-b', '', '-B', ' g, ,r', '--rgb-composites', ''])
    assert args.main_band == 'g'
    assert args.color_bands == ['r']
    assert args.composite_bands == [] and args.composite_band_members == {}


def test_main_band_is_dropped_from_color_bands(tmpdir):
    # BUGS.md 42: -b r -B r,g showed r twice in the panel pickers.
    path = _dataset(tmpdir, ['g', 'r'])
    args = _parse(['-p', path, '-b', 'r', '-B', 'r,g', '--rgb-composites', ''])
    assert (args.main_band, args.color_bands) == ('r', ['g'])
    args = _parse(['-p', path, '-b', '', '-B', 'g,r,g', '--rgb-composites', ''])
    assert (args.main_band, args.color_bands) == ('g', ['r'])
    args = _parse(['-p', path, '-b', 'r', '-B', 'r', '--rgb-composites', ''])
    assert (args.main_band, args.color_bands) == ('r', [])


def test_rgb_composites_keys_and_members(tmpdir):
    path = _dataset(tmpdir, ['g', 'r', 'z'])
    args = _parse(['-p', path, '-b', 'r', '-B', '', '--rgb-composites', 'z,r,g; g, r, z ;'])
    assert args.composite_bands == ['z,r,g', 'g,r,z']
    assert args.composite_band_members == {'z,r,g': ('z', 'r', 'g'), 'g,r,z': ('g', 'r', 'z')}


def test_composite_that_is_not_a_triple_is_a_usage_error(tmpdir):
    path = _dataset(tmpdir, ['g', 'r'])
    assert _exit_code(lambda: _parse(['-p', path, '-b', 'r', '-B', '',
                                      '--rgb-composites', 'g,r'])) == 2


def test_no_main_band_and_no_color_band_is_a_usage_error(tmpdir):
    path = _dataset(tmpdir, ['g'])
    assert _exit_code(lambda: _parse(['-p', path, '-b', '', '-B', '',
                                      '--rgb-composites', ''])) == 2


def test_missing_band_directory_exits_1_but_not_under_mef(tmpdir):
    path = _dataset(tmpdir, ['g'])
    assert _exit_code(lambda: _parse(['-p', path, '-b', 'g', '-B', 'nope',
                                      '--rgb-composites', ''])) == 1
    assert _parse(['-p', path, '--mef', '-b', 'g', '-B', 'nope']).color_bands == ['nope']


def test_state_dir_override_is_resolved(tmpdir):
    path = _dataset(tmpdir, ['g'])
    args = _parse(['-p', path, '-b', 'g', '-B', '', '--rgb-composites', '',
                   '--state-dir', 'rel/state'])
    assert args.state_dir_override == os.path.abspath('rel/state')


def test_reset_flags_forget_preferences_position_and_legacy_config(tmpdir):
    from qtstamp import state
    path = _dataset(os.path.join(tmpdir, 'data'), ['g'])
    cls = os.path.join(tmpdir, 'cls')
    args = _parse(['-p', path, '-b', 'g', '-B', '', '--rgb-composites', '', '-N', 'n', '-s', '3',
                   '--state-dir', os.path.join(tmpdir, 'state'), '--classifications-dir', cls,
                   '--reset-config', '--reset-position'])
    session = state.SessionId('mosaic', path, 'n', 3)
    state.save_preferences('mosaic', {'scale': 'sqrt'}, config_dir=args.state_dir_override)
    state.save_session(session, cls, position_filename='obj1.png', csv=None)
    legacy = _touch(os.path.join(tmpdir, 'legacy_config.json'))

    cli.apply_reset_flags(args, 'mosaic', legacy)

    assert not os.path.exists(state.preferences_path('mosaic', args.state_dir_override))
    assert state.load_session(session, cls) is None
    assert os.path.exists(legacy + '.bak') and not os.path.exists(legacy)


def test_lobby_band_argv_reads_back_as_written(tmpdir):
    "The lobby writes -b/-B/--rgb-composites; the viewers must read back what it meant."
    path = _dataset(tmpdir, ['VIS', 'Y', 'J', 'H'])
    argv = lobby.build_band_argv('VIS', ['Y', ' J'], [['H', 'J', 'Y'], ['H', '', 'Y']])
    args = _parse(['-p', path] + argv)
    assert (args.main_band, args.color_bands) == ('VIS', ['Y', 'J'])
    assert args.composite_bands == ['H,J,Y']    # the incomplete row is dropped

    args = _parse(['-p', path] + lobby.build_band_argv(lobby.NO_MAIN_BAND, ['J', 'H'], []))
    assert (args.main_band, args.color_bands, args.composite_bands) == ('J', ['H'], [])


# --- --classifications ----------------------------------------------------------

def test_parse_classifications_buttons():
    entries = cli.parse_classifications(" A=1;B=2;; X:Merger=a ; A, B ,A:Spiral=s;C:=c")
    major_classes, subclasses = cli.scheme_buttons(entries)
    assert major_classes == [('A', '1'), ('B', '2'), ('C', 'c')]
    assert subclasses == [(('X',), 'Merger', 'a'), (('A', 'B'), 'Spiral', 's')]


def test_duplicates_are_reported_in_declared_order():
    entries = cli.parse_classifications("A=1;A:x=1;B=2;B:x=3")
    assert cli.duplicate_keys(entries) == {'1': ['A', 'A:x']}
    assert cli.duplicate_subclasses(entries) == {'x': 2}


def test_lobby_scheme_reads_back_as_the_viewer_buttons():
    "Scheme rows -> lobby's string -> the buttons the 1-by-1 viewer builds from it."
    rows = [
        {'type': 'major', 'major': 'A', 'sub': '', 'key': '1', 'positive': True},
        {'type': 'major', 'major': 'B', 'sub': '', 'key': '2', 'positive': False},
        {'type': 'subclass', 'major': 'A', 'sub': 'x', 'key': '3', 'positive': False},
        {'type': 'subclass', 'major': ' B , A ,B', 'sub': 'Merger', 'key': 'm', 'positive': False},
    ]
    text, positive = lobby.classifications_string_from_rows(rows)
    assert cli.scheme_buttons(cli.parse_classifications(text)) == (
        [('A', '1'), ('B', '2')],
        [(('A',), 'x', '3'), (('B', 'A'), 'Merger', 'm')])
    assert positive == {'A'}


def test_shipped_presets_read_back_as_written():
    "Every shipped preset's scheme survives the lobby -> viewer round trip unchanged."
    import glob
    import json
    presets = glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'qtstamp', 'defaults', 'presets', '*.json'))
    assert presets
    for preset in presets:
        rows = json.load(open(preset)).get('scheme_rows') or []
        text, _ = lobby.classifications_string_from_rows(rows)
        assert cli.format_classifications(cli.parse_classifications(text)) == text, preset


# --- classification CSV names ----------------------------------------------------

def test_csv_base_names():
    assert classification_csv_base(MOSAIC_CSV_PREFIX, 'n', 12) == 'classification_mosaic_autosave_n_12'
    assert classification_csv_base(SINGLE_CSV_PREFIX, None, 12, 7) == 'classification_single__12_7'


def test_csvs_include_forks_but_not_other_seeds(tmpdir):
    base = classification_csv_base(MOSAIC_CSV_PREFIX, 'n', 5, 7)
    mine = [_touch(os.path.join(tmpdir, f)) for f in
            (f'{base}.csv', f'{base}-new_dataset_1.csv', f'{base}-simultaneous_10.csv',
             f'{base}-simultaneous_2.csv')]
    for other in (f'{base}0.csv', f'{base}8-new_dataset_1.csv', f'{base}.csv.bak'):
        _touch(os.path.join(tmpdir, other))
    found = classification_csvs(tmpdir, base)
    assert sorted(found) == sorted(mine)
    # natural sort: _2 before _10
    assert found.index(mine[3]) < found.index(mine[2])


def test_unseeded_csvs_do_not_claim_seeded_ones(tmpdir):
    base = classification_csv_base(SINGLE_CSV_PREFIX, 'n', 5)
    mine = _touch(os.path.join(tmpdir, f'{base}.csv'))
    _touch(os.path.join(tmpdir, f'{base}_7.csv'))
    assert classification_csvs(tmpdir, base) == [mine]


def test_csv_names_with_glob_characters(tmpdir):
    base = classification_csv_base(SINGLE_CSV_PREFIX, 'run[1]', 5)
    mine = _touch(os.path.join(tmpdir, 'dir[x]', f'{base}.csv'))
    _touch(os.path.join(tmpdir, 'dir[x]', classification_csv_base(SINGLE_CSV_PREFIX, 'run1', 5) + '.csv'))
    assert classification_csvs(os.path.join(tmpdir, 'dir[x]'), base) == [mine]


def test_resume_csv_pick(tmpdir):
    base = classification_csv_base(SINGLE_CSV_PREFIX, 'n', 5)
    assert resume_csv(classification_csvs(tmpdir, base)) is None
    plain = _touch(os.path.join(tmpdir, f'{base}.csv'))
    assert resume_csv(classification_csvs(tmpdir, base)) == plain
    # `-` sorts before `.`: with the file plus one fork, the fork is reopened --
    # right when the plain name belongs to another dataset (BUGS-DONE 11).
    fork = _touch(os.path.join(tmpdir, f'{base}-new_dataset_1.csv'))
    assert classification_csvs(tmpdir, base) == [fork, plain]
    assert resume_csv(classification_csvs(tmpdir, base)) == fork


def _graded(path, file_names, index=True):
    "A classification CSV listing `file_names`, as either viewer writes one."
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write((',' if index else '') + 'file_name,classification\n')
        for i, name in enumerate(file_names):
            f.write((f'{i},' if index else '') + f'{name},Empty\n')
    return path


def test_resume_csv_prefers_the_file_holding_this_dataset(tmpdir):
    "BUGS 26: by sort position alone, the first dataset's own file was never reopened."
    base = classification_csv_base(SINGLE_CSV_PREFIX, 'n', 3)
    mine, theirs = ['a.png', 'b.png', 'c.png'], ['x.png', 'y.png', 'z.png']
    plain = _graded(os.path.join(tmpdir, f'{base}.csv'), mine)
    fork = _graded(os.path.join(tmpdir, f'{base}-new_dataset_1.csv'), theirs, index=False)
    found = classification_csvs(tmpdir, base)
    assert resume_csv(found, mine) == plain
    assert resume_csv(found, theirs) == fork
    # Same objects in another order is another session: nothing matches, so the old
    # rule picks, and the viewer forks.
    assert resume_csv(found, ['c.png', 'b.png', 'a.png']) == fork
    # Two that match (a simultaneous fork): the old rule among them.
    twin = _graded(os.path.join(tmpdir, f'{base}-simultaneous_1.csv'), mine)
    assert resume_csv(classification_csvs(tmpdir, base), mine) == twin


# --- the lobby finding what a viewer wrote ------------------------------------------

def test_mosaic_csv_found_without_a_vis_band(tmpdir):
    "BUGS.md 39: the count in the name is the main band's, whatever it is called."
    data = _dataset(os.path.join(tmpdir, 'data'), ['r'], n=4)
    cls = os.path.join(tmpdir, 'cls')
    written = _touch(os.path.join(cls, classification_csv_base(MOSAIC_CSV_PREFIX, 'n', 4) + '.csv'))
    assert lobby.predict_mosaic_csv_path(data, 'n', None, 'r', False, cls) == written


def test_mosaic_csv_not_taken_from_a_seed_it_prefixes(tmpdir):
    "BUGS.md 40: seed 7 must not pick up a newer seed-70 session."
    data = _dataset(os.path.join(tmpdir, 'data'), ['VIS'], n=4)
    cls = os.path.join(tmpdir, 'cls')
    now = time.time()
    written = _touch(os.path.join(cls, classification_csv_base(MOSAIC_CSV_PREFIX, 'n', 4, 7) + '.csv'),
                     now - 100)
    _touch(os.path.join(cls, classification_csv_base(MOSAIC_CSV_PREFIX, 'n', 4, 70) + '.csv'), now)
    assert lobby.predict_mosaic_csv_path(data, 'n', 7, 'VIS', False, cls) == written


def test_predicted_mosaic_csv_is_the_newest(tmpdir):
    "After a mosaic run: the file it wrote last, even when that is not its resume pick."
    data = _dataset(os.path.join(tmpdir, 'data'), ['VIS'], n=4)
    cls = os.path.join(tmpdir, 'cls')
    base = classification_csv_base(MOSAIC_CSV_PREFIX, '', 4)
    now = time.time()
    newest = _touch(os.path.join(cls, base + '.csv'), now)
    _touch(os.path.join(cls, base + '-simultaneous_1.csv'), now - 100)
    assert lobby.predict_mosaic_csv_path(data, None, None, 'VIS', False, cls) == newest


def test_predicted_single_csv_is_the_one_the_viewer_reopens(tmpdir):
    "Before a 1-by-1 launch: the viewer's own pick, which the scheme check must read."
    data = _dataset(os.path.join(tmpdir, 'data'), ['VIS'], n=4)
    cls = os.path.join(tmpdir, 'cls')
    base = classification_csv_base(SINGLE_CSV_PREFIX, '', 4)
    now = time.time()
    _touch(os.path.join(cls, base + '.csv'), now)
    fork = _touch(os.path.join(cls, base + '-new_dataset_1.csv'), now - 100)
    assert lobby.predict_single_csv_path(data, None, None, 'VIS', False, cls) == fork


def test_lobby_cli_rejects_a_composite_that_is_not_a_triple():
    args = lobby._build_arg_parser().parse_args(['--no-gui', '--rgb-composites', 'H,Y;H,J,Y'])
    try:
        lobby.config_from_cli(args)
    except ValueError as e:
        assert 'H,Y' in str(e)
    else:
        raise AssertionError('accepted a two-band composite')
    args = lobby._build_arg_parser().parse_args(['--no-gui', '--rgb-composites', 'H,Y,I;;'])
    assert lobby.config_from_cli(args)['rgb_composites'] == [['H', 'Y', 'I']]


def test_effective_main_band():
    assert lobby.effective_main_band('r', ['g']) == 'r'
    assert lobby.effective_main_band(lobby.NO_MAIN_BAND, [' ', 'g', 'z']) == 'g'
    assert lobby.effective_main_band(lobby.NO_MAIN_BAND, []) == ''
    assert lobby.effective_main_band('', ['g']) == cli.DEFAULT_MAIN_BAND


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
