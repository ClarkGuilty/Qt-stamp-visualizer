"""Window-level tests for the two viewers, on Qt's offscreen platform.

The viewers parse their arguments in `main()`, not at import, so a test can
import them and build a window from an argv of its own. These check that, and
that the lobby finds the classification CSV a real viewer wrote -- the contract
BUGS.md 39 and 40 broke. Written as plain pytest functions, but also runnable
directly (`python3 tests/test_viewers.py`).
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['QTSTAMP_CONFIG_DIR'] = os.path.join(
    tempfile.gettempdir(), 'qtstamp-test-no-such-user-dir')
os.environ.pop('QTSTAMP_STATE_DIR', None)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
from PIL import Image
from PySide6 import QtWidgets

# An argv neither viewer understands: importing them must not parse it.
_argv, sys.argv = sys.argv, ['viewer', '--not-a-viewer-option']
import mosaic
import single_viewer
sys.argv = _argv

import lobby
from imaging import (MOSAIC_CSV_PREFIX, SINGLE_CSV_PREFIX, classification_csv_base,
                     classification_csvs)

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _png_dataset(root, bands, n=6, stem='obj'):
    rng = np.random.default_rng(0)
    for band in bands:
        os.makedirs(os.path.join(root, band))
        for i in range(n):
            pixels = (rng.random((20, 20)) * 255).astype(np.uint8)
            Image.fromarray(pixels).save(os.path.join(root, band, f'{stem}{i}.png'))
    return root


def _argv(tmpdir, data, *extra):
    return ['-p', data, '-b', '', '-B', 'r,g', '--rgb-composites', '', '-N', 'run', '-s', '7',
            '--classifications-dir', os.path.join(tmpdir, 'cls'),
            '--state-dir', os.path.join(tmpdir, 'state'), *extra]


def _decoy(tmpdir, prefix):
    "A newer CSV for the same name and count under seed 70 -- what BUGS.md 40 picked up."
    path = os.path.join(tmpdir, 'cls', classification_csv_base(prefix, 'run', 6, 70) + '.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').close()


def test_viewers_import_without_parsing_argv():
    assert callable(mosaic.parse_args) and callable(single_viewer.parse_args)


def test_both_viewers_shuffle_alike_for_every_seed(tmpdir):
    "99 included: the mosaic no longer swaps it for another number."
    data = _png_dataset(os.path.join(tmpdir, 'data'), ['r', 'g'], n=12)
    for seed in ('7', '99'):
        argv = ['-p', data, '-b', 'r', '-B', '', '--rgb-composites', '', '-s', seed,
                '--classifications-dir', os.path.join(tmpdir, 'cls'),
                '--state-dir', os.path.join(tmpdir, 'state')]
        grid = mosaic.MosaicVisualizer(mosaic.parse_args(argv + ['-l', '2', '-m', '2']))
        single = single_viewer.ApplicationWindow(
            single_viewer.parse_args(argv + ['--no-legacysurvey']), clipboard=APP.clipboard())
        expected = sorted(os.listdir(os.path.join(data, 'r')))
        np.random.default_rng(int(seed)).shuffle(expected)
        assert list(grid.listimage) == list(single.listimage) == expected
        grid.close()
        single.close()


def test_mosaic_csv_found_by_the_lobby_and_resumed(tmpdir):
    data = _png_dataset(os.path.join(tmpdir, 'data'), ['r', 'g'])
    args = mosaic.parse_args(_argv(tmpdir, data, '-l', '2', '-m', '2'))
    assert args.main_band == 'r'
    win = mosaic.MosaicVisualizer(args)
    win.my_label_clicked(None, 1, mosaic.C_LENS)
    written = win.df_name
    win.close()
    _decoy(tmpdir, MOSAIC_CSV_PREFIX)

    found = lobby.predict_mosaic_csv_path(data, 'run', 7, 'r', False, os.path.join(tmpdir, 'cls'))
    assert found == written

    again = mosaic.MosaicVisualizer(mosaic.parse_args(_argv(tmpdir, data, '-l', '2', '-m', '2')))
    assert again.df_name == written
    assert (again.df['classification'] == mosaic.C_LENS).sum() == 1
    again.close()


def test_single_csv_found_by_the_lobby_and_resumed(tmpdir):
    data = _png_dataset(os.path.join(tmpdir, 'data'), ['r', 'g'])
    argv = _argv(tmpdir, data, '--no-legacysurvey', '--classifications', 'A=1;B=2;A,B:m=3')
    args = single_viewer.parse_args(argv)
    assert args.major_classes == [('A', '1'), ('B', '2')]
    assert args.subclasses == [(('A', 'B'), 'm', '3')]
    win = single_viewer.ApplicationWindow(args, clipboard=APP.clipboard())
    win.classify_major('B')
    written = win.df_name
    win.close()
    _decoy(tmpdir, SINGLE_CSV_PREFIX)

    found = lobby.predict_single_csv_path(data, 'run', 7, 'r', False, os.path.join(tmpdir, 'cls'))
    assert found == written

    args = single_viewer.parse_args(argv)
    again = single_viewer.ApplicationWindow(args, clipboard=APP.clipboard())
    assert again.df_name == written
    assert list(again.df['classification']).count('B') == 1
    again.close()


def test_datasets_sharing_a_csv_name_each_reopen_their_own(tmpdir):
    """BUGS 26: same --name, count and seed, different objects. Alternating between
    them used to fork a new -new_dataset_N on every switch, never reopening a file."""
    first = _png_dataset(os.path.join(tmpdir, 'first'), ['r', 'g'])
    second = _png_dataset(os.path.join(tmpdir, 'second'), ['r', 'g'], stem='other')
    cls = os.path.join(tmpdir, 'cls')

    def mosaic_grade(data):
        win = mosaic.MosaicVisualizer(mosaic.parse_args(_argv(tmpdir, data, '-l', '2', '-m', '2')))
        graded = (win.df['classification'] == mosaic.C_LENS).sum()
        if not graded:
            win.my_label_clicked(None, 1, mosaic.C_LENS)
        return win, graded

    def single_grade(data):
        win = single_viewer.ApplicationWindow(single_viewer.parse_args(
            _argv(tmpdir, data, '--no-legacysurvey', '--classifications', 'A=1;B=2')),
            clipboard=APP.clipboard())
        graded = list(win.df['classification']).count('A')
        if not graded:
            win.classify_major('A')
        return win, graded

    for prefix, grade in ((MOSAIC_CSV_PREFIX, mosaic_grade), (SINGLE_CSV_PREFIX, single_grade)):
        written = {}
        for data in (first, second, first, second):
            win, graded = grade(data)
            if data in written:
                assert (win.df_name, graded) == (written[data], 1), prefix
            else:
                written[data] = win.df_name
            win.close()
        base = classification_csv_base(prefix, 'run', 6, 7)
        assert sorted(classification_csvs(cls, base)) == sorted(written.values()), prefix


def test_single_dropdown_shows_a_saved_asinh(tmpdir):
    "BUGS 43: an asinh preference showed as Log, and choosing Log then did nothing."
    data = _png_dataset(os.path.join(tmpdir, 'data'), ['r', 'g'])
    os.makedirs(os.path.join(tmpdir, 'state'))
    with open(os.path.join(tmpdir, 'state', 'preferences_single.json'), 'w') as f:
        json.dump({'scale': 'asinh2'}, f)          # the 1-by-1's old name for it
    win = single_viewer.ApplicationWindow(
        single_viewer.parse_args(_argv(tmpdir, data, '--no-legacysurvey')), clipboard=APP.clipboard())
    assert win.cbscale.currentText() == 'Asinh' and win.config_dict['scale'] == 'asinh'
    win.cbscale.setCurrentText('Log')
    assert win.config_dict['scale'] == 'log'
    win.close()


def test_panstarrs_large_fov_matches_legacy_survey():
    "BUGS 44: PanSTARRS's Large FoV was 6 px (1.5 arcsec), smaller than the stamp itself."
    for pixel_size, dim in [(0.1, 100), (0.262, 47), (0.1, 1500)]:
        ps, ps_big = single_viewer.panstarrs_number_of_pixels(pixel_size, dim)
        ls, ls_big = single_viewer.legacy_survey_number_of_pixels(pixel_size, dim)
        assert ps_big > ps
        ps_sky = ps_big*single_viewer.PANSTARRS_PIXEL_SIZE
        ls_sky = ls_big*single_viewer.LEGACY_SURVEY_PIXEL_SIZE
        assert abs(ps_sky - ls_sky) <= single_viewer.PANSTARRS_PIXEL_SIZE*2


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
