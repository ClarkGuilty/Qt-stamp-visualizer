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
from qtstamp import mosaic
from qtstamp import single_viewer
sys.argv = _argv

from qtstamp import lobby
from qtstamp.imaging import (MOSAIC_CSV_PREFIX, SINGLE_CSV_PREFIX, classification_csv_base,
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
    from qtstamp import workers
    for pixel_size, dim in [(0.1, 100), (0.262, 47), (0.1, 1500)]:
        ps, ps_big = workers.panstarrs_number_of_pixels(pixel_size, dim)
        ls, ls_big = workers.legacy_survey_number_of_pixels(pixel_size, dim)
        assert ps_big > ps
        ps_sky = ps_big*workers.PANSTARRS_PIXEL_SIZE
        ls_sky = ls_big*workers.LEGACY_SURVEY_PIXEL_SIZE
        assert abs(ps_sky - ls_sky) <= workers.PANSTARRS_PIXEL_SIZE*2



class _FakeResponse:
    def __init__(self, text):
        self._text = text.encode()
    def read(self):
        return self._text
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


def test_panstarrs_filename_lookup_remembered_in_memory_only(tmpdir):
    "One ps1filenames.py call per position, not one per cutout size; failures and no-rows are retried."
    import urllib.request
    from qtstamp import workers
    calls = []
    table = ('projcell subcell ra dec filter mjd type filename shortname badflag\n' +
             ''.join(f'1 2 3 4 {f} 5 stack /x/{f}.fits {f}.fits 0\n' for f in 'grz'))
    def fake_urlopen(url, timeout=None):
        calls.append(url)
        if len(calls) == 2:
            raise TimeoutError('slow STScI')
        if len(calls) == 4:
            return _FakeResponse('')                         # truncated reply: reads as no rows
        return _FakeResponse(table)
    real, urllib.request.urlopen = urllib.request.urlopen, fake_urlopen
    workers._PS1_FILENAMES.clear()
    cwd = os.getcwd()
    os.chdir(tmpdir)                                         # a stray relative write lands here
    try:
        first = workers.get_panstarrs_filenames(10.5, -20.25)
        assert workers.get_panstarrs_filenames(np.float64(10.5), np.float64(-20.25)) == first
        assert len(calls) == 1 and first['z'] == '/x/z.fits'
        try:
            workers.get_panstarrs_filenames(11.0, -20.25)
            raise AssertionError('the timeout should propagate')
        except TimeoutError:
            pass
        workers.get_panstarrs_filenames(11.0, -20.25)      # not remembered: asks again
        assert len(calls) == 3
        assert workers.get_panstarrs_filenames(12.0, -20.25) is None
        assert workers.get_panstarrs_filenames(12.0, -20.25) is not None   # None not remembered
        assert len(calls) == 5
    finally:
        os.chdir(cwd)
        urllib.request.urlopen = real
        workers._PS1_FILENAMES.clear()
    assert os.listdir(tmpdir) == []


def test_failure_title_says_no_data_only_for_a_coverage_gap(tmpdir):
    from qtstamp import workers
    savefile = os.path.join(tmpdir, 'cut.jpg')
    assert 'timed out or failed' in workers.failure_title('PanSTARRS', savefile)  # no marker
    workers.record_miss(savefile, 'transient', 'timed out')
    assert 'timed out or failed' in workers.failure_title('PanSTARRS', savefile)
    workers.record_miss(savefile, 'no_coverage', 'no rows')
    assert workers.failure_title('PanSTARRS', savefile) == 'No PanSTARRS data available'
    with open(savefile + workers.MISS_SUFFIX, 'w') as f:      # expired: says nothing now
        json.dump({'reason': 'no_coverage', 'time': 0}, f)
    assert 'timed out or failed' in workers.failure_title('PanSTARRS', savefile)


def test_fits_main_band_without_wcs_fetches_nothing(tmpdir):
    "Read as-is, a header with no WCS asked PanSTARRS for a 1e6 px cutout at RA/Dec (11, 11)."
    from astropy.io import fits
    import webbrowser
    rng = np.random.default_rng(0)
    data = os.path.join(tmpdir, 'data')
    for band in ('r', 'g'):
        os.makedirs(os.path.join(data, band))
        for i in range(3):
            fits.PrimaryHDU(rng.random((20, 20)).astype(np.float32)).writeto(
                os.path.join(data, band, f'obj{i}.fits'))
    argv = ['-p', data, '-b', 'r', '-B', 'g', '--rgb-composites', '', '-N', 'run', '-s', '7',
            '--classifications-dir', os.path.join(tmpdir, 'cls'),
            '--state-dir', os.path.join(tmpdir, 'state')]
    win = single_viewer.ApplicationWindow(single_viewer.parse_args(argv), clipboard=APP.clipboard())
    opened, real_open = [], webbrowser.open
    webbrowser.open = opened.append
    try:
        assert win.filetype == 'FITS' and np.isnan(win.ra)
        win.set_survey('PanSTARRS')
        assert win.ax['PanSTARRS'].get_title() == 'No RA/Dec:\nno sky WCS in any of its bands'
        assert win.survey_filename['PanSTARRS'] is None and 'PanSTARRS' not in win.survey_threads
        win.set_survey('Legacy Survey')
        assert win.ax['Legacy Survey'].get_title() == 'No RA/Dec:\nno sky WCS in any of its bands'
        win.viewPanSTARRS(); win.viewls(); win.viewESASky()
        assert opened == []
    finally:
        webbrowser.open = real_open
        win.close()


def test_png_input_says_radec_needs_fits(tmpdir):
    "Not 'no sky WCS': a PNG has no WCS to lack."
    import webbrowser
    data = _png_dataset(os.path.join(tmpdir, 'data'), ['r', 'g'])
    win = single_viewer.ApplicationWindow(
        single_viewer.parse_args(_argv(tmpdir, data, '--no-legacysurvey')), clipboard=APP.clipboard())
    opened, real_open = [], webbrowser.open
    webbrowser.open = opened.append
    try:
        assert win._no_radec_reason().startswith('PNG/JPG input')
        win.viewls(); win.viewPanSTARRS(); win.viewESASky()
        assert opened == []
    finally:
        webbrowser.open = real_open
        win.close()


def _fits(path, shape=(20, 20), wcs=False):
    from astropy.io import fits
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = None
    if wcs:
        from sample_data import _wcs_header
        header = _wcs_header(150.25, 2.5, shape[0])
    fits.PrimaryHDU(np.random.default_rng(0).random(shape).astype(np.float32), header).writeto(path)


def _broken_dataset(root):
    """Four objects in VIS, Y, J, H: `a` complete; `b` without J; `c` with an H of another
    size; `d` with a Y that is not a FITS file; `e` only in H."""
    for stem in 'abcd':
        for band in ('VIS', 'Y', 'J', 'H'):
            if (stem, band) == ('b', 'J'):
                continue
            _fits(os.path.join(root, band, stem + '.fits'),
                  shape=(12, 12) if (stem, band) == ('c', 'H') else (20, 20))
    with open(os.path.join(root, 'Y', 'd.fits'), 'wb') as f:
        f.write(b'not a FITS file')
    _fits(os.path.join(root, 'H', 'e.fits'))
    return root


def _broken_argv(tmpdir, data, *extra):
    return ['-p', data, '-b', 'VIS', '-B', 'Y,J,H', '--rgb-composites', 'H,J,Y',
            '--classifications-dir', os.path.join(tmpdir, 'cls'),
            '--state-dir', os.path.join(tmpdir, 'state'), *extra]


def test_mosaic_draws_no_data_tiles_instead_of_stopping(tmpdir):
    """A composite is made only when all three bands are there, FITS and of one size.
    Each of these used to raise while the page was rendered: at startup, no window."""
    data = _broken_dataset(os.path.join(tmpdir, 'data'))
    win = mosaic.MosaicVisualizer(mosaic.parse_args(_broken_argv(tmpdir, data, '-l', '1', '-m', '1')))
    try:
        assert list(win.listimage) == ['a.fits', 'b.fits', 'c.fits', 'd.fits', 'e.fits']
        notes = [win.missing_bands_note.text()]
        for _ in range(4):
            win.next()
            notes.append(win.missing_bands_note.text())
        assert win.page == 4
        assert notes == ['', 'b: no J', '', 'd: unreadable Y', 'e: no VIS, Y, J']
        assert win.df_name and os.path.exists(win.df_name)      # every page turn still saved
    finally:
        win.close()


def test_single_viewer_draws_no_data_panels_and_stays_in_step(tmpdir):
    """The same objects in the 1-by-1. A panel that could not be drawn used to raise out of
    the page turn, leaving the counter and the lit buttons on the previous object."""
    data = _broken_dataset(os.path.join(tmpdir, 'data'))
    win = single_viewer.ApplicationWindow(
        single_viewer.parse_args(_broken_argv(tmpdir, data, '--no-legacysurvey')),
        clipboard=APP.clipboard())
    try:
        win.config_dict['autonext'] = False
        seen = []
        for i in range(5):
            composite = [t.get_text() for t in win.ax['H,J,Y'].texts]
            seen.append((win.counter_widget.text(), win.missing_bands_note.text(), composite))
            win.classify_major('A')
            assert win.df.at[i, 'classification'] == 'A'
            win.next()
        assert seen == [
            ('1/5', '', []),
            ('2/5', 'b: no J', ['no data\nJ']),
            ('3/5', '', ['no data\nsizes differ']),
            ('4/5', 'd: unreadable Y', ['no data\nY']),
            ('5/5', 'e: no VIS, J, Y', ['no data\nJ\nY']),
        ]
        assert [t.get_text() for t in win.ax['VIS'].texts] == ['no data\nVIS']
    finally:
        win.close()


def test_one_band_may_be_fits_for_one_object_and_png_for_the_next(tmpdir):
    "Read as FITS because its directory held FITS, a PNG stopped both viewers."
    data = os.path.join(tmpdir, 'data')
    _fits(os.path.join(data, 'r', 'a.fits'))
    Image.fromarray(np.zeros((20, 20), np.uint8)).save(os.path.join(data, 'r', 'b.PNG'))
    argv = ['-p', data, '-b', 'r', '-B', '', '--rgb-composites', '',
            '--classifications-dir', os.path.join(tmpdir, 'cls'),
            '--state-dir', os.path.join(tmpdir, 'state')]
    grid = mosaic.MosaicVisualizer(mosaic.parse_args(argv + ['-l', '2', '-m', '1']))
    single = single_viewer.ApplicationWindow(
        single_viewer.parse_args(argv + ['--no-legacysurvey']), clipboard=APP.clipboard())
    try:
        assert list(grid.listimage) == list(single.listimage) == ['a.fits', 'b.PNG']
        assert grid.missing_bands_note.text() == ''
        assert grid.cbscale.isEnabled()                 # there is FITS to scale
        single.next()
        assert single.missing_bands_note.text() == '' and not single.stamps['r'].is_fits
    finally:
        grid.close()
        single.close()


def test_radec_comes_from_any_fits_band_with_a_wcs(tmpdir):
    "The default band need not be the one with the WCS, or FITS at all."
    data = os.path.join(tmpdir, 'data')
    os.makedirs(os.path.join(data, 'rgb'))
    for stem in ('a', 'b'):
        Image.fromarray(np.zeros((20, 20), np.uint8)).save(os.path.join(data, 'rgb', stem + '.png'))
    _fits(os.path.join(data, 'VIS', 'a.fits'), shape=(40, 40), wcs=True)
    argv = ['-p', data, '-b', 'rgb', '-B', 'VIS', '--rgb-composites', '', '--no-legacysurvey',
            '--classifications-dir', os.path.join(tmpdir, 'cls'),
            '--state-dir', os.path.join(tmpdir, 'state')]
    win = single_viewer.ApplicationWindow(single_viewer.parse_args(argv), clipboard=APP.clipboard())
    try:
        win.config_dict['autonext'] = False
        assert win._no_radec_reason() is None
        assert abs(win.ra - 150.25) < 1e-3 and abs(win.dec - 2.5) < 1e-3
        assert win.radec_image_dim == 40 and abs(win.image_pixel_size - 0.1) < 1e-6
        win.classify_major('A')
        assert win.df.at[0, 'image_dim'] == 40 and abs(win.df.at[0, 'ra'] - 150.25) < 1e-3
        # The pre-fetch pass finds the same header by itself.
        from qtstamp import workers
        thread = workers.FetchThread(win.df, 0, [win.band_sources[b] for b in win.single_bands],
                                     win.listimage, 'ls', 'ps')
        radec = thread.radec_from_files(1)
        assert np.isnan(radec.ra)
        radec = thread.radec_from_files(0)
        assert abs(radec.ra - 150.25) < 1e-3 and radec.image_dim == 40
        win.next()                                      # b: no FITS stamp anywhere
        assert np.isnan(win.ra) and win._no_radec_reason() == 'no sky WCS in any of its bands'
    finally:
        win.close()


def test_a_class_named_like_a_missing_value_survives_a_resume(tmpdir):
    """pandas reads NA, N/A, None, null, nan as missing: the resumed session showed the
    object as ungraded, and its next save wrote the cell out empty."""
    data = _png_dataset(os.path.join(tmpdir, 'data'), ['r', 'g'], n=3)
    argv = ['-p', data, '-b', 'r', '-B', '', '--rgb-composites', '', '--no-legacysurvey',
            '--classifications', 'NA=1;N/A=2;B=3',
            '--classifications-dir', os.path.join(tmpdir, 'cls'),
            '--state-dir', os.path.join(tmpdir, 'state')]
    win = single_viewer.ApplicationWindow(single_viewer.parse_args(argv), clipboard=APP.clipboard())
    win.config_dict['autonext'] = False
    win.classify_major('NA')
    win.next()
    win.classify_major('N/A')
    csv = win.df_name
    win.close()

    again = single_viewer.ApplicationWindow(single_viewer.parse_args(argv), clipboard=APP.clipboard())
    try:
        again.config_dict['autonext'] = False
        assert list(again.df['classification']) == ['NA', 'N/A', 'Empty']
        again.counter = 0
        again.go_to_counter_page()
        assert again.bactivatedclassification is again.dict_class2button['NA']
        again.counter = 2
        again.go_to_counter_page()
        again.classify_major('B')                       # any save
        with open(csv) as f:
            rows = [line.split(',')[2] for line in f.read().splitlines()[1:]]
        assert rows == ['NA', 'N/A', 'B']
    finally:
        again.close()


def test_ctrl_click_on_a_mac_marks_interesting(tmpdir):
    "Qt on macOS reports the Control key as Meta (and Command as Control)."
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    data = _png_dataset(os.path.join(tmpdir, 'data'), ['r', 'g'])
    win = mosaic.MosaicVisualizer(mosaic.parse_args(_argv(tmpdir, data, '-l', '3', '-m', '2')))
    try:
        for i, modifier in enumerate((Qt.MetaModifier, Qt.ControlModifier, Qt.ShiftModifier)):
            win.buttons[i].mousePressEvent(QMouseEvent(
                QEvent.MouseButtonPress, QPointF(1, 1), QPointF(1, 1),
                Qt.LeftButton, Qt.LeftButton, modifier))
        win.buttons[3].mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, QPointF(1, 1), QPointF(1, 1),
            Qt.LeftButton, Qt.LeftButton, Qt.AltModifier))
        assert list(win.df['classification'])[:4] == [mosaic.C_INTERESTING] * 3 + [mosaic.C_UNINTERESTING]
    finally:
        win.close()


def test_closing_waits_for_every_fetch_still_running(tmpdir):
    """BUGS 49: only each survey's newest fetch was waited for, so an older one could be
    destroyed while running -- which aborts the process."""
    import time
    from PySide6.QtCore import QThread
    data = os.path.join(tmpdir, 'data')
    for stem in 'abc':
        _fits(os.path.join(data, 'r', stem + '.fits'), wcs=True)
    argv = ['-p', data, '-b', 'r', '-B', '', '--rgb-composites', '', '--no-legacysurvey',
            '--classifications-dir', os.path.join(tmpdir, 'cls'),
            '--state-dir', os.path.join(tmpdir, 'state')]
    win = single_viewer.ApplicationWindow(single_viewer.parse_args(argv), clipboard=APP.clipboard())
    real = single_viewer.fetch_panstarrs
    single_viewer.fetch_panstarrs = lambda *a, **k: (time.sleep(0.4), single_viewer.FetchWorker)[1]
    try:
        threads = []
        for _ in range(3):
            win.set_survey('PanSTARRS')
            # Each object has its own cutout name; here one object, so forget the last.
            threads.append(win.survey_threads['PanSTARRS'])
        assert len(win.fetch_threads) == 3 and all(t.isRunning() for t in threads)
        win.close()
        assert not any(t.isRunning() for t in threads)
        APP.processEvents()
        assert win.fetch_threads == set()
    finally:
        single_viewer.fetch_panstarrs = real


def test_chained_run_on_bands_that_differ_in_objects_and_format(tmpdir):
    """The lobby finds the CSV the mosaic wrote -- named after the union of the bands,
    not the main band's count -- and extracts every band's file for the positives."""
    data = os.path.join(tmpdir, 'data')
    for stem in 'ab':
        _fits(os.path.join(data, 'VIS', stem + '.fits'))
    os.makedirs(os.path.join(data, 'rgb'))
    for stem in 'abcd':
        Image.fromarray(np.zeros((20, 20), np.uint8)).save(os.path.join(data, 'rgb', stem + '.JPG'))
    cwd = os.getcwd()
    os.chdir(tmpdir)
    try:
        win = lobby.LobbyWindow(state_dir_override=os.path.join(tmpdir, 'state'))
        win.classifications_edit.setText(os.path.join(tmpdir, 'cls'))
        win.path_edit.setText(data)
        win._on_identity_changed()
        assert win.available_bands == ['VIS', 'rgb']
        win.main_band_combo.setCurrentText('VIS')
        for i in range(win.color_bands_list.count()):
            item = win.color_bands_list.item(i)
            item.setCheckState(lobby.Qt.Checked if item.text() == 'rgb' else lobby.Qt.Unchecked)
        win.output_path_edit.setText(os.path.join(tmpdir, 'out'))
        win.mode_combo.setCurrentIndex(lobby.MODE_CHAINED)

        # Stage 1, as the lobby would launch it, with two stamps clicked.
        argv = lobby.build_mosaic_argv(data, 'run', None, 2, 2, os.path.join(tmpdir, 'cls'),
                                       band_argv=win._band_argv())[1:]
        grid = mosaic.MosaicVisualizer(mosaic.parse_args(
            argv + ['--state-dir', os.path.join(tmpdir, 'state')]))
        assert list(grid.listimage) == ['a.fits', 'b.fits', 'c.JPG', 'd.JPG']
        grid.my_label_clicked(None, 0, mosaic.C_LENS)
        grid.my_label_clicked(None, 2, mosaic.C_LENS)
        written = grid.df_name
        grid.close()

        launched = []
        win._launch_fire_and_forget = launched.append
        win._stage1_extraction_context = {
            'source_path': data, 'name': 'run', 'seed': None, 'bands': win._viewer_bands(data),
            'mef': False, 'ncols': 2, 'nrows': 2, 'classifications_dir': os.path.join(tmpdir, 'cls')}
        win.on_stage1_finished(0, None)
        log = win.log_view.toPlainText()
        assert f"Auto-detected classification CSV: {written}" in log
        out = os.path.join(tmpdir, 'out')
        assert sorted(os.listdir(os.path.join(out, 'VIS'))) == ['a.fits']
        assert sorted(os.listdir(os.path.join(out, 'rgb'))) == ['a.JPG', 'c.JPG']
        assert len(launched) == 1 and launched[0][launched[0].index('-p') + 1] == out
        win.close()
    finally:
        os.chdir(cwd)


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
