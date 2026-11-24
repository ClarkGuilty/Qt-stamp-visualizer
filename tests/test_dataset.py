"""Tests for dataset.py -- how both viewers list the objects and find each band.

Run on fake sets from `sample_data.py` (one MEF directory, one directory-per-band
JPG set), plus a small FITS set for the composite rules. Arguments go
through `cli.finish_dataset_args`, as in the viewers, so `-b ''` resolves the
same way. Qt-free. Plain pytest functions, also runnable directly
(`python tests/test_dataset.py`).
"""

import argparse
import os
import sys
from os.path import join
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from astropy.io import fits

from qtstamp import cli
from qtstamp import fits_io
from PIL import Image

from qtstamp import dataset
from qtstamp.dataset import DatasetError, composite_stack, load_dataset, load_stamp
from sample_data import write_band_dirs, write_mef_set


def _load(argv):
    parser = argparse.ArgumentParser()
    cli.add_dataset_args(parser)
    args = parser.parse_args(argv)
    cli.finish_dataset_args(parser, args)
    return load_dataset(args.path, args.main_band, args.color_bands, args.composite_bands,
                        args.composite_band_members, args.mef, args.seed)


def _fits_band(root, band, stems):
    os.makedirs(join(root, band))
    for stem in stems:
        fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32)).writeto(join(root, band, stem + '.fits'))


def test_mef_bands_are_extensions_of_each_file(tmpdir):
    mef = write_mef_set(join(tmpdir, 'mef'))
    ds = _load(['-p', mef, '--mef', '-b', 'VIS_BGSUB', '-B', 'NIR_Y_BGSUB',
                '--rgb-composites', 'NIR_H_BGSUB,NIR_J_BGSUB,NIR_Y_BGSUB'])
    assert ds.listimage == fits_io.list_fits_files(mef)
    assert ds.filetype == 'FITS'
    assert ds.composite_bands == ['NIR_H_BGSUB,NIR_J_BGSUB,NIR_Y_BGSUB']
    assert ds.all_single_bands == {'VIS_BGSUB', 'NIR_Y_BGSUB', 'NIR_J_BGSUB', 'NIR_H_BGSUB'}
    assert ds.band_sources['NIR_J_BGSUB'] == fits_io.ExtensionBandSource(mef, 'NIR_J_BGSUB')
    assert set(ds.band_filetype.values()) == {'FITS'}


def test_mef_missing_extension_is_an_error_naming_what_there_is(tmpdir):
    mef = write_mef_set(join(tmpdir, 'mef'))
    try:
        _load(['-p', mef, '--mef', '-b', 'VIS_BGSUB', '-B', 'NIR_Z_BGSUB,NIR_K', '--rgb-composites', ''])
    except DatasetError as e:
        message = str(e)
    else:
        raise AssertionError('no DatasetError')
    n = len(fits_io.list_fits_files(mef))
    assert message.startswith(f'Extensions not found in any of the {n} files in {mef}: '
                              'NIR_K, NIR_Z_BGSUB\nAvailable extensions: ')
    assert 'VIS_BGSUB' in message.splitlines()[1]


def test_mef_bands_searched_in_file_order_whatever_the_seed(tmpdir):
    "The seed must not decide which files are opened to find the EXTNAMEs."
    mef = write_mef_set(join(tmpdir, 'mef'))
    real = fits_io.locate_mef_bands
    searched = []

    def spy(directory, filenames, wanted):
        searched.append(list(filenames))
        return real(directory, filenames, wanted)

    with mock.patch('qtstamp.fits_io.locate_mef_bands', spy):
        ds = _load(['-p', mef, '--mef', '-b', 'VIS_BGSUB', '-B', '', '--rgb-composites', '',
                    '-s', '7'])
    assert searched == [fits_io.list_fits_files(mef)]
    assert ds.listimage != searched[0] and sorted(ds.listimage) == searched[0]


def test_seed_shuffles_like_default_rng(tmpdir):
    jpg = write_band_dirs(join(tmpdir, 'jpg'))
    expected = sorted(os.listdir(join(jpg, 'rgb')))
    np.random.default_rng(99).shuffle(expected)
    ds = _load(['-p', jpg, '-b', 'rgb', '-B', '', '--rgb-composites', '', '-s', '99'])
    assert ds.listimage == expected


def test_no_main_band_takes_the_first_color_band(tmpdir):
    jpg = write_band_dirs(join(tmpdir, 'jpg'))
    ds = _load(['-p', jpg, '-b', '', '-B', 'grey,rgb', '--rgb-composites', ''])
    assert ds.listimage == sorted(os.listdir(join(jpg, 'grey')))
    assert ds.filetype == 'COMPRESSED'
    assert ds.all_single_bands == {'grey', 'rgb'}
    assert ds.band_sources['rgb'] == fits_io.DirBandSource(join(jpg, 'rgb'), 'COMPRESSED')


def test_composite_with_a_jpg_member_is_dropped_with_its_members(tmpdir):
    root = join(tmpdir, 'data')
    stems = ['a', 'b', 'c']
    for band in ('r', 'g', 'b'):
        _fits_band(root, band, stems)
    os.makedirs(join(root, 'jpg'))
    for stem in stems:
        open(join(root, 'jpg', stem + '.jpg'), 'wb').close()
    ds = _load(['-p', root, '-b', 'r', '-B', 'g', '--rgb-composites', 'r,g,b;r,g,jpg'])
    assert ds.composite_bands == ['r,g,b']
    assert ds.composite_band_members == {'r,g,b': ('r', 'g', 'b')}
    assert ds.all_single_bands == {'r', 'g', 'b'}            # not 'jpg'
    assert ds.band_filetype['jpg'] == 'COMPRESSED'
    assert set(ds.band_sources) == {'r', 'g', 'b'}


def test_empty_main_band_directory_is_an_error(tmpdir):
    root = join(tmpdir, 'data')
    os.makedirs(join(root, 'r'))
    try:
        load_dataset(root, 'r', [], [], {}, False, None)
    except DatasetError as e:
        assert str(e) == f"No FITS, PNG, or JPG files found in {join(root, 'r')}."
    else:
        raise AssertionError('no DatasetError')


def test_locate_raises_for_a_missing_object(tmpdir):
    _fits_band(tmpdir, 'r', ['a'])
    sources = {'r': fits_io.DirBandSource(join(tmpdir, 'r'), 'FITS')}
    assert fits_io.locate(sources, 'r', 'a') == fits_io.Locator(join(tmpdir, 'r', 'a.fits'), 0)
    try:
        fits_io.locate(sources, 'r', 'b')
    except fits_io.MissingBandError as e:
        assert e.band == 'r'
    else:
        raise AssertionError('no MissingBandError')


def _png(path, size=4):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.zeros((size, size), np.uint8)).save(path)


def test_objects_are_the_union_of_what_the_bands_hold(tmpdir):
    "No band lists the objects: one with a stamp in any band is an object."
    root = join(tmpdir, 'data')
    _fits_band(root, 'VIS', ['a', 'b'])
    for stem in ('a', 'b', 'c', 'd'):
        _png(join(root, 'rgb', stem + '.JPG'))
    ds = _load(['-p', root, '-b', 'VIS', '-B', 'rgb', '--rgb-composites', ''])
    # Named by the default band's file where it has one, else by the next band's.
    assert ds.listimage == ['a.fits', 'b.fits', 'c.JPG', 'd.JPG']
    assert ds.single_bands == ['VIS', 'rgb'] and ds.filetype == 'FITS'
    assert dataset.list_dataset_objects(root, ['VIS', 'rgb'], False) == ds.listimage
    assert dataset.list_dataset_objects(root, 'VIS', False) == ['a.fits', 'b.fits']
    # With the bands the other way round the same four, under the other band's names.
    swapped = _load(['-p', root, '-b', 'rgb', '-B', 'VIS', '--rgb-composites', ''])
    assert swapped.listimage == ['a.JPG', 'b.JPG', 'c.JPG', 'd.JPG']


def test_bands_holding_the_same_objects_list_as_the_default_band_did(tmpdir):
    "The CSV is named after the count and checked against the names: neither may move."
    root = join(tmpdir, 'data')
    stems = ['b-1', 'b', 'a', 'B']
    for band in ('r', 'g'):
        _fits_band(root, band, stems)
    ds = _load(['-p', root, '-b', 'r', '-B', 'g', '--rgb-composites', ''])
    assert ds.listimage == sorted(stem + '.fits' for stem in stems)


def test_one_band_directory_may_mix_fits_and_png(tmpdir):
    root = join(tmpdir, 'data')
    _fits_band(root, 'r', ['a'])
    _png(join(root, 'r', 'b.png'))
    ds = _load(['-p', root, '-b', 'r', '-B', '', '--rgb-composites', ''])
    assert ds.listimage == ['a.fits', 'b.png'] and ds.band_filetype == {'r': 'FITS'}
    a, b = load_stamp(ds.band_sources, 'r', 'a'), load_stamp(ds.band_sources, 'r', 'b')
    assert a.is_fits and a.pixels.shape == (4, 4) and a.header is not None
    assert not b.is_fits and b.pixels.size == (4, 4) and b.header is None


def test_compressed_fits_are_objects(tmpdir):
    "BUGS 60: a band holding only obj.fits.gz is no longer empty."
    root = join(tmpdir, 'data')
    os.makedirs(join(root, 'r'))
    fits.PrimaryHDU(np.ones((4, 4))).writeto(join(root, 'r', 'a.fits.gz'))
    fits.PrimaryHDU(np.ones((4, 4))).writeto(join(root, 'r', 'b.fits.fz'))
    _png(join(root, 'rgb', 'a.png'))
    _png(join(root, 'rgb', 'c.png'))
    ds = _load(['-p', root, '-b', 'r', '-B', 'rgb', '--rgb-composites', ''])
    assert ds.listimage == ['a.fits.gz', 'b.fits.fz', 'c.png']
    assert ds.band_filetype == {'r': 'FITS', 'rgb': 'COMPRESSED'}
    a = load_stamp(ds.band_sources, 'r', fits_io.stamp_stem(ds.listimage[0]))
    assert a.is_fits and a.pixels.shape == (4, 4)
    assert not load_stamp(ds.band_sources, 'rgb', 'a').is_fits
    flat = _load(['-p', join(root, 'r')])   # a flat directory of compressed FITS
    assert flat.listimage == ['a.fits.gz', 'b.fits.fz']


def test_a_file_that_cannot_be_read_is_no_data_not_an_exception(tmpdir):
    root = join(tmpdir, 'data')
    _fits_band(root, 'r', ['a'])
    for name in ('bad.fits', 'bad2.png'):
        with open(join(root, 'r', name), 'wb') as f:
            f.write(b'not an image')
    fits.PrimaryHDU(np.zeros((3, 4, 4), np.float32)).writeto(join(root, 'r', 'cube.fits'))
    ds = _load(['-p', root, '-b', 'r', '-B', '', '--rgb-composites', ''])
    for stem in ('bad', 'bad2', 'cube'):
        try:
            load_stamp(ds.band_sources, 'r', stem)
        except fits_io.MissingBandError as e:
            assert e.unreadable and e.band == 'r' and e.detail
        else:
            raise AssertionError(f'{stem}: no MissingBandError')
    try:
        load_stamp(ds.band_sources, 'r', 'absent')
    except fits_io.MissingBandError as e:
        assert not e.unreadable
    else:
        raise AssertionError('no MissingBandError')


def test_composite_is_made_only_from_three_fits_stamps_of_one_size(tmpdir):
    def stamp(shape=(4, 4), header=True):
        return dataset.Stamp(np.zeros(shape), {} if header else None, 'x')
    members = ('r', 'g', 'b')
    full = {band: stamp() for band in members}
    stacked, why = composite_stack(full, members)
    assert stacked.shape == (4, 4, 3) and why is None
    assert composite_stack({'r': stamp(), 'b': stamp()}, members) == (None, ['g'])
    assert composite_stack({**full, 'g': stamp(header=False)}, members) == (None, ['g', 'not FITS'])
    assert composite_stack({**full, 'g': stamp((5, 4))}, members) == (None, ['sizes differ'])


def test_viewer_bands_are_the_bands_the_viewer_lists_from(tmpdir):
    "Composite members count, unless the viewer drops the composite."
    root = join(tmpdir, 'data')
    for band in ('r', 'g', 'b'):
        _fits_band(root, band, ['a'])
    _png(join(root, 'jpg', 'only_here.jpg'))
    _fits_band(root, 'i', ['only_i'])
    assert dataset.viewer_bands(root, 'r', ['g'], [('r', 'g', 'b'), ('r', 'g', 'jpg')], False) == \
        ['r', 'g', 'b']
    ds = _load(['-p', root, '-b', 'r', '-B', 'g', '--rgb-composites', 'r,g,i;r,g,jpg'])
    assert ds.single_bands == dataset.viewer_bands(
        root, 'r', ['g'], [('r', 'g', 'i'), ('r', 'g', 'jpg')], False) == ['r', 'g', 'i']
    assert ds.listimage == ['a.fits', 'only_i.fits'] == \
        dataset.list_dataset_objects(root, ds.single_bands, False)


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
