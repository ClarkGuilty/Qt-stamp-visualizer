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

import cli
import fits_io
from dataset import DatasetError, load_dataset
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

    with mock.patch('fits_io.locate_mef_bands', spy):
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
