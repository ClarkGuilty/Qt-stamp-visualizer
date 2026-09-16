"""Tests for fits_io.py -- the shared FITS reading module for mosaic.py and
single_viewer.py.

Covers both dataset layouts it supports: directory-per-band (DirBandSource,
resolve/find_band_file, detect_band_filetype/list_objects) and multi-extension
FITS (ExtensionBandSource, discover_mef_bands, is_mef_dataset), plus the reading
layer (read_pixels/read_header/read_pixels_and_header, FITSReadError) and RA/Dec
extraction (get_ra_dec), built against synthetic in-test FITS data -- no fixture
files needed.

Qt-free, so it runs headless. Plain pytest functions, also runnable directly
(`python tests/test_fits_io.py`).
"""

import os
import sys
from os.path import join
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import fits_io


def _write_fits(directory, name, data, header=None):
    path = join(directory, name)
    fits.PrimaryHDU(data=data, header=header).writeto(path)
    return path


def _write_mef(directory, name, extensions):
    "extensions: list of (extname_or_none, data_or_none)."
    hdus = [fits.PrimaryHDU()]
    for extname, data in extensions:
        hdus.append(fits.ImageHDU(data=data, name=extname))
    path = join(directory, name)
    fits.HDUList(hdus).writeto(path)
    return path


def _synthetic_wcs_header(shape=(10, 10), crval=(150.0, 2.0), cdelt_arcsec=0.1):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = [-cdelt_arcsec / 3600, cdelt_arcsec / 3600]
    w.wcs.crval = list(crval)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    header = w.to_header()
    header['NAXIS'] = 2
    header['NAXIS1'] = shape[1]
    header['NAXIS2'] = shape[0]
    return header


# --- directory-per-band discovery -------------------------------------------

def test_detect_band_filetype(tmpdir):
    fits_dir = join(tmpdir, 'fits_band')
    os.makedirs(fits_dir)
    _write_fits(fits_dir, 'a.fits', np.zeros((2, 2)))
    assert fits_io.detect_band_filetype(fits_dir) == 'FITS'

    png_dir = join(tmpdir, 'png_band')
    os.makedirs(png_dir)
    open(join(png_dir, 'a.png'), 'wb').close()
    assert fits_io.detect_band_filetype(png_dir) == 'COMPRESSED'

    empty_dir = join(tmpdir, 'empty_band')
    os.makedirs(empty_dir)
    assert fits_io.detect_band_filetype(empty_dir) is None
    assert fits_io.detect_band_filetype(join(tmpdir, 'does_not_exist')) is None


def test_list_objects_prefers_fits(tmpdir):
    _write_fits(tmpdir, 'b.fits', np.zeros((2, 2)))
    _write_fits(tmpdir, 'a.fits', np.zeros((2, 2)))
    open(join(tmpdir, 'z.png'), 'wb').close()   # ignored -- FITS present
    names, filetype = fits_io.list_objects(tmpdir)
    assert names == ['a.fits', 'b.fits']
    assert filetype == 'FITS'


def test_list_objects_falls_back_to_compressed_combined(tmpdir):
    # png/jpg/jpeg are combined into one sorted list, not tried one at a time.
    open(join(tmpdir, 'b.jpg'), 'wb').close()
    open(join(tmpdir, 'a.png'), 'wb').close()
    open(join(tmpdir, 'c.jpeg'), 'wb').close()
    names, filetype = fits_io.list_objects(tmpdir)
    assert names == ['a.png', 'b.jpg', 'c.jpeg']
    assert filetype == 'COMPRESSED'


def test_list_objects_empty(tmpdir):
    names, filetype = fits_io.list_objects(tmpdir)
    assert names == []
    assert filetype == 'COMPRESSED'


def test_find_band_file(tmpdir):
    band_dir = join(tmpdir, 'Y')
    os.makedirs(band_dir)
    _write_fits(band_dir, 'obj1.fits', np.zeros((2, 2)))
    open(join(band_dir, 'obj2.png'), 'wb').close()

    assert fits_io.find_band_file(tmpdir, 'Y', 'obj1') == join(band_dir, 'obj1.fits')
    assert fits_io.find_band_file(tmpdir, 'Y', 'obj2') == join(band_dir, 'obj2.png')
    assert fits_io.find_band_file(tmpdir, 'Y', 'missing') is None


# --- resolve() / BandSource ---------------------------------------------------

def test_resolve_dir_band_source_fits_and_compressed(tmpdir):
    band_dir = join(tmpdir, 'VIS')
    os.makedirs(band_dir)
    _write_fits(band_dir, 'obj1.fits', np.zeros((2, 2)))
    open(join(band_dir, 'obj2.jpg'), 'wb').close()
    source = fits_io.DirBandSource(band_dir, 'FITS')

    locator = fits_io.resolve(source, 'obj1')
    assert locator == fits_io.Locator(join(band_dir, 'obj1.fits'), 0)

    locator = fits_io.resolve(source, 'obj2')
    assert locator == fits_io.Locator(join(band_dir, 'obj2.jpg'), None)

    assert fits_io.resolve(source, 'missing') is None


def test_resolve_extension_band_source(tmpdir):
    _write_mef(tmpdir, 'obj1.fits', [('VIS', np.zeros((2, 2))), ('Y', np.ones((2, 2)))])
    source = fits_io.ExtensionBandSource(tmpdir, 2)

    locator = fits_io.resolve(source, 'obj1')
    assert locator == fits_io.Locator(join(tmpdir, 'obj1.fits'), 2)

    assert fits_io.resolve(source, 'missing') is None


# --- reading -------------------------------------------------------------

def test_read_pixels_roundtrip(tmpdir):
    data = np.arange(9, dtype='float32').reshape(3, 3)
    path = _write_fits(tmpdir, 'a.fits', data)
    image = fits_io.read_pixels(fits_io.Locator(path, 0))
    np.testing.assert_array_equal(image, data)


def test_read_pixels_closes_the_file(tmpdir):
    "Regression test for single_viewer's old load_fits, which never closed its HDUList."
    path = _write_fits(tmpdir, 'a.fits', np.zeros((2, 2), dtype='float32'))
    opened = []
    real_open = fits.open

    def spying_open(*args, **kwargs):
        hdu_list = real_open(*args, **kwargs)
        opened.append(hdu_list)
        return hdu_list

    with mock.patch('fits_io.fits.open', side_effect=spying_open):
        fits_io.read_pixels(fits_io.Locator(path, 0))

    assert opened and opened[0]._file.closed


def test_read_pixels_missing_file_raises(tmpdir):
    locator = fits_io.Locator(join(tmpdir, 'nope.fits'), 0)
    try:
        fits_io.read_pixels(locator)
        assert False, "expected FITSReadError"
    except fits_io.FITSReadError:
        pass


def test_read_pixels_bad_hdu_raises(tmpdir):
    path = _write_fits(tmpdir, 'a.fits', np.zeros((2, 2)))
    locator = fits_io.Locator(path, 5)   # only HDU 0 exists
    try:
        fits_io.read_pixels(locator)
        assert False, "expected FITSReadError"
    except fits_io.FITSReadError:
        pass


def test_read_header_and_read_pixels_and_header_agree(tmpdir):
    data = np.ones((4, 4), dtype='float32')
    header = _synthetic_wcs_header(shape=(4, 4))
    path = _write_fits(tmpdir, 'a.fits', data, header=header)
    locator = fits_io.Locator(path, 0)

    header_only = fits_io.read_header(locator)
    image, header_combined = fits_io.read_pixels_and_header(locator)

    np.testing.assert_array_equal(image, data)
    assert header_only['CRVAL1'] == header_combined['CRVAL1']
    assert header_only['CRVAL1'] == header['CRVAL1']


# --- WCS / RA-Dec --------------------------------------------------------------

def test_get_ra_dec(tmpdir):
    header = _synthetic_wcs_header(shape=(10, 10), crval=(150.0, 2.0), cdelt_arcsec=0.2)
    radec = fits_io.get_ra_dec(header)
    assert np.isclose(radec.ra, 150.0, atol=1e-3)
    assert np.isclose(radec.dec, 2.0, atol=1e-3)
    assert np.isclose(radec.pixel_size_arcsec, 0.2, atol=1e-4)
    assert radec.image_dim == 10


# --- MEF discovery -------------------------------------------------------------

def test_discover_mef_bands(tmpdir):
    path = _write_mef(tmpdir, 'obj1.fits', [
        ('VIS', np.zeros((2, 2))),
        ('Y', np.ones((2, 2))),
        (None, np.zeros((2, 2))),   # no EXTNAME -- falls back to 'HDU{index}'
    ])
    bands = fits_io.discover_mef_bands(path)
    assert bands == {'VIS': 1, 'Y': 2, 'HDU3': 3}


def test_discover_mef_bands_skips_empty_primary(tmpdir):
    path = _write_mef(tmpdir, 'obj1.fits', [('VIS', np.zeros((2, 2)))])
    bands = fits_io.discover_mef_bands(path)
    assert 0 not in bands.values()   # empty PrimaryHDU excluded
    assert bands == {'VIS': 1}


def test_is_mef_dataset(tmpdir):
    subdir_only = join(tmpdir, 'directories_only')
    os.makedirs(join(subdir_only, 'VIS'))
    assert fits_io.is_mef_dataset(subdir_only) is False

    flat = join(tmpdir, 'flat_fits')
    os.makedirs(flat)
    _write_fits(flat, 'obj1.fits', np.zeros((2, 2)))
    assert fits_io.is_mef_dataset(flat) is True


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
