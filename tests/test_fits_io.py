"""Tests for fits_io.py -- the shared FITS reading module for mosaic.py and
single_viewer.py.

Covers both dataset layouts it supports: directory-per-band (DirBandSource,
resolve/find_band_file, detect_band_filetype/list_objects) and multi-extension
FITS (ExtensionBandSource, discover_mef_bands, is_mef_dataset), plus the reading
layer (read_pixels/read_header/read_pixels_and_header, FITSReadError) and RA/Dec
extraction (get_ra_dec, pixel_scale_arcsec), built against synthetic in-test FITS data -- no fixture
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


def test_get_ra_dec_non_square_image(tmpdir):
    "Regression: array_shape is (ny, nx), pixel_to_world_values wants (x, y) -- see BUGS.md."
    header = _synthetic_wcs_header(shape=(40, 100), crval=(150.0, 2.0), cdelt_arcsec=0.2)
    radec = fits_io.get_ra_dec(header)
    assert np.isclose(radec.ra, 150.0, atol=1e-3)
    assert np.isclose(radec.dec, 2.0, atol=1e-3)
    assert radec.image_dim == 100


def test_get_ra_dec_rotated_image():
    "Regression (BUGS 37): the pixel-scale matrix diagonal read 0.0707 at 45 deg and 0 at 90."
    for c, s in ((np.sqrt(0.5), np.sqrt(0.5)), (0.0, 1.0)):
        header = _synthetic_wcs_header(shape=(120, 120), cdelt_arcsec=0.1)
        header['PC1_1'], header['PC1_2'], header['PC2_1'], header['PC2_2'] = c, -s, s, c
        assert np.isclose(fits_io.get_ra_dec(header).pixel_size_arcsec, 0.1, atol=1e-4)


def test_pixel_scale_arcsec():
    header = _synthetic_wcs_header(shape=(120, 120), cdelt_arcsec=0.1)
    assert np.isclose(fits_io.pixel_scale_arcsec(header), 0.1)


def test_pixel_scale_arcsec_rotated_image():
    "The diagonal of the pixel-scale matrix is 0 at 90 degrees; the scale is not."
    header = _synthetic_wcs_header(shape=(120, 120), cdelt_arcsec=0.1)
    header['PC1_1'], header['PC1_2'], header['PC2_1'], header['PC2_2'] = 0.0, -1.0, 1.0, 0.0
    assert np.isclose(fits_io.pixel_scale_arcsec(header), 0.1)


def test_pixel_scale_arcsec_without_celestial_wcs():
    assert fits_io.pixel_scale_arcsec(fits.Header({'NAXIS': 2, 'NAXIS1': 21, 'NAXIS2': 21})) is None
    linear = fits.Header({'NAXIS': 2, 'CTYPE1': 'LINEAR', 'CTYPE2': 'LINEAR',
                          'CDELT1': 1.0, 'CDELT2': 1.0})
    assert fits_io.pixel_scale_arcsec(linear) is None


def test_pixel_scale_arcsec_cache_follows_the_scale_cards():
    "Cached per survey, not per stamp: a new position reuses it, a new scale does not."
    fits_io._pixel_scale_from_cards.cache_clear()
    assert np.isclose(fits_io.pixel_scale_arcsec(_synthetic_wcs_header(crval=(150.0, 2.0))), 0.1)
    assert np.isclose(fits_io.pixel_scale_arcsec(_synthetic_wcs_header(crval=(10.0, -30.0))), 0.1)
    assert np.isclose(fits_io.pixel_scale_arcsec(_synthetic_wcs_header(cdelt_arcsec=0.262)), 0.262)
    info = fits_io._pixel_scale_from_cards.cache_info()
    assert (info.misses, info.hits) == (2, 1)


def test_pixel_scale_arcsec_ignores_cards_outside_the_scale():
    """Cards outside the scale must not change it: a stray SIP card (which breaks a full WCS
    parse) or a third axis, in either order against a clean header sharing the cache."""
    clean = _synthetic_wcs_header()
    odd = [clean.copy(), clean.copy()]
    odd[0]['A_ORDER'] = 2
    odd[1]['WCSAXES'], odd[1]['NAXIS'] = 3, 3
    for header in odd:
        for order in ((clean, header), (header, clean)):
            fits_io._pixel_scale_from_cards.cache_clear()
            assert [round(fits_io.pixel_scale_arcsec(h), 6) for h in order] == [0.1, 0.1]


# --- MEF discovery -------------------------------------------------------------

def test_discover_mef_bands(tmpdir):
    path = _write_mef(tmpdir, 'obj1.fits', [
        ('VIS', np.zeros((2, 2))),
        ('Y', np.ones((2, 2))),
        (None, np.zeros((2, 2))),   # no EXTNAME -- falls back to 'HDU{index}'
    ])
    bands = fits_io.discover_mef_bands(path)
    # Named bands are keyed by EXTNAME, looked up per file; only the unnamed one by index.
    assert bands == {'VIS': 'VIS', 'Y': 'Y', 'HDU3': 3}


def test_discover_mef_bands_skips_empty_primary(tmpdir):
    path = _write_mef(tmpdir, 'obj1.fits', [('VIS', np.zeros((2, 2)))])
    bands = fits_io.discover_mef_bands(path)
    assert 0 not in bands.values()   # empty PrimaryHDU excluded
    assert bands == {'VIS': 'VIS'}


def test_discover_mef_bands_repeated_extname_keeps_first(tmpdir):
    "The HDU `hdu_list[name]` will read, not the last one enumerated."
    path = _write_mef(tmpdir, 'obj1.fits', [('VIS', np.zeros((2, 2))), ('VIS', np.ones((2, 2)))])
    assert fits_io.discover_mef_bands(path) == {'VIS': 'VIS'}
    np.testing.assert_array_equal(fits_io.read_pixels(fits_io.Locator(path, 'VIS')), np.zeros((2, 2)))


# --- MEF files whose layouts differ (BUGS 36) -----------------------------------

def _two_layouts(tmpdir):
    """Like the shipped test set: obj1 has VIS, J, H; obj2 has no J, so H sits where
    obj1's J does. The sample band map comes from obj1, as the viewers build it."""
    _write_mef(tmpdir, 'obj1.fits', [('VIS', np.full((2, 2), 1.)), ('J', np.full((2, 2), 2.)),
                                     ('H', np.full((2, 2), 3.))])
    _write_mef(tmpdir, 'obj2.fits', [('VIS', np.full((2, 2), 1.)), ('H', np.full((2, 2), 3.))])
    bands = fits_io.discover_mef_bands(join(tmpdir, 'obj1.fits'))
    return {b: fits_io.ExtensionBandSource(tmpdir, key) for b, key in bands.items()}


def test_mef_band_is_read_by_name_not_position(tmpdir):
    sources = _two_layouts(tmpdir)
    # Used to read HDU 3 of obj2 -- which doesn't exist -- and J's HDU 2 was obj2's H.
    image, header = fits_io.read_pixels_and_header(fits_io.resolve(sources['H'], 'obj2'))
    assert header['EXTNAME'] == 'H'
    np.testing.assert_array_equal(image, np.full((2, 2), 3.))
    assert fits_io.read_header(fits_io.resolve(sources['H'], 'obj2'))['EXTNAME'] == 'H'


def test_mef_band_absent_from_one_file_is_missing_not_another_band(tmpdir):
    sources = _two_layouts(tmpdir)
    locator = fits_io.resolve(sources['J'], 'obj2')
    for read in (fits_io.read_pixels, fits_io.read_pixels_and_header, fits_io.read_header):
        try:
            read(locator)
            assert False, f"{read.__name__}: expected MissingBandError"
        except fits_io.MissingBandError as e:
            assert e.band == 'J'
            assert e.filepath == join(tmpdir, 'obj2.fits')
            assert e.available == ['VIS', 'H']


def test_mef_extension_without_data_is_missing(tmpdir):
    path = _write_mef(tmpdir, 'obj1.fits', [('VIS', np.zeros((2, 2))), ('J', None)])
    try:
        fits_io.read_pixels(fits_io.Locator(path, 'J'))
        assert False, "expected MissingBandError"
    except fits_io.MissingBandError as e:
        assert e.available == ['VIS']


def test_locate_mef_bands_looks_past_a_first_file_that_lacks_one(tmpdir):
    "A run whose first (or first shuffled) file lacks J used to refuse to start."
    _two_layouts(tmpdir)
    found, seen = fits_io.locate_mef_bands(tmpdir, ['obj2.fits', 'obj1.fits'], {'VIS', 'J'})
    assert found == {'VIS': 'VIS', 'J': 'J'}
    assert seen == ['H', 'J', 'VIS']


def test_locate_mef_bands_stops_once_everything_is_found(tmpdir):
    _two_layouts(tmpdir)
    with mock.patch('fits_io.discover_mef_bands', wraps=fits_io.discover_mef_bands) as spy:
        found, _ = fits_io.locate_mef_bands(tmpdir, ['obj1.fits', 'obj2.fits'], {'VIS', 'J'})
    assert found == {'VIS': 'VIS', 'J': 'J'}
    assert spy.call_count == 1


def test_locate_mef_bands_reports_a_band_no_file_has_within_the_cap(tmpdir):
    _two_layouts(tmpdir)
    found, seen = fits_io.locate_mef_bands(tmpdir, ['obj1.fits', 'obj2.fits'], {'VIS', 'K'})
    assert 'K' not in found and seen == ['H', 'J', 'VIS']
    found, _ = fits_io.locate_mef_bands(tmpdir, ['obj2.fits', 'obj1.fits'], {'J'}, max_files=1)
    assert found == {}


def test_missing_bands_note():
    mef = {'J': fits_io.MissingBandError('J', '/d/obj2.fits', available=['VIS', 'H']),
           'J_RMS': fits_io.MissingBandError('J_RMS', '/d/obj2.fits', available=['VIS', 'H'])}
    assert fits_io.missing_bands_note(mef) == "no J, J_RMS -- file has: VIS, H"
    # Directory-per-band: no single file to ask what it has.
    assert fits_io.missing_bands_note({'Y': fits_io.MissingBandError('Y')}) == "no Y"


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
