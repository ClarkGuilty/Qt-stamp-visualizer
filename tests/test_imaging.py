"""Tests for imaging.open_rendered_image -- how both viewers read a PNG/JPG stamp --
for corner_box_size/background_rms_image, which set a FITS band's black point, and
for the display scaling both viewers share (scale_limits, rescale_image,
composite_image), against what each viewer drew before REFACTOR.md step 4.

Qt-free, so it runs headless. Plain pytest functions, also runnable directly
(`python tests/test_imaging.py`).
"""

import os
import sys
from os.path import join

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from qtstamp.imaging import (open_rendered_image, corner_box_size, background_rms_image,
                     SCALES, MOSAIC_DISPLAY, SINGLE_DISPLAY,
                     scale_limits, rescale_image, composite_image, get_value_range_asymmetric)


def _stamp():
    "Black sky, a bright orange source, a red marker: the colours a palette would hold."
    yy, xx = np.mgrid[:40, :40]
    src = np.clip(255 * np.exp(-((xx - 20) ** 2 + (yy - 20) ** 2) / 18.), 0, 255).astype('uint8')
    rgb = np.stack([src, (src * 0.7).astype('uint8'), src // 3], -1)
    rgb[2:4, 2:20] = [255, 0, 0]
    return rgb


def test_palette_png_reads_as_colours_not_indices(tmpdir):
    # PIL's quantize puts black at the highest index: read raw, the sky would be the max.
    rgb = _stamp()
    path = join(tmpdir, 'p.png')
    Image.fromarray(rgb).quantize(colors=32).save(path)
    with Image.open(path) as im:
        assert im.mode == 'P' and np.asarray(im)[0, 0] == np.asarray(im).max()
    out = np.asarray(open_rendered_image(path))
    assert out.shape == (40, 40, 3)
    assert (out[0, 0] == 0).all()
    assert (out[2, 10] == [255, 0, 0]).all()


def test_palette_png_with_transparency_keeps_alpha(tmpdir):
    path = join(tmpdir, 'pt.png')
    im = Image.fromarray(_stamp()).quantize(colors=32)
    im.info['transparency'] = int(np.asarray(im)[0, 0])
    im.save(path, transparency=im.info['transparency'])
    out = np.asarray(open_rendered_image(path))
    assert out.shape == (40, 40, 4)
    assert out[0, 0, 3] == 0 and out[20, 20, 3] == 255


def test_cmyk_jpg_becomes_rgb_and_saves_as_png(tmpdir):
    src = join(tmpdir, 'c.jpg')
    Image.fromarray(_stamp()).convert('CMYK').save(src, quality=100)
    im = open_rendered_image(src)
    assert im.mode == 'RGB'
    assert np.asarray(im)[0, 0].max() < 10  # black stays black
    im.save(join(tmpdir, 'copy.png'))       # the mosaic's tile copy: PNG cannot hold CMYK


def test_grey_with_alpha_becomes_rgba(tmpdir):
    path = join(tmpdir, 'la.png')
    grey = np.full((8, 8), 200, 'uint8')
    alpha = np.full((8, 8), 255, 'uint8')
    alpha[:, :4] = 0
    Image.fromarray(np.stack([grey, alpha], -1), 'LA').save(path)
    out = np.asarray(open_rendered_image(path))
    assert out.shape == (8, 8, 4)
    assert (out[0, 7] == [200, 200, 200, 255]).all() and out[0, 0, 3] == 0


def test_bilevel_becomes_greyscale(tmpdir):
    path = join(tmpdir, 'b.png')
    Image.fromarray(np.eye(8, dtype=bool)).save(path)
    out = open_rendered_image(path)
    assert out.mode == 'L'
    assert np.asarray(out)[0, 0] == 255 and np.asarray(out)[0, 1] == 0


def test_display_ready_modes_are_untouched(tmpdir):
    rgb = _stamp()
    cases = {'L': rgb[..., 0], 'RGB': rgb, 'RGBA': np.dstack([rgb, np.full((40, 40), 255, 'uint8')]),
             'I;16': (rgb[..., 0].astype('uint16') * 257)}
    for mode, arr in cases.items():
        path = join(tmpdir, f'{mode.replace(";", "")}.png')
        Image.fromarray(arr).save(path)
        with Image.open(path) as raw:
            expected_mode, expected = raw.mode, np.asarray(raw).copy()
        out = open_rendered_image(path)
        assert out.mode == expected_mode, mode
        assert np.array_equal(np.asarray(out), expected), mode


def test_result_outlives_the_file(tmpdir):
    # Returned from inside the `with`: it must be a loaded copy, not the closed file's image.
    path = join(tmpdir, 'rgb.png')
    Image.fromarray(_stamp()).save(path)
    im = open_rendered_image(path)
    os.remove(path)
    assert np.asarray(im).shape == (40, 40, 3)


# --- black point of a single FITS band -----------------------------------------

def test_corner_box_size_without_wcs():
    assert corner_box_size((120, 120)) == 10       # the old fixed box
    assert corner_box_size((36, 36)) == 9          # a quarter of the side
    assert corner_box_size((500, 500)) == 16       # 0.1% of the area
    assert corner_box_size((1000, 1000)) == 20     # CORNER_BOX_MAX
    assert corner_box_size((36, 400)) == 9         # the shorter side sets the quarter


def test_corner_box_size_with_wcs_caps_at_one_arcsec():
    assert corner_box_size((120, 120), 0.1) == 10
    assert corner_box_size((1000, 1000), 0.1) == 10
    assert corner_box_size((1000, 1000), 0.05) == 20
    assert corner_box_size((2000, 2000), 0.03) == 33   # 1 arcsec can exceed CORNER_BOX_MAX
    assert corner_box_size((120, 120), 0.1 * (1 + 1e-12)) == 10   # float noise in CD


def test_corner_box_size_floor_wins_over_the_arcsec_cap():
    assert corner_box_size((120, 120), 0.262) == 7    # 1 arcsec is 4 px here


def test_background_rms_image_uses_the_box_it_is_given():
    image = np.random.default_rng(0).normal(size=(60, 60))
    for corner in (np.s_[:7, :7], np.s_[-7:, :7], np.s_[:7, -7:], np.s_[-7:, -7:]):
        image[corner] = 1.0
    assert background_rms_image(7, image) == 0
    assert background_rms_image(10, image) > 0


# --- display scaling, one parameter set per viewer -----------------------------

DISPLAYS = {'mosaic': MOSAIC_DISPLAY, 'single': SINGLE_DISPLAY}

# Fingerprints (see _fingerprint) of what each viewer's own methods returned
# before they moved to imaging.py (as of 60baa34), on the stamps from
# _scaling_stamps, in the dtype the viewers pass (FITS float32): float64 copies
# give other values.
FROZEN = {
    'mosaic/band120/asinh/rescale_image': [245.66570060886443, 57.010143281252496, 0.008265714161098003, 0.9178245067596436, 0, 124.42656294868563],
    'mosaic/band120/asinh/rescale_inverted': [14400.0, 14400.0, 1.0, 1.0, 0, 7222.85043893467],
    'mosaic/band120/cbrt/rescale_image': [2092.7647589445114, 351.9207423341663, 0.13075034320354462, 0.9169425368309021, 0, 1051.5304331210814],
    'mosaic/band120/cbrt/rescale_inverted': [14400.0, 14400.0, 1.0, 1.0, 0, 7222.85043893467],
    'mosaic/band120/linear/rescale_image': [88.05988488532603, 18.900083783122327, 0.002235261956229806, 0.7709501385688782, 0, 44.6866189396176],
    'mosaic/band120/linear/rescale_inverted': [14400.0, 14400.0, 1.0, 1.0, 0, 7222.85043893467],
    'mosaic/band120/log/rescale_image': [2995.297684893012, 709.5661482677287, 0.18156813085079193, 0.963197648525238, 0, 1505.6661860423505],
    'mosaic/band120/log/rescale_inverted': [14400.0, 14400.0, 1.0, 1.0, 0, 7222.85043893467],
    'mosaic/band120/scale_val': [0.002604305511340499, 1.1677050590515137],
    'mosaic/band120/sqrt/rescale_image': [832.781416464597, 88.05988052553519, 0.04727855697274208, 0.8780376315116882, 0, 419.05220355760116],
    'mosaic/band120/sqrt/rescale_inverted': [14400.0, 14400.0, 1.0, 1.0, 0, 7222.85043893467],
    'mosaic/composite/asinh': [943.5899913587184, 174.1262220844935, 0.005004289214600943, 0.9994106726421074, 0, 472.68405020047703],
    'mosaic/composite/cbrt': [1123.3283881438242, 217.22259831017834, 0.0, 1.0, 0, 562.3211995641348],
    'mosaic/composite/linear': [799.4529147183241, 76.83673873489869, 0.012021985892503528, 1.0, 0, 400.7536531038063],
    'mosaic/composite/log': [1734.071591558804, 416.38970148950085, 0.0, 1.0, 0, 867.556303681085],
    'mosaic/composite/sqrt': [938.0042532486466, 155.9792034197909, 0.0, 1.0, 0, 469.7985969377585],
    'mosaic/synth172/asinh/rescale_image': [646.1902388368035, 391.34670004429654, 0.001274959067814052, 1.0, 0, 339.4890227670345],
    'mosaic/synth172/asinh/rescale_inverted': [29584.0, 29584.0, 1.0, 1.0, 0, 14825.796520394542],
    'mosaic/synth172/cbrt/rescale_image': [992.9449956919998, 227.3358010782177, 0.019872324541211128, 1.0, 0, 509.4752826582114],
    'mosaic/synth172/cbrt/rescale_inverted': [29584.0, 29584.0, 1.0, 1.0, 0, 14825.796520394542],
    'mosaic/synth172/linear/rescale_image': [139.4890132632636, 64.37064082769477, 7.847765118640382e-06, 1.0, 0, 76.97560231423704],
    'mosaic/synth172/linear/rescale_inverted': [29584.0, 29584.0, 1.0, 1.0, 0, 14825.796520394542],
    'mosaic/synth172/log/rescale_image': [2549.3793840520084, 693.1186674324446, 0.05896470323204994, 1.0, 0, 1294.4135090727154],
    'mosaic/synth172/log/rescale_inverted': [29584.0, 29584.0, 1.0, 1.0, 0, 14825.796520394542],
    'mosaic/synth172/scale_val': [0.0010000000474974513, 127.42581939697266],
    'mosaic/synth172/sqrt/rescale_image': [371.19052535970695, 139.48900920280585, 0.002801386406645179, 1.0, 0, 196.09127331736292],
    'mosaic/synth172/sqrt/rescale_inverted': [29584.0, 29584.0, 1.0, 1.0, 0, 14825.796520394542],
    'single/band120/asinh/rescale_image': [245.66570060886443, 57.010143281252496, 0.008265714161098003, 0.9178245067596436, 0, 124.42656294868563],
    'single/band120/asinh/rescale_inverted': [0.0, 0.0, 0.0, 0.0, 0, 0.0],
    'single/band120/cbrt/rescale_image': [2092.7647589445114, 351.9207423341663, 0.13075034320354462, 0.9169425368309021, 0, 1051.5304331210814],
    'single/band120/cbrt/rescale_inverted': [0.0, 0.0, 0.0, 0.0, 0, 0.0],
    'single/band120/linear/rescale_image': [88.05988488532603, 18.900083783122327, 0.002235261956229806, 0.7709501385688782, 0, 44.6866189396176],
    'single/band120/linear/rescale_inverted': [0.0, 0.0, 0.0, 0.0, 0, 0.0],
    'single/band120/log/rescale_image': [2995.297684893012, 709.5661482677287, 0.18156813085079193, 0.963197648525238, 0, 1505.6661860423505],
    'single/band120/log/rescale_inverted': [0.0, 0.0, 0.0, 0.0, 0, 0.0],
    'single/band120/scale_val': [0.002604305511340499, 1.1677050590515137],
    'single/band120/sqrt/rescale_image': [832.781416464597, 88.05988052553519, 0.04727855697274208, 0.8780376315116882, 0, 419.05220355760116],
    'single/band120/sqrt/rescale_inverted': [0.0, 0.0, 0.0, 0.0, 0, 0.0],
    'single/composite/asinh': [263.28807608479747, 147.2531779895547, 0.0, 0.9993725364289272, 0, 131.6452847766054],
    'single/composite/cbrt': [340.47445839994305, 171.6780496319684, 0.0, 1.0, 0, 170.26358050825593],
    'single/composite/linear': [122.37768173872857, 57.09789827071609, 0.0, 1.0, 0, 61.363940291911575],
    'single/composite/log': [739.0006295653134, 326.22245148537667, 0.0, 1.0, 0, 369.43861556033636],
    'single/composite/sqrt': [255.56389018511373, 123.51825623393172, 0.0, 1.0, 0, 127.82763865287816],
    'single/synth172/asinh/rescale_image': [609.2597698355094, 389.3940908378629, 0.0, 1.0, 0, 320.989766024442],
    'single/synth172/asinh/rescale_inverted': [0.0, 0.0, 0.0, 0.0, 0, 0.0],
    'single/synth172/cbrt/rescale_image': [438.0872672870755, 213.71077367709876, 0.0, 1.0, 0, 231.99204189367066],
    'single/synth172/cbrt/rescale_inverted': [0.0, 0.0, 0.0, 0.0, 0, 0.0],
    'single/synth172/linear/rescale_image': [136.82912516223587, 62.47326420534945, 0.0, 1.0, 0, 75.55508950061639],
    'single/synth172/linear/rescale_inverted': [0.0, 0.0, 0.0, 0.0, 0, 0.0],
    'single/synth172/log/rescale_image': [909.8775384128094, 594.5645549432516, 0.0, 1.0, 0, 474.6764755087461],
    'single/synth172/log/rescale_inverted': [0.0, 0.0, 0.0, 0.0, 0, 0.0],
    'single/synth172/scale_val': [0.0, 130.0],
    'single/synth172/sqrt/rescale_image': [290.60984130203724, 136.82912128379024, 0.0, 1.0, 0, 155.72493883458174],
    'single/synth172/sqrt/rescale_inverted': [0.0, 0.0, 0.0, 0.0, 0, 0.0],
}


def _fingerprint(a):
    "Sum, sum of squares, min, max, NaN count, and a fixed random weighting: moves if any pixel does."
    a = np.asarray(a, dtype=float)
    w = np.random.default_rng(0).random(a.shape)
    return [np.nansum(a), np.nansum(a * a), np.nanmin(a), np.nanmax(a),
            np.isnan(a).sum(), np.nansum(a * w)]


def _scaling_stamps():
    """Fake stamps: a noisy 120 px band at 0.1"/px, like an optical cutout, a 172 px
    one -- between the two viewers' 170 and 173 px, peak just off centre, corners all
    0 -- and three noisy bands as a composite, in larger units. Built here, not by
    sample_data.py, so that changing the generator cannot move FROZEN."""
    rng = np.random.default_rng(1)
    y, x = np.mgrid[:120, :120]
    galaxy = np.exp(-((x - 61) ** 2 / 32. + (y - 58) ** 2 / 12.5))
    companion = np.exp(-((x - 30) ** 2 + (y - 85) ** 2) / 4.)
    band120 = (rng.normal(0, 0.004, (120, 120)) + 0.9 * galaxy + 0.1 * companion).astype(np.float32)
    stack = np.stack([rng.normal(0, 0.7, (120, 120)) + peak * (galaxy + share * companion)
                      for peak, share in ((550., 0.1), (470., 0.2), (240., 0.4))],
                     axis=-1).astype(np.float32)
    y, x = np.mgrid[:172, :172]
    synth = (100 * np.exp(-((x - 86) ** 2 + (y - 93) ** 2) / 50.)).astype(np.float32)
    synth[synth < 1e-3] = 0
    synth[60:64, 100:104] = 1e4      # brighter than the white point: clipped to it
    return {'band120': (band120, 0.1), 'synth172': (synth, None)}, stack


def _assert_frozen(key, got):
    np.testing.assert_allclose(np.asarray(got, dtype=float), FROZEN[key], rtol=1e-9, err_msg=key)


def test_scale_limits_as_each_viewer_had_them():
    stamps, _ = _scaling_stamps()
    for tool, display in DISPLAYS.items():
        for name, (image, pixel_scale) in stamps.items():
            _assert_frozen(f'{tool}/{name}/scale_val', scale_limits(image, display, pixel_scale))


def test_rescale_image_as_each_viewer_had_it():
    stamps, _ = _scaling_stamps()
    for tool, display in DISPLAYS.items():
        for name, (image, pixel_scale) in stamps.items():
            low, high = scale_limits(image, display, pixel_scale)
            for scale, func in SCALES.items():
                _assert_frozen(f'{tool}/{name}/{scale}/rescale_image',
                               _fingerprint(rescale_image(image.copy(), low, high, func, display)))
                # Inverted limits: everything clips to scale_max (REFACTOR.md step 4 review).
                _assert_frozen(f'{tool}/{name}/{scale}/rescale_inverted',
                               _fingerprint(rescale_image(image.copy(), high, low, func, display)))


def test_composite_image_as_each_viewer_had_it():
    _, stack = _scaling_stamps()
    for tool, display in DISPLAYS.items():
        for scale, func in SCALES.items():
            _assert_frozen(f'{tool}/composite/{scale}',
                           _fingerprint(composite_image(stack.copy(), func, display)))


def test_white_point_pixels():
    "The mosaic shows a pixel at the white point as 1.0; the 1-by-1 as scale(max)/scale(max-min)."
    image = np.array([1., 2., 3., 5.])
    linear = SCALES['linear']
    assert list(rescale_image(image.copy(), 1, 3, linear, MOSAIC_DISPLAY)) == [0.5, 1.0, 1.0, 1.0]
    assert list(rescale_image(image.copy(), 1, 3, linear, SINGLE_DISPLAY)) == [0.5, 1.0, 1.5, 1.5]


def test_zero_black_point_is_lifted_in_the_mosaic_only():
    stamps, _ = _scaling_stamps()
    image, _ = stamps['synth172']
    assert scale_limits(image, MOSAIC_DISPLAY)[0] == 1e-3
    assert scale_limits(image, SINGLE_DISPLAY)[0] == 0


def test_white_point_box_on_a_stamp_smaller_than_the_box():
    "BUGS 23: the centre box used to start at a negative index and slice the far edge."
    rng = np.random.default_rng(0)
    image = rng.normal(size=(10, 10))
    image[5, 5] = 100.
    assert scale_limits(image, SINGLE_DISPLAY)[1] == 130.
    stack = rng.normal(size=(5, 5, 3))
    stack[2, 2] = 50.
    assert get_value_range_asymmetric(stack, 2, 0.1)[1] == 50.


def test_flat_or_negative_composite_has_no_nan():
    "BUGS 24: a flat stack (e.g. outside coverage) was all NaN; an all-negative one inf/NaN."
    stacks = {'zeros': np.zeros((50, 50, 3)), 'flat': np.full((50, 50, 3), 5.),
              'negative': -np.abs(np.random.default_rng(1).normal(size=(50, 50, 3)))}
    for display in (MOSAIC_DISPLAY, SINGLE_DISPLAY):
        background = max(0., display.composite_bkg_level)
        for name, stack in stacks.items():
            for scale, func in SCALES.items():
                with np.errstate(all='raise'):
                    rgb = composite_image(stack.copy(), func, display)
                assert np.isfinite(rgb).all(), (name, scale)
                if name != 'negative':
                    assert np.allclose(rgb, background), (name, scale)


def test_flat_or_negative_band_scales_without_warnings():
    """BUGS 47: an empty range (a blank all-zero band, or one below 0) warned in both
    viewers, and was 0/0 in the 1-by-1. What each viewer draws is unchanged."""
    bands = {'zeros': np.zeros((50, 50)), 'flat': np.full((50, 50), 5.),
             'negative': np.full((50, 50), -2.)}
    drawn = {('zeros', 'mosaic'): 1.0, ('zeros', 'single'): np.nan,
             ('negative', 'mosaic'): 1.0, ('negative', 'single'): 0.0}
    for tool, display in (('mosaic', MOSAIC_DISPLAY), ('single', SINGLE_DISPLAY)):
        for name, band in bands.items():
            for scale, func in SCALES.items():
                with np.errstate(all='raise'):
                    scale_min, scale_max = scale_limits(band, display)
                    image = rescale_image(band.copy(), scale_min, scale_max, func, display)
                if name != 'flat':
                    assert np.allclose(image, drawn[name, tool], equal_nan=True), \
                        (name, tool, scale)


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
