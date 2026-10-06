"""Tests for imaging.open_rendered_image -- how both viewers read a PNG/JPG stamp --
and for corner_box_size/background_rms_image, which set a FITS band's black point.

Qt-free, so it runs headless. Plain pytest functions, also runnable directly
(`python tests/test_imaging.py`).
"""

import os
import sys
from os.path import join

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from imaging import open_rendered_image, corner_box_size, background_rms_image


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
