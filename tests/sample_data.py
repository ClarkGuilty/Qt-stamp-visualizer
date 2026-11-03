"""Fake stamps, so the tests and the snapshot need no survey data (none is in the repo).

`write_mef_set` writes the multi-extension layout (`--mef`): one FITS file per object,
one image extension per band, found by EXTNAME, with a north-up TAN WCS. Every third
file has no NIR_J, and one has an all-zero NIR_H (outside coverage). `write_band_dirs`
writes the other layout: a directory per band, one PNG/JPG per object under the same
stems, one band greyscale and one RGB. Both are seeded: a call writes the same pixels
every time.

Also a script, to try the tools without real data:

    python tests/sample_data.py OUT_DIR
"""

import os
import sys
from os.path import join

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

SIZE = 120                  # px
PIXEL_SCALE = 0.1           # arcsec/px
# EXTNAME: (noise sigma, peak of the central source). One optical band in small
# units and three near-IR ones in large units, so each band needs its own limits.
MEF_BANDS = {'VIS_BGSUB': (0.004, 0.9), 'NIR_Y_BGSUB': (0.6, 240.),
             'NIR_J_BGSUB': (0.7, 470.), 'NIR_H_BGSUB': (0.7, 550.)}
MISSING_EVERY = 3           # every third file has no NIR_J
BLANK = 4                   # this file's NIR_H is all zeros


def fake_object(rng, size=SIZE):
    "One object's sources: a galaxy near the centre plus fainter ones, the same in every band."
    n = int(rng.integers(3, 7))
    xy = np.vstack([size / 2 + rng.uniform(-3, 3, 2), rng.uniform(0, size, (n - 1, 2))])
    return {'xy': xy, 'flux': np.r_[1., rng.uniform(0.02, 0.3, n - 1)],
            'radius': rng.uniform(1.5, 5, n), 'axis_ratio': rng.uniform(0.4, 1, n),
            'angle': rng.uniform(0, np.pi, n), 'colour': rng.uniform(0.6, 1.6, n)}


def fake_band(rng, obj, sigma, peak, band_index=0, size=SIZE):
    "Background-subtracted noise plus the object's sources, as float32 like a FITS band."
    y, x = np.mgrid[:size, :size]
    image = rng.normal(0., sigma, (size, size))
    for (x0, y0), flux, r, q, a, c in zip(obj['xy'], obj['flux'], obj['radius'],
                                          obj['axis_ratio'], obj['angle'], obj['colour']):
        u = (x - x0) * np.cos(a) + (y - y0) * np.sin(a)
        v = (-(x - x0) * np.sin(a) + (y - y0) * np.cos(a)) / q
        image += peak * flux * c ** band_index * np.exp(-(u * u + v * v) / (2 * r * r))
    return image.astype(np.float32)


def _wcs_header(ra, dec, size=SIZE):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [(size + 1) / 2] * 2
    wcs.wcs.cd = np.diag([-PIXEL_SCALE, PIXEL_SCALE]) / 3600
    return wcs.to_header()


def write_mef_set(root, n=14, seed=0):
    "`n` files `objNNN.fits` in `root`, each with a `*_BGSUB` and `*_RMS` extension per band."
    os.makedirs(root)
    rng = np.random.default_rng(seed)
    for i in range(n):
        obj = fake_object(rng)
        header = _wcs_header(rng.uniform(150, 151), rng.uniform(2, 3))
        hdus = [fits.PrimaryHDU()]
        for index, (band, (sigma, peak)) in enumerate(MEF_BANDS.items()):
            if band == 'NIR_J_BGSUB' and i % MISSING_EVERY == MISSING_EVERY - 1:
                continue
            data = fake_band(rng, obj, sigma, peak, index)
            if band == 'NIR_H_BGSUB' and i == BLANK:
                data[:] = 0
            hdus.append(fits.ImageHDU(data, header, name=band))
            hdus.append(fits.ImageHDU(np.full((SIZE, SIZE), sigma, np.float32), header,
                                      name=band.replace('_BGSUB', '_RMS')))
        fits.HDUList(hdus).writeto(join(root, f'obj{i:03d}.fits'))
    return root


def _to_8bit(image):
    return (255 * np.clip(np.arcsinh(image / 3) / np.arcsinh(40), 0, 1)).astype(np.uint8)


def write_band_dirs(root, n=14, ext='jpg', seed=0):
    "Band directories `grey` (L) and `rgb` (RGB) in `root`, with `n` stamps `objNNN.<ext>` each."
    rng = np.random.default_rng(seed)
    for band in ('grey', 'rgb'):
        os.makedirs(join(root, band))
    for i in range(n):
        obj = fake_object(rng)
        grey = _to_8bit(fake_band(rng, obj, 1., 60.))
        rgb = np.stack([_to_8bit(fake_band(rng, obj, 1., 60., k)) for k in (2, 1, 0)], -1)
        Image.fromarray(grey, 'L').save(join(root, 'grey', f'obj{i:03d}.{ext}'))
        Image.fromarray(rgb, 'RGB').save(join(root, 'rgb', f'obj{i:03d}.{ext}'))
    return root


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = os.path.abspath(sys.argv[1])
    mef, jpg = write_mef_set(join(out, 'mef')), write_band_dirs(join(out, 'jpg'))
    print(f"Wrote {mef} and {jpg}. Try:\n"
          f"  qtstamp-mosaic -p {mef} --mef -b VIS_BGSUB -B NIR_Y_BGSUB,NIR_J_BGSUB,NIR_H_BGSUB "
          f"--rgb-composites NIR_H_BGSUB,NIR_J_BGSUB,NIR_Y_BGSUB\n"
          f"  qtstamp-single -p {jpg} -b '' -B grey,rgb --rgb-composites ''")
