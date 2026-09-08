# This Python file uses the following encoding: utf-8
"""Shared pure functions for image scaling/normalization and filename
handling, used by both mosaic.py and single_viewer.py.
"""

import glob
import os
from os.path import join
import re

import numpy as np

FITS_EXT = '.fits'
COMPRESSED_EXTS = ('.png', '.jpg', '.jpeg')


def detect_band_filetype(band_dir):
    """Returns 'FITS' if band_dir contains FITS files, 'COMPRESSED' if it contains
    PNG/JPG/JPEG instead, or None if band_dir has neither (or doesn't exist).
    A band's format is a property of its whole directory, not of individual files --
    this lets different bands in the same dataset use different formats."""
    if glob.glob(join(band_dir, '*' + FITS_EXT)):
        return 'FITS'
    if any(glob.glob(join(band_dir, '*' + ext)) for ext in COMPRESSED_EXTS):
        return 'COMPRESSED'
    return None


def find_band_file(stampspath, band, stem):
    """Resolves an object's stem (filename without extension, taken from the main
    band's listing) to the actual file inside another band's directory -- FITS first,
    then PNG/JPG/JPEG -- so bands with different formats/extensions for the same
    object still match up by name. Returns None if nothing matches."""
    base = join(stampspath, band, stem)
    for ext in (FITS_EXT, *COMPRESSED_EXTS):
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate
    return None


def identity(x):
    return x


def log(x, a=1000):
    "Simple log base 1000 function that ignores numbers less than 0"
    return np.log(a * x + 1) / np.log(a)


def asinh2(x):
    return np.arcsinh(10 * x) / 3


def print_range(image):
    return f"{image.min() = }, {image.max() = }"


def get_value_range(x, p=98):
    q = (100 - p) / 2
    low = np.nanpercentile(x, q)
    high = np.nanpercentile(x, 100 - q)
    return low, high


def get_value_range_asymmetric(x, q_low=1, q_high=1):
    low = np.nanpercentile(x, q_low)

    if x.shape[0] > 80:
        pixel_boxsize_low = np.round(np.sqrt(np.prod(x.shape) * 0.01)).astype(int)
    else:
        pixel_boxsize_low = 8
    xl, yl, _ = np.shape(x)
    xmin = int((xl) / 2. - (pixel_boxsize_low / 2.))
    xmax = int((xl) / 2. + (pixel_boxsize_low / 2.))
    ymin = int((yl) / 2. - (pixel_boxsize_low / 2.))
    ymax = int((yl) / 2. + (pixel_boxsize_low / 2.))
    high = np.nanpercentile(x[xmin:xmax, ymin:ymax], 100 - q_high)
    return low, high


def clip_normalize(x, low=None, high=None):
    x = np.clip(x, low, high)
    x = (x - low) / (high - low)
    return x


def contrast_bias_scale(x, contrast, bias):
    x = ((x - bias) * contrast + 0.5)
    x = np.clip(x, 0, 1)
    return x


def get_contrast_bias_reasonable_assumptions(value_at_min, bkg_color, scale_min, scale_max, scale):
    bkg_level = clip_normalize(value_at_min, scale_min, scale_max)
    bkg_level = scale(bkg_level)
    contrast = (bkg_color - 1) / (bkg_level - 1)  # with bkg_level != 1 and bkg_color != 1
    bias = 1 - (bkg_level - 1) / (2 * (bkg_color - 1))
    return contrast, bias


def natural_sort(l):
    "https://stackoverflow.com/a/4836734"
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def find_filename_iteration(latest_filename, max_iterations=100, initial_iteration="-(1)"):
    "Uses regex to find and add 1 to the number in parentheses right before the .csv"
    re_pattern = re.compile('-\\(([^)]+)\\)')
    re_search = re_pattern.search(latest_filename)
    if re_search is None:
        return initial_iteration
    iterations = 0
    while re_search.span()[-1] != len(latest_filename) and (iterations < max_iterations):
        re_search = re_pattern.search(latest_filename, re_search.span()[-1])
        if re_search is None:
            return initial_iteration
    if re_search.span()[-1] == len(latest_filename):  # at this point, re_search cannot be None
        re_match = re_search[1]
    try:
        int_match = int(re_match)
    except:
        return initial_iteration

    return f"-({int_match + 1})"
