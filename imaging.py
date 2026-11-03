# This Python file uses the following encoding: utf-8
"""Shared pure functions for image scaling/normalization and filename
handling, used by both mosaic.py and single_viewer.py.

FITS/band-directory I/O lives in fits_io.py, not here -- this module stays
astropy-free.
"""

import csv
import glob
import os
from os.path import join
import re
import stat
from dataclasses import dataclass
from uuid import uuid4

import numpy as np
from PIL import Image


def identity(x):
    return x


def log(x, a=1000):
    "Log base `a` of `a*x + 1`: 0 at 0, about 1 at 1. Meant for x >= 0; NaN below -1/a."
    return np.log(a * x + 1) / np.log(a)


def asinh2(x):
    return np.arcsinh(10 * x) / 3


def _central_box(shape, box_size):
    """(xmin, xmax, ymin, ymax) of a box_size box at the centre of a 2-D `shape`.
    Clamped at 0: on a stamp smaller than the box a negative start would wrap
    around and slice the far edge instead (BUGS 23)."""
    xl, yl = shape[:2]
    return (max(0, int(xl / 2. - box_size / 2.)), int(xl / 2. + box_size / 2.),
            max(0, int(yl / 2. - box_size / 2.)), int(yl / 2. + box_size / 2.))


def get_value_range_asymmetric(x, q_low=1, q_high=1):
    """(black, white) points of an (ny, nx, 3) composite stack: the q_low percentile
    of everything, and the 100 - q_high percentile of a box at the centre."""
    low = np.nanpercentile(x, q_low)

    # np.prod over all three bands: the box is 3% of the stamp's area, not 1%. A
    # composite always has exactly three bands, so this is a fixed size, and it is
    # how composites have always looked (BUGS 22).
    if x.shape[0] > 80:
        pixel_boxsize_low = np.round(np.sqrt(np.prod(x.shape) * 0.01)).astype(int)
    else:
        pixel_boxsize_low = 8
    xmin, xmax, ymin, ymax = _central_box(x.shape, pixel_boxsize_low)
    high = np.nanpercentile(x[xmin:xmax, ymin:ymax], 100 - q_high)
    return low, high


# --- black point of a single FITS band ----------------------------------------
#
# Both viewers set a FITS band's black point from the noise in the stamp's four
# corners. The corner-box size is the first change this branch makes to how stamps
# are displayed (BUGS-DONE 18): until 2026-09-28 every stamp used a fixed 10 px box,
# as in ERO_edition_2026. POSSIBLY TEMPORARY -- to go back, make corner_box_size
# return 10.

CORNER_BOX_MIN = 7              # px
CORNER_BOX_MAX = 20             # px, for a stamp without WCS
CORNER_BOX_MAX_ARCSEC = 1.0     # for a stamp with WCS


def corner_box_size(shape, pixel_scale_arcsec=None):
    """Side, in pixels, of each corner box background_rms_image measures.

    10 px, or 0.1% of the stamp's area (about 3.2% of a side) once that is larger.
    Capped at a quarter of the shorter side, so the boxes stay clear of the middle
    where the source is, and at CORNER_BOX_MAX px -- or at CORNER_BOX_MAX_ARCSEC,
    rounded to whole pixels, when `pixel_scale_arcsec` comes from a WCS. Never under
    CORNER_BOX_MIN px: that floor wins when a cap falls below it."""
    size = max(10, int(np.round(np.sqrt(np.prod(shape) * 0.001))))
    size = min(size, min(shape) // 4)
    if pixel_scale_arcsec is None:
        upper = CORNER_BOX_MAX
    else:
        upper = int(np.round(CORNER_BOX_MAX_ARCSEC / pixel_scale_arcsec))
    return max(CORNER_BOX_MIN, min(size, upper))


def background_rms_image(cb, image):
    """Standard deviation of the pixels in `image`'s four cb x cb corner boxes.

    While one box's mean is over 5 times another's, drops the highest of the values
    left at each pixel position -- per pixel, not the brightest box -- and tests
    again (see BUGS.md 35)."""
    xg, yg = np.shape(image)
    cut0 = image[0:cb, 0:cb]
    cut1 = image[xg - cb:xg, 0:cb]
    cut2 = image[0:cb, yg - cb:yg]
    cut3 = image[xg - cb:xg, yg - cb:yg]
    l = [cut0, cut1, cut2, cut3]
    while len(l) > 1:
        m = np.nanmean(np.nanmean(l, axis=1), axis=1)
        if max(m) > 5 * min(m):
            s = np.sort(l, axis=0)
            l = s[:-1]
        else:
            std = np.nanstd(l)
            return std
    std = np.nanstd(l)
    return std


def clip_normalize(x, low=None, high=None):
    "`x` clipped to [low, high] and mapped onto [0, 1]. All 0 when low == high (a flat image)."
    if high == low:
        return np.zeros_like(x, dtype=float)
    x = np.clip(x, low, high)
    x = (x - low) / (high - low)
    return x


def contrast_bias_scale(x, contrast, bias):
    x = ((x - bias) * contrast + 0.5)
    x = np.clip(x, 0, 1)
    return x


def get_contrast_bias_reasonable_assumptions(value_at_min, bkg_color, scale_min, scale_max, scale):
    """(contrast, bias) that put `value_at_min` at display value `bkg_color` and keep
    1 at 1. When the background is already at the white point (bkg_level 1: nothing
    above 0 in the centre box) there is no such line; leave the image as it is,
    rather than divide by zero (BUGS 24). `bkg_color` is never 1."""
    bkg_level = clip_normalize(value_at_min, scale_min, scale_max)
    bkg_level = scale(bkg_level)
    if bkg_level == 1:
        return 1.0, 0.5
    contrast = (bkg_color - 1) / (bkg_level - 1)
    bias = 1 - (bkg_level - 1) / (2 * (bkg_color - 1))
    return contrast, bias


# --- display scaling, shared by both viewers --------------------------------
#
# A FITS band is shown by picking a black and white point (scale_limits), then
# stretching between them (rescale_image). An RGB composite stretches its three
# bands together (composite_image). The two viewers differ in four details, kept on
# purpose so that neither one's display changes (REFACTOR.md step 4): one Display
# per viewer names them.

# Stored names, in the mosaic's dropdown order. The 1-by-1 shows its own labels.
SCALES = {'linear': identity, 'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': log, 'asinh': asinh2}
# Names saved before 2026-10-06, read as their current ones.
OLD_SCALE_NAMES = {'identity': 'linear', 'log10': 'log', 'asinh2': 'asinh'}


@dataclass(frozen=True)
class Display:
    vmax_box_grows_above: int    # px: above it, the white point's box is 1% of the stamp's area
    lift_zero_vmin: bool         # a black point of exactly 0 becomes 1e-3
    max_is_white: bool           # pixels at the white point are 1.0, not scale(max)/scale(max-min)
    composite_p_low: float       # percentile of the composite's black point
    composite_bkg_level: float   # display value the composite's background maps to


MOSAIC_DISPLAY = Display(vmax_box_grows_above=173, lift_zero_vmin=True, max_is_white=True,
                         composite_p_low=1, composite_bkg_level=0.015)
SINGLE_DISPLAY = Display(vmax_box_grows_above=170, lift_zero_vmin=False, max_is_white=False,
                         composite_p_low=2, composite_bkg_level=-0.05)


def scale_limits(image, display, pixel_scale_arcsec=None):
    """(black, white) points of one band: the noise in its corners (1 sigma), and
    1.3 times the brightest pixel in a box at its centre."""
    if image.shape[0] > display.vmax_box_grows_above:
        box_size_vmax = np.round(np.sqrt(np.prod(image.shape) * 0.01)).astype(int)
    else:
        box_size_vmax = 14
    box_size_vmin = corner_box_size(image.shape, pixel_scale_arcsec)
    vmin = np.nanmin(background_rms_image(box_size_vmin, image))
    if vmin == 0 and display.lift_zero_vmin:
        vmin += 1e-3
    xmin, xmax, ymin, ymax = _central_box(image.shape, box_size_vmax)
    vmax = np.nanmax(image[xmin:xmax, ymin:ymax])
    return vmin * 1.0, vmax * 1.3


def rescale_image(image, scale_min, scale_max, scale, display):
    """`image` clipped to [scale_min, scale_max], then scale(x) / scale(max - min).
    Clipping first is what keeps bright noise from washing the stamp out. NaNs stay."""
    image = image.clip(min=scale_min, max=scale_max)
    # Below scale_min after the clip only when the limits are inverted (a black point
    # above the white point): the clip then puts every pixel at scale_max. The 1-by-1
    # shows those as 0, the mosaic as 1.
    below = image < scale_min
    if display.max_is_white:
        at_max = image >= scale_max
        inside = (image >= scale_min) & ~at_max
    else:
        at_max = np.zeros_like(below)
        inside = (image >= scale_min) & (image <= scale_max)
    image[below] = 0.0
    image[at_max] = 1.0
    # An empty range has nothing to stretch, and scale() of it warns (BUGS 47): inverted
    # limits leave nothing inside, and a flat band's pixels (an all-zero blank stamp in
    # the 1-by-1) stay NaN, which is what the 0/0 here always made them.
    if scale_max > scale_min:
        image[inside] = scale(image[inside]) / scale(scale_max - scale_min)
    else:
        image[inside] = np.nan
    return image


def _composite_channel(image, scale_min, scale_max, scale, bkg_level):
    "One band of a composite: normalised to [0, 1], scaled, then contrast and bias applied."
    image = clip_normalize(image, scale_min, scale_max)
    image = scale(image)
    contrast, bias = get_contrast_bias_reasonable_assumptions(
        max(0, scale_min), bkg_level, scale_min, scale_max, scale)
    return contrast_bias_scale(image, contrast, bias)


def composite_image(images, scale, display):
    "RGB in [0, 1] from an (ny, nx, 3) stack, all three bands sharing one black and white point."
    composite = np.zeros_like(images, dtype=float)
    scale_min, scale_max = get_value_range_asymmetric(images, display.composite_p_low, 0.1)
    for i in range(images.shape[-1]):
        composite[:, :, i] = _composite_channel(images[:, :, i], scale_min, scale_max, scale,
                                                display.composite_bkg_level)
    return composite


def open_rendered_image(filepath):
    """A PNG/JPG stamp as a loaded PIL image in a mode that shows as the file does:
    greyscale ('L', 16-bit 'I;16'), RGB, or RGBA when the file has transparency.

    Anything else is converted. A palette ('P') image's pixels are palette indices, not
    intensities: read raw, a black sky stored at the highest index renders white
    (BUGS-DONE #34). CMYK and grey-plus-alpha ('LA') are neither greyscale nor RGB(A),
    and PNG cannot hold CMYK at all."""
    with Image.open(filepath) as im:
        if im.mode in ('L', 'RGB', 'RGBA') or im.mode.startswith('I'):
            return im.copy()
        if im.mode == '1':
            return im.convert('L')
        if im.mode in ('LA', 'La', 'PA', 'RGBa') or 'transparency' in im.info:
            return im.convert('RGBA')
        return im.convert('RGB')


def natural_sort(l):
    "https://stackoverflow.com/a/4836734"
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


# --- fork names -------------------------------------------------------------
#
# A classification CSV is named after the dataset, so two runs can legitimately
# want the same name. When that happens one of them has to move aside, and the
# name it moves to says *why* rather than carrying a bare number: there are two
# quite different situations, and a user looking at the directory afterwards has
# no other way to tell which one happened.

SIMULTANEOUS_FORK = 'simultaneous'   # another viewer is grading this same session
NEW_DATASET_FORK = 'new_dataset'     # this name is taken by a different dataset

# `-(N)` is what both of these were called before the markers had words. Still
# recognised, so forking a pre-existing file doesn't stack suffixes on it, but
# never written any more.
_FORK_SUFFIX = re.compile(r'-(?:{})_\d+$|-\(\d+\)$'.format(
    '|'.join((SIMULTANEOUS_FORK, NEW_DATASET_FORK))))


def next_free_csv_path(path, marker, max_attempts=1000):
    """The first free `-{marker}_N` variant of `path`, e.g. `-simultaneous_1`.

    Scanning for a free name rather than incrementing the number found in an
    existing one keeps this correct when the numbering has gaps, and there is no
    parenthesis anywhere in the result: these files get typed at a shell, and
    `-(1)` is a syntax error in bash and zsh unless it is quoted or escaped.

    The marker is introduced by `-`, which is load-bearing. Every tool finds a
    session's files with `classification_csvs`, which globs `{base}-*.csv`, so a
    marker spelled with a leading `_` would read as a seed and hide the fork
    from every tool that goes looking for it.
    """
    base = path[:-len('.csv')] if path.endswith('.csv') else path
    base = _FORK_SUFFIX.sub('', base)      # don't stack markers on a fork
    for n in range(1, max_attempts):
        candidate = '{}-{}_{}.csv'.format(base, marker, n)
        if not os.path.exists(candidate):
            return candidate
    raise RuntimeError('no free filename near {}'.format(path))


# --- session CSV names -------------------------------------------------------
#
# A classification CSV is named after its dataset -- tool, --name, object count
# and seed -- and nothing else. Both viewers open theirs by that name and the
# lobby finds what they wrote by it, so all three go through the functions below:
# the lobby's own copy of this naming drifted from the viewers' (BUGS.md 39, 40).

MOSAIC_CSV_PREFIX = 'classification_mosaic_autosave'
SINGLE_CSV_PREFIX = 'classification_single'


def classification_csv_base(prefix, name, n_objects, seed=None):
    """`{prefix}_{name}_{n_objects}`, plus `_{seed}` for a seeded session: the
    session's CSV name without `.csv`. `_` introduces the seed and `-` a fork
    (`next_free_csv_path`), so the two never meet in a glob."""
    base = '{}_{}_{}'.format(prefix, name or '', n_objects)
    return base if seed is None else '{}_{}'.format(base, seed)


def classification_csvs(classifications_dir, base):
    """`{base}.csv` and its `-`-suffixed forks in `classifications_dir`, natural-sorted.

    Not `{base}*.csv`: for a seeded `base` that also matches every seed it is a
    prefix of, so seed 7 would find seed 70's files (BUGS-DONE 11)."""
    stem = join(glob.escape(classifications_dir), glob.escape(base))
    return natural_sort(set(glob.glob(stem + '.csv')) | set(glob.glob(stem + '-*.csv')))


def csv_holds(path, file_names):
    "True if the CSV at `path` lists exactly `file_names`, in that order, in its `file_name` column."
    try:
        with open(path, newline='', encoding='utf-8') as f:
            held = [row.get('file_name') for row in csv.DictReader(f)]
    except (OSError, UnicodeDecodeError, csv.Error):
        return False
    return held == list(file_names)


def resume_csv(candidates, file_names=None):
    """Which of a session's `classification_csvs` a viewer reopens, or None if there
    are none.

    Given the session's `file_names`, in order, only the CSVs holding exactly those
    are considered, when there are any (BUGS 26). Two datasets can share a CSV name
    (same --name, object count and seed): the second forks a `-new_dataset_N`, and
    each must then reopen its own file. Picking by sort position alone, alternating
    between the two forked again on every switch and never reopened the earlier
    grades. A CSV with the same objects in another order is another session.

    Among those, the only one, else the second-to-last: forks sort before the
    unsuffixed file, so that is the newest fork. When none matches, the same rule
    over all of them, and the viewer, finding a different dataset, forks."""
    if file_names is not None:
        candidates = [c for c in candidates if csv_holds(c, file_names)] or candidates
    if not candidates:
        return None
    return candidates[-2] if len(candidates) > 1 else candidates[0]


# --- classification autosave ------------------------------------------------
#
# Both viewers rewrite their whole classification CSV on every grade -- one page
# at a time in the mosaic, one object at a time in the 1-by-1 tool. That is
# hundreds of writes per session over a file that may hold hours of work, so the
# way it is written matters more than anything else in this module.


def _fingerprint(path):
    "Cheap identity of the file as we last left it, or None if it isn't there."
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)


def atomic_write(path, write, encoding='utf-8', newline='', suffix=''):
    """Write via `write(fileobj)` so that `path` is never a partially written file.

    Streaming straight onto the target truncates it first, so an interruption at
    the wrong moment -- a crash, a kill, a full disk, a laptop lid -- leaves a
    truncated or empty file where completed work used to be. Writing a temp file
    in the same directory and renaming it over the target makes the update
    atomic: a reader sees either the whole previous file or the whole new one,
    and an interrupted write leaves the previous one exactly as it was.

    **Permissions match what a plain `open()` would have produced**: an existing
    target keeps its own mode, a new one gets the umask default. That is why the
    temp file is opened by hand rather than with `tempfile.mkstemp`, which
    hardcodes 0600 -- `os.replace` carries the temp file's mode onto the target,
    so mkstemp would quietly strip group read off a shared classification CSV on
    the very first autosave. Reading the umask to reproduce it instead is not an
    option: `os.umask` is read-modify-write and process-global, and both viewers
    have worker threads creating files of their own.

    Ownership, ACLs and xattrs cannot be carried across a rename and are not
    preserved; in the usual setgid shared directory the group is inherited by
    the temp file anyway.
    """
    directory = os.path.dirname(path) or os.curdir
    os.makedirs(directory, exist_ok=True)
    tmp = join(directory, '.tmp-{}-{}{}'.format(os.getpid(), uuid4().hex, suffix))
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, 'w', encoding=encoding, newline=newline) as f:
            write(f)
            # fsync before the rename, or the rename can land while the contents
            # are still only in the page cache and a power loss yields an empty
            # file.
            f.flush()
            os.fsync(f.fileno())
        try:
            os.chmod(tmp, stat.S_IMODE(os.stat(path).st_mode))
        except OSError:
            pass                # no target yet: keep the umask default
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def atomic_to_csv(df, path, **to_csv_kwargs):
    "Write `df` to `path` atomically, preserving its permissions. See `atomic_write`."
    # newline='' is what pandas uses when it opens a path itself; without it the
    # csv module's \r\n would be translated again on Windows, doubling it.
    encoding = to_csv_kwargs.pop('encoding', 'utf-8')
    return atomic_write(path, lambda f: df.to_csv(f, **to_csv_kwargs),
                        encoding=encoding, suffix='.csv')


class ClassificationWriter:
    """Owns one classification CSV and every autosave to it.

    Two guarantees, in the order they matter:

    1. A save never leaves a half-written file (see `atomic_to_csv`).
    2. A save never overwrites a file this writer did not write. If the target
       changed underneath us -- a second viewer open on the same session, most
       likely -- the other process's grades would otherwise be silently replaced
       by ours on the next click. That covers a target that *appeared* since we
       started just as much as one that changed after we last saved it: running
       the same command twice on a session nobody has started yet is the likeliest
       way to collide at all, and both viewers begin with nothing on disk to
       compare against. So we leave the other file alone and continue in the next
       free `-simultaneous_N` name, and both sets of work survive for the user to
       reconcile.
    """

    def __init__(self, path, **to_csv_kwargs):
        self.path = path
        self._to_csv_kwargs = to_csv_kwargs
        self._fingerprint = _fingerprint(path)

    def save(self, df):
        "Autosave `df`. Returns the path actually written, which may have forked."
        current = _fingerprint(self.path)
        # `current is not None`, not `self._fingerprint is not None`: a file that
        # *appeared* since we started is someone else's too. Two viewers launched
        # on a session nobody has started yet both begin with no fingerprint, and
        # guarding on ours meant the second one's first save replaced the first
        # one's grades outright. A target that has been *deleted* meanwhile is the
        # one case with nothing to protect, so it is simply recreated in place.
        if current is not None and current != self._fingerprint:
            forked = next_free_csv_path(self.path, SIMULTANEOUS_FORK)
            print("{} {} -- another viewer is grading this session.\n"
                  "  Leaving it untouched and continuing in {}".format(
                      self.path,
                      'appeared since this session started' if self._fingerprint is None
                      else 'has been written since this session last saved it',
                      forked))
            self.path = forked
        atomic_to_csv(df, self.path, **self._to_csv_kwargs)
        self._fingerprint = _fingerprint(self.path)
        return self.path
