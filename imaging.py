# This Python file uses the following encoding: utf-8
"""Shared pure functions for image scaling/normalization and filename
handling, used by both mosaic.py and single_viewer.py.

FITS/band-directory I/O lives in fits_io.py, not here -- this module stays
astropy-free.
"""

import os
from os.path import join
import re
import stat
from uuid import uuid4

import numpy as np


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


DEFAULT_CLASSIFICATIONS_DIRNAME = 'Classifications'


def resolve_classifications_dir(path=None):
    """Absolute path of the directory the classification CSVs live in.

    `path` may be absolute or relative (resolved against the current working
    directory) and may start with `~`; None or '' means the default,
    ./Classifications. Callers that are about to write create it themselves --
    this stays a pure function so that merely resolving a path (e.g. for
    lobby's --print-command) never touches the disk.
    """
    return os.path.abspath(os.path.expanduser(path or DEFAULT_CLASSIFICATIONS_DIRNAME))


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

    The marker is introduced by `-`, which is load-bearing. Both viewers glob
    `{base}-*.csv` minus `{base}_*.csv` to find a seedless dataset's files, so a
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
