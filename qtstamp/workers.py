# This Python file uses the following encoding: utf-8
"""Survey cutout fetching: cache protocol, downloads, and the background
QObject workers that drive them, and the cutout sizes. Used by single_viewer.py.

Cache protocol
--------------
A cutout lives at a deterministic path (see `legacy_survey_cache_name` /
`panstarrs_cache_name`). The presence of that .jpg means "we have the image".
A *miss* is never stored as a .jpg -- it is recorded in a sidecar
`<savefile>.miss` JSON marker carrying a reason and a timestamp, so that:

  * a real coverage gap is not re-queried on every single visit, and
  * a transient failure (timeout, 5xx, garbled response) expires quickly and
    is retried once the network comes back.

This separation matters: an earlier version wrote a black 66x66 placeholder
JPEG to the cache path on failure, which every reader then treated as a cache
hit. One outage permanently poisoned those coordinates -- the panel showed a
black square forever, and no retry was ever attempted.
"""

import enum
import json
import os
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image
from PySide6.QtCore import QObject, QThread, Signal, Slot

from . import fits_io
from .imaging import atomic_write

PS1_FILENAMES_URL = 'https://ps1images.stsci.edu/cgi-bin/ps1filenames.py'
PS1_FITSCUT_URL = 'https://ps1images.stsci.edu/cgi-bin/fitscut.cgi'
LEGACY_SURVEY_CUTOUT_URL = 'https://www.legacysurvey.org/viewer/cutout.jpg'

# Per-socket-operation timeout. urlretrieve() takes no timeout argument, which
# is why the downloads it used to do could hang forever on a stalled server and
# park the prefetch thread on a single object for the rest of the session.
NETWORK_TIMEOUT = 15
# STScI's PS1 services are slower: the filename lookup was measured at 4-14 s and a
# 512 px fitscut at ~8 s before its first byte, too close to NETWORK_TIMEOUT. Not
# much more: it is per blocking read, and the prefetch runs its fetches in series.
PS1_NETWORK_TIMEOUT = 30

# How long a recorded miss suppresses re-downloading.
MISS_TTL_SECONDS = {
    # The survey answered correctly and has nothing here. Re-checking weekly
    # costs nothing and picks up new data releases on its own.
    'no_coverage': 7 * 24 * 3600,
    # Network blip, server error, garbled response. Long enough not to hammer a
    # server that is down, short enough that coming back to the object after
    # the connection recovers just works.
    'transient': 15 * 60,
}
DEFAULT_MISS_TTL = MISS_TTL_SECONDS['transient']

MISS_SUFFIX = '.miss'

# Legacy Survey answers out-of-footprint requests with a valid but completely
# flat JPEG. A real cutout of empty sky is noise, never uniform, so a uniform
# image is treated as a coverage gap rather than cached as data. Set to False
# to keep whatever the server returned.
REJECT_UNIFORM_LEGACY_SURVEY_CUTOUTS = True


# --------------------------------------------------------------------------
# Cutout sizes, in survey pixels, for a stamp of `image_dim` px at
# `image_pixel_size` arcsec/px: (same field as the stamp, Large FoV).
# --------------------------------------------------------------------------

LEGACY_SURVEY_PIXEL_SIZE = 0.262
LEGACY_SURVEY_PIXSCALE = str(LEGACY_SURVEY_PIXEL_SIZE)   # as the cutout URL takes it
LEGACY_SURVEY_BIG_FOV_PIXELS = 488
PANSTARRS_PIXEL_SIZE = 0.25


def legacy_survey_number_of_pixels(image_pixel_size,
                                    image_dim,
                                    pixels_big_fov_ls=LEGACY_SURVEY_BIG_FOV_PIXELS): #sizes in ARCSECONDS
    n_pixels_in_ls = int(np.ceil(image_pixel_size*image_dim/LEGACY_SURVEY_PIXEL_SIZE))
    if n_pixels_in_ls >= pixels_big_fov_ls:
        pixels_big_fov_ls = 2*n_pixels_in_ls
    return n_pixels_in_ls, pixels_big_fov_ls

# One "Large FoV" toggle drives both survey panels, so PanSTARRS covers the same
# ~2.1' as Legacy Survey's 488 px, and doubles the stamp when that is wider (BUGS 44).
PANSTARRS_BIG_FOV_ARCSEC = LEGACY_SURVEY_BIG_FOV_PIXELS*LEGACY_SURVEY_PIXEL_SIZE

def panstarrs_number_of_pixels(image_pixel_size, image_dim): #sizes in ARCSECONDS
    n_pixels_in_ps1 = int(np.ceil(image_pixel_size*image_dim/PANSTARRS_PIXEL_SIZE))
    pixels_big_fov_ps1 = int(np.ceil(PANSTARRS_BIG_FOV_ARCSEC/PANSTARRS_PIXEL_SIZE))
    if n_pixels_in_ps1 >= pixels_big_fov_ps1:
        pixels_big_fov_ps1 = 2*n_pixels_in_ps1
    return n_pixels_in_ps1, pixels_big_fov_ps1


class FetchResult(enum.Enum):
    "Outcome of a cutout fetch (or of consulting the cache for one)."
    OK = 'ok'                    # a usable image is on disk
    NO_COVERAGE = 'no_coverage'  # the survey has nothing here
    TRANSIENT = 'transient'      # something failed; worth retrying later


class CacheState(enum.Enum):
    HIT = 'hit'      # real image on disk, use it
    MISS = 'miss'    # unexpired miss marker, do not hit the network
    RETRY = 'retry'  # nothing cached, or the marker expired: go fetch


# --------------------------------------------------------------------------
# Cache names. These strings are the cache keys -- changing them invalidates
# every already-downloaded cutout, so they are kept byte-for-byte as they were,
# missing separators and all.
# --------------------------------------------------------------------------

def legacy_survey_cache_name(ra, dec, size, residual=False):
    res = '-resid' if residual else '-grz'
    return 'N' + '_' + str(ra) + '_' + str(dec) + f"_{size}" + f'ls-dr10{res}.jpg'


def panstarrs_cache_name(ra, dec, size):
    return 'P' + '_' + str(ra) + '_' + str(dec) + f"_{size}" + 'ps1-grz.jpg'


def legacy_survey_url(ra, dec, size, residual=False, pixscale=LEGACY_SURVEY_PIXSCALE):
    res = '-resid' if residual else '-grz'
    return (f'{LEGACY_SURVEY_CUTOUT_URL}?ra={ra}&dec={dec}'
            f'&layer=ls-dr10{res}&size={size}&pixscale={pixscale}')


# --------------------------------------------------------------------------
# Miss markers
# --------------------------------------------------------------------------

def _miss_path(savefile):
    return savefile + MISS_SUFFIX


def record_miss(savefile, reason, detail=''):
    """Record that `savefile` could not be fetched, and why. Atomic, like the
    download itself: two workers missing the same cutout can't interleave (BUGS 27)."""
    marker = {'reason': reason, 'time': time.time(), 'detail': str(detail)[:500]}
    try:
        atomic_write(_miss_path(savefile), lambda f: json.dump(marker, f))
    except OSError:
        pass  # an unwritable cache dir must not break the fetch itself


def clear_miss(savefile):
    "Drop any stale marker once the image has actually been retrieved."
    try:
        os.remove(_miss_path(savefile))
    except OSError:
        pass


def read_miss(savefile):
    "Return the marker dict for `savefile`, or None if there is no usable one."
    try:
        with open(_miss_path(savefile)) as f:
            marker = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(marker, dict):
        return None
    return marker


def miss_is_current(marker):
    if marker is None:
        return False
    reason = marker.get('reason', 'transient')
    ttl = MISS_TTL_SECONDS.get(reason, DEFAULT_MISS_TTL)
    try:
        age = time.time() - float(marker.get('time', 0))
    except (TypeError, ValueError):
        return False
    return 0 <= age < ttl


def failure_title(survey, savefile):
    """Panel title for a cutout that could not be shown. Only a recorded 'no_coverage'
    means the survey has nothing here; a timeout or server error must not say so."""
    marker = read_miss(savefile)
    if miss_is_current(marker) and marker.get('reason') == 'no_coverage':
        return f'No {survey} data available'
    return (f'{survey} timed out or failed\n'
            f'(retried after {MISS_TTL_SECONDS["transient"] // 60} min)')


def cache_state(savefile):
    """Tri-state cache lookup: HIT (image on disk), MISS (known-bad and still
    fresh) or RETRY (nothing known, or the miss expired)."""
    if os.path.exists(savefile):
        return CacheState.HIT
    if miss_is_current(read_miss(savefile)):
        return CacheState.MISS
    return CacheState.RETRY


# --------------------------------------------------------------------------
# Downloading
# --------------------------------------------------------------------------

def _download_to(url, savefile, timeout=NETWORK_TIMEOUT):
    """Fetch `url` into `savefile` atomically and with a bounded timeout.

    The bytes land in a temp file in the destination directory and are moved
    into place only once complete, so an interrupted transfer can never leave a
    truncated image behind for the next run to treat as a valid cache hit."""
    os.makedirs(os.path.dirname(savefile) or '.', exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(savefile) or '.',
                                        suffix='.part')
    os.close(tmp_fd)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, \
                open(tmp_path, 'wb') as out:
            shutil.copyfileobj(response, out)
        os.replace(tmp_path, savefile)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _classify_exception(exc):
    """Map a download failure onto a miss reason.

    HTTP 400/404 means the service understood the query and has nothing;
    everything else -- including the ValueError/IndexError raised when
    ps1filenames.py answers with an HTML error page instead of a table -- is
    treated as transient and retried later."""
    if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 404):
        return FetchResult.NO_COVERAGE
    return FetchResult.TRANSIENT


def _is_uniform_image(path):
    "True if every pixel of the image is identical (an out-of-footprint tile)."
    try:
        with Image.open(path) as im:
            extrema = im.convert('RGB').getextrema()
    except (OSError, ValueError):
        return False
    return all(lo == hi for lo, hi in extrema)


def get_panstarrs_filenames(ra, dec, filters='grz'):
    """Look up PS1 stack image filenames for ra/dec via the STScI ps1filenames.py service.

    Remembered for the life of the process, in memory only, never on disk: the answer
    depends on ra/dec alone, and the normal and Large FoV cutouts of one object would
    otherwise each pay for the lookup. Only a full answer is remembered: an exception
    or None (no rows -- also what a truncated reply looks like) asks again next time."""
    key = (float(ra), float(dec), filters)
    with _PS1_FILENAMES_LOCK:
        if key in _PS1_FILENAMES:
            return dict(_PS1_FILENAMES[key])
    filenames = _query_panstarrs_filenames(*key)
    if filenames is not None:
        with _PS1_FILENAMES_LOCK:
            if len(_PS1_FILENAMES) >= _PS1_FILENAMES_MAX:
                _PS1_FILENAMES.clear()
            _PS1_FILENAMES[key] = dict(filenames)
    return filenames


_PS1_FILENAMES = {}
_PS1_FILENAMES_MAX = 4096
_PS1_FILENAMES_LOCK = threading.Lock()


def _query_panstarrs_filenames(ra, dec, filters):
    query_url = f"{PS1_FILENAMES_URL}?ra={ra}&dec={dec}&filters={filters}&type=stack"
    with urllib.request.urlopen(query_url, timeout=PS1_NETWORK_TIMEOUT) as response:
        lines = response.read().decode('utf-8').splitlines()
    if len(lines) < 2:
        return None
    header = lines[0].split()
    filename_col = header.index('filename')
    filter_col = header.index('filter')
    filenames = {}
    for line in lines[1:]:
        fields = line.split()
        filenames[fields[filter_col]] = fields[filename_col]
    if not all(f in filenames for f in filters):
        return None
    return filenames


def _finish_fetch(savefile, verbose=False, reject_uniform=False):
    "Post-download validation shared by both surveys."
    if reject_uniform and _is_uniform_image(savefile):
        try:
            os.remove(savefile)
        except OSError:
            pass
        record_miss(savefile, 'no_coverage', 'server returned a blank cutout')
        return FetchResult.NO_COVERAGE
    clear_miss(savefile)
    return FetchResult.OK


def fetch_legacy_survey(savefile, ra, dec, size, residual=False,
                        pixscale=LEGACY_SURVEY_PIXSCALE, verbose=False):
    """Ensure a Legacy Survey cutout for ra/dec is cached at `savefile`.

    Never raises: every failure is recorded as a miss marker and reported
    through the return value."""
    state = cache_state(savefile)
    if state is CacheState.HIT:
        print('File already exists:', savefile) if verbose else False
        return FetchResult.OK
    if state is CacheState.MISS:
        return FetchResult.NO_COVERAGE

    url = legacy_survey_url(ra, dec, size, residual=residual, pixscale=pixscale)
    print(url) if verbose else False
    try:
        _download_to(url, savefile)
    except Exception as exc:
        result = _classify_exception(exc)
        record_miss(savefile, result.value, exc)
        return result
    return _finish_fetch(savefile, verbose=verbose,
                         reject_uniform=REJECT_UNIFORM_LEGACY_SURVEY_CUTOUTS)


def fetch_panstarrs(savefile, ra, dec, size, verbose=False):
    """Ensure a PanSTARRS grz cutout for ra/dec is cached at `savefile`.

    Never raises: see fetch_legacy_survey."""
    state = cache_state(savefile)
    if state is CacheState.HIT:
        print('File already exists:', savefile) if verbose else False
        return FetchResult.OK
    if state is CacheState.MISS:
        return FetchResult.NO_COVERAGE

    try:
        filenames = get_panstarrs_filenames(ra, dec, filters='grz')
    except Exception as exc:
        result = _classify_exception(exc)
        record_miss(savefile, result.value, exc)
        return result
    if filenames is None:
        # The service answered properly and has no stack images here.
        record_miss(savefile, 'no_coverage', 'ps1filenames returned no rows')
        return FetchResult.NO_COVERAGE

    url = (f"{PS1_FITSCUT_URL}?red={filenames['z']}&green={filenames['r']}"
           f"&blue={filenames['g']}&ra={ra}&dec={dec}&size={size}"
           f"&output_size=256&autoscale=99.5&format=jpg")
    print(url) if verbose else False
    try:
        _download_to(url, savefile, timeout=PS1_NETWORK_TIMEOUT)
    except Exception as exc:
        result = _classify_exception(exc)
        record_miss(savefile, result.value, exc)
        return result
    return _finish_fetch(savefile, verbose=verbose)


# --------------------------------------------------------------------------
# Migration of the old failure placeholders
# --------------------------------------------------------------------------

_PLACEHOLDER_SIDE = 66


def _is_legacy_placeholder(path):
    """True for the black 66x66 JPEG that older versions wrote into the cache
    on a failed download. Real cutouts come back at output_size=256 or at a
    size derived from the stamp, never 66x66, so this cannot match data."""
    try:
        if os.path.getsize(path) > 1024:
            return False
        with Image.open(path) as im:
            if im.size != (_PLACEHOLDER_SIDE, _PLACEHOLDER_SIDE):
                return False
            extrema = im.convert('RGB').getextrema()
    except (OSError, ValueError):
        return False
    return all(lo == hi == 0 for lo, hi in extrema)


def purge_placeholder_cutouts(*directories):
    """Delete failure placeholders left by older versions, which the cache
    would otherwise keep serving as if they were real images. Returns the
    number removed."""
    removed = 0
    for directory in directories:
        try:
            entries = os.listdir(directory)
        except OSError:
            continue
        for entry in entries:
            if not entry.endswith('.jpg'):
                continue
            path = os.path.join(directory, entry)
            if _is_legacy_placeholder(path):
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass
    return removed


# --------------------------------------------------------------------------
# Qt workers. FetchWorker runs one fetch on a worker thread and reports back by
# signal. It may not raise: a slot that dies takes has_finished with it and leaves
# the panel stuck on "Waiting for data" forever. FetchThread is the prefetch pass.
# --------------------------------------------------------------------------

class FetchWorker(QObject):
    """Runs one cutout fetch off the GUI thread. `fetch` is a fetch_* call with its
    arguments bound. For PanSTARRS that includes the ps1filenames.py lookup -- a
    network call too, which used to run on the GUI thread and froze the UI for up
    to 10s whenever a stamp had no PS1 coverage."""
    successful_download = Signal()
    failed_download = Signal()
    has_finished = Signal()

    def __init__(self, survey, fetch):
        super().__init__()
        self.survey = survey
        self.fetch = fetch

    @Slot()
    def run(self):
        try:
            result = self.fetch()
        except Exception as E:  # defensive: fetch_* is not supposed to raise
            print(f"Unexpected error fetching the {self.survey} cutout:", E)
            result = FetchResult.TRANSIENT
        if result is FetchResult.OK:
            self.successful_download.emit()
        else:
            self.failed_download.emit()
        self.has_finished.emit()


class FetchThread(QThread):
    "The prefetch pass: every cutout wanted for each object from `initial_counter` on."
    def __init__(self, df, initial_counter, main_band_source, listimage,
                 legacy_survey_path, panstarrs_path,
                 fetch_panstarrs=True, fetch_legacysurvey=True,
                 fetch_big_fov_residuals=False, verbose=False, parent=None):
            QThread.__init__(self, parent)

            self.df = df
            self.initial_counter = initial_counter
            self.legacy_survey_path = legacy_survey_path
            self.panstarrs_path = panstarrs_path
            self.verbose = verbose
            # Passed in rather than rediscovered here, so this thread's view of "the
            # main band's file for object N" can never drift from ApplicationWindow's.
            self.main_band_source = main_band_source
            self.listimage = listimage
            self.fetch_panstarrs = fetch_panstarrs
            self.fetch_legacysurvey = fetch_legacysurvey
            self.fetch_big_fov_residuals = fetch_big_fov_residuals
            # interrupt() can arrive before run() has started (isRunning() is
            # already true by then), so this must exist from construction.
            self._active = True

    def download_legacy_survey(self,ra,dec,size=47,residual=False,pixscale=LEGACY_SURVEY_PIXSCALE):
        savefile = os.path.join(self.legacy_survey_path,
                                legacy_survey_cache_name(ra, dec, size, residual=residual))
        return fetch_legacy_survey(savefile, ra, dec, size, residual=residual,
                                   pixscale=pixscale, verbose=self.verbose)

    def download_panstarrs(self,ra,dec,size=240):
        savefile = os.path.join(self.panstarrs_path,
                                panstarrs_cache_name(ra, dec, size))
        return fetch_panstarrs(savefile, ra, dec, size, verbose=self.verbose)

    def interrupt(self):
        self._active = False

    def fetch_one(self, index):
        "Download every cutout wanted for the object at `index`."
        stamp = self.df.iloc[index]
        if np.isnan(stamp['ra']) or np.isnan(stamp['dec']): #TODO: add smt for when there is no RADec.
            stem = os.path.splitext(self.listimage[index])[0]
            locator = fits_io.resolve(self.main_band_source, stem)
            ra,dec,image_pixel_size,image_dim = fits_io.get_ra_dec(fits_io.read_header(locator))
        else:
            ra,dec,image_pixel_size,image_dim = stamp[['ra','dec','pixel_size','image_dim']]
        if any(pd.isna(v) for v in (ra, dec, image_pixel_size, image_dim)):
            return  # no sky WCS in the main band, or a CSV row missing its scale: nothing to fetch
        if self.fetch_panstarrs:
            n_pixels_ps1, n_pixels_big_ps1 = panstarrs_number_of_pixels(image_pixel_size, image_dim)
            self.download_panstarrs(ra,dec,size=n_pixels_ps1)
            self.download_panstarrs(ra,dec,size=n_pixels_big_ps1)

        if self.fetch_legacysurvey:
            n_pixels_ls, n_pixels_big_ls = legacy_survey_number_of_pixels(image_pixel_size,
                                    image_dim)
            self.download_legacy_survey(ra,dec,size=n_pixels_ls)
            self.download_legacy_survey(ra,dec,size=n_pixels_ls,residual=True)
            self.download_legacy_survey(ra,dec,size=n_pixels_big_ls)
            if self.fetch_big_fov_residuals:
                self.download_legacy_survey(ra,dec,size=n_pixels_big_ls,residual=True)

    def run(self):
        index = self.initial_counter
        self._active = True
        while self._active and index < len(self.df):
            # One unreadable header or malformed survey response must not end
            # the prefetch pass: everything after this index would silently
            # never be prefetched, for the rest of the session.
            try:
                self.fetch_one(index)
            except Exception as E:
                print(f"Skipping pre-fetch for object {index}: {type(E).__name__}: {E}")
            index+=1
        return 0
