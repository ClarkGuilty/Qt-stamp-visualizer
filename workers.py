# This Python file uses the following encoding: utf-8
"""Survey cutout fetching: cache protocol, downloads, and the background
QObject workers that drive them. Used by single_viewer.py.

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
import time
import urllib.error
import urllib.request

from PIL import Image
from PySide6.QtCore import QObject, Signal, Slot

PS1_FILENAMES_URL = 'https://ps1images.stsci.edu/cgi-bin/ps1filenames.py'
PS1_FITSCUT_URL = 'https://ps1images.stsci.edu/cgi-bin/fitscut.cgi'
LEGACY_SURVEY_CUTOUT_URL = 'https://www.legacysurvey.org/viewer/cutout.jpg'

# Per-socket-operation timeout. urlretrieve() takes no timeout argument, which
# is why the downloads it used to do could hang forever on a stalled server and
# park the prefetch thread on a single object for the rest of the session.
NETWORK_TIMEOUT = 15

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


def legacy_survey_url(ra, dec, size, residual=False, pixscale='0.262'):
    res = '-resid' if residual else '-grz'
    return (f'{LEGACY_SURVEY_CUTOUT_URL}?ra={ra}&dec={dec}'
            f'&layer=ls-dr10{res}&size={size}&pixscale={pixscale}')


# --------------------------------------------------------------------------
# Miss markers
# --------------------------------------------------------------------------

def _miss_path(savefile):
    return savefile + MISS_SUFFIX


def record_miss(savefile, reason, detail=''):
    "Record that `savefile` could not be fetched, and why."
    try:
        os.makedirs(os.path.dirname(savefile) or '.', exist_ok=True)
        with open(_miss_path(savefile), 'w') as f:
            json.dump({'reason': reason, 'time': time.time(),
                       'detail': str(detail)[:500]}, f)
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
    "Look up PS1 stack image filenames for ra/dec via the STScI ps1filenames.py service."
    query_url = f"{PS1_FILENAMES_URL}?ra={ra}&dec={dec}&filters={filters}&type=stack"
    with urllib.request.urlopen(query_url, timeout=NETWORK_TIMEOUT) as response:
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
                        pixscale='0.262', verbose=False):
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
        _download_to(url, savefile)
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
# Qt workers. Both run one fetch on a worker thread and report back by signal.
# Neither may raise: a slot that dies takes has_finished with it and leaves the
# panel stuck on "Waiting for data" forever.
# --------------------------------------------------------------------------

class SingleFetchWorker(QObject):
    "Fetches one Legacy Survey cutout off the GUI thread."
    successful_download = Signal()
    failed_download = Signal()
    has_finished = Signal()

    def __init__(self, savefile, ra, dec, size, residual=False,
                 pixscale='0.262', verbose=False):
        super(SingleFetchWorker, self).__init__()
        self.savefile = savefile
        self.ra = ra
        self.dec = dec
        self.size = size
        self.residual = residual
        self.pixscale = pixscale
        self.verbose = verbose

    @Slot()
    def run(self):
        try:
            result = fetch_legacy_survey(self.savefile, self.ra, self.dec,
                                         self.size, residual=self.residual,
                                         pixscale=self.pixscale,
                                         verbose=self.verbose)
        except Exception as E:  # defensive: fetch_* is not supposed to raise
            print("Unexpected error fetching the Legacy Survey cutout:", E)
            result = FetchResult.TRANSIENT
        if result is FetchResult.OK:
            self.successful_download.emit()
        else:
            self.failed_download.emit()
        self.has_finished.emit()


class PanstarrsFetchWorker(QObject):
    """Fetches one PanSTARRS cutout off the GUI thread, including the
    ps1filenames.py lookup -- that lookup is a network call too, and must not
    run on the GUI thread (it used to, and froze the UI for up to 10s whenever
    a stamp had no PS1 coverage)."""
    successful_download = Signal()
    failed_download = Signal()
    has_finished = Signal()

    def __init__(self, ra, dec, savefile, size, verbose=False):
        super(PanstarrsFetchWorker, self).__init__()
        self.ra = ra
        self.dec = dec
        self.savefile = savefile
        self.size = size
        self.verbose = verbose

    @Slot()
    def run(self):
        try:
            result = fetch_panstarrs(self.savefile, self.ra, self.dec,
                                     self.size, verbose=self.verbose)
        except Exception as E:  # defensive: fetch_* is not supposed to raise
            print("Unexpected error fetching the PanSTARRS cutout:", E)
            result = FetchResult.TRANSIENT
        if result is FetchResult.OK:
            self.successful_download.emit()
        else:
            self.failed_download.emit()
        self.has_finished.emit()
