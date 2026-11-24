# This Python file uses the following encoding: utf-8
"""All astropy.io.fits / astropy.wcs access lives here.

Both mosaic.py and single_viewer.py go through this module instead of calling
astropy directly, so there is exactly one place that knows how to open a FITS
file, one memmap policy, and one answer to "what does a band mean" for either of
the two dataset layouts the tools support:

- directory-per-band (the original scheme): one directory under --path per band,
  each holding one FITS or PNG/JPG file per object, matched across bands by
  filename stem. A directory may hold both formats: each file is read as what it
  is. See DirBandSource.
- multi-extension FITS, --mef (additive): one directory of per-object files under
  --path, each file holding every band as an HDU extension. Bands are looked up
  by EXTNAME in each object's own file, so files may order their extensions
  differently or lack some; a lacking one raises MissingBandError. See
  ExtensionBandSource and discover_mef_bands.

A BandSource is resolved once per band at startup (not per object); resolve()
then turns a BandSource plus one object's filename stem into a Locator, and
read_pixels/read_header/read_pixels_and_header turn a Locator into pixel data
and/or a header without either caller needing to know which scheme produced it.
"""

import functools
import os
import re
from dataclasses import dataclass
from os.path import join
from typing import Dict, NamedTuple, Optional, Union

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

FITS_EXT = '.fits'
FITS_EXTS = (FITS_EXT, '.fit', '.fts')
COMPRESSED_EXTS = ('.png', '.jpg', '.jpeg')
# A band name meaning "the stamps directly under --path", for a dataset that is one
# flat directory of images: join(path, '.') is the path itself.
FLAT_BAND = '.'


# --- what counts as a stamp ---------------------------------------------------

def stamp_kind(filename):
    """'FITS' or 'COMPRESSED' for a file a viewer can show, by its extension in any
    case (`.JPG`, `.FITS`), else None. A hidden file is never a stamp: macOS leaves
    `._name.fits` resource forks and `.DS_Store` beside the real files on drives that
    are not its own."""
    if filename.startswith('.'):
        return None
    ext = os.path.splitext(filename)[1].lower()
    if ext in FITS_EXTS:
        return 'FITS'
    if ext in COMPRESSED_EXTS:
        return 'COMPRESSED'
    return None


def scan_stamps(directory, fits_only=False):
    """{stem: filename} of the stamps directly in `directory`; {} if it is not one.

    One object per stem: where `obj.fits` and `obj.png` sit side by side, the FITS
    file is the object's stamp, then PNG, JPG, JPEG; between two spellings of one
    name (`obj.fits`, `obj.FITS`) the first in sorted order. A link to nothing is
    not a stamp."""
    rank = {ext: i for i, ext in enumerate((*FITS_EXTS, *COMPRESSED_EXTS))}
    try:
        with os.scandir(directory) as entries:
            names = sorted(e.name for e in entries
                           if stamp_kind(e.name) and e.is_file())
    except OSError:
        return {}
    found = {}
    for name in names:
        kind = stamp_kind(name)
        if fits_only and kind != 'FITS':
            continue
        stem, ext = os.path.splitext(name)
        held = found.get(stem)
        if held is None or rank[ext.lower()] < rank[os.path.splitext(held)[1].lower()]:
            found[stem] = name
    return found


def has_stamps(directory):
    "True if `directory` holds at least one stamp. Stops at the first."
    try:
        with os.scandir(directory) as entries:
            return any(stamp_kind(e.name) and e.is_file() for e in entries)
    except OSError:
        return False


def list_band_dirs(path):
    """Sorted names of the subdirectories of `path` that hold stamps: the bands of a
    directory-per-band dataset. Hidden ones (`.qtstamp`) and ones without an image
    (`Classifications`, `__MACOSX`) are not bands."""
    try:
        with os.scandir(path) as entries:
            dirs = [e.name for e in entries if not e.name.startswith('.') and e.is_dir()]
    except OSError:
        return []
    return sorted(d for d in dirs if has_stamps(join(path, d)))


# --- directory-per-band discovery -------------------------------------------

def stamps_filetype(stamps):
    """'FITS' if a band's {stem: filename} (`scan_stamps`) holds any FITS stamp,
    'COMPRESSED' if it holds only PNG/JPG/JPEG, None if it is empty.

    'FITS' says the band can be scaled, and can be a composite's member, for the
    objects that have a FITS file there. It does not say every file is FITS: each
    object's own file decides how it is read (see resolve)."""
    kinds = {stamp_kind(name) for name in stamps.values()}
    if 'FITS' in kinds:
        return 'FITS'
    return 'COMPRESSED' if kinds else None


def detect_band_filetype(band_dir):
    "stamps_filetype of the directory `band_dir`; None too if it doesn't exist."
    return stamps_filetype(scan_stamps(band_dir))


def list_fits_files(directory):
    "Sorted basenames of every FITS file directly under `directory`."
    return sorted(scan_stamps(directory, fits_only=True).values())


# --- MEF (multi-extension FITS) discovery -----------------------------------

def _image_hdus(hdu_list) -> Dict[str, Union[str, int]]:
    """{band_name: key} for every HDU in an open file that holds image data (skips
    an empty PrimaryHDU). A band's name is the HDU's EXTNAME, and so is its key. An
    HDU with no EXTNAME is named 'HDU{index}' and keyed by that index, having no
    name to look up. A repeated EXTNAME keeps its first HDU, as `hdu_list[name]` does."""
    bands = {}
    for index, hdu in enumerate(hdu_list):
        if hdu.data is None:
            continue
        bands.setdefault(hdu.name or f'HDU{index}', hdu.name or index)
    return bands


def discover_mef_bands(sample_filepath: str) -> Dict[str, Union[str, int]]:
    """Opens one representative FITS file and returns {band_name: key} for its
    image HDUs (see _image_hdus); a key is what ExtensionBandSource takes.

    Lobby's preview lists these; the viewers go through locate_mef_bands. Each
    object's own file is still asked for its bands by name (BUGS 36), so neither
    calls this per object."""
    with fits.open(sample_filepath, memmap=False) as hdu_list:
        return _image_hdus(hdu_list)


MEF_BAND_SEARCH_FILES = 100


def locate_mef_bands(directory, filenames, wanted, max_files=MEF_BAND_SEARCH_FILES):
    """({band: key} for each band in `wanted` that some file has, every band name
    seen), from the first `max_files` of `filenames`, stopping once all are found.

    Usually the first file has every band and is the only one opened. Looking
    further is what lets a run start when that file happens to lack one (BUGS 36):
    it would get a placeholder like any other. The cap bounds the wait before a
    mistyped band name is reported (~2.5 ms a file)."""
    found, seen = {}, {}
    for name in filenames[:max_files]:
        if set(wanted) <= found.keys():
            break
        bands = discover_mef_bands(join(directory, name))
        for band, key in bands.items():
            seen.setdefault(band, key)
            if band in wanted:
                found.setdefault(band, key)
    return found, sorted(seen)


def is_mef_dataset(path: str) -> bool:
    "True if `path` holds *.fits files directly, rather than only subdirectories."
    return bool(list_fits_files(path))


# --- band sources -------------------------------------------------------------

class _StampIndex:
    """{stem: filename} for a band's directory, listed once, on first use, and kept:
    resolving an object is then a lookup, not four `os.path.exists` calls, and it is
    the same answer for `obj.JPG` on any filesystem. Files added later are not seen."""

    _fits_only = False

    def stamps(self):
        index = self.__dict__.get('_index')
        if index is None:
            index = scan_stamps(self.directory, fits_only=self._fits_only)
            self.use_index(index)
        return index

    def use_index(self, index):
        "Share one listing between sources of the same directory (every --mef band)."
        object.__setattr__(self, '_index', index)


@dataclass(frozen=True)
class DirBandSource(_StampIndex):
    "A band backed by one file per object, all in one directory (the original scheme)."
    directory: str
    filetype: Optional[str] = None   # detect_band_filetype's answer for the directory


@dataclass(frozen=True)
class ExtensionBandSource(_StampIndex):
    """A band backed by one HDU extension inside a per-object multi-extension FITS
    file (--mef). Every band in an MEF run shares the same `directory` -- there are
    no per-band subdirectories -- and differs only in which HDU it reads.

    `extension` is an EXTNAME, looked up in each object's own file. Only an HDU the
    sample file left unnamed is an index, and that one is still read by position."""
    directory: str
    extension: Union[str, int]   # from discover_mef_bands

    _fits_only = True


BandSource = Union[DirBandSource, ExtensionBandSource]


class Locator(NamedTuple):
    "Where one object's data for one band lives."
    filepath: str
    hdu: Union[str, int, None]   # EXTNAME or index; None = not FITS, no HDU concept applies

    @property
    def is_fits(self):
        return self.hdu is not None


def resolve(source: BandSource, stem: str) -> Optional[Locator]:
    """Where object `stem`'s data lives for this band, or None if it doesn't exist.
    The only place that branches on which kind of BandSource this is. The object's
    own file decides whether it is read as FITS: a directory band may hold both."""
    name = source.stamps().get(stem)
    if name is None:
        return None
    filepath = join(source.directory, name)
    if isinstance(source, ExtensionBandSource):
        return Locator(filepath, source.extension)
    return Locator(filepath, 0 if stamp_kind(name) == 'FITS' else None)


def locate(band_sources: Dict[str, BandSource], band: str, stem: str) -> Locator:
    "Where object `stem`'s data lives for `band`; MissingBandError if nowhere."
    locator = resolve(band_sources[band], stem)
    if locator is None:
        raise MissingBandError(band)
    return locator


# --- reading -----------------------------------------------------------------

class FITSReadError(Exception):
    "A FITS file could not be opened, or its data/header could not be read."


class MissingBandError(LookupError):
    """One object has no data for one band: its file lacks that extension (MEF), or
    there is no file for it in that band's directory. Both viewers draw a placeholder
    panel for it instead of stopping (BUGS 36). `available` is the band names the
    object's file does have, when there is a file to ask; otherwise None.

    `unreadable` marks the other way of having no data: there is a file, and it could
    not be read as an image (`detail` says why). The viewers treat both alike."""

    def __init__(self, band, filepath=None, available=None, unreadable=False, detail=''):
        self.band = band
        self.filepath = filepath
        self.available = available
        self.unreadable = unreadable
        self.detail = detail
        where = f"in {os.path.basename(filepath)}" if filepath else "for this object"
        if unreadable:
            super().__init__(f"Could not read '{band}' {where}: {detail}")
        else:
            super().__init__(f"No '{band}' {where}.")


def _hdu_key(locator: Locator) -> Union[str, int]:
    return locator.hdu if locator.hdu is not None else 0


def _has_image(hdu):
    "From the header alone: asking for `.data` would read the pixels."
    return getattr(hdu, 'is_image', False) and hdu.header.get('NAXIS', 0) > 0


def _find_hdu(hdu_list, locator: Locator):
    """The HDU `locator` points at in an open file, or None when the file has no such
    name. A directory band's file is asked for HDU 0; when its primary HDU is empty
    (a compressed or extension-only stamp) its first image HDU stands in."""
    key = _hdu_key(locator)
    try:
        hdu = hdu_list[key]
    except KeyError:
        return None
    if key == 0 and not _has_image(hdu):
        return next((h for h in hdu_list[1:] if _has_image(h)), hdu)
    return hdu


def _open_hdu(hdu_list, locator: Locator):
    """The HDU `locator` points at in an open file. A name the file doesn't have, or
    an HDU without data, is a missing band, not a read error; a bad index still is."""
    hdu = _find_hdu(hdu_list, locator)
    if hdu is None or hdu.data is None:
        raise MissingBandError(locator.hdu, locator.filepath, available=list(_image_hdus(hdu_list)))
    return hdu


def _stamp_pixels(hdu, locator):
    """The HDU's pixels as the 2-D image every caller expects. Axes of length 1 are
    dropped (a 1 x ny x nx stamp); anything still not 2-D, or with no pixels at all,
    cannot be shown as a stamp."""
    data = hdu.data
    if data.ndim != 2:
        data = np.squeeze(data)
    if data.ndim != 2 or data.size == 0:
        raise FITSReadError(f"{locator.filepath} (HDU {locator.hdu}) is not a 2-D image: "
                            f"shape {hdu.data.shape}")
    return data


# Not only OSError: a truncated or damaged file fails in astropy in more ways than
# that (ValueError on a bad card, TypeError or EOFError on a short data block).
_READ_ERRORS = (OSError, IndexError, ValueError, TypeError, EOFError)


def read_pixels(locator: Locator, *, memmap: bool = False) -> np.ndarray:
    """Opens `locator.filepath` and returns the pixel array at `locator.hdu` (HDU 0
    if unset). Always closes the file -- the only place fits.open is called for
    pixel data in this codebase. memmap defaults to False: much faster than True
    when opening/closing many small single-HDU files, which is the common case."""
    try:
        with fits.open(locator.filepath, memmap=memmap) as hdu_list:
            return _stamp_pixels(_open_hdu(hdu_list, locator), locator)
    except _READ_ERRORS as e:
        raise FITSReadError(f"Could not read {locator.filepath} (HDU {locator.hdu}): {e}") from e


def read_header(locator: Locator) -> fits.Header:
    """Header-only read (no pixel data) -- for RA/Dec-only callers like FetchThread.
    Not fits.getheader, which refuses a bare EXTNAME."""
    try:
        with fits.open(locator.filepath, memmap=False) as hdu_list:
            hdu = _find_hdu(hdu_list, locator)
            if hdu is None:
                raise MissingBandError(locator.hdu, locator.filepath,
                                       available=list(_image_hdus(hdu_list)))
            return hdu.header
    except _READ_ERRORS as e:
        raise FITSReadError(f"Could not read header of {locator.filepath} (HDU {locator.hdu}): {e}") from e


def read_pixels_and_header(locator: Locator, *, memmap: bool = False):
    "One file open, both pixels and header -- avoids opening the same file twice."
    try:
        with fits.open(locator.filepath, memmap=memmap) as hdu_list:
            hdu = _open_hdu(hdu_list, locator)
            return _stamp_pixels(hdu, locator), hdu.header
    except _READ_ERRORS as e:
        raise FITSReadError(f"Could not read {locator.filepath} (HDU {locator.hdu}): {e}") from e


def missing_bands_note(missing: Dict[str, MissingBandError]) -> str:
    """One status-bar line for one object, from {band: MissingBandError}: the bands
    it lacks, then the ones whose file could not be read, then what its file does
    have (MEF only -- a directory-per-band object has no single file to ask)."""
    absent = [band for band, e in missing.items() if not e.unreadable]
    unreadable = [band for band, e in missing.items() if e.unreadable]
    parts = []
    if absent:
        parts.append(f"no {', '.join(absent)}")
    if unreadable:
        parts.append(f"unreadable {', '.join(unreadable)}")
    note = '; '.join(parts)
    available = next((e.available for e in missing.values() if e.available is not None), None)
    if available is not None:
        note += f" -- file has: {', '.join(available) or '(no image extensions)'}"
    return note


# --- WCS / RA-Dec --------------------------------------------------------------

class RaDec(NamedTuple):
    ra: float
    dec: float
    pixel_size_arcsec: float
    image_dim: int


def get_ra_dec(header) -> RaDec:
    """RA/Dec of the image center, arcsec/pixel scale, and image size, from `header`'s
    WCS. Unifies single_viewer's two near-duplicate versions (one a 2-tuple, one a
    4-tuple) into the one superset shape; callers destructure what they need.

    A header without a celestial WCS (a PSF extension, a stamp saved without one) gives
    NaN RA/Dec/scale: read as-is it is pixel coordinates at 3600"/px, and a survey
    cutout of that is ~1e6 px at a meaningless position."""
    w = WCS(header, fix=False)
    if not w.has_celestial:
        return RaDec(np.nan, np.nan, np.nan, np.max(w.array_shape) if w.array_shape else 0)
    # array_shape is numpy-order (ny, nx); pixel_to_world_values wants WCS-axis
    # order (x, y) = (nx, ny). Swapping these was a longstanding bug that only
    # showed up for non-square images -- see BUGS.md.
    sky = w.pixel_to_world_values([w.array_shape[1] // 2], [w.array_shape[0] // 2])
    # Not the diagonal of pixel_scale_matrix: that is scale x cos(rotation), 0 at 90 deg (BUGS 37).
    pixel_size = np.round(np.max(proj_plane_pixel_scales(w)) * 3600, decimals=4)
    return RaDec(sky[0][0], sky[1][0], pixel_size, np.max(w.array_shape))


def sky_position(header) -> Optional[RaDec]:
    """get_ra_dec(header) when the header places the stamp on the sky, else None: no
    celestial WCS, or one astropy cannot evaluate. What the viewer and the pre-fetch
    both ask of each candidate header, so they agree on which band answers."""
    try:
        radec = get_ra_dec(header)
    except Exception:       # any way a WCS can fail to evaluate is "no position" here
        return None
    if any(v is None or np.isnan(v) for v in (radec.ra, radec.dec, radec.pixel_size_arcsec)):
        return None
    return radec


def find_sky_position(sources, stem, headers=None) -> Optional[RaDec]:
    """Object `stem`'s position on the sky: `sky_position` of the first of its FITS
    stamps that has one, taking `sources` (BandSources) in order -- the default band
    first. `headers`, parallel to `sources`, holds the headers already read (None
    where not), so a caller that has drawn a band does not open its file again.
    None when no stamp has one. Headers only: whether a band's pixels can be read
    does not change the answer."""
    for i, source in enumerate(sources):
        header = headers[i] if headers else None
        if header is None:
            locator = resolve(source, stem)
            if locator is None or not locator.is_fits:
                continue
            try:
                header = read_header(locator)
            except (MissingBandError, FITSReadError):
                continue
        radec = sky_position(header)
        if radec is not None:
            return radec
    return None


# The pixel scale depends only on these cards, and every stamp from one survey shares
# them. Parsing a WCS costs ~1.4 ms, and the mosaic needs a scale for every band of
# every tile on each page: uncached, that took a 40-tile page from 0.5 s to 0.8 s.
# The WCS is built from these cards alone, so the cached answer is a function of its
# key: nothing else in a header (distortion cards, a third axis) can change it.
_SCALE_CARDS = re.compile(r'(CTYPE|CUNIT|CDELT|CROTA|CD|PC)\d')


def pixel_scale_arcsec(header) -> Optional[float]:
    """Arcsec per pixel from `header`'s celestial WCS, or None when there is none (a PSF
    extension, a stamp saved without WCS) or it can't be used. Non-square pixels give
    the larger of the two scales. proj_plane_pixel_scales, unlike the diagonal of
    pixel_scale_matrix, is right for a rotated image too."""
    return _pixel_scale_from_cards(tuple((k, header[k]) for k in header if _SCALE_CARDS.match(k)))


@functools.lru_cache(maxsize=1024)
def _pixel_scale_from_cards(cards):
    try:
        w = WCS(fits.Header(cards), fix=False)
    except ValueError:      # astropy's WcsError and everything under it
        return None
    if not w.has_celestial:
        return None
    scale = float(np.max(proj_plane_pixel_scales(w.celestial))) * 3600
    return scale if np.isfinite(scale) and scale > 0 else None
