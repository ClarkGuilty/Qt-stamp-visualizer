# This Python file uses the following encoding: utf-8
"""All astropy.io.fits / astropy.wcs access lives here.

Both mosaic.py and single_viewer.py go through this module instead of calling
astropy directly, so there is exactly one place that knows how to open a FITS
file, one memmap policy, and one answer to "what does a band mean" for either of
the two dataset layouts the tools support:

- directory-per-band (the original scheme): one directory under --path per band,
  each holding one single-extension FITS (or PNG/JPG) file per object, matched
  across bands by filename stem. See DirBandSource.
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
import glob
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
COMPRESSED_EXTS = ('.png', '.jpg', '.jpeg')


# --- directory-per-band discovery -------------------------------------------

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


def list_fits_files(directory):
    "Sorted basenames of every *.fits file directly under `directory`."
    return sorted(os.path.basename(x) for x in glob.glob(join(directory, '*' + FITS_EXT)))


def list_objects(directory):
    """Sorted basenames of every FITS file in `directory`; if none, sorted
    basenames of every PNG/JPG/JPEG file instead (all three extensions combined
    into one sorted list, not tried one at a time). Returns (names, filetype)
    where filetype is 'FITS' or 'COMPRESSED', matching detect_band_filetype."""
    fits_files = list_fits_files(directory)
    if fits_files:
        return fits_files, 'FITS'
    compressed = sorted(os.path.basename(x)
                         for ext in COMPRESSED_EXTS
                         for x in glob.glob(join(directory, '*' + ext)))
    return compressed, 'COMPRESSED'


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

@dataclass(frozen=True)
class DirBandSource:
    "A band backed by one file per object, all in one directory (the original scheme)."
    directory: str
    filetype: str   # 'FITS' or 'COMPRESSED', from detect_band_filetype/list_objects


@dataclass(frozen=True)
class ExtensionBandSource:
    """A band backed by one HDU extension inside a per-object multi-extension FITS
    file (--mef). Every band in an MEF run shares the same `directory` -- there are
    no per-band subdirectories -- and differs only in which HDU it reads.

    `extension` is an EXTNAME, looked up in each object's own file. Only an HDU the
    sample file left unnamed is an index, and that one is still read by position."""
    directory: str
    extension: Union[str, int]   # from discover_mef_bands


BandSource = Union[DirBandSource, ExtensionBandSource]


class Locator(NamedTuple):
    "Where one object's data for one band lives."
    filepath: str
    hdu: Union[str, int, None]   # EXTNAME or index; None = not FITS, no HDU concept applies


def resolve(source: BandSource, stem: str) -> Optional[Locator]:
    """Where object `stem`'s data lives for this band, or None if it doesn't exist.
    The only place that branches on which kind of BandSource this is."""
    if isinstance(source, ExtensionBandSource):
        filepath = join(source.directory, stem + FITS_EXT)
        return Locator(filepath, source.extension) if os.path.exists(filepath) else None
    for ext in (FITS_EXT, *COMPRESSED_EXTS):
        filepath = join(source.directory, stem + ext)
        if os.path.exists(filepath):
            return Locator(filepath, 0 if ext == FITS_EXT else None)
    return None


# --- reading -----------------------------------------------------------------

class FITSReadError(Exception):
    "A FITS file could not be opened, or its data/header could not be read."


class MissingBandError(LookupError):
    """One object has no data for one band: its file lacks that extension (MEF), or
    there is no file for it in that band's directory. Both viewers draw a placeholder
    panel for it instead of stopping (BUGS 36). `available` is the band names the
    object's file does have, when there is a file to ask; otherwise None."""

    def __init__(self, band, filepath=None, available=None):
        self.band = band
        self.filepath = filepath
        self.available = available
        where = f"in {os.path.basename(filepath)}" if filepath else "for this object"
        super().__init__(f"No '{band}' {where}.")


def _hdu_key(locator: Locator) -> Union[str, int]:
    return locator.hdu if locator.hdu is not None else 0


def _open_hdu(hdu_list, locator: Locator):
    """The HDU `locator` points at in an open file. A name the file doesn't have, or
    an HDU without data, is a missing band, not a read error; a bad index still is."""
    try:
        hdu = hdu_list[_hdu_key(locator)]
    except KeyError:
        hdu = None
    if hdu is None or hdu.data is None:
        raise MissingBandError(locator.hdu, locator.filepath, available=list(_image_hdus(hdu_list)))
    return hdu


def read_pixels(locator: Locator, *, memmap: bool = False) -> np.ndarray:
    """Opens `locator.filepath` and returns the pixel array at `locator.hdu` (HDU 0
    if unset). Always closes the file -- the only place fits.open is called for
    pixel data in this codebase. memmap defaults to False: much faster than True
    when opening/closing many small single-HDU files, which is the common case."""
    try:
        with fits.open(locator.filepath, memmap=memmap) as hdu_list:
            return _open_hdu(hdu_list, locator).data
    except (OSError, IndexError) as e:
        raise FITSReadError(f"Could not read {locator.filepath} (HDU {locator.hdu}): {e}") from e


def read_header(locator: Locator) -> fits.Header:
    """Header-only read (no pixel data) -- for RA/Dec-only callers like FetchThread.
    Not fits.getheader, which refuses a bare EXTNAME."""
    try:
        with fits.open(locator.filepath, memmap=False) as hdu_list:
            try:
                return hdu_list[_hdu_key(locator)].header
            except KeyError:
                raise MissingBandError(locator.hdu, locator.filepath,
                                       available=list(_image_hdus(hdu_list))) from None
    except (OSError, IndexError) as e:
        raise FITSReadError(f"Could not read header of {locator.filepath} (HDU {locator.hdu}): {e}") from e


def read_pixels_and_header(locator: Locator, *, memmap: bool = False):
    "One file open, both pixels and header -- avoids opening the same file twice."
    try:
        with fits.open(locator.filepath, memmap=memmap) as hdu_list:
            hdu = _open_hdu(hdu_list, locator)
            return hdu.data, hdu.header
    except (OSError, IndexError) as e:
        raise FITSReadError(f"Could not read {locator.filepath} (HDU {locator.hdu}): {e}") from e


def missing_bands_note(missing: Dict[str, MissingBandError]) -> str:
    """One status-bar line for one object, from {band: MissingBandError}: the bands
    it lacks, then what its file does have (MEF only -- a directory-per-band object
    has no single file to ask)."""
    note = f"no {', '.join(missing)}"
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
    4-tuple) into the one superset shape; callers destructure what they need."""
    w = WCS(header, fix=False)
    # array_shape is numpy-order (ny, nx); pixel_to_world_values wants WCS-axis
    # order (x, y) = (nx, ny). Swapping these was a longstanding bug that only
    # showed up for non-square images -- see BUGS.md.
    sky = w.pixel_to_world_values([w.array_shape[1] // 2], [w.array_shape[0] // 2])
    # Not the diagonal of pixel_scale_matrix: that is scale x cos(rotation), 0 at 90 deg (BUGS 37).
    pixel_size = np.round(np.max(proj_plane_pixel_scales(w)) * 3600, decimals=4)
    return RaDec(sky[0][0], sky[1][0], pixel_size, np.max(w.array_shape))


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
