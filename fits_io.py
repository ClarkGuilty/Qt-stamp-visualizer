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
  --path, each file holding every band as an HDU extension. Every object's file
  is assumed to share the same extension layout. See ExtensionBandSource and
  discover_mef_bands.

A BandSource is resolved once per band at startup (not per object); resolve()
then turns a BandSource plus one object's filename stem into a Locator, and
read_pixels/read_header/read_pixels_and_header turn a Locator into pixel data
and/or a header without either caller needing to know which scheme produced it.
"""

import glob
import os
from dataclasses import dataclass
from os.path import join
from typing import Dict, NamedTuple, Optional, Union

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

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

def discover_mef_bands(sample_filepath: str) -> Dict[str, int]:
    """Opens one representative FITS file and returns {band_name: hdu_index} for
    every HDU that actually holds image data (skips an empty PrimaryHDU). A band's
    name is the HDU's EXTNAME if it has one, else 'HDU{index}'.

    --mef assumes every object's file shares this same extension layout, so this
    is called once per run (or once for Lobby's preview), never per object."""
    bands = {}
    with fits.open(sample_filepath, memmap=False) as hdu_list:
        for index, hdu in enumerate(hdu_list):
            if hdu.data is None:
                continue
            bands[hdu.name or f'HDU{index}'] = index
    return bands


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
    no per-band subdirectories -- and differs only in which HDU it reads."""
    directory: str
    extension: int   # HDU index, from discover_mef_bands


BandSource = Union[DirBandSource, ExtensionBandSource]


class Locator(NamedTuple):
    "Where one object's data for one band lives."
    filepath: str
    hdu: Optional[int]   # None means "not FITS, no HDU concept applies"


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


def _hdu_index(locator: Locator) -> int:
    return locator.hdu if locator.hdu is not None else 0


def read_pixels(locator: Locator, *, memmap: bool = False) -> np.ndarray:
    """Opens `locator.filepath` and returns the pixel array at `locator.hdu` (HDU 0
    if unset). Always closes the file -- the only place fits.open is called for
    pixel data in this codebase. memmap defaults to False: much faster than True
    when opening/closing many small single-HDU files, which is the common case."""
    try:
        with fits.open(locator.filepath, memmap=memmap) as hdu_list:
            return hdu_list[_hdu_index(locator)].data
    except (OSError, IndexError) as e:
        raise FITSReadError(f"Could not read {locator.filepath} (HDU {locator.hdu}): {e}") from e


def read_header(locator: Locator) -> fits.Header:
    "Header-only read (no pixel data) -- for RA/Dec-only callers like FetchThread."
    try:
        return fits.getheader(locator.filepath, ext=_hdu_index(locator), memmap=False)
    except (OSError, IndexError) as e:
        raise FITSReadError(f"Could not read header of {locator.filepath} (HDU {locator.hdu}): {e}") from e


def read_pixels_and_header(locator: Locator, *, memmap: bool = False):
    "One file open, both pixels and header -- avoids opening the same file twice."
    try:
        with fits.open(locator.filepath, memmap=memmap) as hdu_list:
            hdu = hdu_list[_hdu_index(locator)]
            return hdu.data, hdu.header
    except (OSError, IndexError) as e:
        raise FITSReadError(f"Could not read {locator.filepath} (HDU {locator.hdu}): {e}") from e


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
    sky = w.pixel_to_world_values([w.array_shape[0] // 2], [w.array_shape[1] // 2])
    pixel_size = np.round(np.max(np.diag(np.abs(w.pixel_scale_matrix))) * 3600, decimals=4)
    return RaDec(sky[0][0], sky[1][0], pixel_size, np.max(w.array_shape))
