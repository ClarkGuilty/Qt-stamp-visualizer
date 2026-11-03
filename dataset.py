"""The objects a viewer classifies, in order, and where each band's data lives.

Both viewers open a dataset the same way, so they do it here (REFACTOR.md step 3).
Qt-free: what goes wrong is a `DatasetError` whose message the viewer prints
before exiting, and the notes along the way (object count, shuffle, a composite
dropped) are printed as they always were.
"""

from dataclasses import dataclass
from os.path import join
from typing import Dict, List, Optional, Set

import numpy as np

import fits_io


class DatasetError(Exception):
    "The dataset cannot be classified. The message says why, ready to print."


@dataclass
class Dataset:
    listimage: List[str]          # object filenames, shuffled if seeded
    filetype: str                 # the main band's, 'FITS' or 'COMPRESSED'
    band_filetype: Dict[str, Optional[str]]
    composite_bands: List[str]    # only the composites that can be drawn
    composite_band_members: Dict[str, List[str]]
    all_single_bands: Set[str]    # main, color, and kept composites' members
    band_sources: Dict[str, fits_io.BandSource]


def list_dataset_objects(path, main_band, mef):
    """(names, filetype) of the objects under `path`, unshuffled: every FITS file
    directly under it with `mef`, else the main band directory's listing."""
    if mef:
        return fits_io.list_fits_files(path), 'FITS'
    return fits_io.list_objects(join(path, main_band))


def shuffle(listimage, seed):
    "`listimage` in the order a session with `seed` classifies it, in place. None keeps it."
    if seed is not None:
        np.random.default_rng(seed).shuffle(listimage)
    return listimage


def load_dataset(path, main_band, color_bands, composite_bands, composite_band_members,
                 mef, seed) -> Dataset:
    """List the objects under `path`, find every band's source, and shuffle with `seed`.

    With `mef`, each object is one multi-extension FITS file and a band is an
    EXTNAME; a band no searched file has is a `DatasetError`, one that only some
    files lack becomes a placeholder panel there (BUGS 36). Otherwise each band is
    a directory, the main band's lists the objects, and an RGB composite with a
    member that is not FITS is dropped with a note.
    """
    listimage, filetype = list_dataset_objects(path, main_band, mef)
    if mef:
        print(f"Classifying {len(listimage)} sources.")
        if not listimage:
            raise DatasetError(f"No FITS files found in {path}.")

        all_single_bands = ({main_band} | set(color_bands) |
                            {b for members in composite_band_members.values() for b in members})
        # Searched in file order, before the shuffle: which files are opened to find
        # the EXTNAMEs (the first MEF_BAND_SEARCH_FILES) must not depend on the seed.
        mef_bands, seen = fits_io.locate_mef_bands(path, listimage, all_single_bands)
        missing = sorted(b for b in all_single_bands if b not in mef_bands)
        if missing:
            searched = min(len(listimage), fits_io.MEF_BAND_SEARCH_FILES)
            where = (f"any of the {searched}" if searched == len(listimage)
                     else f"the first {searched} of the {len(listimage)}")
            raise DatasetError(
                f"Extension{'s' if len(missing) != 1 else ''} not found in {where} files "
                f"in {path}: {', '.join(missing)}\n"
                f"Available extensions: {', '.join(seen) if seen else '(none found)'}")
        band_filetype = {b: 'FITS' for b in all_single_bands}
        band_sources = {b: fits_io.ExtensionBandSource(path, mef_bands[b])
                        for b in all_single_bands}
        # MEF bands are always FITS, so every declared composite is eligible.
        kept_composites = list(composite_bands)
        kept_members = dict(composite_band_members)
    else:
        print(f"Classifying {len(listimage)} sources.")
        if not listimage:
            raise DatasetError(f"No FITS, PNG, or JPG files found in {join(path, main_band)}.")

        # Each band's format is a property of its own directory, so bands can mix
        # FITS and PNG/JPG in one session. Only the main band's lists the objects,
        # and gives the RA/Dec the 1-by-1 viewer's tools use.
        band_filetype = {main_band: filetype}
        for band in (set(color_bands) |
                     {b for members in composite_band_members.values() for b in members}):
            band_filetype.setdefault(band, fits_io.detect_band_filetype(join(path, band)))

        # An RGB composite needs all three members to be FITS: drop one that isn't,
        # rather than crashing or mixing formats into one image.
        kept_composites, kept_members = [], {}
        for key in composite_bands:
            members = composite_band_members[key]
            non_fits = [b for b in members if band_filetype.get(b) != 'FITS']
            if non_fits:
                print(f"RGB composite '{key}' skipped -- requires FITS bands, but "
                      f"{', '.join(non_fits)} {'is' if len(non_fits) == 1 else 'are'} not FITS.")
                continue
            kept_composites.append(key)
            kept_members[key] = members

        all_single_bands = ({main_band} | set(color_bands) |
                            {b for members in kept_members.values() for b in members})
        band_sources = {band: fits_io.DirBandSource(join(path, band), band_filetype[band])
                        for band in all_single_bands}

    if seed is not None:
        print(f"Shuffling with seed {seed}")
    shuffle(listimage, seed)

    return Dataset(listimage, filetype, band_filetype, kept_composites, kept_members,
                   all_single_bands, band_sources)
