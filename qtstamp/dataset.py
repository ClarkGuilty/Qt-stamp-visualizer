"""The objects a viewer classifies, in order, and where each band's data lives.

Both viewers open a dataset the same way, so they do it here (REFACTOR.md step 3).
Qt-free: what goes wrong is a `DatasetError` whose message the viewer prints
before exiting, and the notes along the way (object count, shuffle, a composite
dropped) are printed as they always were.

An object is a filename stem with a stamp in at least one band. No band is
required to have every object: the list is the union over the bands in use, and a
band an object lacks -- or whose file cannot be read -- is a "no data" panel
(`load_stamp` raises `MissingBandError`). The default band (-b) only names the
panel shown first, and comes first when an object's name or RA/Dec is looked up.
"""

from dataclasses import dataclass
from os.path import join
from typing import Dict, List, NamedTuple, Optional, Set

import numpy as np

from . import fits_io
from .imaging import open_rendered_image


class DatasetError(Exception):
    "The dataset cannot be classified. The message says why, ready to print."


@dataclass
class Dataset:
    listimage: List[str]          # object filenames, shuffled if seeded
    filetype: str                 # 'FITS' if any band holds FITS stamps, else 'COMPRESSED'
    band_filetype: Dict[str, Optional[str]]
    composite_bands: List[str]    # only the composites that can be drawn
    composite_band_members: Dict[str, List[str]]
    all_single_bands: Set[str]    # main, color, and kept composites' members
    band_sources: Dict[str, fits_io.BandSource]
    single_bands: List[str]       # all_single_bands in order: default band, color bands, members


def ordered_bands(main_band, color_bands, composite_members):
    """The single bands a viewer reads, once each: the default band, the color bands,
    then the members of `composite_members` (R,G,B triples), in that order."""
    bands = [main_band, *color_bands, *(b for members in composite_members for b in members)]
    return list(dict.fromkeys(b for b in bands if b))


def drawable_composites(composite_bands, composite_band_members, band_filetype, log=print):
    """(keys, {key: members}) of the composites a viewer can draw at all, given
    {band: filetype} for their member bands (`fits_io.stamps_filetype`).

    An RGB composite stacks three FITS bands. One with a member band that holds no
    FITS stamp can never be drawn, so it is dropped here, with a note, rather than
    kept as a column of placeholders. A composite kept here is still drawn per
    object: one lacking a member is a "no data" panel."""
    kept, kept_members = [], {}
    for key in composite_bands:
        members = composite_band_members[key]
        non_fits = [b for b in members if band_filetype.get(b) != 'FITS']
        if non_fits:
            log(f"RGB composite '{key}' skipped -- requires FITS bands, but "
                f"{', '.join(non_fits)} {'has' if len(non_fits) == 1 else 'have'} no FITS stamps.")
            continue
        kept.append(key)
        kept_members[key] = members
    return kept, kept_members


def _dir_sources(path, bands):
    "{band: DirBandSource} with each band's directory listed once, its filetype from that listing."
    sources = {}
    for band in bands:
        stamps = fits_io.scan_stamps(join(path, band))
        sources[band] = fits_io.DirBandSource(join(path, band), fits_io.stamps_filetype(stamps))
        sources[band].use_index(stamps)
    return sources


def viewer_bands(path, main_band, color_bands, composites, mef):
    """The single bands a viewer started with this band setup reads, in order
    (`ordered_bands`), after it has dropped the composites it cannot draw.
    `composites` is a list of R,G,B triples. What the lobby passes to
    `list_dataset_objects` to list the objects exactly as the viewer will."""
    members = {','.join(t): tuple(t) for t in composites}
    if not mef:
        filetypes = {band: fits_io.detect_band_filetype(join(path, band))
                     for band in ordered_bands('', [], members.values())}
        _, members = drawable_composites(list(members), members, filetypes, log=lambda _note: None)
    return ordered_bands(main_band, color_bands, members.values())


def _union_names(sources):
    """Sorted filenames of every object with a stamp in at least one of `sources`.
    An object's name is its file's in the first source that has it, so with the
    default band first a dataset whose bands all hold the same objects is listed
    exactly as that band's directory."""
    names = {}
    for source in sources:
        for stem, name in source.stamps().items():
            names.setdefault(stem, name)
    return sorted(names.values())


def list_dataset_objects(path, bands, mef):
    """The objects under `path`, unshuffled: every FITS file directly under it with
    `mef`, else every stamp in the directory of any of `bands` -- the viewer's single
    bands in order (`viewer_bands`), or one band's name."""
    if mef:
        return fits_io.list_fits_files(path)
    if isinstance(bands, str):
        bands = [bands]
    return _union_names(fits_io.DirBandSource(join(path, band)) for band in bands)


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
    a directory and the objects are the union of what the bands hold; an RGB
    composite with a member that has no FITS stamp is dropped with a note.
    """
    if mef:
        # MEF bands are always FITS, so every declared composite is eligible.
        kept_composites, kept_members = list(composite_bands), dict(composite_band_members)
        single_bands = ordered_bands(main_band, color_bands, kept_members.values())
        all_single_bands = set(single_bands)
        # One listing of the directory, shared by every band below.
        index = fits_io.scan_stamps(path, fits_only=True)
        listimage = sorted(index.values())
        print(f"Classifying {len(listimage)} sources.")
        if not listimage:
            raise DatasetError(f"No FITS files found in {path}.")

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
                        for b in single_bands}
        for source in band_sources.values():
            source.use_index(index)
    else:
        # A band's stamps are read as what each file is, so FITS and PNG/JPG can mix
        # within a band as well as between bands. band_filetype says whether a band
        # holds any FITS at all. Every band named is listed once, here.
        sources = _dir_sources(path, ordered_bands(main_band, color_bands,
                                                   composite_band_members.values()))
        band_filetype = {band: source.filetype for band, source in sources.items()}
        kept_composites, kept_members = drawable_composites(
            composite_bands, composite_band_members, band_filetype)
        single_bands = ordered_bands(main_band, color_bands, kept_members.values())
        all_single_bands = set(single_bands)
        band_sources = {band: sources[band] for band in single_bands}
        listimage = _union_names(band_sources[band] for band in single_bands)
        print(f"Classifying {len(listimage)} sources.")
        if not listimage:
            where = ', '.join(join(path, band) for band in single_bands)
            raise DatasetError(f"No FITS, PNG, or JPG files found in {where}.")

    if seed is not None:
        print(f"Shuffling with seed {seed}")
    shuffle(listimage, seed)

    filetype = ('FITS' if any(band_filetype[b] == 'FITS' for b in single_bands)
                else 'COMPRESSED')
    return Dataset(listimage, filetype, band_filetype, kept_composites, kept_members,
                   all_single_bands, band_sources, single_bands)


# --- one object's stamp ---------------------------------------------------------

class Stamp(NamedTuple):
    """One object's data in one band. From FITS: `pixels` is the 2-D array and `header`
    its header. From PNG/JPG: `pixels` is the loaded PIL image, already rendered, and
    `header` is None."""
    pixels: object
    header: object
    filepath: str

    @property
    def is_fits(self):
        return self.header is not None


def load_stamp(band_sources, band, stem) -> Stamp:
    """Object `stem`'s stamp in `band`, read as what its file is.

    MissingBandError when there is nothing to show: no file, no such extension, or
    (`unreadable`) a file that cannot be read as an image. Nothing else is raised
    for a bad file, so a viewer need only draw its "no data" panel."""
    locator = fits_io.locate(band_sources, band, stem)
    try:
        if locator.is_fits:
            pixels, header = fits_io.read_pixels_and_header(locator)
            return Stamp(pixels, header, locator.filepath)
        return Stamp(open_rendered_image(locator.filepath), None, locator.filepath)
    except (fits_io.FITSReadError, OSError, ValueError, SyntaxError) as e:
        # SyntaxError is what PIL raises for a damaged PNG.
        raise fits_io.MissingBandError(band, locator.filepath, unreadable=True,
                                       detail=str(e)) from e


def composite_stack(stamps, members):
    """The (ny, nx, 3) stack for an RGB composite of `members`, from {band: Stamp}, or
    (None, why) when it cannot be made: a member with no stamp, one that is not FITS
    for this object, or members of different sizes. `why` is a short list of lines
    for the placeholder."""
    absent = [b for b in members if b not in stamps]
    if absent:
        return None, absent
    not_fits = [b for b in members if not stamps[b].is_fits]
    if not_fits:
        return None, [*not_fits, 'not FITS']
    shapes = {stamps[b].pixels.shape for b in members}
    if len(shapes) > 1:
        return None, ['sizes differ']
    return np.stack([stamps[b].pixels for b in members], axis=-1), None
