# This Python file uses the following encoding: utf-8
"""Shared, Qt-free logic for filtering a classification CSV down to the
"positive" objects and materializing them (symlinked or copied) into a fresh
band-folder tree for the next classification stage.

Used by lobby.py; kept free of any Qt imports so it can be exercised with a
plain `python` interpreter, no `QT_QPA_PLATFORM=offscreen` needed.
"""

import errno
import os
import shutil
from os.path import join

import pandas as pd

from . import fits_io


class ExtractionResult:
    def __init__(self):
        self.linked = 0
        self.copied = 0
        self.missing = []  # list of (band, filename)
        self.bands_processed = []

    def summary(self):
        return (f"Linked: {self.linked}, Copied: {self.copied}, "
                f"Missing: {len(self.missing)}, Bands: {', '.join(self.bands_processed)}")


def _symlinks_refused(error):
    """True when an os.symlink failure means the filesystem (exFAT, FAT) or the
    platform (Windows without the privilege) does not do symlinks at all -- not one
    file's own trouble, which is reported for that file and leaves the rest linked."""
    return (error.errno in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP)
            or getattr(error, 'winerror', None) == 1314)


def select_positive_filenames(csv_path, positive_values):
    """Return the file_name values whose 'classification' matches positive_values.

    Handles both the mosaic tool's numeric codes (loaded as floats, e.g. 1.0)
    and the single-viewer's string codes (e.g. 'A') -- a plain
    `astype(str).isin(...)` would silently match nothing for the numeric case,
    since str(1.0) == "1.0", not "1".
    """
    # Read as text, then tested for being numeric: left to pandas, a class named NA,
    # N/A, None or null is a missing value and matches nothing.
    df = pd.read_csv(csv_path, converters={'classification': str, 'file_name': str})
    col = df['classification']
    numeric = pd.to_numeric(col, errors='coerce')
    if numeric.notna().mean() > 0.5:
        mask = numeric.isin({float(v) for v in positive_values})
    else:
        mask = col.astype(str).isin({str(v) for v in positive_values})
    return df.loc[mask, 'file_name'].tolist()


def extract(csv_path, source_root, output_root, positive_values,
            use_symlink=True, on_progress=None):
    """Filter csv_path by positive_values and materialize matching files from
    every band subfolder of source_root into output_root, symlinking by
    default (use_symlink=False to copy instead). Returns an ExtractionResult.

    An object is matched in each band by its filename stem, not its full name:
    the CSV names its file in one band, and another band may hold it in another
    format (`obj.fits` beside `obj.jpg`). Only directories that hold stamps are
    bands (`fits_io.list_band_dirs`).

    A multi-extension FITS dataset (`fits_io.is_mef_dataset`: FITS files directly
    under source_root, one per object, as the lobby detects it) has no band
    subfolders: its matching files land directly in output_root, where the next
    stage's --mef finds them (BUGS 41). Reported as one band, '(MEF)'. So does a
    flat directory of PNG/JPGs, reported as '(flat)'.

    Where the filesystem refuses symlinks (exFAT and FAT drives), the files are
    copied instead, and `on_progress` is told once.

    Raises ValueError, before touching anything, when output_root is source_root.
    """
    def progress(msg):
        if on_progress is not None:
            on_progress(msg)

    if os.path.realpath(output_root) == os.path.realpath(source_root):
        # Each destination would be its own source: removed, then linked to itself.
        raise ValueError(f"The output path is the data path ({source_root}); "
                         "extracting there would delete the stamps. Pick another output path.")

    result = ExtractionResult()
    filenames = select_positive_filenames(csv_path, positive_values)
    progress(f"{len(filenames)} positive file(s) selected from {csv_path}")

    if fits_io.is_mef_dataset(source_root):
        groups = [('(MEF)', source_root, output_root)]
    else:
        groups = [(b, join(source_root, b), join(output_root, b))
                  for b in fits_io.list_band_dirs(source_root)]
        if not groups and fits_io.has_stamps(source_root):
            groups = [('(flat)', source_root, output_root)]
    result.bands_processed = [band for band, _, _ in groups]

    for band, src_band_dir, out_band_dir in groups:
        os.makedirs(out_band_dir, exist_ok=True)
        stamps = fits_io.scan_stamps(src_band_dir)

        for filename in filenames:
            name = stamps.get(fits_io.stamp_stem(filename))
            if name is None:
                result.missing.append((band, filename))
                progress(f"Missing: {fits_io.stamp_stem(filename)} in {src_band_dir}")
                continue
            src = join(src_band_dir, name)
            dst = join(out_band_dir, name)

            if os.path.lexists(dst):
                os.remove(dst)

            try:
                if use_symlink:
                    try:
                        os.symlink(os.path.abspath(src), dst)
                        result.linked += 1
                        continue
                    except OSError as e:
                        if not _symlinks_refused(e):
                            raise
                        progress(f"Symlinks are not possible in {out_band_dir} ({e}) -- "
                                 "copying the files instead.")
                        use_symlink = False
                shutil.copy2(src, dst)
                result.copied += 1
            except OSError as e:
                progress(f"Error linking/copying {src} -> {dst}: {e}")

        progress(f"Band {band}: done ({len(filenames)} candidate file(s)).")

    return result
