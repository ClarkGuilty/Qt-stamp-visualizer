# This Python file uses the following encoding: utf-8
"""Shared, Qt-free logic for filtering a classification CSV down to the
"positive" objects and materializing them (symlinked or copied) into a fresh
band-folder tree for the next classification stage.

Used by lobby.py; kept free of any Qt imports so it can be exercised with a
plain `python` interpreter, no `QT_QPA_PLATFORM=offscreen` needed.
"""

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


def select_positive_filenames(csv_path, positive_values):
    """Return the file_name values whose 'classification' matches positive_values.

    Handles both the mosaic tool's numeric codes (loaded as floats, e.g. 1.0)
    and the single-viewer's string codes (e.g. 'A') -- a plain
    `astype(str).isin(...)` would silently match nothing for the numeric case,
    since str(1.0) == "1.0", not "1".
    """
    df = pd.read_csv(csv_path)
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

    A multi-extension FITS dataset (`fits_io.is_mef_dataset`: FITS files directly
    under source_root, one per object, as the lobby detects it) has no band
    subfolders: its matching files land directly in output_root, where the next
    stage's --mef finds them (BUGS 41). Reported as one band, '(MEF)'.

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
                  for b in sorted(os.listdir(source_root)) if os.path.isdir(join(source_root, b))]
    result.bands_processed = [band for band, _, _ in groups]

    for band, src_band_dir, out_band_dir in groups:
        os.makedirs(out_band_dir, exist_ok=True)

        for filename in filenames:
            src = join(src_band_dir, filename)
            dst = join(out_band_dir, filename)

            if not os.path.exists(src):
                result.missing.append((band, filename))
                progress(f"Missing: {src}")
                continue

            if os.path.lexists(dst):
                os.remove(dst)

            try:
                if use_symlink:
                    os.symlink(os.path.abspath(src), dst)
                    result.linked += 1
                else:
                    shutil.copy2(src, dst)
                    result.copied += 1
            except OSError as e:
                progress(f"Error linking/copying {src} -> {dst}: {e}")

        progress(f"Band {band}: done ({len(filenames)} candidate file(s)).")

    return result
