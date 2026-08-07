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
    """
    def progress(msg):
        if on_progress is not None:
            on_progress(msg)

    result = ExtractionResult()
    filenames = select_positive_filenames(csv_path, positive_values)
    progress(f"{len(filenames)} positive file(s) selected from {csv_path}")

    bands = sorted(b for b in os.listdir(source_root)
                    if os.path.isdir(join(source_root, b)))
    result.bands_processed = bands

    for band in bands:
        src_band_dir = join(source_root, band)
        out_band_dir = join(output_root, band)
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
