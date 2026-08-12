# This Python file uses the following encoding: utf-8
"""Background QObject workers for fetching external cutouts (PanSTARRS),
meant to be moved to a QThread by the caller. Used by
single_viewer_multiband_ERO_edition.py.
"""

import urllib.error
import urllib.request

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, Signal, Slot

PS1_FILENAMES_URL = 'https://ps1images.stsci.edu/cgi-bin/ps1filenames.py'
PS1_FITSCUT_URL = 'https://ps1images.stsci.edu/cgi-bin/fitscut.cgi'


def get_panstarrs_filenames(ra, dec, filters='grz'):
    "Look up PS1 stack image filenames for ra/dec via the STScI ps1filenames.py service."
    query_url = f"{PS1_FILENAMES_URL}?ra={ra}&dec={dec}&filters={filters}&type=stack"
    with urllib.request.urlopen(query_url, timeout=10) as response:
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


class SingleFetchWorker(QObject):
    successful_download = Signal()
    failed_download = Signal()
    has_finished = Signal()

    def __init__(self, url, savefile, title):
        super(SingleFetchWorker, self).__init__()
        self.url = url
        self.savefile = savefile
        self.title = title

    @Slot()
    def run(self):
        if self.url == '':
            self.successful_download.emit()
        else:
            try:
                urllib.request.urlretrieve(self.url, self.savefile)
                self.successful_download.emit()
            except (urllib.error.URLError, OSError):
                with open(self.savefile,'w') as f:
                    Image.fromarray(np.zeros((66,66),dtype=np.uint8)).save(f)
                self.failed_download.emit()
        self.has_finished.emit()


class PanstarrsFetchWorker(QObject):
    """Like SingleFetchWorker, but also does the ps1filenames.py lookup in the
    background thread -- that lookup is a network call too, and must not run
    on the GUI thread (it used to, and froze the UI for up to 10s whenever a
    stamp had no PS1 coverage, since a "no data" result is never cached)."""
    successful_download = Signal()
    failed_download = Signal()
    has_finished = Signal()

    def __init__(self, ra, dec, savefile, size):
        super(PanstarrsFetchWorker, self).__init__()
        self.ra = ra
        self.dec = dec
        self.savefile = savefile
        self.size = size

    @Slot()
    def run(self):
        try:
            filenames = get_panstarrs_filenames(self.ra, self.dec, filters='grz')
            if filenames is None:
                raise urllib.error.URLError('no PS1 filenames found')
            url = (f"{PS1_FITSCUT_URL}?red={filenames['z']}&green={filenames['r']}&blue={filenames['g']}"
                   f"&ra={self.ra}&dec={self.dec}&size={self.size}&output_size=256&autoscale=99.5&format=jpg")
            urllib.request.urlretrieve(url, self.savefile)
            self.successful_download.emit()
        except (urllib.error.URLError, OSError):
            with open(self.savefile,'w') as f:
                Image.fromarray(np.zeros((66,66),dtype=np.uint8)).save(f)
            self.failed_download.emit()
        self.has_finished.emit()
