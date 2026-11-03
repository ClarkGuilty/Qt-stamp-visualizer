# This Python file uses the following encoding: utf-8

import argparse

import numpy as np
import astropy.units as u


import glob
from functools import partial


import pandas as pd
import subprocess

# PySide6 must be imported before matplotlib, so the Qt backend below binds to
# these bindings. #TODO remove: rewrite without matplotlib widgets
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Slot, QThread
from PySide6.QtGui import QKeySequence, QShortcut

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import image as mpimg

import filecmp
import os
from os.path import join
import shutil

import sys
from time import time
import webbrowser

import cli
from dataset import DatasetError, load_dataset
import fits_io
from imaging import (
    SCALES, SINGLE_DISPLAY, scale_limits, rescale_image, composite_image,
    open_rendered_image,
    ClassificationWriter,
    next_free_csv_path, NEW_DATASET_FORK,
    SINGLE_CSV_PREFIX, classification_csv_base, classification_csvs, resume_csv,
)
import paths
import state
from widgets import PanelRowPicker, SettingsMenu, BandNamesLabel, MissingBandsNote
from workers import (
    CacheState, SingleFetchWorker, PanstarrsFetchWorker,
    cache_state, fetch_legacy_survey, fetch_panstarrs,
    legacy_survey_cache_name, panstarrs_cache_name,
    purge_placeholder_cutouts,
)

LEGACY_SURVEY_PIXEL_SIZE=0.262

PANSTARRS_PIXEL_SIZE = 0.25
PS1_CUTOUTS_URL = 'https://ps1images.stsci.edu/cgi-bin/ps1cutouts'

SINGLE_BAND = 'single_band'
MAIN_BAND = 'main_band'
COMPOSITE_BAND = 'composite_band'
EXTERNAL_BAND = 'external_band'
_LEGACY_SURVEY_KEY = "Legacy Survey"
_PANSTARRS_KEY = "PanSTARRS"
DOWNLOADING_LS_MESSAGE = "Downloading legacy survey jpeg."
DOWNLOADING_PS_MESSAGE = "Downloading PanSTARRS jpeg."

SESSION_TOOL = 'single'
# Pre-split config file. Read once, at the first startup after the split, then
# kept as a .bak -- see state.migrate_legacy_config.
LEGACY_CONFIG_FILE = ".config.json"

def parse_args(argv=None):
    """The 1-by-1 viewer's arguments, normalised (see cli.finish_dataset_args), with
    --classifications split into `major_classes` and `subclasses`. `argv` defaults
    to sys.argv."""
    parser = argparse.ArgumentParser(description='configure the parameters of the execution.')
    cli.add_dataset_args(parser)
    parser.add_argument("--verbose", help="activates loging to terminal",
                        action="store_true", default=False)
    parser.add_argument("--clean", help="Empties this workspace's Legacy Survey and PanSTARRS "
                        "cutout caches (.qtstamp/cache/, or under --state-dir) before starting.",
                        action="store_true")
    parser.add_argument("--legacysurvey",
                        help="Enables the Legacy Survey panel and downloads. On by default; "
                        "pass --no-legacysurvey to disable (e.g. if the Legacy Survey server is unreliable).",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--ls-big-fov-residuals",
                        help="Also pre-fetch the large-field-of-view Legacy Survey residual "
                        "cutout, so that turning on both 'Large FoV' and 'Residuals' shows a "
                        "cached image instead of downloading on demand. Off by default: it "
                        "adds a fourth Legacy Survey download per object.",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--classifications',
                        help='Classification buttons: semicolon-separated MAJOR=KEY or MAJOR:SUB=KEY entries. '
                        'A bare MAJOR=KEY (or empty SUB) makes a major-class button; MAJOR:SUB=KEY makes a '
                        'subclass button under that major, setting both fields when clicked. A subclass '
                        'shared by several majors lists them comma-separated, MAJOR1,MAJOR2:SUB=KEY: '
                        'clicking it highlights those majors and dims every other button, and the major '
                        'clicked next sets both fields; clicking the subclass again cancels. '
                        'Example: "A=1;B=2;C=3;X=4;I=5;X:Merger=a;A,B:Spiral=s"',
                        default="A=1;B=2;C=3;X=4;I=5")
    args = parser.parse_args(argv)
    cli.finish_dataset_args(parser, args)

    entries = cli.parse_classifications(args.classifications)
    # major_classes: [(major, key)]; subclasses: [(majors, sub, key)], majors a tuple.
    args.major_classes, args.subclasses = cli.scheme_buttons(entries)
    for key, labels in cli.duplicate_keys(entries).items():
        print(f"Warning: keyboard shortcut '{key}' is assigned to more than one button: {', '.join(labels)}")
    for sub, count in cli.duplicate_subclasses(entries).items():
        print(f"Warning: subclass '{sub}' is declared {count} times -- declare it once "
              f"with its majors comma-separated (e.g. \"A,B:{sub}=KEY\") instead.")
    return args


def survey_cache_paths(state_dir_override=None):
    "(Legacy Survey, PanSTARRS) cutout cache directories for this workspace."
    # Downloaded cutouts live with the workspace (2b-v), under the state dir's
    # cache/. `paths.cache_dir` falls back to the per-user cache dir if the local
    # one is unusable -- unlike settings, a cache has no user intent to respect and
    # no fetch can succeed without somewhere to land. Cutouts already downloaded
    # into the per-user dir by a 2b-iii-era run are not reused here: moving them
    # would strand every other workspace that reads them, so they re-download.
    cache = paths.cache_dir(override=state_dir_override)
    return os.path.join(cache, 'Legacy_survey'), os.path.join(cache, 'PanSTARRS')


def prepare_survey_caches(args):
    """Startup housekeeping on the cutout caches: bring old cutouts over from the
    CWD, then empty the caches (--clean) or clear out the failure placeholders
    older versions left in them."""
    legacy_survey_path, panstarrs_path = survey_cache_paths(args.state_dir_override)
    paths.migrate_once(os.path.join(os.curdir, 'Legacy_survey'), legacy_survey_path)
    paths.migrate_once(os.path.join(os.curdir, 'PanSTARRS'), panstarrs_path)

    if args.clean:
        for f in (glob.glob(join(legacy_survey_path,"*.jpg")) +
                  glob.glob(join(panstarrs_path,"*.jpg")) +
                  glob.glob(join(legacy_survey_path,"*.miss")) +
                  glob.glob(join(panstarrs_path,"*.miss"))):
            if os.path.exists(f):
                os.remove(f)
    else:
        # Older versions wrote a black 66x66 JPEG into the cache whenever a
        # download failed, and every reader then treated that file's existence as a
        # cache hit -- so one outage blacked out those coordinates permanently.
        # Misses are sidecar markers now; clear the placeholders left behind.
        purged = purge_placeholder_cutouts(legacy_survey_path, panstarrs_path)
        if purged:
            print(f"Removed {purged} cached failure placeholder(s); "
                  "those cutouts will be downloaded again when needed.")

def legacy_survey_number_of_pixels(image_pixel_size,
                                    image_dim,
                                    pixels_big_fov_ls=488): #sizes in ARCSECONDS
    n_pixels_in_ls = int(np.ceil(image_pixel_size*image_dim/LEGACY_SURVEY_PIXEL_SIZE))
    if n_pixels_in_ls >= pixels_big_fov_ls:
        pixels_big_fov_ls = 2*n_pixels_in_ls
    return n_pixels_in_ls, pixels_big_fov_ls

# One "Large FoV" toggle drives both survey panels, so PanSTARRS covers the same
# ~2.1' as Legacy Survey's 488 px, and doubles the stamp when that is wider (BUGS 44).
PANSTARRS_BIG_FOV_ARCSEC = 488*LEGACY_SURVEY_PIXEL_SIZE

def panstarrs_number_of_pixels(image_pixel_size, image_dim): #sizes in ARCSECONDS
    n_pixels_in_ps1 = int(np.ceil(image_pixel_size*image_dim/PANSTARRS_PIXEL_SIZE))
    pixels_big_fov_ps1 = int(np.ceil(PANSTARRS_BIG_FOV_ARCSEC/PANSTARRS_PIXEL_SIZE))
    if n_pixels_in_ps1 >= pixels_big_fov_ps1:
        pixels_big_fov_ps1 = 2*n_pixels_in_ps1
    return n_pixels_in_ps1, pixels_big_fov_ps1


class FetchThread(QThread):
    def __init__(self, df, initial_counter, main_band_source, listimage,
                 legacy_survey_path, panstarrs_path,
                 fetch_panstarrs=True, fetch_legacysurvey=True,
                 fetch_big_fov_residuals=False, verbose=False, parent=None):
            QThread.__init__(self, parent)

            self.df = df
            self.initial_counter = initial_counter
            self.legacy_survey_path = legacy_survey_path
            self.panstarrs_path = panstarrs_path
            self.verbose = verbose
            # Passed in rather than rediscovered here, so this thread's view of "the
            # main band's file for object N" can never drift from ApplicationWindow's.
            self.main_band_source = main_band_source
            self.listimage = listimage
            self.fetch_panstarrs = fetch_panstarrs
            self.fetch_legacysurvey = fetch_legacysurvey
            self.fetch_big_fov_residuals = fetch_big_fov_residuals
            # interrupt() can arrive before run() has started (isRunning() is
            # already true by then), so this must exist from construction.
            self._active = True

    def download_legacy_survey(self,ra,dec,size=47,residual=False,pixscale='0.262'):
        savefile = os.path.join(self.legacy_survey_path,
                                legacy_survey_cache_name(ra, dec, size, residual=residual))
        return fetch_legacy_survey(savefile, ra, dec, size, residual=residual,
                                   pixscale=pixscale, verbose=self.verbose)

    def download_panstarrs(self,ra,dec,size=240):
        savefile = os.path.join(self.panstarrs_path,
                                panstarrs_cache_name(ra, dec, size))
        return fetch_panstarrs(savefile, ra, dec, size, verbose=self.verbose)

    def interrupt(self):
        self._active = False

    def fetch_one(self, index):
        "Download every cutout wanted for the object at `index`."
        stamp = self.df.iloc[index]
        if np.isnan(stamp['ra']) or np.isnan(stamp['dec']): #TODO: add smt for when there is no RADec.
            stem = os.path.splitext(self.listimage[index])[0]
            locator = fits_io.resolve(self.main_band_source, stem)
            ra,dec,image_pixel_size,image_dim = fits_io.get_ra_dec(fits_io.read_header(locator))
        else:
            ra,dec,image_pixel_size,image_dim = stamp[['ra','dec','pixel_size','image_dim']]
        if self.fetch_panstarrs:
            n_pixels_ps1, n_pixels_big_ps1 = panstarrs_number_of_pixels(image_pixel_size, image_dim)
            self.download_panstarrs(ra,dec,size=n_pixels_ps1)
            self.download_panstarrs(ra,dec,size=n_pixels_big_ps1)

        if self.fetch_legacysurvey:
            n_pixels_ls, n_pixels_big_ls = legacy_survey_number_of_pixels(image_pixel_size,
                                    image_dim,
                                    pixels_big_fov_ls=488)
            self.download_legacy_survey(ra,dec,size=n_pixels_ls)
            self.download_legacy_survey(ra,dec,size=n_pixels_ls,residual=True)
            self.download_legacy_survey(ra,dec,size=n_pixels_big_ls)
            if self.fetch_big_fov_residuals:
                self.download_legacy_survey(ra,dec,size=n_pixels_big_ls,residual=True)

    def run(self):
        index = self.initial_counter
        self._active = True
        while self._active and index < len(self.df):
            # One unreadable header or malformed survey response must not end
            # the prefetch pass: everything after this index would silently
            # never be prefetched, for the rest of the session.
            try:
                self.fetch_one(index)
            except Exception as E:
                print(f"Skipping pre-fetch for object {index}: {type(E).__name__}: {E}")
            index+=1
        return 0

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, args, clipboard=None):
        super().__init__()
        self.args = args
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.status = self.statusBar()

        self.clipboard = clipboard

        title_strings = ["1-by-1 classifier ERO edition"]
        if args.name is not None:
            self.name = args.name
            title_strings.append(self.name)
        else:
            self.name = ''
        self.setWindowTitle(' - '.join(title_strings))
        
        # Preferences only: they follow the *user*, so they survive a change of
        # --path/--name/--seed. The resume position follows the *dataset* and
        # lives in the session store instead (see state.py).
        self.defaults = {
                    'legacysurvey':False,
                    'legacybigarea':False,
                    'legacyresiduals':False,
                    'panstarrs':False,
                    'prefetch_panstarrs':False,
                    'prefetch_legacysurvey':False,
                    'autonext':True,
                    'colormap':'gist_gray',
                    'scale':'log',
                    'keyboardshortcuts':False,
                    'colorbandsvisible':False,
                    'nisprgbvisible':False,
                    'row_1':'',
                    'row_2':'',
                    'row_3':'',
                        }
        self.ds9_comm_backend = "xpa"
        self.is_ds9_open = False
        self.singlefetchthread_active = False
        self.buttoncolor = "darkRed"
        self.buttonclasscolor = "darkRed"
        self.buttonoptioncolor = "darkOrange"  # majors offered by a pending shared subclass


        self.stampspath = args.path
        self.main_band = args.main_band
        self.color_bands = args.color_bands
        self.legacy_survey_path, self.panstarrs_path = survey_cache_paths(args.state_dir_override)
        self.random_seed = args.seed

        try:
            dataset = load_dataset(self.stampspath, self.main_band, self.color_bands,
                                   args.composite_bands, args.composite_band_members,
                                   args.mef, self.random_seed)
        except DatasetError as e:
            print(e)
            sys.exit(1)
        self.listimage = dataset.listimage
        self.filetype = dataset.filetype
        self.band_filetype = dataset.band_filetype
        self.composite_bands = dataset.composite_bands
        self.composite_band_members = dataset.composite_band_members
        self.all_single_bands = dataset.all_single_bands
        self.band_sources = dataset.band_sources

        self.external_bands = [_LEGACY_SURVEY_KEY] if args.legacysurvey else []
        self.all_bands = [self.main_band,
                          *self.composite_bands,
                          *self.color_bands,
                          *self.external_bands,
                          _PANSTARRS_KEY]
        self.band_types = ({self.main_band: MAIN_BAND} |
                          {band: COMPOSITE_BAND for band in self.composite_bands} |
                          {band: SINGLE_BAND for band in self.color_bands} |
                          {band: EXTERNAL_BAND for band in self.external_bands} |
                          {_PANSTARRS_KEY: EXTERNAL_BAND})

        self.classifications_dir = state.open_classifications_dir(args.classifications_dir)

        self.session_id = state.SessionId(SESSION_TOOL, self.stampspath, self.name, self.random_seed)
        self.df = self.obtain_df()  # sets self.df_name
        # Every autosave goes through this: atomic, and never clobbers a file
        # another process has written since we last saved it.
        self.csv_writer = ClassificationWriter(self.df_name)

        self.config_dict, self.counter = state.load_viewer_state(
            self.session_id, self.defaults, self.listimage, LEGACY_CONFIG_FILE,
            self.legacy_counter, csv=self.df_name,
            classifications_dir=self.classifications_dir, config_dir=args.state_dir_override)
        if not args.legacysurvey:
            self.config_dict['legacysurvey'] = False
        self.colormap = self.config_dict['colormap']

        self.number_graded = 0
        self.COUNTER_MIN = 0
        self.COUNTER_MAX = len(self.listimage)
        self.filename = self.listimage[self.counter]

        main_layout = QtWidgets.QVBoxLayout(self._main)
        self.label_layout = QtWidgets.QHBoxLayout()
        self.plot_layout_area = QtWidgets.QVBoxLayout()
        self.plot_layout_0_Widget = QtWidgets.QWidget()
        self.plot_layout_0 = QtWidgets.QHBoxLayout(self.plot_layout_0_Widget)
        self.plot_layout_1_Widget = QtWidgets.QWidget()
        self.plot_layout_1 = QtWidgets.QHBoxLayout(self.plot_layout_1_Widget)
        self.plot_layout_2_Widget = QtWidgets.QWidget()
        self.plot_layout_2 = QtWidgets.QHBoxLayout(self.plot_layout_2_Widget)
        self.panel_row_layout = {'row_1': self.plot_layout_0, 'row_2': self.plot_layout_1, 'row_3': self.plot_layout_2}
        button_layout = QtWidgets.QVBoxLayout()
        button_layout.setSpacing(0)
        button_layout.setContentsMargins(0,0,0,0)
        button_row0_layout = QtWidgets.QHBoxLayout()
        button_row10_layout = QtWidgets.QHBoxLayout()
        button_row11_layout = QtWidgets.QHBoxLayout()
        for row_layout in (button_row0_layout, button_row10_layout, button_row11_layout):
            row_layout.setSpacing(10)
            row_layout.setContentsMargins(0,0,0,0)

        self.counter_widget = QtWidgets.QLabel("{}/{}".format(self.counter+1,self.COUNTER_MAX))
        self.counter_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed) #QLabels have different default size policy. Better to use the policy of buttons.
        self.counter_widget.setStyleSheet("font-size: 14px")
        
        self.band2bandname_dict = {band: band for band in self.all_bands}

        self.label_plot = {band: QtWidgets.QLabel(f"{band}", alignment=Qt.AlignCenter) for band in [self.main_band]}
        self.rows_summary_label = BandNamesLabel(alignment=Qt.AlignCenter)
        font = self.label_plot[self.main_band].font()
        font.setPointSize(16)
        self.label_plot[self.main_band].setFont(font)
        summary_font = self.rows_summary_label.font()
        summary_font.setPointSize(16)
        self.rows_summary_label.setFont(summary_font)

        self.label_layout.addWidget(self.label_plot[self.main_band])
        self.label_layout.addWidget(self.rows_summary_label)


        self.plot_layout_area.setSpacing(0)
        self.plot_layout_area.setContentsMargins(0,0,0,0)
        for row_layout in self.panel_row_layout.values():
            row_layout.setSpacing(0)
            row_layout.setContentsMargins(0,0,0,0)


        self.figure = {band: Figure(figsize=(5,3),layout="constrained",facecolor='black') for band in self.all_bands}
        self.canvas = {band: FigureCanvas(self.figure[band]) for band in self.all_bands}
        
        self.panel_keys = [self.main_band, *self.composite_bands, *self.color_bands,
                          *self.external_bands, _PANSTARRS_KEY]
        for band in self.panel_keys:
            self.canvas[band].setStyleSheet('background-color: black')

        default_row_1 = [self.main_band] + self.composite_bands[:1]
        row_panels_by_key = {row_key: [b for b in self.config_dict[row_key].split(';') if b in self.panel_keys]
                             for row_key in self.panel_row_layout}
        # The layout is saved per tool, not per dataset: one naming only bands this dataset
        # lacks (VIS;H,Y,I in a PNG session), or only the survey panels, would open with no
        # stamp in view. The default row is put back in front whenever no band is shown.
        image_panels = {self.main_band, *self.composite_bands, *self.color_bands}
        if not any(b in image_panels for row in row_panels_by_key.values() for b in row):
            row_panels_by_key['row_1'] = default_row_1 + row_panels_by_key['row_1']
        visible_panels = set()
        for row_key, row_layout in self.panel_row_layout.items():
            row_panels = row_panels_by_key[row_key]
            for band in row_panels:
                row_layout.addWidget(self.canvas[band],1)
            self.config_dict[row_key] = ';'.join(row_panels)
            visible_panels.update(row_panels)

        for band in self.panel_keys: #Panels not assigned to any row start parked (hidden) in row 1.
            if band not in visible_panels:
                self.plot_layout_0.addWidget(self.canvas[band],1)
                self.canvas[band].hide()

        self.config_dict['colorbandsvisible'] = any(band in visible_panels for band in self.color_bands)
        self.config_dict['nisprgbvisible'] = bool(self.composite_bands) and self.composite_bands[-1] in visible_panels

        self.ax = {band: self.figure[band].subplots() for band in self.all_bands}
        self.images = {}
        self.scale_mins = {}
        self.scale_maxs = {}
        # {band: MissingBandError} for the object on screen (BUGS 36), and the note that
        # names them where the page messages go.
        self.missing_bands = {}
        self.missing_bands_note = MissingBandsNote()
        self.status.addWidget(self.missing_bands_note)

        self.bottom_row_bands_already_plotted = False
        self.plot()


        list_button_row0_layout=[]

        self.bgoto = QtWidgets.QPushButton('Go to')
        self.bgoto.clicked.connect(self.goto)
        list_button_row0_layout.append(self.bgoto)

        self.bprev = QtWidgets.QPushButton('Prev')
        self.bprev.clicked.connect(self.prev)
        list_button_row0_layout.append(self.bprev)

        self.bnext = QtWidgets.QPushButton('Next')
        self.bnext.clicked.connect(self.next)
        list_button_row0_layout.append(self.bnext)

        self.tools_button = QtWidgets.QToolButton()
        self.tools_button.setText("Tools")
        self.tools_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        tools_menu = QtWidgets.QMenu(self.tools_button)
        tools_menu.addAction("Open ds9", self.open_ds9)
        tools_menu.addAction("Open LS", self.viewls)
        tools_menu.addAction("Open PanSTARRS", self.viewPanSTARRS)
        tools_menu.addAction("Open ESASky", self.viewESASky)
        if self.filetype != 'FITS': #These all need FITS pixel data and/or RA/Dec from WCS.
            for action in tools_menu.actions():
                action.setEnabled(False)
        self.tools_button.setMenu(tools_menu)
        list_button_row0_layout.append(self.tools_button)

        self.panel_picker = PanelRowPicker(self.panel_keys, self.band2bandname_dict, n_rows=3)
        for row_key in self.panel_row_layout:
            self.panel_picker.set_row_panels(row_key, [b for b in self.config_dict[row_key].split(';') if b])
        self.panel_picker.rowChanged.connect(self.on_panel_row_changed)
        list_button_row0_layout.append(self.panel_picker)

        #PanSTARRS and Legacy Survey are both self.panel_keys now, so their visibility/placement
        #is already handled by the row-assignment above -- just fetch each if it started out visible.
        self.config_dict['panstarrs'] = _PANSTARRS_KEY in visible_panels
        if self.config_dict['panstarrs']:
            self.set_panstarrs()

        self.config_dict['legacysurvey'] = _LEGACY_SURVEY_KEY in visible_panels
        if self.config_dict['legacysurvey']:
            self.set_legacy_survey()

        #Beside Residuals, but not under args.legacysurvey: it sizes the PanSTARRS panel too.
        self.blsarea = QtWidgets.QCheckBox("Large FoV")
        self.blsarea.setChecked(self.config_dict['legacybigarea'])
        self.blsarea.clicked.connect(self.checkbox_ls_change_area)
        list_button_row0_layout.append(self.blsarea)

        if args.legacysurvey:
            self.blsresidual = QtWidgets.QCheckBox("Residuals")
            self.blsresidual.clicked.connect(self.checkbox_ls_use_residuals)
            if self.filetype == 'FITS':
                if self.config_dict['legacyresiduals']:
                    self.blsresidual.toggle()
                    if self.config_dict['legacysurvey']:
                        self.set_legacy_survey()
            else:
                self.blsresidual.setEnabled(False)
                self.config_dict['legacyresiduals'] = False
            list_button_row0_layout.append(self.blsresidual)

        self.settings_menu = SettingsMenu("Settings")
        self.bprefetch_ps = self.settings_menu.add_toggle("Pre-fetch PanSTARRS")
        self.bprefetch_ps.clicked.connect(self.toggle_prefetch_panstarrs)
        if self.filetype != 'FITS':
            self.bprefetch_ps.setEnabled(False)
            self.config_dict['prefetch_panstarrs'] = False
        elif self.config_dict['prefetch_panstarrs']:
            self.config_dict['prefetch_panstarrs'] = False
            self.toggle_prefetch_panstarrs()
            self.bprefetch_ps.setChecked(True)

        if args.legacysurvey:
            self.bprefetch_ls = self.settings_menu.add_toggle("Pre-fetch Legacy Survey")
            self.bprefetch_ls.clicked.connect(self.toggle_prefetch_legacysurvey)
            if self.filetype != 'FITS':
                self.bprefetch_ls.setEnabled(False)
                self.config_dict['prefetch_legacysurvey'] = False
            elif self.config_dict['prefetch_legacysurvey']:
                self.config_dict['prefetch_legacysurvey'] = False
                self.toggle_prefetch_legacysurvey()
                self.bprefetch_ls.setChecked(True)
        else:
            self.config_dict['prefetch_legacysurvey'] = False

        self.bautopass = self.settings_menu.add_toggle("Auto-next", checked=self.config_dict['autonext'])
        self.bautopass.clicked.connect(self.checkbox_auto_next)

        self.bkeyboardshortcuts = self.settings_menu.add_toggle("Keyboard shortcuts", checked=self.config_dict['keyboardshortcuts'])
        self.bkeyboardshortcuts.clicked.connect(self.checkbox_keyboard_shortcuts)
        list_button_row0_layout.append(self.settings_menu)

        list_classifications = []
        self.dict_class2button = {'None': None}
        self.original_button_style = None
        for major, key in args.major_classes:
            button = QtWidgets.QPushButton(major)
            button.clicked.connect(partial(self.classify_major, major))
            list_classifications.append(button)
            self.dict_class2button[major] = button
            if self.original_button_style is None:
                self.original_button_style = button.styleSheet()

        list_subclassifications = []
        self.dict_subclass2button = {'None': None}
        for majors, sub, key in args.subclasses:
            button = QtWidgets.QPushButton(sub)
            button.clicked.connect(partial(self.classify_subclass, majors, sub))
            list_subclassifications.append(button)
            self.dict_subclass2button[sub] = button
            if self.original_button_style is None:
                self.original_button_style = button.styleSheet()

        for major, _ in args.major_classes:
            self.dict_subclass2button.setdefault(major, None)

        self.scale_options = {'Linear':'linear', 'Sqrt':'sqrt', 'Cbrt':'cbrt', 'Log':'log', 'Asinh':'asinh'}
        scale2display = {v: k for k, v in self.scale_options.items()}
        self.cbscale = QtWidgets.QComboBox()
        self.cbscale.addItems(self.scale_options.keys())
        self.cbscale.setCurrentText(scale2display.get(self.config_dict['scale'], 'Log'))
        self.cbscale.currentTextChanged.connect(self.change_scale)
        list_button_row0_layout.append(self.cbscale)

        self.colormap_options = {'Yarg':'gist_yarg', 'Hot':'hot', 'Gray':'gist_gray', 'Viridis':'viridis'}
        colormap2display = {v: k for k, v in self.colormap_options.items()}
        self.cbcolormap = QtWidgets.QComboBox()
        self.cbcolormap.addItems(self.colormap_options.keys())
        self.cbcolormap.setCurrentText(colormap2display.get(self.config_dict['colormap'], 'Gray'))
        self.cbcolormap.currentTextChanged.connect(self.change_colormap)
        list_button_row0_layout.append(self.cbcolormap)

        self.bactivatedclassification = None
        self.bactivatedsubclassification = None
        self.pending_subclass = None  # (majors, sub) of a shared subclass awaiting its major

        self._report_unknown_classifications()

        grade = self.df.at[self.counter,'classification']
        self.bactivatedclassification = self._button_for_grade(grade, self.dict_class2button)
        if self.bactivatedclassification is not None:
            self.bactivatedclassification.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))

        subgrade = self.df.at[self.counter,'subclassification']
        self.bactivatedsubclassification = self._button_for_grade(subgrade, self.dict_subclass2button)
        if self.bactivatedsubclassification is not None:
            self.bactivatedsubclassification.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))

        #Keyboard shortcuts
        self.classification_shortcuts = []
        for major, key in args.major_classes:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(partial(self.keyClassify, major))
            self.classification_shortcuts.append(shortcut)

        for majors, sub, key in args.subclasses:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(partial(self.keyClassifySubclass, majors, sub))
            self.classification_shortcuts.append(shortcut)


        self.kNext = QShortcut(QKeySequence(QKeySequence.MoveToPreviousPage), self)
        self.kNext.activated.connect(self.keyPrev)

        self.kNext = QShortcut(QKeySequence(QKeySequence.MoveToNextPage), self)
        self.kNext.activated.connect(self.keyNext)

        self.kNext = QShortcut(QKeySequence('k'), self)
        self.kNext.activated.connect(self.keyPrev)

        self.kNext = QShortcut(QKeySequence('j'), self)
        self.kNext.activated.connect(self.keyNext)


        self.kCopyRADec = QShortcut(QKeySequence(QKeySequence.Copy), self)
        self.kCopyRADec.activated.connect(self.copy_RADec_to_keyboard)

        self.kCopyRADec = QShortcut(QKeySequence('c'), self)
        self.kCopyRADec.activated.connect(self.copy_filename_to_keyboard)

        for button in list_button_row0_layout:
            button_row0_layout.addWidget(button)

        button_row0_layout.addWidget(self.counter_widget,alignment=Qt.AlignRight)

        for button in list_classifications:
            button_row10_layout.addWidget(button)

        for button in list_subclassifications:
            button_row11_layout.addWidget(button)

        button_layout_spacing = 0
        button_layout.addLayout(button_row0_layout, button_layout_spacing)
        button_layout.addLayout(button_row10_layout, button_layout_spacing)
        button_layout.addLayout(button_row11_layout, button_layout_spacing)


        self.plot_layout_area.addWidget(self.plot_layout_0_Widget,1)
        self.plot_layout_area.addWidget(self.plot_layout_1_Widget,1)
        self.plot_layout_area.addWidget(self.plot_layout_2_Widget,1)
        self.update_row_visibility()

        main_layout.addLayout(self.label_layout, 2)
        main_layout.addLayout(self.plot_layout_area, 88)
        main_layout.addLayout(button_layout, 10)

        self.timer_0 = time()

    def stop_prefetch_thread(self, thread_attr):
        """Interrupt a prefetch pass if it is still running.

        `finished` is wired to deleteLater, so a pass that ran to completion
        leaves the Python attribute pointing at a destroyed C++ object; calling
        into it raises RuntimeError instead of doing nothing."""
        thread = getattr(self, thread_attr, None)
        if thread is None:
            return
        try:
            thread.interrupt()
        except RuntimeError:
            pass

    def on_prefetch_finished(self, config_key, button):
        "Keep the toggle's state honest once a prefetch pass ends by itself."
        self.config_dict[config_key] = False
        try:
            button.setChecked(False)
        except RuntimeError:
            pass

    @Slot()
    def toggle_prefetch_panstarrs(self):
        if self.filetype != 'FITS':
            self.status.showMessage("Pre-fetching PanSTARRS requires FITS input.",5000)
            return
        if self.config_dict['prefetch_panstarrs']:
            self.stop_prefetch_thread('fetchthread_ps')
            self.config_dict['prefetch_panstarrs'] = False
        else:
            self.fetchthread_ps = FetchThread(self.df,self.counter,
                                        self.band_sources[self.main_band], self.listimage,
                                        self.legacy_survey_path, self.panstarrs_path,
                                        fetch_panstarrs=True, fetch_legacysurvey=False,
                                        verbose=self.args.verbose) #Always store in an object.
            self.fetchthread_ps.finished.connect(self.fetchthread_ps.deleteLater)
            # A finished pass leaves the config flag claiming a thread is still
            # running; the next toggle would then try to interrupt a destroyed
            # object instead of starting a new pass.
            self.fetchthread_ps.finished.connect(
                partial(self.on_prefetch_finished, 'prefetch_panstarrs', self.bprefetch_ps))
            self.fetchthread_ps.setTerminationEnabled(True)
            self.fetchthread_ps.start()
            self.config_dict['prefetch_panstarrs'] = True

    @Slot()
    def toggle_prefetch_legacysurvey(self):
        if self.filetype != 'FITS':
            self.status.showMessage("Pre-fetching Legacy Survey requires FITS input.",5000)
            return
        if self.config_dict['prefetch_legacysurvey']:
            self.stop_prefetch_thread('fetchthread_ls')
            self.config_dict['prefetch_legacysurvey'] = False
        else:
            self.fetchthread_ls = FetchThread(self.df,self.counter,
                                        self.band_sources[self.main_band], self.listimage,
                                        self.legacy_survey_path, self.panstarrs_path,
                                        fetch_panstarrs=False, fetch_legacysurvey=True,
                                        fetch_big_fov_residuals=self.args.ls_big_fov_residuals,
                                        verbose=self.args.verbose) #Always store in an object.
            self.fetchthread_ls.finished.connect(self.fetchthread_ls.deleteLater)
            self.fetchthread_ls.finished.connect(
                partial(self.on_prefetch_finished, 'prefetch_legacysurvey', self.bprefetch_ls))
            self.fetchthread_ls.setTerminationEnabled(True)
            self.fetchthread_ls.start()
            self.config_dict['prefetch_legacysurvey'] = True

    def closeEvent(self, event):
        # Each of these threads is wired to `finished.connect(thread.deleteLater)`,
        # so once a thread finishes on its own, Qt destroys the underlying C++
        # object on the next event-loop turn -- but the Python attribute is left
        # pointing at that now-dead wrapper. Calling isRunning()/wait() on it then
        # raises "Internal C++ object already deleted", so every check here must
        # tolerate that instead of crashing.
        for thread_attr in ('fetchthread_ps', 'fetchthread_ls'):
            thread = getattr(self, thread_attr, None)
            if thread is None:
                continue
            try:
                if thread.isRunning():
                    thread.interrupt()
                    thread.wait(5000)
            except RuntimeError:
                pass
        # workerThread/workerThreadPS run a single one-shot network call (Legacy
        # Survey / PanSTARRS lookup+download) with no loop to cooperatively
        # interrupt -- just wait for them so Qt doesn't destroy a still-running
        # QThread (which aborts the process with "QThread: Destroyed while
        # thread is still running").
        for thread_attr in ('workerThread', 'workerThreadPS'):
            thread = getattr(self, thread_attr, None)
            if thread is None:
                continue
            try:
                if thread.isRunning():
                    thread.wait(5000)
            except RuntimeError:
                pass
        event.accept()

    def save_preferences(self):
        state.save_preferences(SESSION_TOOL, self.config_dict,
                               config_dir=self.args.state_dir_override)

    def save_position(self):
        "Persist the resume position as a filename, so it survives a reshuffle-free relist."
        state.save_session(self.session_id, self.classifications_dir,
                           position_filename=self.listimage[self.counter],
                           csv=self.df_name)

    def legacy_counter(self, legacy):
        """The index the pre-split `.config.json` was pointing at, or None to drop it.

        A `--name` comparison is the only guard the old file ever had, so it is
        also the most that can honestly be checked when importing it.
        """
        if (legacy.get('name') or '') != self.name:
            return None
        counter = legacy.get('counter')
        return counter if isinstance(counter, int) and not isinstance(counter, bool) else None

    def update_counter(self):
        self.counter_widget.setText("{}/{}".format(self.counter+1,self.COUNTER_MAX))

    @Slot()
    def keyClassify(self, major):
        if self.config_dict['keyboardshortcuts'] == True:
            self.classify_major(major)

    @Slot()
    def keyClassifySubclass(self, majors, subgrade):
        if self.config_dict['keyboardshortcuts'] == True:
            self.classify_subclass(majors, subgrade)
        
    @Slot()
    def keyNext(self):
        if self.config_dict['keyboardshortcuts'] == True:
            self.next()

    @Slot()
    def keyPrev(self):
        if self.config_dict['keyboardshortcuts'] == True:
            self.prev()

    @Slot()
    def copy_RADec_to_keyboard(self):
        if self.filetype != 'FITS':
            self.status.showMessage("RA,Dec is only available for FITS input.",5000)
            return
        if self._main_band_missing("Copying RA,Dec"):
            return
        to_copy = f"{self.ra},{self.dec}"
        self.clipboard.setText(to_copy)
        self.status.showMessage(f'RA,Dec copied to clipboard: {self.ra},{self.dec}',10000)


    @Slot()
    def copy_filename_to_keyboard(self):
        to_copy = f"{self.filename}"
        self.clipboard.setText(to_copy)
        self.status.showMessage(f'Filename copied to clipboard: {self.filename}',10000)

    @Slot()
    def classify(self, grade, subgrade):
        # The df/listimage disagreement this used to assert on was the saved
        # counter pointing into a dataset it never belonged to. obtain_df only
        # returns a df whose file_name column matches listimage exactly, and the
        # position is now resolved by filename against that same list, so the
        # two cannot drift apart any more.
        self._clear_pending_subclass()
        cnt = self.counter
        self.df.at[cnt,'classification'] = grade
        self.df.at[cnt,'subclassification'] = subgrade
        if self.filetype == 'FITS':
            self.df.at[cnt,'ra'] = self.ra
            self.df.at[cnt,'dec'] = self.dec
            self.df.at[cnt,'pixel_size'] = self.image_pixel_size #Arcsec/pixel comes from the WCS, so FITS only.
        #The image itself is loaded for every filetype, so its size is always known.
        #PNG/JPG arrays carry a trailing channel axis -- only the first two axes are the stamp.
        main_image = self.images.get(self.main_band)
        self.df.at[cnt,'image_dim'] = np.max(main_image.shape[:2]) if main_image is not None else pd.NA
        self.df.at[cnt,'time'] += (time() - self.timer_0)
        self.timer_0 = time()
        self.df_name = self.csv_writer.save(self.df)

        self.update_classification_buttoms()
        self.update_subclassification_buttoms()

        if self.config_dict['autonext']:
            self.next()

    @Slot()
    def classify_major(self, major):
        """Major button: that is the class -- unless a shared subclass is waiting
        for one of its majors, in which case this one completes it, keeping the
        subclass. Other majors are dimmed while one waits; their shortcuts, which
        a disabled button does not stop, are ignored the same way."""
        pending = self.pending_subclass
        if pending is None:
            self.classify(major, major)
        elif major in pending[0]:
            self.classify(major, pending[1])

    @Slot()
    def classify_subclass(self, majors, subgrade):
        """Subclass button. With one major it sets both fields at once. Shared by
        several, nothing is written yet: its majors are highlighted, every other
        classification button is dimmed, and the next major clicked
        (classify_major) supplies the class. Clicking it again cancels."""
        pending = self.pending_subclass
        if pending is not None:
            if pending[1] == subgrade:
                self._clear_pending_subclass()
            return  # another subclass is dimmed while one waits
        if len(majors) == 1:
            self.classify(majors[0], subgrade)
            return
        self.pending_subclass = (majors, subgrade)
        option_style = "background-color : {};color : white;".format(self.buttonoptioncolor)
        lit = {self.dict_subclass2button.get(subgrade)} | \
              {self.dict_class2button.get(m) for m in majors}
        for button in self._classification_buttons():
            if button in lit:
                button.setStyleSheet(option_style)
            else:
                button.setEnabled(False)
        self.status.showMessage(f"'{subgrade}': pick its class -- {' or '.join(majors)} "
                                f"(click '{subgrade}' again to cancel).")

    def _classification_buttons(self):
        "Every major and subclass button, once each."
        buttons = list(self.dict_class2button.values()) + list(self.dict_subclass2button.values())
        return [b for b in dict.fromkeys(buttons) if b is not None]

    def _clear_pending_subclass(self):
        "Drops a shared subclass still waiting for its major, and its highlights."
        if self.pending_subclass is None:
            return
        self.pending_subclass = None
        for button in self._classification_buttons():
            button.setEnabled(True)
            button.setStyleSheet(self.original_button_style)
        self.status.clearMessage()
        # The highlights above may have covered the row's own classification.
        self.update_classification_buttoms()
        self.update_subclassification_buttoms()
        

    def legacy_survey_cache_lookup(self,ra,dec,residual=False,size=47):
        "Cache path for this cutout plus its HIT/MISS/RETRY state."
        savename = legacy_survey_cache_name(ra, dec, size, residual=residual)
        savefile = os.path.join(self.legacy_survey_path, savename)
        print(f"Quering for {savename} ")
        return savefile, cache_state(savefile)

    def generate_title(self, size_in_sky, residuals=False, bigarea=False):
        units = 'arcmin' if bigarea else 'arcsec'
        if residuals:
            return "Residuals, {0:.2f} x {0:.2f}".format(size_in_sky.to(units))
        if bigarea:
            return '{0:.2f} x {0:.2f}'.format(size_in_sky.to(units))
        return "{0:.2f} x {0:.2f}".format(size_in_sky.to(units))

    def plot_legacy_survey(self, savefile, title):
        # Guard before clearing: a worker for a previously-shown object can
        # finish after the user has moved on, and must leave this panel alone.
        if savefile != self.legacy_filename:
            return
        self._clear_status_message(DOWNLOADING_LS_MESSAGE)
        self.ax[_LEGACY_SURVEY_KEY].cla()
        try:
            image = mpimg.imread(savefile)
        except (OSError, ValueError, SyntaxError) as E:
            print("Could not read the Legacy Survey cutout:", E)
            self.plot_no_legacy_survey(title='No Legacy Survey data available',
                                       colormap='viridis')
            return
        self.ax[_LEGACY_SURVEY_KEY].imshow(image)
        self.ax[_LEGACY_SURVEY_KEY].set_title(title, color='white', fontsize=10)
        self.ax[_LEGACY_SURVEY_KEY].set_axis_off()
        self.canvas[_LEGACY_SURVEY_KEY].draw()

    def _blank_survey_panel(self):
        "Zeros the shape of the main band's stamp; a square when this object has no main band."
        image = self.images.get(self.main_band)
        return np.zeros(image.shape if image is not None else (64, 64))

    def _clear_status_message(self, message):
        """Ends `message` if it is still the one shown -- not whatever replaced it. The
        download messages have no timeout and would otherwise cover the missing-bands
        note for the rest of the session."""
        if self.status.currentMessage() == message:
            self.status.clearMessage()

    def plot_no_legacy_survey(self, title='Waiting for data',
                            colormap='Greys_r', savefile=None):
        # Same staleness guard as above: a late failure from the previous
        # object used to wipe out the current object's perfectly good panel.
        if savefile is not None and savefile != self.legacy_filename:
            return
        self._clear_status_message(DOWNLOADING_LS_MESSAGE)
        self.ax[_LEGACY_SURVEY_KEY].cla()
        self.ax[_LEGACY_SURVEY_KEY].imshow(self._blank_survey_panel(), cmap=colormap)
        self.ax[_LEGACY_SURVEY_KEY].set_title(title, color='white', fontsize=10)
        self.ax[_LEGACY_SURVEY_KEY].set_axis_off()
        self.canvas[_LEGACY_SURVEY_KEY].draw()

    @Slot()
    def set_legacy_survey(self):
        if self.filetype != 'FITS':
            self.status.showMessage("Legacy Survey requires FITS input (RA/Dec from WCS).",5000)
            # Drawn anyway: an axes nothing was drawn on shows up as a white box.
            self.plot_no_legacy_survey(title='Needs FITS input\n(RA/Dec from WCS)')
            return
        if self.main_band in self.missing_bands:
            # No stale cutout from the previous object may land here either.
            self.legacy_filename = None
            self.plot_no_legacy_survey(title=f'No RA/Dec:\nno {self.main_band}')
            return
        pixscale = str(LEGACY_SURVEY_PIXEL_SIZE)
        n_pixels_in_ls, pixels_big_fov_ls = legacy_survey_number_of_pixels(self.image_pixel_size, 
                                    np.max(self.images[self.main_band].shape),
                                    pixels_big_fov_ls=488)

        size = pixels_big_fov_ls if self.config_dict['legacybigarea'] else n_pixels_in_ls
        size_in_sky = (LEGACY_SURVEY_PIXEL_SIZE * u.arcsec) * size
        try:
            savefile, state = self.legacy_survey_cache_lookup(self.ra,self.dec,
                                        residual=self.config_dict['legacyresiduals'],
                                        size=size)

            title = self.generate_title(
                                        size_in_sky = size_in_sky, 
                                        residuals=self.config_dict['legacyresiduals'],
                                        bigarea=self.config_dict['legacybigarea'])
            self.legacy_filename = savefile
            if state is CacheState.HIT:
                self.plot_legacy_survey(savefile, title)
                return
            if state is CacheState.MISS:
                # Known gap in coverage, recorded recently enough to trust.
                # Say so at once instead of spawning a thread to re-confirm it.
                self.plot_no_legacy_survey(title='No Legacy Survey data available',
                                           colormap='viridis', savefile=savefile)
                return
            self.plot_no_legacy_survey(savefile=savefile)
            self.status.showMessage(DOWNLOADING_LS_MESSAGE)
            self.workerThread = QThread(parent=self)
            self.singleFetchWorker = SingleFetchWorker(savefile, self.ra, self.dec, size,
                            residual=self.config_dict['legacyresiduals'],
                            pixscale=pixscale, verbose=self.args.verbose)
            self.workerThread.finished.connect(self.singleFetchWorker.deleteLater)
            self.workerThread.started.connect(self.singleFetchWorker.run)
        
            self.singleFetchWorker.moveToThread(self.workerThread)

            self.singleFetchWorker.successful_download.connect(partial(self.plot_legacy_survey, savefile, title))
            self.singleFetchWorker.failed_download.connect(partial(self.plot_no_legacy_survey,title='No Legacy Survey data available',
                            colormap='viridis', savefile=savefile))
            self.workerThread.finished.connect(self.workerThread.deleteLater)
            self.workerThread.setTerminationEnabled(True)

            self.workerThread.start()
            self.workerThread.quit()
        
        except FileNotFoundError:
            self.plot_no_legacy_survey()
        except Exception as E:
            print("Exception while setting up the Legacy Survey image:")
            print(E.args)
            print(type(E))

    def panstarrs_cache_lookup(self,ra,dec,size=240):
        "Cache path for this cutout plus its HIT/MISS/RETRY state."
        savename = panstarrs_cache_name(ra, dec, size)
        savefile = os.path.join(self.panstarrs_path, savename)
        print(f"Quering for {savename} ")
        return savefile, cache_state(savefile)

    def plot_panstarrs(self, savefile, title):
        # Guard before clearing: a worker for a previously-shown object can
        # finish after the user has moved on, and must leave this panel alone.
        if savefile != self.panstarrs_filename:
            return
        self._clear_status_message(DOWNLOADING_PS_MESSAGE)
        self.ax[_PANSTARRS_KEY].cla()
        try:
            image = mpimg.imread(savefile)
        except (OSError, ValueError, SyntaxError) as E:
            print("Could not read the PanSTARRS cutout:", E)
            self.plot_no_panstarrs(title='No PanSTARRS data available',
                                   colormap='viridis')
            return
        self.ax[_PANSTARRS_KEY].imshow(image)
        self.ax[_PANSTARRS_KEY].set_title(title, color='white', fontsize=10)
        self.ax[_PANSTARRS_KEY].set_axis_off()
        self.canvas[_PANSTARRS_KEY].draw()

    def plot_no_panstarrs(self, title='Waiting for data',
                            colormap='Greys_r', savefile=None):
        # Same staleness guard as above: a late failure from the previous
        # object used to wipe out the current object's perfectly good panel.
        if savefile is not None and savefile != self.panstarrs_filename:
            return
        self._clear_status_message(DOWNLOADING_PS_MESSAGE)
        self.ax[_PANSTARRS_KEY].cla()
        self.ax[_PANSTARRS_KEY].imshow(self._blank_survey_panel(), cmap=colormap)
        self.ax[_PANSTARRS_KEY].set_title(title, color='white', fontsize=10)
        self.ax[_PANSTARRS_KEY].set_axis_off()
        self.canvas[_PANSTARRS_KEY].draw()

    @Slot()
    def set_panstarrs(self):
        if self.filetype != 'FITS':
            self.status.showMessage("PanSTARRS requires FITS input (RA/Dec from WCS).",5000)
            # Drawn anyway: an axes nothing was drawn on shows up as a white box.
            self.plot_no_panstarrs(title='Needs FITS input\n(RA/Dec from WCS)')
            return
        if self.main_band in self.missing_bands:
            # No stale cutout from the previous object may land here either.
            self.panstarrs_filename = None
            self.plot_no_panstarrs(title=f'No RA/Dec:\nno {self.main_band}')
            return
        n_pixels_in_ps1, pixels_big_fov_ps1 = panstarrs_number_of_pixels(self.image_pixel_size,
                                    np.max(self.images[self.main_band].shape))

        size = pixels_big_fov_ps1 if self.config_dict['legacybigarea'] else n_pixels_in_ps1
        size_in_sky = (PANSTARRS_PIXEL_SIZE * u.arcsec) * size
        try:
            savefile, state = self.panstarrs_cache_lookup(self.ra,self.dec,size=size)

            title = self.generate_title(
                                        size_in_sky = size_in_sky,
                                        bigarea=self.config_dict['legacybigarea'])
            self.panstarrs_filename = savefile
            if state is CacheState.HIT:
                self.plot_panstarrs(savefile, title)
                return
            if state is CacheState.MISS:
                # Known gap in coverage, recorded recently enough to trust.
                # Say so at once instead of spawning a thread to re-confirm it.
                self.plot_no_panstarrs(title='No PanSTARRS data available',
                                       colormap='viridis', savefile=savefile)
                return
            self.plot_no_panstarrs(savefile=savefile)
            self.status.showMessage(DOWNLOADING_PS_MESSAGE)
            self.workerThreadPS = QThread(parent=self)
            self.singleFetchWorkerPS = PanstarrsFetchWorker(self.ra, self.dec, savefile, size,
                                                            verbose=self.args.verbose)
            self.workerThreadPS.finished.connect(self.singleFetchWorkerPS.deleteLater)
            self.workerThreadPS.started.connect(self.singleFetchWorkerPS.run)

            self.singleFetchWorkerPS.moveToThread(self.workerThreadPS)

            self.singleFetchWorkerPS.successful_download.connect(partial(self.plot_panstarrs, savefile, title))
            self.singleFetchWorkerPS.failed_download.connect(partial(self.plot_no_panstarrs,title='No PanSTARRS data available',
                            colormap='viridis', savefile=savefile))
            self.workerThreadPS.finished.connect(self.workerThreadPS.deleteLater)
            self.workerThreadPS.setTerminationEnabled(True)

            self.workerThreadPS.start()
            self.workerThreadPS.quit()

        except FileNotFoundError:
            self.plot_no_panstarrs()
        except Exception as E:
            print("Exception while setting up the PanSTARRS image:")
            print(E.args)
            print(type(E))


    def update_row_visibility(self):
        "Row 2/3 show themselves automatically whenever a panel is checked into them."
        self.plot_layout_1_Widget.setVisible(bool(self.config_dict['row_2']))
        self.plot_layout_2_Widget.setVisible(bool(self.config_dict['row_3']))

    def visible_rows_summary(self):
        "List-of-lists of friendly panel names, one list per currently-shown row, empty rows dropped."
        rows = [[self.band2bandname_dict[b] for b in self.config_dict[row_key].split(';') if b]
                for row_key in self.panel_row_layout]
        return [row for row in rows if row]

    def _row_band_order(self, row_key):
        "Checked bands for the row, in their actual left-to-right widget order (not the fixed panel_keys order)."
        visible = set(self.panel_picker.row_panels(row_key))
        canvas_to_band = {canvas: band for band, canvas in self.canvas.items()}
        row_layout = self.panel_row_layout[row_key]
        ordered = [canvas_to_band[row_layout.itemAt(i).widget()] for i in range(row_layout.count())]
        return [band for band in ordered if band in visible]

    @Slot()
    def on_panel_row_changed(self, row_key, panel_key, checked):
        canvas = self.canvas[panel_key]
        for row_layout in self.panel_row_layout.values():
            row_layout.removeWidget(canvas)
        if checked:
            self.panel_row_layout[row_key].addWidget(canvas,1)
            canvas.show()
        else:
            canvas.hide()
            self.plot_layout_0.addWidget(canvas,1) #park hidden panels in row 1

        for rk in self.panel_row_layout:
            self.config_dict[rk] = ';'.join(self._row_band_order(rk))
        visible_panels = set(self.config_dict['row_1'].split(';') +
                             self.config_dict['row_2'].split(';') +
                             self.config_dict['row_3'].split(';'))

        was_colorbandsvisible = self.config_dict['colorbandsvisible']
        was_panstarrs = self.config_dict['panstarrs']
        was_legacysurvey = self.config_dict['legacysurvey']
        self.config_dict['colorbandsvisible'] = any(band in visible_panels for band in self.color_bands)
        self.config_dict['nisprgbvisible'] = bool(self.composite_bands) and self.composite_bands[-1] in visible_panels
        self.config_dict['panstarrs'] = _PANSTARRS_KEY in visible_panels
        self.config_dict['legacysurvey'] = _LEGACY_SURVEY_KEY in visible_panels
        if self.config_dict['colorbandsvisible'] and not was_colorbandsvisible and not self.color_bands_already_plotted:
            self.plot()
            self.color_bands_already_plotted = True
        if self.config_dict['panstarrs'] and not was_panstarrs:
            self.set_panstarrs()
        if self.config_dict['legacysurvey'] and not was_legacysurvey:
            self.set_legacy_survey()
        self.update_row_visibility()
        self.rows_summary_label.updateText(self.visible_rows_summary())


    @Slot()
    def checkbox_ls_change_area(self):
        self.config_dict['legacybigarea'] = not self.config_dict['legacybigarea']
        if self.config_dict['legacysurvey']:
            self.set_legacy_survey()
        if self.config_dict['panstarrs']:
            self.set_panstarrs()

    @Slot()
    def checkbox_ls_use_residuals(self):
        self.config_dict['legacyresiduals'] = not self.config_dict['legacyresiduals']
        if self.config_dict['legacysurvey']:
            self.set_legacy_survey()

    @Slot()
    def checkbox_auto_next(self):
        self.config_dict['autonext'] = not self.config_dict['autonext']

    @Slot()
    def checkbox_keyboard_shortcuts(self):
        self.config_dict['keyboardshortcuts'] = not self.config_dict['keyboardshortcuts']

    @Slot()
    def open_ds9(self):
        band2zoom = {'VIS': 4,
                    'I':12,
                    'H':12,
                    'J':12,
                    'Y':12,
                    }
        default_zoom = 12
        all_referenced_bands = ({self.main_band} | set(self.color_bands) |
                                {b for members in self.composite_band_members.values() for b in members})
        arguments = ["ds9", '-fits']
        for band in sorted(all_referenced_bands):
            band_filetype = self.band_filetype.get(band)
            if band_filetype != 'FITS': #ds9 only understands FITS -- skip any non-FITS band.
                continue
            try:
                located = fits_io.locate(self.band_sources, band, self.stem)
            except fits_io.MissingBandError:
                continue
            filename = located.filepath
            if located.hdu:   # an --mef extension: ds9's file[EXTNAME], or it opens the empty primary
                filename += f'[{located.hdu}]'
            arguments += [filename, '-zoom', 'to',str(band2zoom.get(band, default_zoom)), '-colorbar', 'no']
        print(" ".join(arguments))
        # The handle is dropped on purpose: CPython reaps a finished child on the
        # next Popen, so at most one exited ds9 waits as a zombie (BUGS 25).
        try:
            subprocess.Popen(arguments)
        except OSError as e:
            self.status.showMessage(f"Could not start ds9: {e}", 8000)

    @Slot()
    def viewls(self):
        if self._main_band_missing("Open LS"):
            return
        webbrowser.open("https://www.legacysurvey.org/viewer?ra={}&dec={}&layer=ls-dr10&zoom=14&manga&spectra&desi-spec-edr&desi-spec-dr1".format(self.ra,self.dec))

    @Slot()
    def viewPanSTARRS(self):
        if self._main_band_missing("Open PanSTARRS"):
            return
        n_pixels_in_ps1, pixels_big_fov_ps1 = panstarrs_number_of_pixels(self.image_pixel_size,
                                    np.max(self.images[self.main_band].shape))
        size = pixels_big_fov_ps1 if self.config_dict['legacybigarea'] else n_pixels_in_ps1
        webbrowser.open(f"{PS1_CUTOUTS_URL}?pos={self.ra}+{self.dec}&filter=color&filetypes=stack"+
                         f"&size={size}&output_size=0&verbose=0&autoscale=99.500000&catlist=")

    @Slot()
    def viewESASky(self):
        if self._main_band_missing("Open ESASky"):
            return
        fov = (self.image_pixel_size * np.max(self.images[self.main_band].shape)) / 3600
        website = f"https://sky.esa.int/esasky/?target={self.ra}%20{self.dec}&hips=PanSTARRS+DR1+color+(i%2C+r%2C+g)&fov={fov}&cooframe=J2000&sci=true&lang=en&"
        webbrowser.open(website)

    @Slot()
    def change_scale(self, display_text):
        scale = self.scale_options[display_text]
        self.config_dict['scale'] = scale
        self.replot()
        self.save_preferences()

    @Slot()
    def change_colormap(self, display_text):
        self.config_dict['colormap'] = self.colormap_options[display_text]
        self.replot()
        self.save_preferences()

    def load_fits(self, locator, get_radec=False):
        "Pixels and header. With get_radec, also records the object's RA/Dec and pixel size."
        image, header = fits_io.read_pixels_and_header(locator)
        if get_radec:
            radec = fits_io.get_ra_dec(header)
            self.ra, self.dec = radec.ra, radec.dec
            self.image_pixel_size = radec.pixel_size_arcsec
        return image, header

    def plot(self):
        self.label_plot[self.main_band].setText(f"{self.listimage[self.counter]}")
        if self.config_dict['colorbandsvisible']:
            for band in self.color_bands:
                self.plot_band(band)
            self.color_bands_already_plotted = True
        else:
            self.color_bands_already_plotted = False

        if not self.bottom_row_bands_already_plotted:
            for band in [self.main_band]:
                self.plot_band(band)
            for band in self.composite_bands:
                self.plot_composite_band(band)
            self.bottom_row_bands_already_plotted = True

        self.rows_summary_label.updateText(self.visible_rows_summary())
        self.show_missing_bands()

    @property
    def scale(self):
        "The scale function the saved preference names."
        return SCALES[self.config_dict['scale']]

    @property
    def stem(self):
        "The current object's filename without its extension: its name in every band."
        return os.path.splitext(self.filename)[0]

    def show_missing_bands(self):
        "Names the bands this object has no data for in the status-bar note, or empties it."
        self.missing_bands_note.set_note(
            f"{self.stem}: {fits_io.missing_bands_note(self.missing_bands)}" if self.missing_bands else '')

    def _draw_missing_band(self, panel_key, bands):
        "Placeholder panel naming the band(s) this object has no data for (BUGS 36)."
        ax = self.ax[panel_key]
        ax.cla()
        ax.imshow([[0.19]], cmap='gray', vmin=0, vmax=1)
        ax.text(0.5, 0.5, '\n'.join(['no', *bands]), transform=ax.transAxes,
                ha='center', va='center', color='lightgray', fontsize=11)
        ax.set_axis_off()
        self.canvas[panel_key].draw()

    def _record_missing_band(self, band, error):
        self.missing_bands[band] = error
        self.images[band] = None
        if band == self.main_band:
            # RA/Dec come from the main band's header. Without it they are unknown, and
            # the previous object's must not carry over into the CSV or the cutouts.
            self.ra = self.dec = self.image_pixel_size = np.nan

    def _main_band_missing(self, action):
        "True, saying so, when this object has no RA/Dec because its file lacks the main band."
        if self.main_band not in self.missing_bands:
            return False
        self.status.showMessage(f"{action} needs RA/Dec, but this object has no {self.main_band}.", 5000)
        return True

    def plot_band(self, band, scale_min = None, scale_max = None):
        self.ax[band].cla()
        get_radec = True if band == self.main_band else False
        band_filetype = self.band_filetype.get(band)
        try:
            locator = fits_io.locate(self.band_sources, band, self.stem)
            if band_filetype == 'FITS':
                image, header = self.load_fits(locator, get_radec)
        except fits_io.MissingBandError as e:
            self._record_missing_band(band, e)
            self._draw_missing_band(band, [band])
            return
        if band_filetype == 'FITS':
            self.images[band] = np.copy(image)
            if scale_min is None or scale_max is None:
                scale_min, scale_max = scale_limits(image, SINGLE_DISPLAY,
                                                    fits_io.pixel_scale_arcsec(header))
            self.scale_mins[band] = scale_min
            self.scale_maxs[band] = scale_max
            image = rescale_image(image, scale_min, scale_max, self.scale, SINGLE_DISPLAY)
            self.ax[band].imshow(image,cmap=self.config_dict['colormap'], origin='lower')
        else:
            image = np.asarray(open_rendered_image(locator.filepath))
            self.images[band] = np.copy(image)
            self._imshow_rendered(band, image)
        self.ax[band].set_axis_off() #Always before .draw()!
        self.canvas[band].draw()

    def _imshow_rendered(self, band, image):
        "PNG/JPG stamps are already rendered: shown as-is, and a greyscale one (2-D) in grey, not viridis."
        cmap = 'gray' if image.ndim == 2 else None
        self.ax[band].imshow(image, cmap=cmap, origin='upper') #For pngs this is best.

    def plot_composite_band(self, composite_band, scale_min = None, scale_max = None):
        # self.composite_bands only ever contains composites whose 3 members are all
        # FITS bands (filtered at startup), so no format branching is needed here.
        base_bands = self.composite_band_members[composite_band]
        self.ax[composite_band].cla()

        cached_bands = {self.main_band}
        if self.color_bands_already_plotted:
            cached_bands |= set(self.color_bands)
        absent = [band for band in base_bands if band in self.missing_bands]
        if not absent and not set(base_bands).issubset(cached_bands):
            images = {}
            for band in base_bands:
                try:
                    images[band] = self.load_fits(fits_io.locate(self.band_sources, band, self.stem),
                                                  get_radec=False)[0]
                except fits_io.MissingBandError as e:
                    self._record_missing_band(band, e)
                    absent.append(band)
        else:
            images = self.images
        if absent:
            self._draw_missing_band(composite_band, absent)
            return
        try:
            stacked = np.stack([images[band] for band in base_bands],axis=2)
        except ValueError:
            shapes = {band: images[band].shape for band in base_bands}
            raise ValueError(
                f"RGB composite '{composite_band}' requires all member bands to have the exact same "
                f"image dimensions -- got {shapes}")
        image = composite_image(stacked, self.scale, SINGLE_DISPLAY)
        self.ax[composite_band].imshow(image, origin='lower')
        self.ax[composite_band].set_axis_off() #Always before .draw()!
        self.canvas[composite_band].draw()

    def replot(self, scale_min = None, scale_max = None):
        if self.config_dict['colorbandsvisible']:
            for band in self.color_bands:
                self.replot_band(band)
                self.color_bands_already_plotted = False
        else:
            self.color_bands_already_plotted = False

        for band in [self.main_band]:
            self.replot_band(band)
        for band in self.composite_bands:
            self.plot_composite_band(band)

    def replot_band(self, band, scale_min = None, scale_max = None):
        if band in self.missing_bands:
            self._draw_missing_band(band, [band])
            return
        self.ax[band].cla()
        image = np.copy(self.images[band])
        band_filetype = self.band_filetype.get(band)
        if band_filetype == 'FITS':
            image = rescale_image(image, self.scale_mins[band], self.scale_maxs[band], self.scale,
                                  SINGLE_DISPLAY)
            self.ax[band].imshow(image,cmap=self.config_dict['colormap'], origin='lower')
        else:
            self._imshow_rendered(band, image)
        self.ax[band].set_axis_off()
        self.canvas[band].draw()


    def obtain_df(self):
        base_filename = classification_csv_base(SINGLE_CSV_PREFIX, self.name,
                                                len(self.listimage), self.random_seed)
        fresh_name = join(self.classifications_dir, f'{base_filename}.csv')
        new_dataset_name = ''
        existing = resume_csv(classification_csvs(self.classifications_dir, base_filename),
                              self.listimage)
        if existing is not None:
            self.df_name = existing
            print('Reading '+ self.df_name)
            df = pd.read_csv(self.df_name,index_col=0)
            keys_to_drop = []
            for key in df.keys():
                if "Unnamed:" in key:
                    keys_to_drop.append(key)
            df = df.drop(keys_to_drop, axis=1)
            if 'subclassification' not in df.columns:
                df['subclassification'] = 'Empty'
            if (len(self.listimage) == len(df) and
                np.all(self.listimage == df['file_name'].values)):
                return df
            else:
                print("Classification file corresponds to a different dataset.")
                if os.path.exists(fresh_name):
                    new_dataset_name = next_free_csv_path(fresh_name, NEW_DATASET_FORK)

        self.df_name = new_dataset_name or fresh_name
        print('A new csv will be created', self.df_name)
        if new_dataset_name:
            print("To avoid this in the future use the argument `-N name` and give different names to different datasets.")
        dfc = ['file_name', 'classification',
                'subclassification',
                'ra','dec',
                'pixel_size',
                'image_dim',
                'time']
        df = pd.DataFrame(columns=dfc)
        df['file_name'] = self.listimage
        df['classification'] = ['Empty'] * len(self.listimage)
        df['subclassification'] = ['Empty'] * len(self.listimage)
        df['ra'] = np.full(len(self.listimage),np.nan)
        df['dec'] = np.full(len(self.listimage),np.nan)
        df['pixel_size'] = np.full(len(self.listimage),np.nan)
        df['image_dim'] = np.full(len(self.listimage),pd.NA)
        df['time'] = np.full(len(self.listimage),0.0)
        return df

    def go_to_counter_page(self):
        self._clear_pending_subclass()
        self.filename = self.listimage[self.counter]
        self.bottom_row_bands_already_plotted = False
        self.missing_bands = {}
        self.plot()
        if self.config_dict['legacysurvey']:
            self.set_legacy_survey()
        if self.config_dict['panstarrs']:
            self.set_panstarrs()
        self.update_classification_buttoms()
        self.update_subclassification_buttoms()
        self.update_counter()
        self.save_preferences()
        self.save_position()
        cnt = self.counter# - 1
        self.df.at[cnt,'time'] += (time() - self.timer_0)
        self.timer_0 = time()

    @Slot()
    def goto(self):
        # 1-based in the dialog, 0-based in `counter`. COUNTER_MAX is a count,
        # not a last index, so the inclusive maximum here is COUNTER_MAX itself.
        i, ok = QtWidgets.QInputDialog.getInt(self,
                                             'Visual inspection',
                                             '',
                                             self.counter+1,
                                             1,
                                             self.COUNTER_MAX)
        if ok:
            self.counter = i-1
            self.go_to_counter_page()

    @Slot()
    def next(self):
        self.counter = self.counter + 1

        if self.counter>self.COUNTER_MAX-1:
            self.counter=self.COUNTER_MAX-1
            self.status.showMessage('Last image', 10000)
        else:
            self.go_to_counter_page()

    @Slot()
    def prev(self):
        self.counter = self.counter - 1

        if self.counter<self.COUNTER_MIN:
            self.counter=self.COUNTER_MIN
            self.status.showMessage('First image', 10000)

        else:
            self.go_to_counter_page()


    def _unknown_classification_counts(self):
        """{label: number of rows} for every classification in the CSV that the
        current --classifications scheme does not declare. Both columns are
        counted together -- what matters to the user is the label, not which
        column it sat in. The button dicts deliberately map some known keys to
        None (a major with no subclass button, 'None' itself), so membership,
        not truthiness, is what separates known from unknown."""
        counts = {}
        for column, dict_grade2button in (('classification', self.dict_class2button),
                                          ('subclassification', self.dict_subclass2button)):
            for label in self.df[column]:
                if label is None or pd.isna(label) or label in ('None', 'Empty'):
                    continue
                if label not in dict_grade2button:
                    counts[label] = counts.get(label, 0) + 1
        return counts

    def _backup_paths(self):
        """(copies already written, next free name), oldest copy first.

        The first copy is `<name>.csv.bak` -- the plain name -- and later ones
        add `.bak.1`, `.bak.2`, ... Deliberately not `imaging.next_free_csv_path`,
        whose `-marker_N.csv` names exist *to be* found by the viewers' globs:
        a backup must be the opposite, invisible to them, or reopening the
        session could resume one. Nothing here ends in `.csv`, so nothing does.
        """
        candidate = self.df_name + '.bak'
        existing = []
        n = 0
        while os.path.exists(candidate):
            existing.append(candidate)
            n += 1
            candidate = f"{self.df_name}.bak.{n}"
        return existing, candidate

    def _backup_classification_csv(self):
        """Copy the CSV aside when a scheme mismatch is found, before this
        session can overwrite any of the labels it has no button for. Every
        copy is kept, under the next free number: an earlier one may be the
        only remaining record of a label that has since been graded over, so
        none of them is ever replaced.

        A copy is skipped when the newest one already matches the file byte for
        byte -- relaunching without grading anything should not pile up
        identical files.

        Returns `(path of the copy holding the file as found, line for the
        warning block)`; the path is None if there is nothing to copy or the
        copy failed.
        """
        if not os.path.exists(self.df_name):
            return None, None

        existing, backup_path = self._backup_paths()
        if existing and filecmp.cmp(self.df_name, existing[-1], shallow=False):
            return existing[-1], f"  Backup: {existing[-1]} (already matches this file)"

        try:
            shutil.copy2(self.df_name, backup_path)
        except OSError as e:
            return None, (f"  Backup FAILED ({e})\n"
                          f"  Copy {self.df_name} by hand before grading anything.")
        kept = f" ({len(existing)} earlier cop{'y' if len(existing) == 1 else 'ies'} kept)" if existing else ""
        return backup_path, f"  Backup: {backup_path}{kept}"

    def _report_unknown_classifications(self):
        """Say at startup -- in the terminal and at the bottom of the window --
        that this CSV holds classifications the current scheme cannot render a
        button for (BUGS.md #15). Reported up front, over the whole CSV, rather
        than only when the user happens to navigate onto such a row: the label
        is valid data that this session will not change, but nothing lights up
        for it, which is otherwise indistinguishable from unclassified and
        invites overwriting it. Not a reason to refuse to start -- just
        something the user has to know before grading.
        """
        counts = self._unknown_classification_counts()
        if not counts:
            return

        summary = ', '.join(f"'{label}' ({n} row{'s' if n != 1 else ''})"
                            for label, n in sorted(counts.items()))
        scheme = ', '.join(major for major, _ in self.args.major_classes) or '(none)'
        backup_path, backup_line = self._backup_classification_csv()
        print("=" * 72)
        print("WARNING: this session's classification file holds labels that the")
        print("current scheme (--classifications) does not declare:")
        for label, n in sorted(counts.items()):
            print(f"    {label}  ({n} row{'s' if n != 1 else ''})")
        print(f"  Current majors: {scheme}")
        print(f"  File: {self.df_name}")
        if backup_line:
            print(backup_line)
        print("  They are kept exactly as they are, but no button lights up for them,")
        print("  and reclassifying such a row overwrites the label it had.")
        print("=" * 72)

        # A permanent status-bar widget, not showMessage(): it has to outlive
        # every transient message the session prints over it ("Downloading
        # ...", "Last image"), because it is the only thing on screen saying
        # those rows are not simply unclassified.
        notice = QtWidgets.QLabel(f"Not in scheme: {summary}")
        notice.setStyleSheet("color : white; background-color : #a04000; padding : 2px 6px;")
        notice.setToolTip(
            "These classifications were read from " + self.df_name + " but are not in "
            "the current --classifications scheme, so they have no button here. They are "
            "left untouched; classifying such a row replaces the label it had." +
            (f"\nA copy of the file as it was found is kept at {backup_path}"
             if backup_path else ""))
        self.status.addPermanentWidget(notice)

    def _button_for_grade(self, grade, dict_grade2button):
        """dict_grade2button.get(grade), tolerating a label outside the current
        --classifications scheme (BUGS.md #15): the CSV may hold a classification
        written under a different scheme (or a lobby-launched session where the
        mismatch was ignored rather than added), which is valid data with no
        button to light up. `dict_grade2button` deliberately maps some known
        keys to None already (a major with no subclass button, 'None' itself),
        so only a key genuinely absent from the dict is reported as unknown.
        """
        if grade is None or grade in ('None', 'Empty'):
            return None
        if grade not in dict_grade2button:
            # Terminal already has the full list from startup; this one names
            # the row the user is looking at right now.
            self.status.showMessage(
                f"'{grade}' is not in the current classification scheme "
                "(--classifications) -- kept as it is, but no button is highlighted "
                "for it.", 8000)
            return None
        return dict_grade2button[grade]

    def update_classification_buttoms(self):
        grade = self.df.at[self.counter,'classification']

        if self.bactivatedclassification is not None:
            self.bactivatedclassification.setStyleSheet(self.original_button_style)

        button = self._button_for_grade(grade, self.dict_class2button)
        if button is not None:
            button.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))
        self.bactivatedclassification = button

    def update_subclassification_buttoms(self):
        subgrade = self.df.at[self.counter,'subclassification']
        if self.bactivatedsubclassification is not None:
            self.bactivatedsubclassification.setStyleSheet(self.original_button_style)

        button = self._button_for_grade(subgrade, self.dict_subclass2button)
        if button is not None:
            button.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))
        self.bactivatedsubclassification = button

            
def main(argv=None):
    args = parse_args(argv)
    cli.apply_reset_flags(args, SESSION_TOOL, LEGACY_CONFIG_FILE)
    prepare_survey_caches(args)

    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    clipboard = QtWidgets.QApplication.clipboard()
    app = ApplicationWindow(args, clipboard=clipboard)
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()


if __name__ == "__main__":
    main()
