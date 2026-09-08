# This Python file uses the following encoding: utf-8

import argparse
import PySide6 #Must be imported before matplotlib. #TODO remove rewrite without matplotlib widgets

#import time
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS


import glob
from functools import partial

import json

import pandas as pd
import subprocess
from PIL import Image

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Slot, QObject, QThread, Signal
from PySide6.QtGui import QPixmap, QKeySequence, QShortcut, QClipboard

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import image as mpimg

import os
from os.path import join

import re
import sys
from time import time
import urllib
import webbrowser

from imaging import (
    identity, log, asinh2, print_range, get_value_range,
    get_value_range_asymmetric, clip_normalize, contrast_bias_scale,
    get_contrast_bias_reasonable_assumptions, natural_sort,
    find_filename_iteration, detect_band_filetype, find_band_file,
)
from widgets import PanelRowPicker, SettingsMenu, BandNamesLabel
from workers import (
    PS1_FITSCUT_URL, get_panstarrs_filenames, SingleFetchWorker,
    PanstarrsFetchWorker,
)

parser = argparse.ArgumentParser(description='configure the parameters of the execution.')
parser.add_argument('-p',"--path", help="path to the images to inspect",
                    default="Color_stamps_to_inspect")
parser.add_argument('-N',"--name", help="name of the classifying session.",
                    default=None)
parser.add_argument('-b',"--main_band", help='High resolution band. Example: "VIS". This is also the '
                    "tool's default band: the panel that's always individually shown and pre-selected.",
                    default="VIS")
parser.add_argument('-B',"--color_bands", help='Comma-separated bands to show individually via "Show NISP '
                    'bands". Example: "Y,J,H"',
                    default="Y,J,H")
parser.add_argument("--reset-config", help="removes the configuration dictionary during startup.",
                    action="store_true", default=False)
parser.add_argument("--verbose", help="activates loging to terminal",
                    action="store_true", default=False)
parser.add_argument("--clean", help="cleans the legacy survey folder.",
                    action="store_true")
parser.add_argument("--legacysurvey",
                    help="Enables the Legacy Survey panel and downloads. On by default; "
                    "pass --no-legacysurvey to disable (e.g. if the Legacy Survey server is unreliable).",
                    action=argparse.BooleanOptionalAction,
                    default=True)
parser.add_argument('-s',"--seed", help="seed used to shuffle the images.",type=int,
                    default=None)
parser.add_argument('--classifications',
                    help='Classification buttons: semicolon-separated MAJOR=KEY or MAJOR:SUB=KEY entries. '
                    'A bare MAJOR=KEY (or empty SUB) makes a major-class button; MAJOR:SUB=KEY makes a '
                    'subclass button under that major, setting both fields when clicked. '
                    'Example: "A=1;B=2;C=3;X=4;I=5;X:Merger=a;X:Spiral=s"',
                    default="A=1;B=2;C=3;X=4;I=5")
parser.add_argument('--rgb-composites',
                    help='RGB composites: semicolon-separated R,G,B band-name triples (any directory name '
                    "under --path can be a band -- not just VIS/Y/J/H/I). A composite's own name/label is "
                    'simply its comma-joined member list. Example: "H,Y,I;H,J,Y". '
                    'All three bands in a composite must have the exact same image dimensions '
                    '(and ideally the same zero-point) -- mismatched bands cannot be stacked into one RGB image.',
                    default="H,Y,I;H,J,Y")

args = parser.parse_args()

# Empty entries dropped: -B '' means "no color bands", not one band named ''
# (which passes the missing-directory check below -- it resolves to --path itself
# -- and then fails at image load).
args.color_bands = [b.strip() for b in args.color_bands.split(',') if b.strip()]

args.composite_bands = []  # ordered list of composite keys ("H,Y,I")
args.composite_band_members = {}  # key -> (r, g, b) tuple
for entry in args.rgb_composites.split(';'):
    entry = entry.strip()
    if not entry:
        continue
    members = tuple(b.strip() for b in entry.split(','))
    if len(members) != 3:
        parser.error(f'--rgb-composites entry "{entry}" must have exactly 3 comma-separated bands (R,G,B).')
    key = ','.join(members)
    args.composite_bands.append(key)
    args.composite_band_members[key] = members

_referenced_bands = ({args.main_band} | set(args.color_bands) |
                      {b for members in args.composite_band_members.values() for b in members})
_missing_bands = [b for b in _referenced_bands if not os.path.isdir(join(args.path, b))]
if _missing_bands:
    _available = sorted(d for d in os.listdir(args.path) if os.path.isdir(join(args.path, d))) \
        if os.path.isdir(args.path) else []
    print(f"Band director{'y' if len(_missing_bands) == 1 else 'ies'} not found under {args.path}: "
          f"{', '.join(sorted(_missing_bands))}")
    print(f"Available subdirectories: {', '.join(_available) if _available else '(none found)'}")
    sys.exit(1)

args.major_classes = []  # list of (major, key) tuples, in declared order
args.subclasses = []  # list of (major, sub, key) tuples, in declared order
seen_keys = {}  # key -> list of labels, for the duplicate-key warning
for entry in args.classifications.split(';'):
    entry = entry.strip()
    if not entry:
        continue
    rest, _, key = entry.rpartition('=')
    key = key.strip()
    major, _, sub = rest.partition(':')
    major, sub = major.strip(), sub.strip()
    label = f"{major}:{sub}" if sub else major
    seen_keys.setdefault(key, []).append(label)
    if sub:
        args.subclasses.append((major, sub, key))
    else:
        args.major_classes.append((major, key))
for key, labels in seen_keys.items():
    if len(labels) > 1:
        print(f"Warning: keyboard shortcut '{key}' is assigned to more than one button: {', '.join(labels)}")


LEGACY_SURVEY_PATH = './Legacy_survey/'
LEGACY_SURVEY_PIXEL_SIZE=0.262

PANSTARRS_PATH = './PanSTARRS/'
PANSTARRS_PIXEL_SIZE = 0.25
PS1_CUTOUTS_URL = 'https://ps1images.stsci.edu/cgi-bin/ps1cutouts'

SINGLE_BAND = 'single_band'
MAIN_BAND = 'main_band'
COMPOSITE_BAND = 'composite_band'
EXTERNAL_BAND = 'external_band'
_LEGACY_SURVEY_KEY = "Legacy Survey"
_PANSTARRS_KEY = "PanSTARRS"

PATH_TO_CONFIG_FILE = ".config.json"

if args.reset_config:
    if os.path.exists(PATH_TO_CONFIG_FILE):
        os.remove(PATH_TO_CONFIG_FILE)

if args.clean:
    for f in (glob.glob(join(LEGACY_SURVEY_PATH,"*.jpg")) +
              glob.glob(join(PANSTARRS_PATH,"*.jpg"))):
        if os.path.exists(f):
            os.remove(f)

def legacy_survey_number_of_pixels(image_pixel_size,
                                    image_dim,
                                    pixels_big_fov_ls=488): #sizes in ARCSECONDS
    n_pixels_in_ls = int(np.ceil(image_pixel_size*image_dim/LEGACY_SURVEY_PIXEL_SIZE))
    if n_pixels_in_ls >= pixels_big_fov_ls:
        pixels_big_fov_ls = 2*n_pixels_in_ls
    return n_pixels_in_ls, pixels_big_fov_ls

PANSTARRS_BIG_FOV_ARCSEC = 1.4

def panstarrs_number_of_pixels(image_pixel_size, image_dim): #sizes in ARCSECONDS
    n_pixels_in_ps1 = int(np.ceil(image_pixel_size*image_dim/PANSTARRS_PIXEL_SIZE))
    pixels_big_fov_ps1 = int(np.ceil(PANSTARRS_BIG_FOV_ARCSEC/PANSTARRS_PIXEL_SIZE))
    return n_pixels_in_ps1, pixels_big_fov_ps1


class FetchThread(QThread):
    def __init__(self, df, initial_counter, fetch_panstarrs=True, fetch_legacysurvey=True, parent=None):
            QThread.__init__(self, parent)

            self.df = df
            self.initial_counter = initial_counter
            self.legacy_survey_path = LEGACY_SURVEY_PATH
            self.panstarrs_path = PANSTARRS_PATH
            self.stampspath = args.path
            self.main_band = args.main_band
            self.listimage = sorted([os.path.basename(x) for x in glob.glob(join(self.stampspath,self.main_band,'*.fits'))])
            self.im = Image.fromarray(np.zeros((66,66),dtype=np.uint8))
            self.fetch_panstarrs = fetch_panstarrs
            self.fetch_legacysurvey = fetch_legacysurvey
    def download_legacy_survey(self,ra,dec,size=47,residual=False,pixscale='0.262'):
        # residual = (residual and size == 47)
        res = '-resid' if residual else '-grz'
        savename = 'N' + '_' + str(ra) + '_' + str(dec) +f"_{size}" + f'ls-dr10{res}.jpg'
        savefile = os.path.join(self.legacy_survey_path, savename)        
        if os.path.exists(savefile):
            print('File already exists:', savefile) if args.verbose else False
            return True
        url = (f'http://legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}'+
         f'&layer=ls-dr10{res}&size={size}&pixscale={pixscale}')
        print(url) if args.verbose else False
        try:
            urllib.request.urlretrieve(url, savefile)
        except (urllib.error.URLError, OSError):
            with open(savefile,'w') as f:
                self.im.save(f)
            return False

        return True

    def download_panstarrs(self,ra,dec,size=240):
        savename = 'P' + '_' + str(ra) + '_' + str(dec) + f"_{size}" + 'ps1-grz.jpg'
        savefile = os.path.join(self.panstarrs_path, savename)
        if os.path.exists(savefile):
            print('File already exists:', savefile) if args.verbose else False
            return True
        try:
            filenames = get_panstarrs_filenames(ra,dec,filters='grz')
            if filenames is None:
                raise urllib.error.URLError('no PS1 filenames found')
            url = (f"{PS1_FITSCUT_URL}?red={filenames['z']}&green={filenames['r']}&blue={filenames['g']}"+
                   f"&ra={ra}&dec={dec}&size={size}&output_size=256&autoscale=99.5&format=jpg")
            print(url) if args.verbose else False
            urllib.request.urlretrieve(url, savefile)
        except (urllib.error.URLError, OSError):
            with open(savefile,'w') as f:
                self.im.save(f)
            return False

        return True

    def get_ra_dec(self,header):
        w = WCS(header,fix=False)
        sky = w.pixel_to_world_values([w.array_shape[0]//2], [w.array_shape[1]//2])
        image_pixel_size = np.max(np.diag(np.abs(w.pixel_scale_matrix))) * 3600
        return (sky[0][0], sky[1][0],
                np.round(image_pixel_size,decimals=4),
                np.max(w.array_shape)
               )


    def interrupt(self):
        self._active = False

    def run(self):
        index = self.initial_counter
        self._active = True
        while self._active and index < len(self.df): 
            stamp = self.df.iloc[index]
            if np.isnan(stamp['ra']) or np.isnan(stamp['dec']): #TODO: add smt for when there is no RADec.
                f = join(self.stampspath,self.main_band,self.listimage[index])
                ra,dec,image_pixel_size,image_dim = self.get_ra_dec(fits.getheader(f,memmap=False))
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
                # self.download_legacy_survey(ra,dec,size=n_pixels_big_ls, residual=True) #uncomment for large FoV residuals.
            index+=1
        return 0

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, clipboard=None):
        super().__init__()
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
        
        self.defaults = {
                    'name': self.name,
                    'counter':0,
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
        self.config_dict = self.load_dict()
        if not args.legacysurvey:
            self.config_dict['legacysurvey'] = False
        self.im = Image.fromarray(np.zeros((66,66),dtype=np.uint8))

        self.ds9_comm_backend = "xpa"
        self.is_ds9_open = False
        self.singlefetchthread_active = False
        self.colormap = self.config_dict['colormap']
        self.buttoncolor = "darkRed"
        self.buttonclasscolor = "darkRed"
        # self.scratchpath = './.temp_multiband'
        # os.makedirs(self.scratchpath,exist_ok=True)
        self.scale2funct = {'identity':identity,
                            'sqrt':np.sqrt,
                            'log':log,
                            'log10':log,
                            'cbrt':np.cbrt,
                            'asinh2':asinh2}
        self.scale = self.scale2funct[self.config_dict['scale']]


        self.stampspath = args.path
        self.main_band = args.main_band
        self.color_bands = args.color_bands
        self.legacy_survey_path = LEGACY_SURVEY_PATH
        self.panstarrs_path = PANSTARRS_PATH
        self.random_seed = args.seed

        base_band_path = join(self.stampspath, self.main_band)
        self.listimage = sorted(os.path.basename(x) for x in glob.glob(join(base_band_path, '*.fits')))
        self.filetype='FITS'
        if len(self.listimage) < 1:
            self.listimage = sorted(os.path.basename(x)
                            for x in (glob.glob(join(base_band_path, '*.png')) +
                                      glob.glob(join(base_band_path, '*.jpg')) +
                                      glob.glob(join(base_band_path, '*.jpeg'))))
            self.filetype='COMPRESSED'
        print(f"Classifying {len(self.listimage)} sources.")

        if len(self.listimage) < 1:
            print(f"No FITS, PNG, or JPG files found in {base_band_path}.")
            sys.exit(1)
        if self.config_dict['counter'] > len(self.listimage):
            self.config_dict['counter'] = 0

        if self.random_seed is not None:
            print(f"Shuffling with seed {self.random_seed}")
            rng = np.random.default_rng(self.random_seed)
            rng.shuffle(self.listimage) #inplace shuffling

        # Each band's format is a property of its own directory (detected independently),
        # so bands can mix FITS and PNG/JPG within the same session -- only the main band
        # (self.filetype) governs the object list and RA/Dec-dependent tools.
        self.band_filetype = {self.main_band: self.filetype}
        for band in (set(self.color_bands) |
                     {b for members in args.composite_band_members.values() for b in members}):
            self.band_filetype.setdefault(band, detect_band_filetype(join(self.stampspath, band)))

        # RGB composites require all three member bands to be FITS -- drop any composite
        # that isn't, rather than crashing or silently mixing formats into one image.
        self.composite_bands = []
        self.composite_band_members = {}
        for key in args.composite_bands:
            members = args.composite_band_members[key]
            non_fits = [b for b in members if self.band_filetype.get(b) != 'FITS']
            if non_fits:
                print(f"RGB composite '{key}' skipped -- requires FITS bands, but "
                      f"{', '.join(non_fits)} {'is' if len(non_fits) == 1 else 'are'} not FITS.")
                continue
            self.composite_bands.append(key)
            self.composite_band_members[key] = members

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

        # print(self.all_bands)
        self.df = self.obtain_df()

        self.number_graded = 0
        self.COUNTER_MIN = 0
        self.COUNTER_MAX = len(self.listimage)
        # self.filename = join(self.stampspath, 'VIS',self.listimage[self.config_dict['counter']])
        self.filename = join(self.listimage[self.config_dict['counter']])
        # self.status.showMessage(self.listimage[self.config_dict['counter']],)


        main_layout = QtWidgets.QVBoxLayout(self._main)
        self.label_layout = QtWidgets.QHBoxLayout()
        # self.plot_layout_area = QtWidgets.QGridLayout()
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

        self.counter_widget = QtWidgets.QLabel("{}/{}".format(self.config_dict['counter']+1,self.COUNTER_MAX))
        self.counter_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed) #QLabels have different default size policy. Better to use the policy of buttons.
        self.counter_widget.setStyleSheet("font-size: 14px")
        
        # self.label_plot = {band: QtWidgets.QLabel(f"{self.listimage[self.config_dict['counter']]} - {band}", alignment=Qt.AlignCenter) for band in self.all_bands}
        self.band2bandname_dict = {band: band for band in self.all_bands}

        self.label_plot = {band: QtWidgets.QLabel(f"{band}", alignment=Qt.AlignCenter) for band in [self.main_band]}
        self.rows_summary_label = BandNamesLabel(alignment=Qt.AlignCenter)
        # print(f"{self.all_bands = }")
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
        
        # bands_positions = {
        #                 'VIS': (0,0),
        #                 'J': (1,0),
        #                 'H': (1,1),
        #                 'Y': (1,2),
        # }
        # for band in self.all_bands:
            # self.plot_layout_0.addWidget(self.canvas[band], *bands_positions[band]) #Use this if the layout is a grid


        # print(f"{self.composite_bands = }")
        self.panel_keys = [self.main_band, *self.composite_bands, *self.color_bands,
                          *self.external_bands, _PANSTARRS_KEY]
        for band in self.panel_keys:
            self.canvas[band].setStyleSheet('background-color: black')

        no_row_config_saved = not any(self.config_dict[row_key] for row_key in self.panel_row_layout)
        default_row_panels = {'row_1': [self.main_band] + self.composite_bands[:1], 'row_2': [], 'row_3': []}
        visible_panels = set()
        for row_key, row_layout in self.panel_row_layout.items():
            saved = self.config_dict[row_key]
            row_panels = default_row_panels[row_key] if no_row_config_saved else [b for b in saved.split(';') if b in self.panel_keys]
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

        # print(f"{self.all_bands = }")

        self.ax = {band: self.figure[band].subplots() for band in self.all_bands}
        self.images = {}
        self.scale_mins = {}
        self.scale_maxs = {}

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
        self.blsarea = self.panel_picker.add_toggle("Large FoV", checked=self.config_dict['legacybigarea'])
        self.blsarea.clicked.connect(self.checkbox_ls_change_area)
        list_button_row0_layout.append(self.panel_picker)

        #PanSTARRS and Legacy Survey are both self.panel_keys now, so their visibility/placement
        #is already handled by the row-assignment above -- just fetch each if it started out visible.
        self.config_dict['panstarrs'] = _PANSTARRS_KEY in visible_panels
        if self.config_dict['panstarrs']:
            self.set_panstarrs()

        self.config_dict['legacysurvey'] = _LEGACY_SURVEY_KEY in visible_panels
        if self.config_dict['legacysurvey']:
            self.set_legacy_survey()

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
            button.clicked.connect(partial(self.classify, major, major))
            list_classifications.append(button)
            self.dict_class2button[major] = button
            if self.original_button_style is None:
                self.original_button_style = button.styleSheet()

        list_subclassifications = []
        self.dict_subclass2button = {'None': None}
        for major, sub, key in args.subclasses:
            button = QtWidgets.QPushButton(sub)
            button.clicked.connect(partial(self.classify, major, sub))
            list_subclassifications.append(button)
            self.dict_subclass2button[sub] = button
            if self.original_button_style is None:
                self.original_button_style = button.styleSheet()

        for major, _ in args.major_classes:
            self.dict_subclass2button.setdefault(major, None)

        self.scale_options = {'Linear':'identity', 'Sqrt':'sqrt', 'Cbrt':'cbrt', 'Log':'log'}
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

        grade = self.df.at[self.config_dict['counter'],'classification']
        if grade is not None and grade != 'None' and grade != 'Empty':
            self.bactivatedclassification = self.dict_class2button[grade]
            self.bactivatedclassification.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))

        subgrade = self.df.at[self.config_dict['counter'],'subclassification']
        if subgrade is not None and subgrade != 'None' and subgrade != 'Empty':
            self.bactivatedsubclassification = self.dict_subclass2button[subgrade]
            if self.bactivatedsubclassification is not None:
                self.bactivatedsubclassification.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))

        #Keyboard shortcuts
        self.classification_shortcuts = []
        for major, key in args.major_classes:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(partial(self.keyClassify, major, major))
            self.classification_shortcuts.append(shortcut)

        for major, sub, key in args.subclasses:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(partial(self.keyClassify, major, sub))
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


        # self.plot_layout_area.addLayout(self.plot_layout_0,1,0)
        # self.plot_layout_area.addWidget(self.plot_layout_1_Widget,0,0)

        self.plot_layout_area.addWidget(self.plot_layout_0_Widget,1)
        self.plot_layout_area.addWidget(self.plot_layout_1_Widget,1)
        self.plot_layout_area.addWidget(self.plot_layout_2_Widget,1)
        self.update_row_visibility()

        main_layout.addLayout(self.label_layout, 2)
        main_layout.addLayout(self.plot_layout_area, 88)
        main_layout.addLayout(button_layout, 10)

        self.timer_0 = time()

    @Slot()
    def toggle_prefetch_panstarrs(self):
        if self.filetype != 'FITS':
            self.status.showMessage("Pre-fetching PanSTARRS requires FITS input.",5000)
            return
        if self.config_dict['prefetch_panstarrs']:
            self.fetchthread_ps.interrupt()
            self.config_dict['prefetch_panstarrs'] = False
        else:
            self.fetchthread_ps = FetchThread(self.df,self.config_dict['counter'],
                                        fetch_panstarrs=True, fetch_legacysurvey=False) #Always store in an object.
            self.fetchthread_ps.finished.connect(self.fetchthread_ps.deleteLater)
            self.fetchthread_ps.setTerminationEnabled(True)
            self.fetchthread_ps.start()
            self.config_dict['prefetch_panstarrs'] = True

    @Slot()
    def toggle_prefetch_legacysurvey(self):
        if self.filetype != 'FITS':
            self.status.showMessage("Pre-fetching Legacy Survey requires FITS input.",5000)
            return
        if self.config_dict['prefetch_legacysurvey']:
            self.fetchthread_ls.interrupt()
            self.config_dict['prefetch_legacysurvey'] = False
        else:
            self.fetchthread_ls = FetchThread(self.df,self.config_dict['counter'],
                                        fetch_panstarrs=False, fetch_legacysurvey=True) #Always store in an object.
            self.fetchthread_ls.finished.connect(self.fetchthread_ls.deleteLater)
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

    def save_dict(self):
        with open(PATH_TO_CONFIG_FILE, 'w') as f:
            json.dump(self.config_dict, f, ensure_ascii=False, indent=4)

    def load_dict(self):
        try:
            with open(PATH_TO_CONFIG_FILE, ) as f:
                temp_dict = json.load(f)
                if temp_dict['colormap'] == 'gray':
                    temp_dict['colormap'] = "gist_gray"
                if 'name' in temp_dict.keys():
                    if temp_dict['name'] != self.name:
                        temp_dict['name'] = self.name
                        temp_dict['counter'] = 0
                for key in self.defaults.keys():
                    if key not in temp_dict.keys():
                        temp_dict[key] = self.defaults[key]
                
                
                return temp_dict
        except FileNotFoundError:
            return self.defaults


    def update_counter(self):
        self.counter_widget.setText("{}/{}".format(self.config_dict['counter']+1,self.COUNTER_MAX))

    @Slot()
    def keyClassify(self, grade, subgrade):
        if self.config_dict['keyboardshortcuts'] == True:
            self.classify(grade, subgrade)
        
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
        t0 = time()
        cnt = self.config_dict['counter']# - 1
        assert self.df.at[cnt,'file_name'] == self.listimage[self.config_dict['counter']] #TODO handling this possibility better.
        self.df.at[cnt,'classification'] = grade
        self.df.at[cnt,'subclassification'] = subgrade
        if self.filetype == 'FITS':
            self.df.at[cnt,'ra'] = self.ra
            self.df.at[cnt,'dec'] = self.dec
        # self.df.at[cnt,'comment'] = grade
        self.df.at[cnt,'pixel_size'] = self.image_pixel_size
        self.df.at[cnt,'image_dim'] = self.image_size
        self.df.at[cnt,'time'] += (time() - self.timer_0)
        self.timer_0 = time()
        self.df.to_csv(self.df_name)

        self.update_classification_buttoms()
        self.update_subclassification_buttoms()
        
        if self.config_dict['autonext']:
            self.next()
        

    def generate_legacy_survey_filename_url(self,ra,dec,pixscale='0.262',residual=False,size=47):
        # pixscale = '0.262'
        residual = residual
        # residual = (residual and size == 47) #Uncomment to deactivate large FoV residuals.
        res = '-resid' if residual else '-grz'
        savename = 'N' + '_' + str(ra) + '_' + str(dec) +f"_{size}" + f'ls-dr10{res}.jpg'
        savefile = os.path.join(self.legacy_survey_path, savename) 
        print(f"Quering for {savename} ")       
        if os.path.exists(savefile):
            return savefile, ''
        self.status.showMessage("Downloading legacy survey jpeg.")
        url = (f'http://legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}'+
         f'&layer=ls-dr10{res}&size={size}&pixscale={pixscale}')
        return savefile, url

    def generate_title(self, size_in_sky, residuals=False, bigarea=False):
        units = 'arcmin' if bigarea else 'arcsec'
        if residuals:
            return "Residuals, {0:.2f} x {0:.2f}".format(size_in_sky.to(units))
        if bigarea:
            return '{0:.2f} x {0:.2f}'.format(size_in_sky.to(units))
        return "{0:.2f} x {0:.2f}".format(size_in_sky.to(units))

    def plot_legacy_survey(self, savefile, title):
        self.ax[_LEGACY_SURVEY_KEY].cla()
        if savefile != self.legacy_filename:
            return
        self.ax[_LEGACY_SURVEY_KEY].imshow(mpimg.imread(savefile))
        self.ax[_LEGACY_SURVEY_KEY].set_title(title, color='white', fontsize=10)
        self.ax[_LEGACY_SURVEY_KEY].set_axis_off()
        self.canvas[_LEGACY_SURVEY_KEY].draw()

    def plot_no_legacy_survey(self, title='Waiting for data',
                            colormap='Greys_r'):
        self.ax[_LEGACY_SURVEY_KEY].cla()
        self.ax[_LEGACY_SURVEY_KEY].imshow(np.zeros(self.images[self.main_band].shape), cmap=colormap)
        self.ax[_LEGACY_SURVEY_KEY].set_title(title, color='white', fontsize=10)
        self.ax[_LEGACY_SURVEY_KEY].set_axis_off()
        self.canvas[_LEGACY_SURVEY_KEY].draw()

    @Slot()
    def set_legacy_survey(self):
        if self.filetype != 'FITS':
            self.status.showMessage("Legacy Survey requires FITS input (RA/Dec from WCS).",5000)
            return
        pixscale = str(LEGACY_SURVEY_PIXEL_SIZE)
        n_pixels_in_ls, pixels_big_fov_ls = legacy_survey_number_of_pixels(self.image_pixel_size, 
                                    np.max(self.images[self.main_band].shape),
                                    pixels_big_fov_ls=488)

        size = pixels_big_fov_ls if self.config_dict['legacybigarea'] else n_pixels_in_ls
        size_in_sky = (LEGACY_SURVEY_PIXEL_SIZE * u.arcsec) * size
        try:
            savefile, url = self.generate_legacy_survey_filename_url(self.ra,self.dec,
                                        pixscale=pixscale,
                                        residual=self.config_dict['legacyresiduals'],
                                        size=size) 

            title = self.generate_title(
                                        size_in_sky = size_in_sky, 
                                        residuals=self.config_dict['legacyresiduals'],
                                        bigarea=self.config_dict['legacybigarea'])
            if url == '':
                self.legacy_filename = savefile
                self.plot_legacy_survey(savefile, title)
                return
            self.plot_no_legacy_survey()
            self.legacy_filename = savefile
            self.workerThread = QThread(parent=self)
            self.singleFetchWorker = SingleFetchWorker(url, savefile, title)
            self.workerThread.finished.connect(self.singleFetchWorker.deleteLater)
            self.workerThread.started.connect(self.singleFetchWorker.run)
        
            self.singleFetchWorker.moveToThread(self.workerThread)

            self.singleFetchWorker.successful_download.connect(partial(self.plot_legacy_survey, savefile, title))
            self.singleFetchWorker.failed_download.connect(partial(self.plot_no_legacy_survey,title='No Legacy Survey data available',
                            colormap='viridis'))
            self.workerThread.finished.connect(self.workerThread.deleteLater)
            self.workerThread.setTerminationEnabled(True)

            self.workerThread.start()
            self.workerThread.quit()
        
        except FileNotFoundError as E:
            self.plot_no_legacy_survey()
            # raise
        except Exception as E:
            print("Exception while setting up the Legacy Survey image:")
            print(E.args)
            print(type(E))
            # raise

    def generate_panstarrs_filename_url(self,ra,dec,size=240):
        savename = 'P' + '_' + str(ra) + '_' + str(dec) + f"_{size}" + 'ps1-grz.jpg'
        savefile = os.path.join(self.panstarrs_path, savename)
        print(f"Quering for {savename} ")
        return savefile, os.path.exists(savefile)

    def plot_panstarrs(self, savefile, title):
        self.ax[_PANSTARRS_KEY].cla()
        if savefile != self.panstarrs_filename:
            return
        self.ax[_PANSTARRS_KEY].imshow(mpimg.imread(savefile))
        self.ax[_PANSTARRS_KEY].set_title(title, color='white', fontsize=10)
        self.ax[_PANSTARRS_KEY].set_axis_off()
        self.canvas[_PANSTARRS_KEY].draw()

    def plot_no_panstarrs(self, title='Waiting for data',
                            colormap='Greys_r'):
        self.ax[_PANSTARRS_KEY].cla()
        self.ax[_PANSTARRS_KEY].imshow(np.zeros(self.images[self.main_band].shape), cmap=colormap)
        self.ax[_PANSTARRS_KEY].set_title(title, color='white', fontsize=10)
        self.ax[_PANSTARRS_KEY].set_axis_off()
        self.canvas[_PANSTARRS_KEY].draw()

    @Slot()
    def set_panstarrs(self):
        if self.filetype != 'FITS':
            self.status.showMessage("PanSTARRS requires FITS input (RA/Dec from WCS).",5000)
            return
        n_pixels_in_ps1, pixels_big_fov_ps1 = panstarrs_number_of_pixels(self.image_pixel_size,
                                    np.max(self.images[self.main_band].shape))

        size = pixels_big_fov_ps1 if self.config_dict['legacybigarea'] else n_pixels_in_ps1
        size_in_sky = (PANSTARRS_PIXEL_SIZE * u.arcsec) * size
        try:
            savefile, cached = self.generate_panstarrs_filename_url(self.ra,self.dec,size=size)

            title = self.generate_title(
                                        size_in_sky = size_in_sky,
                                        bigarea=self.config_dict['legacybigarea'])
            self.panstarrs_filename = savefile
            if cached:
                self.plot_panstarrs(savefile, title)
                return
            self.plot_no_panstarrs()
            self.status.showMessage("Downloading PanSTARRS jpeg.")
            self.workerThreadPS = QThread(parent=self)
            self.singleFetchWorkerPS = PanstarrsFetchWorker(self.ra, self.dec, savefile, size)
            self.workerThreadPS.finished.connect(self.singleFetchWorkerPS.deleteLater)
            self.workerThreadPS.started.connect(self.singleFetchWorkerPS.run)

            self.singleFetchWorkerPS.moveToThread(self.workerThreadPS)

            self.singleFetchWorkerPS.successful_download.connect(partial(self.plot_panstarrs, savefile, title))
            self.singleFetchWorkerPS.failed_download.connect(partial(self.plot_no_panstarrs,title='No PanSTARRS data available',
                            colormap='viridis'))
            self.workerThreadPS.finished.connect(self.workerThreadPS.deleteLater)
            self.workerThreadPS.setTerminationEnabled(True)

            self.workerThreadPS.start()
            self.workerThreadPS.quit()

        except FileNotFoundError as E:
            self.plot_no_panstarrs()
            # raise
        except Exception as E:
            print("Exception while setting up the PanSTARRS image:")
            print(E.args)
            print(type(E))
            # raise


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
            band_filetype = self.filetype if band == self.main_band else self.band_filetype.get(band)
            if band_filetype != 'FITS': #ds9 only understands FITS -- skip any non-FITS band.
                continue
            filename = self._band_filepath(band)
            arguments += [filename, '-zoom', 'to',str(band2zoom.get(band, default_zoom)), '-colorbar', 'no']
        print(" ".join(arguments))
        subprocess.Popen(arguments)

    @Slot()
    def viewls(self):
        webbrowser.open("https://www.legacysurvey.org/viewer?ra={}&dec={}&layer=ls-dr10&zoom=14&manga&spectra&desi-spec-edr&desi-spec-dr1".format(self.ra,self.dec))

    @Slot()
    def viewPanSTARRS(self):
        n_pixels_in_ps1, pixels_big_fov_ps1 = panstarrs_number_of_pixels(self.image_pixel_size,
                                    np.max(self.images[self.main_band].shape))
        size = pixels_big_fov_ps1 if self.config_dict['legacybigarea'] else n_pixels_in_ps1
        webbrowser.open(f"{PS1_CUTOUTS_URL}?pos={self.ra}+{self.dec}&filter=color&filetypes=stack"+
                         f"&size={size}&output_size=0&verbose=0&autoscale=99.500000&catlist=")

    @Slot()
    def viewESASky(self):
        fov = (self.image_pixel_size * np.max(self.images[self.main_band].shape)) / 3600
        website = f"https://sky.esa.int/esasky/?target={self.ra}%20{self.dec}&hips=PanSTARRS+DR1+color+(i%2C+r%2C+g)&fov={fov}&cooframe=J2000&sci=true&lang=en&"
        # website += "&euclid_image=perseus" #Use this to add the Euclid ERO overlay. Sadly, this is always centered on the same coordinate.
        webbrowser.open(website)

    @Slot()
    def change_scale(self, display_text):
        scale = self.scale_options[display_text]
        self.scale = self.scale2funct[scale]
        self.config_dict['scale'] = scale
        self.replot()
        self.save_dict()

    @Slot()
    def change_colormap(self, display_text):
        self.config_dict['colormap'] = self.colormap_options[display_text]
        self.replot()
        self.save_dict()

    def background_rms_image(self, cb, image):
        xg, yg = np.shape(image)
        cb=10
        cut0 = image[0:cb, 0:cb]
        cut1 = image[xg - cb:xg, 0:cb]
        cut2 = image[0:cb, yg - cb:yg]
        cut3 = image[xg - cb:xg, yg - cb:yg]
        l = [cut0, cut1, cut2, cut3]
        while len(l) > 1:
            m = np.nanmean(np.nanmean(l, axis=1), axis=1)
            if max(m) > 5 * min(m):
                s = np.sort(l, axis=0)
                l = s[:-1]
            else:
                std = np.nanstd(l)
                return std
        std = np.nanstd(l)
        return std

    def background_rms_image_old(self,cb, image):
        xg, yg = np.shape(image)
        cut0 = image[0:cb, 0:cb]
        cut1 = image[xg - cb:xg, 0:cb]
        cut2 = image[0:cb, yg - cb:yg]
        cut3 = image[xg - cb:xg, yg - cb:yg]
        l = [cut0, cut1, cut2, cut3]
        m = np.nanmean(np.nanmean(l, axis=1), axis=1)
        ml = min(m)
        mm = max(m)
        if ml is np.nan or mm is np.nan:
            print(f"WARNING: {ml = }, {mm = }")
        if mm > 5 * ml:
            s = np.sort(l, axis=0)
            nl = s[:-1]
            std = np.nanstd(nl)
        else:
            std = np.nanstd([cut0, cut1, cut2, cut3])
        return std
    
    def scale_val(self,image_array):
        if len(np.shape(image_array)) == 2:
            image_array = [image_array]

        if image_array[0].shape[0] > 170:
            box_size_vmin = np.round(np.sqrt(np.prod(image_array[0].shape) * 0.001)).astype(int)
            box_size_vmax = np.round(np.sqrt(np.prod(image_array[0].shape) * 0.01)).astype(int)
        else:
            #Sensible default values
            box_size_vmin = 5
            box_size_vmax = 14
        # print(len(image_array))
        vmin = np.nanmin([self.background_rms_image(box_size_vmin, image) for image in image_array])
        
        # print(f"{box_size_vmin = }, {box_size_vmax = }")
        xl, yl = np.shape(image_array[0])
        xmin = int((xl) / 2. - (box_size_vmax / 2.))
        xmax = int((xl) / 2. + (box_size_vmax / 2.))
        ymin = int((yl) / 2. - (box_size_vmax / 2.))
        ymax = int((yl) / 2. + (box_size_vmax / 2.))
        vmax = np.nanmax([image[xmin:xmax, ymin:ymax] for image in image_array])
        return vmin*1.0, vmax*1.3 #vmin is 1 sigma of noise.

    def rescale_image_composite(self, image, scale_min, scale_max, composite = False):
        factor = self.scale(scale_max - scale_min)
        # print(f"{scale_min = }, {scale_max = }")
        image = np.clip(image, scale_min, scale_max)
        image -= scale_min

        indices1 = np.where(image > 0)
        image[indices1] = self.scale(image[indices1]) / (factor * 1.0)
        return image

    def rescale_image_composite2(self, image,
                                p_low = 1,
                                p_high = 1,
                                value_at_min = 0,
                                color_bkg_level = 0.1):
        # scale_min, scale_max = get_value_range(image,p)
        scale_min, scale_max = get_value_range_asymmetric(image,p_low,p_high)

        image = clip_normalize(image,scale_min,scale_max)
        image = self.scale(image)
        contrast, bias = get_contrast_bias_reasonable_assumptions(
                                                                    # 0,
                                                                    max(value_at_min,scale_min),
                                                                    color_bkg_level,
                                                                    scale_min,
                                                                    scale_max,
                                                                    self.scale)
        image = contrast_bias_scale(image, contrast, bias)

        return image

    def rescale_single_band(self, image,
                            scale_min,
                            scale_max,
                            value_at_min=0,
                            color_bkg_level=-0.05):

        image = clip_normalize(image,scale_min,scale_max)
        image = self.scale(image)
        contrast, bias = get_contrast_bias_reasonable_assumptions(
                                                                    max(value_at_min,scale_min),
                                                                    color_bkg_level,
                                                                    scale_min,
                                                                    scale_max,
                                                                    self.scale)
        return contrast_bias_scale(image, contrast, bias)

    def prepare_composite_image(self, images,
                                p_low=2, p_high=0.1,
                                value_at_min=0,
                                color_bkg_level=-0.05,
                                ):
        # composite_image = np.zeros((*images[0].shape, 3),
        #                             dtype=float)
        composite_image = np.zeros_like(images,
                                    dtype=float)
        scale_min, scale_max = get_value_range_asymmetric(images,p_low,p_high)
        # print(f"{scale_min = }, {scale_max = }")
        # for i, image in enumerate(images):
        # print(images.shape)
        for i in range(images.shape[-1]):
            composite_image[:,:,i] = self.rescale_single_band(
                            images[:,:,i],
                            scale_min,
                            scale_max,
                            value_at_min,
                            color_bkg_level)
        return composite_image


    def rescale_image(self, image, scale_min, scale_max):
            factor = self.scale(scale_max - scale_min)
            image = image.clip(min=scale_min, max=scale_max)
            indices0 = np.where(image < scale_min)
            indices1 = np.where((image >= scale_min) & (image <= scale_max))
            indices2 = np.where(image > scale_max)
            # image = image - scale_min
            image[indices0] = 0.0 #Why would there be a value below scale_min?
            image[indices2] = 1.0 #This is probably useless
            image[indices1] = self.scale(image[indices1]) / (factor * 1.0)

            return image

    def load_fits(self,filepath, get_radec=False):
        opened_fits = fits.open(filepath)
        if get_radec:
            self.ra,self.dec = self.get_ra_dec(opened_fits[0].header)
        return opened_fits[0].data

    def get_ra_dec(self,header):
        w = WCS(header,fix=False)
        sky = w.pixel_to_world_values([w.array_shape[0]//2], [w.array_shape[1]//2])
        self.image_pixel_size = np.round(np.max(np.diag(np.abs(w.pixel_scale_matrix))) * 3600, decimals=4)
        self.image_size = np.max(w.array_shape)
        return sky[0][0], sky[1][0]#, image_pixel_size

    def _band_filepath(self, band):
        "Resolves band's file for the current object -- main band uses self.filename directly, other bands are matched by stem since their format/extension can differ."
        if band == self.main_band:
            return join(self.stampspath, band, self.filename)
        stem = os.path.splitext(self.filename)[0]
        filepath = find_band_file(self.stampspath, band, stem)
        if filepath is None:
            raise FileNotFoundError(f"No FITS/PNG/JPG file found for '{stem}' in band '{band}'.")
        return filepath

    def plot(self, scale_min = None, scale_max = None, band = None):
        self.label_plot[self.main_band].setText(f"{self.listimage[self.config_dict['counter']]}")
        # label = ""
        if self.config_dict['colorbandsvisible']:
            for band in self.color_bands:
                self.plot_band(band)
                # label += f"{band}-"
            self.color_bands_already_plotted = True
            # label = label[:-1]+'\n'
        else:
            self.color_bands_already_plotted = False
        # label += f'{self.main_band}-'

        if not self.bottom_row_bands_already_plotted:
            for band in [self.main_band]:
                self.plot_band(band)
            for band in self.composite_bands:
                self.plot_composite_band(band)
            self.bottom_row_bands_already_plotted = True

        self.rows_summary_label.updateText(self.visible_rows_summary())

    def plot_band(self, band, scale_min = None, scale_max = None):
        # self.label_plot[band].setText(self.listimage[self.config_dict['counter']])
        self.ax[band].cla()
        get_radec = True if band == self.main_band else False
        band_filetype = self.filetype if band == self.main_band else self.band_filetype.get(band)
        filepath = self._band_filepath(band)
        if band_filetype == 'FITS':
            image = self.load_fits(filepath, get_radec)
            # scaling_factor = np.nanpercentile(image,q=90)
            # if scaling_factor == 0:
            #     # scaling_factor = np.nanpercentile(image,q=99)
            #     scaling_factor = 1
            # image = image / scaling_factor*300 #Rescaling for better visualization.
            self.images[band] = np.copy(image)
            if scale_min is None or scale_max is None:
                scale_min, scale_max = self.scale_val(image)
            # print(f"{band}: {scale_min = }, {scale_max = }, {image.max()}")
            self.scale_mins[band] = scale_min
            self.scale_maxs[band] = scale_max
            image = self.rescale_image(image, scale_min, scale_max)
            self.ax[band].imshow(image,cmap=self.config_dict['colormap'], origin='lower')
        else:
            image = np.asarray(Image.open(filepath))
            self.images[band] = np.copy(image)
            self.ax[band].imshow(image, origin='upper') #For pngs this is best.
        self.ax[band].set_axis_off() #Always before .draw()!
        self.canvas[band].draw()

    def plot_composite_band(self, composite_band, scale_min = None, scale_max = None):
        # self.composite_bands only ever contains composites whose 3 members are all
        # FITS bands (filtered at startup), so no format branching is needed here.
        base_bands = self.composite_band_members[composite_band]

        # self.label_plot[composite_band].setText(self.listimage[self.config_dict['counter']])
        self.ax[composite_band].cla()

        cached_bands = {self.main_band}
        if self.color_bands_already_plotted:
            cached_bands |= set(self.color_bands)
        if not set(base_bands).issubset(cached_bands):
            images = {band: self.load_fits(self._band_filepath(band),get_radec=False) for band in base_bands}
        else:
            images = self.images
        try:
            stacked = np.stack([images[band] for band in base_bands],axis=2)
        except ValueError:
            shapes = {band: images[band].shape for band in base_bands}
            raise ValueError(
                f"RGB composite '{composite_band}' requires all member bands to have the exact same "
                f"image dimensions -- got {shapes}")
        image = self.prepare_composite_image(stacked)
        self.ax[composite_band].imshow(image, origin='lower')
        self.ax[composite_band].set_axis_off() #Always before .draw()!
        self.canvas[composite_band].draw()

    def replot(self, scale_min = None, scale_max = None):
        # for band in self.all_bands:
        #     if self.band_types[band] in [COMPOSITE_BAND,
        #                                  EXTERNAL_BAND]:
        #         continue
        #     self.replot_band(band)
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
        # self.label_plot[band].setText(self.listimage[self.config_dict['counter']])
        self.ax[band].cla()
        image = np.copy(self.images[band])
        band_filetype = self.filetype if band == self.main_band else self.band_filetype.get(band)
        if band_filetype == 'FITS':
            image = self.rescale_image(image, self.scale_mins[band], self.scale_maxs[band])
            self.ax[band].imshow(image,cmap=self.config_dict['colormap'], origin='lower')
        else:
            self.ax[band].imshow(image, origin='upper') #For pngs this is best.
        self.ax[band].set_axis_off()
        self.canvas[band].draw()


    def obtain_df(self):
        if self.random_seed is None:
            base_filename = f'classification_single_{self.name}_{len(self.listimage)}'
            string_to_glob = f'./Classifications/{base_filename}-*.csv'
            # print("Globing for", string_to_glob)
            # string_to_glob_for_files_with_seed = f'./Classifications/{base_filename}_*.csv'
            # glob_results = set(glob.glob(string_to_glob)) - set(glob.glob(string_to_glob_for_files_with_seed))
            string_to_glob_for_files_with_seed = f'./Classifications/{base_filename}_*.csv'
            glob_results = (set(glob.glob(string_to_glob)) -
                            set(glob.glob(string_to_glob_for_files_with_seed)) |
                            set(glob.glob(f'./Classifications/{base_filename}.csv')))
            # print("first glob:", set(glob.glob(string_to_glob)))
        else:
            base_filename = f'classification_single_{self.name}_{len(self.listimage)}_{self.random_seed}'
            string_to_glob = f'./Classifications/{base_filename}*.csv'
            glob_results = glob.glob(string_to_glob)
        
        file_iteration = ""
        class_file = np.array(natural_sort(glob_results)) #better to use natural sort.
        # print(class_file)
        if len(class_file) >= 1:
            file_index = 0
            if len(class_file) > 1:
                file_index = -2
            self.df_name = class_file[file_index]
            print('Reading '+ self.df_name)
            df = pd.read_csv(self.df_name,index_col=0)
            keys_to_drop = []
            for key in df.keys():
                if "Unnamed:" in key:
                    keys_to_drop.append(key)
            df.drop(keys_to_drop,axis=1)
            if 'subclassification' not in df.columns:
                df['subclassification'] = 'Empty'
            if (len(self.listimage) == len(df) and
                np.all(self.listimage == df['file_name'].values)):
                return df
            else:
                print("Classification file corresponds to a different dataset.")
                string_tested = os.path.basename(self.df_name).split(".csv")[0]
                file_iteration = find_filename_iteration(string_tested) if f'./Classifications/{base_filename}.csv' in class_file else ''

        # self.config_dict['counter'] = 0
        # self.update_counter()

        self.dfc = ['file_name', 'classification', 'grid_pos','page']
        self.df_name = f'./Classifications/{base_filename}{file_iteration}.csv'
        print('A new csv will be created', self.df_name)
        if file_iteration != "":
            print("To avoid this in the future use the argument `-N name` and give different names to different datasets.")
        self.config_dict['counter'] = 0
        dfc = ['file_name', 'classification',
                'subclassification',
                'ra','dec',
                # 'comment',
                'image_dim',
                'time']
        df = pd.DataFrame(columns=dfc)
        df['file_name'] = self.listimage
        df['classification'] = ['Empty'] * len(self.listimage)
        df['subclassification'] = ['Empty'] * len(self.listimage)
        df['ra'] = np.full(len(self.listimage),np.nan)
        df['dec'] = np.full(len(self.listimage),np.nan)
        # df['comment'] = ['Empty'] * len(self.listimage)
        df['image_dim'] = np.full(len(self.listimage),pd.NA)
        df['time'] = np.full(len(self.listimage),0.0)
        return df

    def go_to_counter_page(self):
        self.filename = self.listimage[self.config_dict['counter']]
        self.bottom_row_bands_already_plotted = False
        self.plot()
        if self.config_dict['legacysurvey']:
            self.set_legacy_survey()
        if self.config_dict['panstarrs']:
            self.set_panstarrs()
        self.update_classification_buttoms()
        self.update_subclassification_buttoms()
        self.update_counter()
        self.save_dict()
        cnt = self.config_dict['counter']# - 1
        self.df.at[cnt,'time'] += (time() - self.timer_0)
        self.timer_0 = time()

    @Slot()
    def goto(self):
        i, ok = QtWidgets.QInputDialog.getInt(self,
                                             'Visual inspection',
                                             '',
                                             self.config_dict['counter']+1,
                                             1,
                                             self.COUNTER_MAX+1)
        if ok:
            self.config_dict['counter'] = i-1
            self.go_to_counter_page()

    @Slot()
    def next(self):
        self.config_dict['counter'] = self.config_dict['counter'] + 1

        if self.config_dict['counter']>self.COUNTER_MAX-1:
            self.config_dict['counter']=self.COUNTER_MAX-1
            self.status.showMessage('Last image')
        else:
            self.go_to_counter_page()

    @Slot()
    def prev(self):
        self.config_dict['counter'] = self.config_dict['counter'] - 1

        if self.config_dict['counter']<self.COUNTER_MIN:
            self.config_dict['counter']=self.COUNTER_MIN
            self.status.showMessage('First image')

        else:
            self.go_to_counter_page()


    def update_classification_buttoms(self):
        grade = self.df.at[self.config_dict['counter'],'classification']


        if self.bactivatedclassification is not None:
            # self.bactivatedclassification.setStyleSheet("background-color : white;color : black;")
            self.bactivatedclassification.setStyleSheet(self.original_button_style)

        #if grade is not None and not np.isnan(float(grade)) and grade != 'None':
        if grade is not None and grade != 'None' and grade != 'Empty':
            button = self.dict_class2button[grade]
            if button is not None:
                button.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))
                self.bactivatedclassification = button

    def update_subclassification_buttoms(self):
        subgrade = self.df.at[self.config_dict['counter'],'subclassification']
        if self.bactivatedsubclassification is not None:
            self.bactivatedsubclassification.setStyleSheet(self.original_button_style)

#        if subgrade is not None and not np.isnan(subgrade) and subgrade != 'None':
        if subgrade is not None and subgrade != 'None' and subgrade != 'Empty':
            button = self.dict_subclass2button[subgrade]
            if button is not None:
                button.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))
                self.bactivatedsubclassification = button

            
def main():
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    clipboard = QtWidgets.QApplication.clipboard()
    app = ApplicationWindow(clipboard=clipboard)
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()


if __name__ == "__main__":
    main()
