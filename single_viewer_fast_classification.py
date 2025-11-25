
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
from PySide6.QtGui import (QPixmap, QKeySequence,
                           QShortcut, QClipboard, QFont,
                           QAction, QPalette, QColor, QPainter)



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

parser = argparse.ArgumentParser(description='configure the parameters of the execution.')
parser.add_argument('-p',"--path", help="path to the images to inspect",
                    default="Color_stamps_to_inspect")
parser.add_argument('-N',"--name", help="name of the classifying session.",
                    default=None)
# parser.add_argument('-b',"--main_band", help='High resolution band. Example: "VIS"',
#                     default="VIS")
# parser.add_argument('-B',"--color_bands", help='Comma-separated photometric bands, Bluer to Redder. Example: "Y,J,H"',
#                     default="Y,J,H")
parser.add_argument("--reset-config", help="removes the configuration dictionary during startup.",
                    action="store_true", default=False)
# parser.add_argument("--verbose", help="activates loging to terminal",
#                     action="store_true", default=False)
# parser.add_argument("--clean", help="cleans the legacy survey folder.",
#                     action="store_true")
# parser.add_argument('--fits',
#                     help=("forces app to only use fits (--fits) or png/jp(e)g (--no-fits). "+
#                     "If unset, the app searches for fits files in the path, but defaults to "+
#                     "png/jp(e)g if no fits files are found."),
#                     action=argparse.BooleanOptionalAction,
#                     default=False) # Only jpg support for now.
parser.add_argument('-s',"--seed", help="seed used to shuffle the images.",type=int,
                    default=None)
parser.add_argument("--extension", help="File extension for compressed images (e.g., jpg, png, jpeg)",
                    default='jpg')

args = parser.parse_args()

args.verbose = False
args.fits = False


LEGACY_SURVEY_PATH = './Legacy_survey/'
LEGACY_SURVEY_PIXEL_SIZE=0.262

SINGLE_BAND = 'single_band'
MAIN_BAND = 'main_band'
COMPOSITE_BAND = 'composite_band'
EXTERNAL_BAND = 'external_band'
_LEGACY_SURVEY_KEY = "Legacy Survey"
# _VIS_RESAMPLED_BAND = 'I'
_BAND_FILENAMES_KEY = "Bands_filenames"

_FILETYPE_FITS = 'FITS'
_FILETYPE_COMPRESSED = 'COMPRESSED'

PATH_TO_CONFIG_FILE = ".config.json"

if args.reset_config:
    if os.path.exists(PATH_TO_CONFIG_FILE):
        os.remove(PATH_TO_CONFIG_FILE)

# if args.clean:
#     for f in glob.glob(join(LEGACY_SURVEY_PATH,"*.jpg")):
#         if os.path.exists(f):
#             os.remove(f)

def identity(x):
    return x

# def log(x):
#     "Simple log base 1000 function that ignores numbers less than 0"
#     return np.log(x, out=np.zeros_like(x), where=(x>0)) / np.log(1000)

def log(x,a=1000):
    "Simple log base 1000 function that ignores numbers less than 0"
    return np.log(a*x+1) / np.log(a)

# def log(x):
#     return np.arcsinh(10*x)/3

def asinh2(x):
    return np.arcsinh(10*x)/3


def print_range(image):
    return f"{image.min() = }, {image.max() = }"

def get_value_range(x, p=98):
    q = (100 - p)/2
    low = np.nanpercentile(x, q)
    high = np.nanpercentile(x, 100-q)
    return low, high

def get_value_range_asymmetric(x, q_low=1, q_high=1,
                              pixel_boxsize_low = None):
    
    low = np.nanpercentile(x, q_low)
    
    # if pixel_boxsize_low is :
    # if pixel_boxsize_low is None:
    #     high = np.nanpercentile(x, 100-q_high)
    # else:
    if x.shape[0] > 80:
        pixel_boxsize_low = np.round(np.sqrt(np.prod(x.shape) * 0.01)).astype(int)
    else:
        pixel_boxsize_low = 8
    xl, yl, _ = np.shape(x)
    xmin = int((xl) / 2. - (pixel_boxsize_low / 2.))
    xmax = int((xl) / 2. + (pixel_boxsize_low / 2.))
    ymin = int((yl) / 2. - (pixel_boxsize_low / 2.))
    ymax = int((yl) / 2. + (pixel_boxsize_low / 2.))
    high = np.nanpercentile(x[xmin:xmax,ymin:ymax], 100-q_high)
    # print(pixel_boxsize_low)
    return low, high

def clip_normalize(x, low=None, high=None):
    x = np.clip(x, low, high)
    x = (x - low)/(high - low)
    return x 

def contrast_bias_scale(x, contrast, bias):
    x = ((x - bias) * contrast + 0.5 )
    x = np.clip(x, 0, 1)
    return x

def get_contrast_bias_reasonable_assumptions(value_at_min, bkg_color, scale_min, scale_max, scale):
    bkg_level = clip_normalize(value_at_min, scale_min, scale_max)
    bkg_level = scale(bkg_level)
    contrast = (bkg_color - 1) / (bkg_level - 1) # with bkg_level != 1 and bkg_color != 1
    bias = 1 - (bkg_level-1)/(2*(bkg_color-1))
    return contrast, bias

def natural_sort(l): 
    "https://stackoverflow.com/a/4836734"
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def find_filename_iteration(latest_filename, max_iterations = 100, initial_iteration = "-(1)"):
    "Uses regex to find and add 1 to the number in parentheses right before the .csv"
    re_pattern = re.compile('-\\(([^)]+)\\)')
    re_search = re_pattern.search(latest_filename)
    if re_search is None:
        return initial_iteration
    iterations = 0
    while re_search.span()[-1] != len(latest_filename) and (iterations < max_iterations):
        re_search = re_pattern.search(latest_filename, re_search.span()[-1])
        if re_search is None:
            return initial_iteration
    if re_search.span()[-1] == len(latest_filename): #at this point, re_search cannot be None
        re_match = re_search[1]
    try:
        int_match = int(re_match)
    except:
        return initial_iteration
    
    return f"-({int_match+1})"

def legacy_survey_number_of_pixels(image_pixel_size, 
                                    image_dim,
                                    pixels_big_fov_ls=488): #sizes in ARCSECONDS
    n_pixels_in_ls = int(np.ceil(image_pixel_size*image_dim/LEGACY_SURVEY_PIXEL_SIZE))
    if n_pixels_in_ls >= pixels_big_fov_ls:
        pixels_big_fov_ls = 2*n_pixels_in_ls
    return n_pixels_in_ls, pixels_big_fov_ls

def join_list_of_lists(list_of_lists, separator=','): #into a string.
    return f'{separator}'.join(item for sublist in list_of_lists for item in sublist)

def join_nested(lines, sep=' | ', line_sep='\n'):#into a string for labels.
    return line_sep.join(sep.join(map(str, sub)) for sub in lines)
class BandNamesLabel(QtWidgets.QLabel):
    def __init__(self,
                no_of_rows,
                *args,
                **kwargs
                ):
        QtWidgets.QLabel.__init__(self, *args, **kwargs)
        self.row_texts = [''] * no_of_rows

    def updateText(self,
                    status_plot_rows,
                    ):
        label = join_nested(status_plot_rows)
        self.setText(label)




class BandNamesLabel_old(QtWidgets.QLabel):
    def __init__(self,
                main_band,
                color_bands,
                standard_color_band,
                resampled_color_band,
                *args,
                **kwargs
                ):
        QtWidgets.QLabel.__init__(self, *args, **kwargs)
        self.main_band = main_band
        self.color_bands = color_bands
        self.standard_color_band = standard_color_band
        self.resampled_color_band = resampled_color_band

        # self.bands = args

        # print('band names label')
        # print(self.main_band)
        # print(self.color_bands)
        # print(self.standard_color_band)
        # print(self.resampled_color_band)
        # print('end')

    def updateText(self,
                    color_bands_status,
                    standard_color_band_status):
        label = ''
        # if color_bands_status:
        #     for band in self.color_bands:
        #         label += f"{band} | "
        #     label = label[:-3]+'\n'

        if color_bands_status:
            label += f'{self.color_bands[0]} | '
            label += self.color_bands[1]
            if standard_color_band_status:
                label += f' | {self.color_bands[2]}'
            label += '\n'

        label += f'{self.main_band} | '
        label += self.resampled_color_band

        if standard_color_band_status:
            label += f' | {self.standard_color_band}'

        self.setText(label)

class CheckableSubMenu(QtWidgets.QMenu):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self._checkboxes = []

    def add_check_item(self, text, checked=False):
        action = QtWidgets.QWidgetAction(self)
        box = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(box)
        layout.setContentsMargins(8, 2, 8, 2)
        cb = QtWidgets.QCheckBox(text, box)
        cb.setChecked(checked)
        layout.addWidget(cb)
        action.setDefaultWidget(box)
        self.addAction(action)
        self._checkboxes.append(cb)
        return cb

    def selected_texts(self):
        return [cb.text() for cb in self._checkboxes if cb.isChecked()]

    def set_all(self, state):
        for cb in self._checkboxes:
            cb.setChecked(state)


class MultiSelectDropdown(QtWidgets.QWidget):
    itemToggled = Signal(str, str, bool)
    def __init__(self, list_of_bands, parent=None):
        super().__init__(parent)
        self.button = QtWidgets.QToolButton(self)
        self.button.setText("Choose...")
        self.button.setPopupMode(QtWidgets.QToolButton.InstantPopup)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.button, alignment=Qt.AlignLeft)
        layout.setContentsMargins(12, 12, 12, 12)

        self.categories = {
            "Row 1": list_of_bands,
            "Row 2": list_of_bands,
            "Row 3": list_of_bands,
        }

        self.menu = QtWidgets.QMenu(self.button)
        self.button.setMenu(self.menu)

        self.submenus = {}
        self._build_menu()

    def _build_menu(self):
        for cat, items in self.categories.items():
            sub = CheckableSubMenu(cat, self.menu)

            select_all = QAction("Select all", sub)
            clear_all = QAction("Clear all", sub)
            select_all.triggered.connect(lambda _, s=sub: (s.set_all(True), self._update_label()))
            clear_all.triggered.connect(lambda _, s=sub: (s.set_all(False), self._update_label()))
            sub.addAction(select_all)
            sub.addAction(clear_all)
            sub.addSeparator()

            for it in items:
                cb = sub.add_check_item(it, checked=False)
                cb.toggled.connect(self._update_label)
                cb.toggled.connect(lambda checked, c=cat, i=it: self.itemToggled.emit(c, i, checked))
                
            self.menu.addMenu(sub)
            self.submenus[cat] = sub

        self.menu.addSeparator()
        done_action = QAction("Done", self.menu)
        done_action.triggered.connect(self.menu.close)
        self.menu.addAction(done_action)

    def _update_label(self):
        parts = []
        for cat, sub in self.submenus.items():
            sel = sub.selected_texts()
            if sel:
                parts.append(f"{cat}({len(sel)})")
        self.button.setText(", ".join(parts) if parts else "Choose...")

    def selections(self):
        return {cat: sub.selected_texts() for cat, sub in self.submenus.items()}



class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, clipboard=None):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.status = self.statusBar()
        
        self.clipboard = clipboard

        title_strings = ["One-by-one classifier Euclid jpg edition"]
        self.at_launch = True
        if args.name is not None:
            self.name = args.name
            title_strings.append(self.name)
        else:
            self.name = ''
        self.setWindowTitle(' - '.join(title_strings))
        
        self.no_plotting_rows = 3

        self.defaults = {
                    'name': self.name,
                    'counter':0,
                    'legacysurvey':False,
                    'legacybigarea':False,
                    'legacyresiduals':False,
                    'prefetch':False,
                    'autonext':True,
                    'colormap':'gist_gray',
                    'scale':'log',
                    'keyboardshortcuts':False,
                    'colorbandsvisible':False,
                    'nisprgbvisible':False,
                    'no_rows': 1, #Make no assumptions on what are the bands.
                        } | {f'row_{i+1}' : '' for i in range(0, self.no_plotting_rows)}
        self.config_dict = self.load_dict()
        self.im = Image.fromarray(np.zeros((66,66),dtype=np.uint8))
        
        self.ds9_comm_backend = "xpa"
        self.is_ds9_open = False
        self.singlefetchthread_active = False
        self.background_downloading = self.config_dict['prefetch']
        self.colormap = self.config_dict['colormap']
        self.buttoncolor = "darkRed"
        self.buttonclasscolor = "darkRed"
        self.scale2funct = {'identity':identity,
                            'sqrt':np.sqrt,
                            'log':log,
                            'log10':log,
                            'cbrt':np.cbrt,
                            'asinh2':asinh2}
        self.scale = self.scale2funct[self.config_dict['scale']]


        self.stampspath = args.path

        self.all_bands = sorted(os.listdir(self.stampspath))
        self.all_bands = [elem for elem in self.all_bands if elem!='.DS_Store'] # Thank you Phil Holloway!

        # self.color_bands = args.color_bands.split(",")
        self.legacy_survey_path = LEGACY_SURVEY_PATH
        self.random_seed = args.seed

        self.pre_scalings = ['asinh', 'mtf'] # Write first the main scaling.
        self.bands = ['vis_only', 'vis_y', 'vis_y_h'] # Write first the main band.
        self.main_band = self.pre_scalings[0] + '_' + self.bands[0]


        self.paths_to_images = ([join(self.stampspath, pre_scaling+'_'+band) 
                                for pre_scaling in self.pre_scalings for band in self.bands])


        self.color_bands = [self.pre_scalings[1]+'_'+band for band in self.bands]
        self.composite_bands = [self.pre_scalings[0]+'_'+band for band in self.bands[1:]] 

        # self.band_types = ({self.main_band: MAIN_BAND} |
        #                   {band: COMPOSITE_BAND for band in self.composite_bands} |
        #                   {band: SINGLE_BAND for band in self.color_bands} | 
        #                   {band: EXTERNAL_BAND for band in self.external_bands})

        if args.fits is None:
            print("args.fits is None.")
            sys.exit()
        elif args.fits:
            print("At the moment, only jpg/png files are supported")
            sys.exit()
        else:
            print(f"Trying to load {args.extension} files")
            self.listimage = sum([glob.glob(join(stamps_path,f"*.{args.extension}")) for stamps_path in self.paths_to_images], [])
            # self.listimage = sorted(list(set(map( lambda s: s.split('/')[-1], self.listimage)))) # Removing paths.
            self.listimage = sorted(list(set(map(os.path.basename, self.listimage)))) # Removing paths and duplicates.


            self.filetype=_FILETYPE_COMPRESSED


        if len(self.listimage) < 1:
            print(f"Failed to find useful files in {args.path}")
            sys.exit()
        if self.config_dict['counter'] > len(self.listimage):
            self.config_dict['counter'] = 0
        
        if self.random_seed is not None:
            print(f"Shuffling with seed {self.random_seed}")
            rng = np.random.default_rng(self.random_seed)
            rng.shuffle(self.listimage) #inplace shuffling
        

        self.df = self.obtain_df()
        self.number_graded = 0
        self.COUNTER_MIN = 0
        self.COUNTER_MAX = len(self.listimage)
        self.filename = join(self.listimage[self.config_dict['counter']])
        # self.status.showMessage(self.listimage[self.config_dict['counter']],)


        main_layout = QtWidgets.QVBoxLayout(self._main)
        self.label_layout = QtWidgets.QHBoxLayout()
        self.label_layout_container = QtWidgets.QWidget()
        self.label_layout_container.setLayout(self.label_layout)

        self.plot_layout_area = QtWidgets.QVBoxLayout()
        self.plot_layout_area.setSpacing(0)
        self.plot_layout_area.setContentsMargins(0,0,0,0)

        self.plot_layout_area_container = QtWidgets.QWidget()
        self.plot_layout_area_container.setLayout(self.plot_layout_area)
        # self.plot_layout_area_container.setSizePolicy(
        #     QtWidgets.QSizePolicy.Ignored,
        #     QtWidgets.QSizePolicy.Ignored,
        # )

        self.plot_layout_rows_widgets = [QtWidgets.QWidget() for i in range(self.no_plotting_rows)]
        self.plot_layout_rows_layouts = [QtWidgets.QHBoxLayout(widget) for
                                         widget in self.plot_layout_rows_widgets]
        
        self.status_plot_rows = [ [] for _ in range(self.no_plotting_rows)]

        for layout in self.plot_layout_rows_layouts:
            layout.setSpacing(0)
            layout.setContentsMargins(0,0,0,0)

        for widget in self.plot_layout_rows_widgets:
            self.plot_layout_area.addWidget(widget,1)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                 QtWidgets.QSizePolicy.Expanding)

        button_layout = QtWidgets.QVBoxLayout()
        button_layout_container = QtWidgets.QWidget()
        button_layout_container.setLayout(button_layout)
        button_row0_layout = QtWidgets.QHBoxLayout()
        button_row10_layout = QtWidgets.QHBoxLayout()
        button_row11_layout = QtWidgets.QHBoxLayout()
        button_row2_layout = QtWidgets.QHBoxLayout()
        button_row3_layout = QtWidgets.QHBoxLayout()

        self.counter_widget = QtWidgets.QLabel("{}/{}".format(self.config_dict['counter']+1,self.COUNTER_MAX))
        self.counter_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed) #QLabels have different default size policy. Better to use the policy of buttons.
        self.counter_widget.setStyleSheet("font-size: 14px")
        
        self.label_plot = {band: QtWidgets.QLabel(f"{band}", alignment=Qt.AlignCenter) for band in [self.main_band]}
        self.label_plot[_BAND_FILENAMES_KEY] = BandNamesLabel(no_of_rows=self.no_plotting_rows,
                                                             alignment=Qt.AlignCenter)

        font = {band: self.label_plot[band].font() for band in [self.main_band, _BAND_FILENAMES_KEY]}
        for band in [self.main_band, _BAND_FILENAMES_KEY]:
            self.label_plot[band].setFont(font[band])

        self.label_layout.addWidget(self.label_plot[self.main_band])
        self.label_layout.addWidget(self.label_plot[_BAND_FILENAMES_KEY])


        self.size_of_figure_on_screen = 120
        default_params_figure = {'figsize': (self.size_of_figure_on_screen/100,
                                            self.size_of_figure_on_screen/100),
                                'layout':"constrained",
                                'facecolor' :'black'}

        self.row_figures = [{band: Figure(**default_params_figure) for band in self.all_bands}
                                 for i in range(self.no_plotting_rows)]
        self.row_canvas = [{band: FigureCanvas(self.row_figures[i][band]) for band in self.all_bands}
                                 for i in range(self.no_plotting_rows)] #Multiple rows different figures
                                 
        for row in range(self.no_plotting_rows):
            for band in self.all_bands:
                widget = self.row_canvas[row][band]
                widget.setStyleSheet('background-color: black')
                self.plot_layout_rows_layouts[row].addWidget(widget,1)
                widget.hide()

                # sizePolicy = QtWidgets.QSizePolicy.MinimumExpanding  
                # widget.setSizePolicy(sizePolicy,sizePolicy)
                # widget.setMinimumWidth(300) #Policy is implicitely set by the Matplotlib figure
                # widget.updateGeometry()



        self.axes = [{band: self.row_figures[i][band].subplots() for band in self.all_bands}
                                                                for i in range(self.no_plotting_rows)]
        

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

        self.cbnumberofrows = QtWidgets.QComboBox()
        self.cbnumberofrows.addItems(['1','2','3'])
        self.cbnumberofrows.setStyleSheet('background-color: gray')
        self.cbnumberofrows.setCurrentIndex(self.config_dict['no_rows']-1)  # index starts from 0
        self.cbnumberofrows.currentIndexChanged.connect(self.change_number_of_band_rows)
        self.show_number_of_rows(self.config_dict['no_rows']-1)
        list_button_row0_layout.append(self.cbnumberofrows)


        self.cbcontentofrows = MultiSelectDropdown(self.all_bands)
        self.cbcontentofrows.itemToggled.connect(self.change_bands_shown_in_row)   # connect signal to method
        list_button_row0_layout.append(self.cbcontentofrows)

        #  # Process the row information in the config.json
        # if join_list_of_lists(self.status_plot_rows) == '':
        #     self.status_plot_rows[0].append(self.main_band)
        
        initial_state_string = (self.config_dict['row_1'] + 
                self.config_dict['row_2'] +
                self.config_dict['row_3'])

        # print(initial_state_string)
        if initial_state_string == "":
            self.config_dict['row_1'] = self.main_band

        # print(f"Before {self.config_dict['row_1'] = }")
        # print(f"Before {self.config_dict['row_2'] = }")
        # print(f"Before {self.config_dict['row_3'] = }")

        # Manually activate the previous configuration
        for i in range(self.no_plotting_rows):
            for cb in self.cbcontentofrows.submenus[f'Row {i+1}']._checkboxes:
                # if cb.text() in self.status_plot_rows[i]:
                # print(i, f"{cb.text() = }")
                if cb.text() in self.config_dict[f'row_{i+1}']:
                    # print(f"{cb.text()=} was in config_dict")
                    cb.setChecked(True)
                else:
                    cb.setChecked(False)

        self.process_rows_in_config()


        self.bautopass = QtWidgets.QCheckBox("Auto-next")
        self.bautopass.clicked.connect(self.checkbox_auto_next)
        if self.config_dict['autonext']:
            self.bautopass.toggle()
        list_button_row0_layout.append(self.bautopass)

        self.bkeyboardshortcuts = QtWidgets.QCheckBox("Keyboard shortcuts")
        self.bkeyboardshortcuts.clicked.connect(self.checkbox_keyboard_shortcuts)
        if self.config_dict['keyboardshortcuts']:
            self.bkeyboardshortcuts.toggle()
        list_button_row0_layout.append(self.bkeyboardshortcuts)

        list_classifications = []
   
   
   
        self.bsurelens = QtWidgets.QPushButton('A/B [1]')
        self.original_button_style = self.bsurelens.styleSheet()
        # print(self.original_button_style)
        self.bsurelens.clicked.connect(partial(self.classify, 'A/B','A/B') )
        self.bsurelens.setFocusPolicy(Qt.NoFocus)
        # from pprint import pprint
        # bg = self.bsurelens.palette().color(QPalette.Button)
        # pprint(bg.name())
        list_classifications.append(self.bsurelens)

        self.bnonlens = QtWidgets.QPushButton('C/X [4]')
        self.bnonlens.clicked.connect(partial(self.classify, 'C/X','C/X'))
        # self.bnonlens.setFocusPolicy(Qt.NoFocus)

        list_classifications.append(self.bnonlens)


        self.dict_class2button = {
                                    'A': self.bsurelens,
                                    'B': self.bmaybelens,
                                    'C': self.bflexion,
                                    'X': self.bnonlens,

                                 'None':None}

        list_scales_buttons = []
        self.blinear = QtWidgets.QPushButton('Linear')
        self.blinear.clicked.connect(partial(self.set_scale,self.blinear,'identity'))
        list_scales_buttons.append(self.blinear)

        self.bsqrt = QtWidgets.QPushButton('Sqrt')
        self.bsqrt.clicked.connect(partial(self.set_scale,self.bsqrt,'sqrt'))
        list_scales_buttons.append(self.bsqrt)

        self.bcbrt = QtWidgets.QPushButton('Cbrt')
        self.bcbrt.clicked.connect(partial(self.set_scale,self.bcbrt,'cbrt'))
        list_scales_buttons.append(self.bcbrt)

        self.blog = QtWidgets.QPushButton('Log')
        self.blog.clicked.connect(partial(self.set_scale,self.blog,'log'))
        list_scales_buttons.append(self.blog)

        list_colormap_buttons = []
        self.bInverted = QtWidgets.QPushButton('Yarg')
        self.bInverted.clicked.connect(partial(self.set_colormap,self.bInverted,'gist_yarg'))
        list_colormap_buttons.append(self.bInverted)

        self.bBb8 = QtWidgets.QPushButton('Hot')
        self.bBb8.clicked.connect(partial(self.set_colormap,self.bBb8,'hot'))
        list_colormap_buttons.append(self.bBb8)

        self.bGray = QtWidgets.QPushButton('Gray')
        self.bGray.clicked.connect(partial(self.set_colormap,self.bGray,'gist_gray'))
        list_colormap_buttons.append(self.bGray)

        self.bViridis = QtWidgets.QPushButton('Viridis')
        self.bViridis.clicked.connect(partial(self.set_colormap,self.bViridis,'viridis'))
        list_colormap_buttons.append(self.bViridis)

        self.scale2button = {'identity':self.blinear,
                            'sqrt':self.bsqrt,
                            'log':self.blog,
                            'log10':self.blog,
                            'cbrt':self.bcbrt,
                            # 'asinh2': self.basinh
                            }
        self.colormap2button = {'gist_yarg':self.bInverted,
                                'hot':self.bBb8,
                                'gist_gray':self.bGray,
                                'viridis': self.bViridis}

        self.bactivatedclassification = None
        self.bactivatedsubclassification = None
        self.bactivatedscale = self.scale2button[self.config_dict['scale']]
        self.bactivatedcolormap = self.colormap2button[self.config_dict['colormap']]

        grade = self.df.at[self.config_dict['counter'],'classification']
        if grade is not None and grade != 'None' and grade != 'Empty':
            self.bactivatedclassification = self.dict_class2button[grade]
            self.bactivatedclassification.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))
 
        self.bactivatedscale.setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))
        self.bactivatedcolormap.setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))

        #Keyboard shortcuts
        self.ksurelens = QShortcut(QKeySequence('1'), self)
        self.ksurelens.activated.connect(partial(self.keyClassify, 'A','A'))

        self.kmaybelens = QShortcut(QKeySequence('2'), self)
        self.kmaybelens.activated.connect(partial(self.keyClassify, 'B','B'))

        self.kflexion = QShortcut(QKeySequence('3'), self)
        self.kflexion.activated.connect(partial(self.keyClassify, 'C','C'))

        self.knonlens = QShortcut(QKeySequence('4'), self)
        self.knonlens.activated.connect(partial(self.keyClassify, 'X','X'))

        self.kPrev = QShortcut(QKeySequence(QKeySequence.MoveToPreviousPage), self)
        self.kPrev.activated.connect(self.keyPrev)

        self.kNext = QShortcut(QKeySequence(QKeySequence.MoveToNextPage), self)
        self.kNext.activated.connect(self.keyNext)

        self.kPrev = QShortcut(QKeySequence('k'), self)
        self.kPrev.activated.connect(self.keyPrev)

        self.kNext = QShortcut(QKeySequence('j'), self)
        self.kNext.activated.connect(self.keyNext)

        self.kCopyRADec = QShortcut(QKeySequence('c'), self)
        self.kCopyRADec.activated.connect(self.copy_filename_to_keyboard)

        for button in list_button_row0_layout:
            button_row0_layout.addWidget(button)

        button_row0_layout.addWidget(self.counter_widget,alignment=Qt.AlignRight)

        for button in list_classifications:
            button_row10_layout.addWidget(button)

        for button in list_scales_buttons:
            button_row2_layout.addWidget(button)

        for button in list_colormap_buttons:
            button_row3_layout.addWidget(button)

        button_layout_spacing = 0
        button_layout.addLayout(button_row0_layout, button_layout_spacing)
        button_layout.addLayout(button_row10_layout, button_layout_spacing)
        button_layout.addLayout(button_row11_layout, button_layout_spacing)

            
        if self.filetype == _FILETYPE_FITS:
            button_layout.addLayout(button_row2_layout, button_layout_spacing)
            button_layout.addLayout(button_row3_layout, button_layout_spacing)
        else:
            button_layout.addLayout(button_row3_layout, button_layout_spacing)


        main_layout.addWidget(self.label_layout_container, 2)
        main_layout.addWidget(self.plot_layout_area_container, 88)
        # main_layout.addLayout(self.plot_layout_area, 88)
        main_layout.addWidget(button_layout_container, 10)
        # main_layout.addLayout(button_layout, 10)

        self.timer_0 = time()
        self.at_launch = False

    def change_bands_shown_in_row(self,category, item, checked ):
        # print(category, item, checked)
        row = int(category.split(' ')[-1]) - 1
        widget = self.row_canvas[row][item]
        if checked:
            # self.plot_layout_rows_layouts[row].addWidget(widget,1)
            self.plot_layout_rows_layouts[row].removeWidget(widget)
            self.plot_layout_rows_layouts[row].addWidget(widget,1)

            widget.show()
            self.status_plot_rows[row].append(item)
        else:
            widget.hide()
            self.status_plot_rows[row].remove(item)
            # widget.updateGeometry()
        self.save_dict()
        self.label_plot[_BAND_FILENAMES_KEY].updateText(self.status_plot_rows)

    def change_number_of_band_rows(self, number_of_rows_to_show):
        self.show_number_of_rows(int(number_of_rows_to_show))
        self.config_dict['no_rows'] = int(number_of_rows_to_show)+1
        self.save_dict()

        

    def show_number_of_rows(self,number_of_rows_to_show):
        for i, widget in enumerate(self.plot_layout_rows_widgets):
            widget.setVisible(i <= number_of_rows_to_show)
            # print(f"{widget.sizeHint()}")


    def process_rows_in_config(self):
        for i in range(self.no_plotting_rows):
            line_data = self.config_dict[f'row_{i+1}']
            self.config_dict[f'row_{i+1}'] = ''
            # print(f"{line_data = }")
            self.status_plot_rows[i] = (line_data.split(',') 
                                            if len(line_data) > 0 else [])
 

    def save_dict(self):
        # state_string = (self.config_dict['row_1'] + 
        #         self.config_dict['row_2'] +
        #         self.config_dict['row_3'])
        
        # if state_string == '':
        if not self.at_launch:
            for i in range(self.no_plotting_rows):
                self.config_dict[f'row_{i+1}'] = ','.join(self.status_plot_rows[i])

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
            # print(self.defaults)
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
        # clipboard = QClipboard()
        to_copy = f"{self.ra},{self.dec}"
        self.clipboard.setText(to_copy)
        self.status.showMessage(f'RA,Dec copied to clipboard: {self.ra},{self.dec}',10000)


    @Slot()
    def copy_filename_to_keyboard(self):
        # clipboard = QClipboard()
        to_copy = f"{self.filename}"
        self.clipboard.setText(to_copy)
        self.status.showMessage(f'Filename copied to clipboard: {self.filename}',10000)

    @Slot()
    def classify(self, grade, subgrade):
        t0 = time()
        cnt = self.config_dict['counter']# - 1
        assert self.df.at[cnt,'file_name'] == self.listimage[self.config_dict['counter']] #TODO handling this possibility better.
        self.df.at[cnt,'classification'] = grade
        # self.df.at[cnt,'subclassification'] = subgrade
        if self.filetype == _FILETYPE_FITS:
            self.df.at[cnt,'ra'] = self.ra
            self.df.at[cnt,'dec'] = self.dec
            self.df.at[cnt,'pixel_size'] = self.image_pixel_size
            self.df.at[cnt,'image_dim'] = self.image_size
        # self.df.at[cnt,'comment'] = grade
        self.df.at[cnt,'time'] += (time() - self.timer_0)
        self.timer_0 = time()
        self.df.to_csv(self.df_name)

        self.update_classification_buttons()
        # self.update_subclassification_buttoms()
        
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
        self.label_plot[_LEGACY_SURVEY_KEY].setText(title)
        self.ax[_LEGACY_SURVEY_KEY].cla()
        if savefile != self.legacy_filename:
            return
        self.ax[_LEGACY_SURVEY_KEY].imshow(mpimg.imread(savefile))
        self.ax[_LEGACY_SURVEY_KEY].set_axis_off()
        self.canvas[_LEGACY_SURVEY_KEY].draw()

    def plot_no_legacy_survey(self, title='Waiting for data',
                            colormap='Greys_r'):
        self.label_plot[_LEGACY_SURVEY_KEY].setText(title)
        self.ax[_LEGACY_SURVEY_KEY].cla()
        self.ax[_LEGACY_SURVEY_KEY].imshow(np.zeros(self.images[self.main_band].shape), cmap=colormap)
        self.ax[_LEGACY_SURVEY_KEY].set_axis_off()
        self.canvas[_LEGACY_SURVEY_KEY].draw()

    @Slot()
    def set_legacy_survey(self):
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


    @Slot()
    def checkbox_show_color_bands(self):
        self.config_dict['colorbandsvisible'] = not self.config_dict['colorbandsvisible']
        if not self.config_dict['colorbandsvisible']:
                self.plot()
                self.plot_layout_1_Widget.hide()
        else:
            if not self.color_bands_already_plotted:
                # for band in self.color_bands:
                #     self.plot_band(band)
                self.plot()
                self.color_bands_already_plotted = True
            self.plot_layout_1_Widget.show()




    @Slot()
    def checkbox_legacy_survey(self):
        if self.config_dict['legacysurvey']:
                self.label_plot[_LEGACY_SURVEY_KEY].hide()
                self.canvas[_LEGACY_SURVEY_KEY].hide()
        else:
                self.label_plot[_LEGACY_SURVEY_KEY].show()
                self.canvas[_LEGACY_SURVEY_KEY].show()
                self.set_legacy_survey()
        self.config_dict['legacysurvey'] = not self.config_dict['legacysurvey']

    @Slot()
    def checkbox_ls_change_area(self):
        self.config_dict['legacybigarea'] = not self.config_dict['legacybigarea']
        if self.config_dict['legacysurvey']:
            self.set_legacy_survey()

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
                    _VIS_RESAMPLED_BAND:12,
                    'H':12,
                    'J':12,
                    'Y':12,
                    }
        arguments = ["ds9", '-fits']
        for band in [self.main_band, _VIS_RESAMPLED_BAND, *self.color_bands]:
            filename = f"{join(self.stampspath,band,self.filename)}"
            arguments += [filename, '-zoom', 'to',str(band2zoom[band]), '-colorbar', 'no']
        print(" ".join(arguments))
        subprocess.Popen(arguments)

    @Slot()
    def viewls(self):
        webbrowser.open("https://www.legacysurvey.org/viewer?ra={}&dec={}&layer=ls-dr10-grz&zoom=16&spectra".format(self.ra,self.dec))

    @Slot()
    def viewESASky(self):
        website = f"https://sky.esa.int/esasky/?target={self.ra}%20{self.dec}&hips=PanSTARRS+DR1+color+(i%2C+r%2C+g)&fov=0.02&cooframe=J2000&sci=true&lang=en&"
        # website += "&euclid_image=perseus" #Use this to add the Euclid ERO overlay. Sadly, this is always centered on the same coordinate.
        webbrowser.open(website)

    @Slot()
    def set_scale(self, button, scale):
        if button != self.bactivatedscale:
            self.scale = self.scale2funct[scale]
            self.replot()
            button.setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))
            self.bactivatedscale.setStyleSheet("background-color : white;color : black;")
            self.bactivatedscale = button
            self.config_dict['scale']= scale
            self.save_dict()

    @Slot()
    def set_colormap(self, button, colormap):
        if button != self.bactivatedcolormap:
            self.config_dict['colormap'] = colormap
            self.replot()
            button.setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))
            # self.bactivatedcolormap.setStyleSheet("background-color : white;color : black;")
            self.bactivatedcolormap.setStyleSheet(self.original_button_style)
            self.bactivatedcolormap = button
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
        scale_min, scale_max = get_value_range_asymmetric(images,p_low,p_high,
                    pixel_boxsize_low=None)
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

    def plot(self, scale_min = None, scale_max = None, band = None):
        self.label_plot[self.main_band].setText(f"{self.listimage[self.config_dict['counter']]}")
        
        for band in self.all_bands:
            for row in range(self.no_plotting_rows):
                self.plot_band(band,row)

    def plot_old(self, scale_min = None, scale_max = None, band = None):
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
        
        for band in self.composite_bands[:-1]:
            band = band.replace(_BAND_FILENAMES_KEY,self.main_band)
            # label += f'{band}-'
        
        self.label_plot[_BAND_FILENAMES_KEY].updateText(self.config_dict['colorbandsvisible'],
                                                        self.config_dict['nisprgbvisible'])
        # self.label_plot[_BAND_FILENAMES_KEY].setText(label[:-1])
        # print(self.label_plot[_BAND_FILENAMES_KEY].text())

    def plot_band(self, band, row, scale_min = None, scale_max = None):
        # self.label_plot[band].setText(self.listimage[self.config_dict['counter']])
        ax = self.axes[row][band]
        ax.cla()
        get_radec = True if band == self.main_band else False
        if self.filetype == _FILETYPE_FITS:
            # image = self.load_fits(join(self.stampspath, band, self.filename),get_radec)
            print('No support for FITS files')
        else:
            image = np.asarray(Image.open(join(self.stampspath, band, self.filename)))
            self.image = np.copy(image)
            self.images[band] = np.copy(image)
            ax.imshow(image, origin='upper', cmap = self.config_dict['colormap'], vmin=0, vmax=255) #For jpg/pngs this is best.
        ax.set_axis_off() #Always before .draw()!
        # self.canvas[band].draw()
        self.row_canvas[row][band].draw()
        # for row in self.row_canvas:
        #     row[band].draw()

    def plot_band_old(self, band, scale_min = None, scale_max = None):
        # self.label_plot[band].setText(self.listimage[self.config_dict['counter']])
        self.ax[band].cla()
        get_radec = True if band == self.main_band else False
        if self.filetype == _FILETYPE_FITS:
            image = self.load_fits(join(self.stampspath, band, self.filename),get_radec)
            print('No support for FITS files')
        else:
            image = np.asarray(Image.open(join(self.stampspath, band, self.filename)))
            self.image = np.copy(image)
            self.images[band] = np.copy(image)
            self.ax[band].imshow(image, origin='upper', cmap = self.config_dict['colormap'], vmin=0, vmax=255) #For jpg/pngs this is best.
        self.ax[band].set_axis_off() #Always before .draw()!
        # self.canvas[band].draw()
        for row in self.row_canvas:
            row[band].draw()

    def plot_composite_band_old(self, composite_band, scale_min = None, scale_max = None):
        # base_bands = list(composite_band)
        
        self.ax[composite_band].cla()
        
        if self.filetype == _FILETYPE_FITS:
            print('FITS files are not supported')
            # if (not self.color_bands_already_plotted) or (_VIS_RESAMPLED_BAND in base_bands):
            #     images = {band: self.load_fits(join(self.stampspath, band, self.filename),get_radec=False) for band in base_bands}
            # else:
            #     images = self.images

            # image = self.prepare_composite_image(np.stack([images[band] for band in base_bands],axis=2))
            # self.ax[composite_band].imshow(image, origin='lower')
        else:
            image = np.asarray(Image.open(join(self.stampspath, composite_band, self.filename)))
            self.images[composite_band] = image.copy()
            self.ax[composite_band].imshow(image, origin='upper')

        self.ax[composite_band].set_axis_off() #Always before .draw()!
        # self.canvas[composite_band].draw()
        for row in self.row_canvas:
            row[composite_band].draw()

    def replot(self, scale_min = None, scale_max = None):

        if self.filetype == _FILETYPE_COMPRESSED:
            for band in self.all_bands:
                for row in range(self.no_plotting_rows):
                    self.plot_band(band,row)
        else:
            print("Only JPG/PNG format supported currently.")
            raise Exception('FITS files not suported')

    def replot_old(self, scale_min = None, scale_max = None):
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

    def replot_band(self, band, row, scale_min = None, scale_max = None):
        ax = self.axes[row][band]

        ax.cla()
        image = np.copy(self.images[band])
        if self.filetype == _FILETYPE_FITS:
            image = self.rescale_image(image, self.scale_mins[band], self.scale_maxs[band])
            ax.imshow(image,cmap=self.config_dict['colormap'], origin='lower')
        else:
            ax.imshow(image, origin='upper', cmap = self.config_dict['colormap'], vmin=0, vmax=255) #For jpg/pngs this is best.

        ax.set_axis_off()
        self.row_canvas[row][band].draw()

    def replot_band_old(self, band, scale_min = None, scale_max = None):
        # self.label_plot[band].setText(self.listimage[self.config_dict['counter']])
        self.ax[band].cla()
        image = np.copy(self.images[band])
        if self.filetype == _FILETYPE_FITS:
            image = self.rescale_image(image, self.scale_mins[band], self.scale_maxs[band])
            self.ax[band].imshow(image,cmap=self.config_dict['colormap'], origin='lower')
        else:
            self.ax[band].imshow(image, origin='upper', cmap = self.config_dict['colormap'], vmin=0, vmax=255) #For jpg/pngs this is best.

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
                # 'subclassification',
                # 'ra','dec',
                # 'comment',
                # 'image_dim',
                'time']
        df = pd.DataFrame(columns=dfc)
        df['file_name'] = self.listimage
        df['classification'] = ['Empty'] * len(self.listimage)
        # df['subclassification'] = ['Empty'] * len(self.listimage)
        if self.filetype == _FILETYPE_FITS:
            print(self.filetype )
            df['ra'] = np.full(len(self.listimage),np.nan)
            df['dec'] = np.full(len(self.listimage),np.nan)
        # df['comment'] = ['Empty'] * len(self.listimage)
        # df['image_dim'] = np.full(len(self.listimage),pd.NA)
        df['time'] = np.full(len(self.listimage),0.0)
        return df

    def go_to_counter_page(self):
        self.filename = self.listimage[self.config_dict['counter']]
        self.bottom_row_bands_already_plotted = False
        self.plot()
        # if self.config_dict['legacysurvey']:
        #     self.set_legacy_survey()
        self.update_classification_buttons()
        # self.update_subclassification_buttoms()
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


    def update_classification_buttons(self):
        grade = self.df.at[self.config_dict['counter'],'classification']


        if self.bactivatedclassification is not None:
            # self.bactivatedclassification.setStyleSheet("background-color : white;color : black;")
            self.bactivatedclassification.setStyleSheet(self.original_button_style)


        #if grade is not None and not np.isnan(float(grade)) and grade != 'None':
        if grade is not None and grade != 'None' and grade != 'Empty':
            button = self.dict_class2button[grade]
            if button is not None:
                button.setStyleSheet("QPushButton {{background-color : {};color : white;}}".format(self.buttonclasscolor))
                # self.change_button_colors(button,self.buttonclasscolor, 'lightBlue')
                
                self.bactivatedclassification = button

    def update_subclassification_buttoms(self):
        subgrade = self.df.at[self.config_dict['counter'],'subclassification']
        if self.bactivatedsubclassification is not None:
            self.bactivatedsubclassification.setStyleSheet("background-color : white;color : black;")

#        if subgrade is not None and not np.isnan(subgrade) and subgrade != 'None':
        if subgrade is not None and subgrade != 'None' and subgrade != 'Empty':
            button = self.dict_subclass2button[subgrade]
            if button is not None:
                button.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))
                self.bactivatedsubclassification = button

    def change_button_colors(self, button, textColor, backgroundColor):
        palette = button.palette()
        palette.setColor(QPalette.ButtonText, QColor(textColor))
        palette.setColor(QPalette.Button, QColor(backgroundColor))
        button.setAutoFillBackground(True)
        button.update()
        button.setPalette(palette)
        return button
            
if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    
    clipboard = QtWidgets.QApplication.clipboard()
    app = ApplicationWindow(clipboard=clipboard)
    # app = ApplicationWindow()
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
