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
#                     default=None)
parser.add_argument('-s',"--seed", help="seed used to shuffle the images.",type=int,
                    default=None)

args = parser.parse_args()

args.main_band = 'VIS'
args.color_bands = 'Y,J,H'
args.verbose = False
args.fits = None


LEGACY_SURVEY_PATH = './Legacy_survey/'
LEGACY_SURVEY_PIXEL_SIZE=0.262

SINGLE_BAND = 'single_band'
MAIN_BAND = 'main_band'
COMPOSITE_BAND = 'composite_band'
EXTERNAL_BAND = 'external_band'
_LEGACY_SURVEY_KEY = "Legacy Survey"
_VIS_RESAMPLED_BAND = 'I'

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
            except urllib.error.HTTPError:
                with open(self.savefile,'w') as f:
                    Image.fromarray(np.zeros((66,66),dtype=np.uint8)).save(f)
                # self.failed_download.emit('No Legacy Survey data available.')
                self.failed_download.emit()
        self.has_finished.emit()


class BandNamesLabel(QtWidgets.QLabel):
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

    def updateText(self,
                    color_bands_status,
                    standard_color_band_status):
        label = ''
        if color_bands_status:
            for band in self.color_bands:
                label += f"{band}-"
            label = label[:-1]+'\n'

        label += f'{self.main_band}-'
        label += self.resampled_color_band.replace(_VIS_RESAMPLED_BAND,self.main_band)

        if standard_color_band_status:
            label += f'-{self.standard_color_band}'

        self.setText(label)

class FetchThread(QThread):
    def __init__(self, df, initial_counter, parent=None):
            QThread.__init__(self, parent)

            self.df = df
            self.initial_counter = initial_counter
            self.legacy_survey_path = LEGACY_SURVEY_PATH
            self.stampspath = args.path
            self.listimage = sorted([os.path.basename(x) for x in glob.glob(join(self.stampspath,'*.fits'))])
            self.im = Image.fromarray(np.zeros((66,66),dtype=np.uint8))
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
        except urllib.error.HTTPError:
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
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.status = self.statusBar()

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
                    'prefetch':False,
                    'autonext':True,
                    'colormap':'gist_gray',
                    'scale':'log',
                    'keyboardshortcuts':False,
                    'colorbandsvisible':False,
                    'nisprgbvisible':False,
                        }
        self.config_dict = self.load_dict()
        self.im = Image.fromarray(np.zeros((66,66),dtype=np.uint8))

        self.ds9_comm_backend = "xpa"
        self.is_ds9_open = False
        self.singlefetchthread_active = False
        self.background_downloading = self.config_dict['prefetch']
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
        self.color_bands = args.color_bands.split(",")
        self.color_bands_vis = [_VIS_RESAMPLED_BAND,'Y','H']
        self.legacy_survey_path = LEGACY_SURVEY_PATH
        self.random_seed = args.seed

        color_bands_path = join(self.stampspath, f'[{",".join(self.color_bands+["VIS_resampled"])}]')
        base_band_path = join(self.stampspath, self.main_band)
        if args.fits is None:
            print("No filetype was specified, defaulting to .fits")
            # print(join(self.stampspath, f'[{",".join(self.bands)}]','*.fits'))
            

            self.listimage = sorted(set([os.path.basename(x) for x in (
                                            glob.glob(join(color_bands_path, "*.fits"))+
                                            glob.glob(join(base_band_path,'*.fits')) 
                                            )]))
            self.filetype='FITS'
            print(f"Classifying {len(self.listimage)} sources.")
            if len(self.listimage) == 0:
                print("No fits files were found, trying with .png, .jpg, and .jpeg")
                self.listimage = sorted([os.path.basename(x)
                                for x in (glob.glob(join(base_band_path, '*.png')) +
                                          glob.glob(join(base_band_path, '*.jpg')) +
                                          glob.glob(join(base_band_path, '*.jpeg')) +
                                          glob.glob(join(color_bands_path, '*.png')) +
                                          glob.glob(join(color_bands_path, '*.jpg')) +
                                          glob.glob(join(color_bands_path, '*.jpeg'))
                                         )])
                self.filetype='COMPRESSED'

        elif args.fits:
            self.listimage = sorted(set([os.path.basename(x) for x in (
                                            glob.glob(join(color_bands_path, "*.fits"))+
                                            glob.glob(join(base_band_path,'*.fits')) 
                                            )]))
            self.filetype='FITS'
        else:
            self.listimage = sorted([os.path.basename(x)
                                for x in (glob.glob(join(base_band_path, '*.png')) +
                                          glob.glob(join(base_band_path, '*.jpg')) +
                                          glob.glob(join(base_band_path, '*.jpeg')) +
                                          glob.glob(join(color_bands_path, '*.png')) +
                                          glob.glob(join(color_bands_path, '*.jpg')) +
                                          glob.glob(join(color_bands_path, '*.jpeg'))
                                         )])
            self.filetype='COMPRESSED'


        if len(self.listimage) < 1:
            sys.exit()
        if self.config_dict['counter'] > len(self.listimage):
            self.config_dict['counter'] = 0
        
        if self.random_seed is not None:
            print(f"Shuffling with seed {self.random_seed}")
            rng = np.random.default_rng(self.random_seed)
            rng.shuffle(self.listimage) #inplace shuffling

        self.composite_bands = [
                                "".join(self.color_bands_vis[::-1]),
                                "".join(self.color_bands[::-1]),
                                ] #For now, only one composite band.
        # self.external_bands = [_LEGACY_SURVEY_KEY]
        self.external_bands = [] #I deactivated LS for this version
        self.all_bands = [self.main_band,
                          *self.composite_bands,
                          *self.color_bands,
                          _VIS_RESAMPLED_BAND,
                          *self.external_bands]
        self.band_types = ({self.main_band: MAIN_BAND} |
                          {band: COMPOSITE_BAND for band in self.composite_bands} |
                          {band: SINGLE_BAND for band in self.color_bands} | 
                          {band: EXTERNAL_BAND for band in self.external_bands})

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
        self.plot_layout_0 = QtWidgets.QHBoxLayout()
        self.plot_layout_1_Widget = QtWidgets.QWidget()
        self.plot_layout_1 = QtWidgets.QHBoxLayout(self.plot_layout_1_Widget)
        button_layout = QtWidgets.QVBoxLayout()
        button_row0_layout = QtWidgets.QHBoxLayout()
        button_row10_layout = QtWidgets.QHBoxLayout()
        button_row11_layout = QtWidgets.QHBoxLayout()
        button_row2_layout = QtWidgets.QHBoxLayout()
        button_row3_layout = QtWidgets.QHBoxLayout()

        self.counter_widget = QtWidgets.QLabel("{}/{}".format(self.config_dict['counter']+1,self.COUNTER_MAX))
        self.counter_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed) #QLabels have different default size policy. Better to use the policy of buttons.
        self.counter_widget.setStyleSheet("font-size: 14px")
        
        # self.label_plot = {band: QtWidgets.QLabel(f"{self.listimage[self.config_dict['counter']]} - {band}", alignment=Qt.AlignCenter) for band in self.all_bands}
        band2bandname_dict = {band: band for band in self.all_bands}
        band2bandname_dict['HYI'] = 'HYVIS'

        # self.label_plot = {band: QtWidgets.QLabel(f"{band}", alignment=Qt.AlignCenter) for band in [self.main_band, _VIS_RESAMPLED_BAND]}
        self.label_plot = {band: QtWidgets.QLabel(f"{band}", alignment=Qt.AlignCenter) for band in [self.main_band]}
        self.label_plot[_VIS_RESAMPLED_BAND] = BandNamesLabel(self.main_band,
                                                             self.color_bands,
                                                             self.composite_bands[0],
                                                             self.composite_bands[1],
                                                             alignment=Qt.AlignCenter)
        # print(f"{self.all_bands = }")
        font = {band: self.label_plot[band].font() for band in [self.main_band, _VIS_RESAMPLED_BAND]}
        for band in [self.main_band, _VIS_RESAMPLED_BAND]:
            font[band].setPointSize(16)
            self.label_plot[band].setFont(font[band])

        # self.label_layout.addWidget(self.label_plot[_LEGACY_SURVEY_KEY])
        self.label_layout.addWidget(self.label_plot[self.main_band])
        self.label_layout.addWidget(self.label_plot[_VIS_RESAMPLED_BAND])


        self.plot_layout_area.setSpacing(0)
        self.plot_layout_area.setContentsMargins(0,0,0,0)
        self.plot_layout_0.setSpacing(0)
        self.plot_layout_0.setContentsMargins(0,0,0,0)
        self.plot_layout_1.setSpacing(0)
        self.plot_layout_1.setContentsMargins(0,0,0,0)


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
        for band in [self.main_band, *self.composite_bands]:
            self.canvas[band].setStyleSheet('background-color: black')
            self.plot_layout_0.addWidget(self.canvas[band],1)

        for band in self.color_bands:
            self.canvas[band].setStyleSheet('background-color: black')
            self.plot_layout_1.addWidget(self.canvas[band],1)

        for band in self.external_bands:
            self.canvas[band].setStyleSheet('background-color: black')
            self.plot_layout_0.addWidget(self.canvas[band],1)

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

        self.bds9 = QtWidgets.QPushButton('ds9')
        self.bds9.clicked.connect(self.open_ds9)
        if self.filetype != 'FITS':
            self.bds9.setEnabled(False)
        list_button_row0_layout.append(self.bds9)

        # self.bviewls = QtWidgets.QPushButton('Open LS')
        # self.bviewls.clicked.connect(self.viewls)
        # if self.filetype != 'FITS':
        #     self.bviewls.setEnabled(False)
        # list_button_row0_layout.append(self.bviewls)

        self.bviewESA = QtWidgets.QPushButton('Open ESASky')
        self.bviewESA.clicked.connect(self.viewESASky)
        if self.filetype != 'FITS':
            self.bviewESA.setEnabled(False)
        list_button_row0_layout.append(self.bviewESA)

        self.bhidecolorbands = QtWidgets.QCheckBox('Show NISP bands')
        self.bhidecolorbands.clicked.connect(self.checkbox_show_color_bands)
        if self.filetype == 'FITS':
            if not self.config_dict['colorbandsvisible']:
                    self.plot_layout_1_Widget.hide()
            else:
                    self.plot_layout_1_Widget.show()
                    self.bhidecolorbands.toggle()
        else:
            self.config_dict['colorbandsvisible'] = False
            self.bhidecolorbands.setEnabled(False)
            self.plot_layout_1_Widget.hide()
        list_button_row0_layout.append(self.bhidecolorbands)


        self.bshownisprgb = QtWidgets.QCheckBox('Show NISP RGB')
        self.bshownisprgb.clicked.connect(self.checkbox_show_nisp_band)
        if self.filetype == 'FITS':
            if not self.config_dict['nisprgbvisible']:
                    self.canvas[self.composite_bands[-1]].hide()
            else:
                    self.canvas[self.composite_bands[-1]].show()
                    self.bshownisprgb.toggle()
        else:
            self.config_dict['nisprgbvisible'] = False
            self.bshownisprgb.setEnabled(False)
            self.canvas[self.composite_bands[-1]].hide()
        list_button_row0_layout.append(self.bshownisprgb)

        # self.blegsur = QtWidgets.QCheckBox('Legacy Survey (LS)')
        # self.blegsur.clicked.connect(self.checkbox_legacy_survey)
        # if self.filetype == 'FITS':
        #     if not self.config_dict['legacysurvey']:
        #             self.label_plot[_LEGACY_SURVEY_KEY].hide()
        #             self.canvas[_LEGACY_SURVEY_KEY].hide()
        #     else:
        #             self.label_plot[_LEGACY_SURVEY_KEY].show()
        #             self.canvas[_LEGACY_SURVEY_KEY].show()
        #             self.blegsur.toggle()
        #             self.set_legacy_survey()
        # else:
        #     self.config_dict['legacysurvey'] = False
        #     self.blegsur.setEnabled(False)
        #     self.label_plot[_LEGACY_SURVEY_KEY].hide()
        #     self.canvas[_LEGACY_SURVEY_KEY].hide()
        # list_button_row0_layout.append(self.blegsur)


    #     self.blsarea = QtWidgets.QCheckBox("Large FoV")
    #     self.blsarea.clicked.connect(self.checkbox_ls_change_area)
    #     if self.filetype == 'FITS':
    #         if self.config_dict['legacybigarea']:
    #             self.blsarea.toggle()
    #             if self.config_dict['legacysurvey']:
    #                 self.set_legacy_survey()
    #     else:
    #         self.blsarea.setEnabled(False)
    #         self.config_dict['legacybigarea'] = False
    #     list_button_row0_layout.append(self.blsarea)

    #     self.blsresidual = QtWidgets.QCheckBox("Residuals")
    #     self.blsresidual.clicked.connect(self.checkbox_ls_use_residuals)
    #     if self.filetype == 'FITS':
    #         if self.config_dict['legacyresiduals']:
    #             self.blsresidual.toggle()
    #             if self.config_dict['legacysurvey']:
    #                 self.set_legacy_survey()
    #     else:
    #         self.blsresidual.setEnabled(False)
    #         self.config_dict['legacyresiduals'] = False
    #     list_button_row0_layout.append(self.blsresidual)

    #     self.bprefetch = QtWidgets.QCheckBox("Pre-fetch")
    #     self.bprefetch.clicked.connect(self.prefetch_legacysurvey)
    #     if self.filetype == 'FITS':
    #         if self.config_dict['prefetch']:
    #             self.config_dict['prefetch'] = False
    #             self.prefetch_legacysurvey()
    #             self.bprefetch.toggle()
    #     else:
    #         self.bprefetch.setEnabled(False)
    #         self.config_dict['prefetch'] = False
    #     list_button_row0_layout.append(self.bprefetch)


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
        self.bsurelens = QtWidgets.QPushButton('A')
        self.bsurelens.clicked.connect(partial(self.classify, 'A','A') )
        list_classifications.append(self.bsurelens)

        self.bmaybelens = QtWidgets.QPushButton('B')
        self.bmaybelens.clicked.connect(partial(self.classify, 'B','B'))
        list_classifications.append(self.bmaybelens)

        self.bflexion = QtWidgets.QPushButton('C')
        self.bflexion.clicked.connect(partial(self.classify, 'C','C'))
        list_classifications.append(self.bflexion)

        self.bnonlens = QtWidgets.QPushButton('X')
        self.bnonlens.clicked.connect(partial(self.classify, 'X','X'))
        list_classifications.append(self.bnonlens)

        # list_subclassifications = []
        # self.bMerger = QtWidgets.QPushButton('Merger')
        # self.bMerger.clicked.connect(partial(self.classify, 'X','Merger') )
        # list_subclassifications.append(self.bMerger)

        # self.bSpiral = QtWidgets.QPushButton('Spiral')
        # self.bSpiral.clicked.connect(partial(self.classify, 'X','Spiral'))
        # list_subclassifications.append(self.bSpiral)

        # self.bRing = QtWidgets.QPushButton('Ring')
        # self.bRing.clicked.connect(partial(self.classify, 'X','Ring'))
        # list_subclassifications.append(self.bRing)

        # self.bElliptical = QtWidgets.QPushButton('Elliptical')
        # self.bElliptical.clicked.connect(partial(self.classify, 'X','Elliptical'))
        # list_subclassifications.append(self.bElliptical)

        # self.bDisc = QtWidgets.QPushButton('Disc')
        # self.bDisc.clicked.connect(partial(self.classify, 'X','Disc'))
        # list_subclassifications.append(self.bDisc)

        # self.bEdgeon = QtWidgets.QPushButton('Edge-on')
        # self.bEdgeon.clicked.connect(partial(self.classify, 'X','Edge-on'))
        # list_subclassifications.append(self.bEdgeon)

        self.dict_class2button = {
                                 'A':self.bsurelens,
                                  'B':self.bmaybelens,
                                  'C':self.bflexion,
                                  'X':self.bnonlens,
                                 'SL':self.bsurelens,
                                  'ML':self.bmaybelens,
                                  'FL':self.bflexion,
                                  'NL':self.bnonlens,

                                 'None':None}

        # self.dict_subclass2button = {'Merger':self.bMerger,
        #                           'Spiral':self.bSpiral,
        #                           'Ring':self.bRing,
        #                           'Elliptical':self.bElliptical,
        #                           'Disc':self.bDisc,
        #                           'Edge-on':self.bEdgeon,
        #                           'A':None,
        #                           'B':None,
        #                           'C':None,
        #                           'X':None,
        #                           'SL':None,
        #                           'ML':None,
        #                           'FL':None,
        #                           'NL':None,
        #                           'None':None}

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
        self.bInverted = QtWidgets.QPushButton('Inverted')
        self.bInverted.clicked.connect(partial(self.set_colormap,self.bInverted,'gist_yarg'))
        list_colormap_buttons.append(self.bInverted)

        self.bBb8 = QtWidgets.QPushButton('Bb8')
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
 
        # subgrade = self.df.at[self.config_dict['counter'],'subclassification']
        # if subgrade is not None and subgrade != 'None' and grade != 'Empty':
        #     self.bactivatedsubclassification = self.dict_subclass2button[subgrade]
        #     if self.bactivatedsubclassification is not None:
        #         self.bactivatedsubclassification.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))


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

#         self.kMerger = QShortcut(QKeySequence('a'), self)
#         self.kMerger.activated.connect(partial(self.keyClassify, 'X','Merger'))

#         self.kSpiral = QShortcut(QKeySequence('s'), self)
#         self.kSpiral.activated.connect(partial(self.keyClassify, 'X','Spiral'))

#         self.kRing = QShortcut(QKeySequence('d'), self)
#         self.kRing.activated.connect(partial(self.keyClassify, 'X','Ring'))

#         self.kElliptical = QShortcut(QKeySequence('f'), self)
#         self.kElliptical.activated.connect(partial(self.keyClassify, 'X','Elliptical'))

#         self.kDisc = QShortcut(QKeySequence('g'), self)
#         self.kDisc.activated.connect(partial(self.keyClassify, 'X','Disc'))

#         self.kEdgeon = QShortcut(QKeySequence('h'), self)
#         self.kEdgeon.activated.connect(partial(self.keyClassify, 'X','Edge-on'))


        self.kNext = QShortcut(QKeySequence(QKeySequence.MoveToPreviousPage), self)
        self.kNext.activated.connect(self.prev)

        self.kNext = QShortcut(QKeySequence(QKeySequence.MoveToNextPage), self)
        self.kNext.activated.connect(self.next)

        self.kNext = QShortcut(QKeySequence('k'), self)
        self.kNext.activated.connect(self.prev)

        self.kNext = QShortcut(QKeySequence('j'), self)
        self.kNext.activated.connect(self.next)


        self.kCopyRADec = QShortcut(QKeySequence(QKeySequence.Copy), self)
        self.kCopyRADec.activated.connect(self.copy_RADec_to_keyboard)

        self.kCopyRADec = QShortcut(QKeySequence('c'), self)
        self.kCopyRADec.activated.connect(self.copy_filename_to_keyboard)

        for button in list_button_row0_layout:
            button_row0_layout.addWidget(button)

        button_row0_layout.addWidget(self.counter_widget,alignment=Qt.AlignRight)

        for button in list_classifications:
            button_row10_layout.addWidget(button)

        # for button in list_subclassifications:
        #     button_row11_layout.addWidget(button)

        for button in list_scales_buttons:
            button_row2_layout.addWidget(button)

        for button in list_colormap_buttons:
            button_row3_layout.addWidget(button)

        button_layout_spacing = 0
        button_layout.addLayout(button_row0_layout, button_layout_spacing)
        button_layout.addLayout(button_row10_layout, button_layout_spacing)
        button_layout.addLayout(button_row11_layout, button_layout_spacing)
        if self.filetype == 'FITS':
            button_layout.addLayout(button_row2_layout, button_layout_spacing)
            button_layout.addLayout(button_row3_layout, button_layout_spacing)
        else:
            print("Use fits images to change colormap and colorscale.")


        # self.plot_layout_area.addLayout(self.plot_layout_0,1,0)
        # self.plot_layout_area.addWidget(self.plot_layout_1_Widget,0,0)

        self.plot_layout_area.addWidget(self.plot_layout_1_Widget,1)
        self.plot_layout_area.addLayout(self.plot_layout_0,1,)

        main_layout.addLayout(self.label_layout, 2)
        main_layout.addLayout(self.plot_layout_area, 88)
        main_layout.addLayout(button_layout, 10)

        self.timer_0 = time()

    @Slot()
    def prefetch_legacysurvey(self):
        if self.config_dict['prefetch']:
            self.fetchthread.terminate()
            self.config_dict['prefetch'] = False
        else:
            self.fetchthread = FetchThread(self.df,self.config_dict['counter'],) #Always store in an object.
            self.fetchthread.finished.connect(self.fetchthread.deleteLater)
            self.fetchthread.setTerminationEnabled(True)
            self.fetchthread.start()
            self.config_dict['prefetch'] = True

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
    def copy_RADec_to_keyboard(self):
        clipboard = QClipboard()
        to_copy = f"{self.ra},{self.dec}"
        clipboard.setText(to_copy)
        self.status.showMessage(f'RA,Dec copied to clipboard: {self.ra},{self.dec}',10000)


    @Slot()
    def copy_filename_to_keyboard(self):
        clipboard = QClipboard()
        to_copy = f"{self.filename}"
        clipboard.setText(to_copy)
        self.status.showMessage(f'Filename copied to clipboard: {self.filename}',10000)

    @Slot()
    def classify(self, grade, subgrade):
        t0 = time()
        cnt = self.config_dict['counter']# - 1
        assert self.df.at[cnt,'file_name'] == self.listimage[self.config_dict['counter']] #TODO handling this possibility better.
        self.df.at[cnt,'classification'] = grade
        # self.df.at[cnt,'subclassification'] = subgrade
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
    def checkbox_show_nisp_band(self):
        self.config_dict['nisprgbvisible'] = not self.config_dict['nisprgbvisible']
        relevantWidget = self.canvas[self.composite_bands[-1]]
        if not self.config_dict['nisprgbvisible']:
            # print(self.canvas[self.composite_bands[-1]].sizePolicy())
            # print(self.canvas[self.composite_bands[-1]].size())
            
            relevantWidget.hide()
            self.plot_layout_0.removeWidget(relevantWidget)
            self.label_plot[_VIS_RESAMPLED_BAND].updateText(self.config_dict['colorbandsvisible'],
                                                        self.config_dict['nisprgbvisible'])
            # for band in [self.main_band, *self.composite_bands]:
            #     widget = self.canvas[band]
            #     print(band, widget.minimumSize(), widget.sizeHint(), widget.sizePolicy())

            # for band in [self.main_band, *self.composite_bands]:
            #     self.canvas[band].hide()
            #     self.plot_layout_0.removeWidget(self.canvas[band])
            # for band in [self.main_band, *self.composite_bands[:-1]]:
            #     self.canvas[band].setStyleSheet('background-color: black')
            #     self.plot_layout_0.addWidget(self.canvas[band])
            #     self.canvas[band].show()

        else:
            relevantWidget.show()
            self.plot_layout_0.addWidget(relevantWidget,1)
            self.label_plot[_VIS_RESAMPLED_BAND].updateText(self.config_dict['colorbandsvisible'],
                                                        self.config_dict['nisprgbvisible'])
            # self.clear_layout()
            # relevantWidget.
            # self.canvas[self.composite_bands[-1]].resize(self.canvas[self.main_band].width(),
            #                                              self.canvas[self.main_band].height())
            
            # for band in [self.main_band, *self.composite_bands]:
            #     self.canvas[band].hide()
            #     self.plot_layout_0.removeWidget(self.canvas[band])

            # for band in [self.main_band, *self.composite_bands]:
            #     self.canvas[band].show()
                
            # for band in [self.main_band, *self.composite_bands]:
            #     self.plot_layout_0.addWidget(self.canvas[band],1)
        # self.updateGeometry()
            


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
            self.bactivatedcolormap.setStyleSheet("background-color : white;color : black;")
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
                                color_bkg_level = -0.05):
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
            band = band.replace(_VIS_RESAMPLED_BAND,self.main_band)
            # label += f'{band}-'
        
        self.label_plot[_VIS_RESAMPLED_BAND].updateText(self.config_dict['colorbandsvisible'],
                                                        self.config_dict['nisprgbvisible'])
        # self.label_plot[_VIS_RESAMPLED_BAND].setText(label[:-1])
        # print(self.label_plot[_VIS_RESAMPLED_BAND].text())

    def plot_band(self, band, scale_min = None, scale_max = None):
        # self.label_plot[band].setText(self.listimage[self.config_dict['counter']])
        self.ax[band].cla()
        get_radec = True if band == self.main_band else False
        if self.filetype == 'FITS':
            image = self.load_fits(join(self.stampspath, band, self.filename),get_radec)
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
            image = np.asarray(Image.open(self.filename))
            self.image = np.copy(image)
            self.ax[band].imshow(image, origin='upper') #For pngs this is best.
        self.ax[band].set_axis_off() #Always before .draw()!
        self.canvas[band].draw()

    def plot_composite_band(self, composite_band, scale_min = None, scale_max = None):
        # base_bands = [band if band != _VIS_RESAMPLED_BAND else 'VIS' for band in list(composite_band)]
        base_bands = list(composite_band)
        # print(base_bands)
        if len(base_bands) != 3:
            print(f"RGB image requires exactly 3 images. Bands provided: {base_bands}")
        
        # self.label_plot[composite_band].setText(self.listimage[self.config_dict['counter']])
        self.ax[composite_band].cla()
        
        if self.filetype == 'FITS':
            if (not self.color_bands_already_plotted) or (_VIS_RESAMPLED_BAND in base_bands):
                images = {band: self.load_fits(join(self.stampspath, band, self.filename),get_radec=False) for band in base_bands}
            else:
                images = self.images
            image = self.prepare_composite_image(np.stack([images[band] for band in base_bands],axis=2))
            self.ax[composite_band].imshow(image, origin='lower')
        else:
            raise Exception("Color RPGs in the form of PNGs are no supported in the ERO edition.")
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
        if self.filetype == 'FITS':
            image = self.rescale_image(image, self.scale_mins[band], self.scale_maxs[band])
            self.ax[band].imshow(image,cmap=self.config_dict['colormap'], origin='lower')
        else:
            self.ax[band].imshow(image, origin='lower')
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
            df = pd.read_csv(self.df_name)
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
                'ra','dec',
                # 'comment',
                'image_dim',
                'time']
        df = pd.DataFrame(columns=dfc)
        df['file_name'] = self.listimage
        df['classification'] = ['Empty'] * len(self.listimage)
        # df['subclassification'] = ['Empty'] * len(self.listimage)
        df['ra'] = np.full(len(self.listimage),np.nan)
        df['dec'] = np.full(len(self.listimage),np.nan)
        # df['comment'] = ['Empty'] * len(self.listimage)
        df['image_dim'] = np.full(len(self.listimage),pd.NA)
        df['time'] = np.full(len(self.listimage),0)
        return df

    def go_to_counter_page(self):
        self.filename = self.listimage[self.config_dict['counter']]
        self.bottom_row_bands_already_plotted = False
        self.plot()
        # if self.config_dict['legacysurvey']:
        #     self.set_legacy_survey()
        self.update_classification_buttoms()
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


    def update_classification_buttoms(self):
        grade = self.df.at[self.config_dict['counter'],'classification']


        if self.bactivatedclassification is not None:
            self.bactivatedclassification.setStyleSheet("background-color : white;color : black;")

        #if grade is not None and not np.isnan(float(grade)) and grade != 'None':
        if grade is not None and grade != 'None' and grade != 'Empty':
            button = self.dict_class2button[grade]
            if button is not None:
                button.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))
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

            
if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    
    app = ApplicationWindow()
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
