# This Python file uses the following encoding: utf-8
import argparse
from astropy.units.physical import _standardize_physical_type_names

from astropy.wcs import WCS
from astropy.io import fits

from functools import partial

import glob
import json

from matplotlib.figure import Figure
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from PIL import Image

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Slot, QObject, QThread, Signal, QEvent, QSize
from PySide6.QtGui import QPixmap, QFont, QKeySequence, QShortcut, QIntValidator

import shutil

import os
from os.path import join

import re
import subprocess
import sys
from time import time
import urllib
import webbrowser


parser = argparse.ArgumentParser(description='Configure the parameters of the execution.')
parser.add_argument('-p',"--path", help="Path to the images to inspect.",
                    default="Stamps_to_inspect")
parser.add_argument('-N',"--name", help="Name of the classifying session.",
                    default=None)
parser.add_argument('-b',"--main_band", help='High resolution band. Example: "VIS"',
                    default="VIS")
parser.add_argument('-B',"--color_bands", help='Comma-separated photometric bands, Bluer to Redder. Example: "Y,J,H"',
                    default="Y,J,H")
parser.add_argument('-l',"--ncols","--gridsize", help="Number of columns per page.",type=int,
                    default=5)
parser.add_argument('-m',"--nrows", 
                    help="Number of rows per page. Leave unset if you want a squared grid",type=int,
                    default=8)
parser.add_argument('-s',"--seed", help="Seed used to shuffle the images.",type=int,
                    default=None)
parser.add_argument("--minimum_size", help="Minimum size of the stamps in the mosaic. The optimal value depends on your screen and the stampsize.",type=int,
                    default=None)
parser.add_argument("--printname", help="Whether to print the name when you click.",
                    action=argparse.BooleanOptionalAction,
                    default=False)
parser.add_argument("--page", help="Initial page.",type=int,
                    default=None)
parser.add_argument('--resize',
                    help="Set to allow the resizing of the stamps with the window.",
                    action=argparse.BooleanOptionalAction,
                    default=False)
parser.add_argument('--fits',
                    help=("forces app to only use fits (--fits) "+
                          "or png/jp(e)g (--no-fits). "+
                          "If unset, the app searches for fits files "+
                          "in the path, but defaults to png/jp(e)g "+
                          "if no fits files are found."),
                    action=argparse.BooleanOptionalAction,
                    default=None)
# parser.add_argument('--crop',
#                     help="Lenth of the side of the cropped cutout in arcsec. Defaults to the whole frame",
#                     type=float,
#                     default=None)


args = parser.parse_args()

C_INTERESTING = 2
C_LENS = 1
C_UNINTERESTING = 0


SINGLE_BAND = 'single_band'
MAIN_BAND = 'main_band'
COMPOSITE_BAND = 'composite_band'
EXTERNAL_BAND = 'external_band'
_VIS_RESAMPLED_BAND = 'I'

def identity(x):
    return x

def log_0(x):
    "Simple log base 1000 function that ignores numbers less than 0"
    return np.log(x, out=np.zeros_like(x), where=(x>0)) / np.log(1000)

def log(x,a=1000):
    "Simple log base 1000 function that ignores numbers less than 0"
    return np.log(a*x+1) / np.log(a)

def asinh2(x):
    return np.arcsinh(10*x)/3


def get_value_range_asymmetric(x, q_low=1, q_high=1,
                              ):
    
    low = np.nanpercentile(x, q_low)
    
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
        # print(re_search.span()[-1], len(latest_filename))
        re_search = re_pattern.search(latest_filename, re_search.span()[-1])
        if re_search is None:
            return initial_iteration
    if re_search.span()[-1] == len(latest_filename): #at this point, re_search cannot be None
        re_match = re_search[1]
    try:
        int_match = int(re_match)
    except:
        return initial_iteration
    return "-({})".format(int_match+1)

def iloc_to_page_and_grid_pos(iloc, gridarea):
    return iloc // gridarea, iloc % gridarea

class LabelledIntField(QtWidgets.QWidget):
    "Widget for the page number."
    "https://www.fundza.com/pyqt_pyside2/pyqt5_int_lineedit/index.html"
    def __init__(self, title, initial_value,  total_pages):
        QtWidgets.QWidget.__init__(self)
        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        
        self.label = QtWidgets.QLabel()
        self.label.setText(title)
        # self.label.setFixedWidth(100)
        self.label.setFont(QFont("Arial",20,weight=QFont.Bold))
        layout.addWidget(self.label)
        
        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setFixedWidth(50)
        self.lineEdit.setValidator(QIntValidator(1,total_pages))
        self.lineEdit.setText(str(initial_value+1))
        self.lineEdit.setFont(QFont("Arial",20))
        self.lineEdit.setStyleSheet('background-color: black; color: gray')
        self.lineEdit.setAlignment(Qt.AlignRight)
        layout.addWidget(self.lineEdit)

        self.total_pages = QtWidgets.QLabel()
        self.total_pages.setText("/ "+str(total_pages))
        self.total_pages.setFont(QFont("Arial",20))
        layout.addWidget(self.total_pages)

        # layout.addStretch()

    def setInputText(self, input):
        self.lineEdit.setText(str(input+1))
        
    def getValue(self):
        return int(self.lineEdit.text())-1

class NamedLabel(QtWidgets.QWidget):
    "Widget to show unclickable label."
    "https://www.fundza.com/pyqt_pyside2/pyqt5_int_lineedit/index.html"
    def __init__(self, title, initial_value):
        QtWidgets.QWidget.__init__(self)
        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        
        self.name = QtWidgets.QLabel()
        self.name.setText(title)
        # self.title.setFixedWidth(100)
        self.name.setFont(QFont("Arial",20,weight=QFont.Bold))
        layout.addWidget(self.name)
        
        self.label = QtWidgets.QLineEdit(self)
        self.label.setFixedWidth(50)
        self.label.setEnabled(False)
        self.label.setText(str(initial_value))
        self.label.setFont(QFont("Arial",20))
        self.label.setStyleSheet('background-color: black; color: gray')
        self.label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.label)

    def setText(self, input):
        self.label.setText(str(input))

        
    def getValue(self):
        return int(self.lineEdit.text())-1

class AlignDelegate(QtWidgets.QStyledItemDelegate):
    "https://stackoverflow.com/a/54262963/10555034"
    def initStyleOption(self, option, index):
        super(AlignDelegate, self).initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter


class MiniMosaicLabels(QtWidgets.QLabel):
    def __init__(self,
                aspectRatioPolicy,
                minimum_size,
                sizePolicy,
                name = None,
                parent=None):
        QtWidgets.QLabel.__init__(self, parent)
        self.name = name
        # test_sizePolicy = QtWidgets.QSizePolicy(sizePolicy,sizePolicy)

        # test_sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        # test_sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        # test_sizePolicy.setHeightForWidth(True)
        # test_sizePolicy.transpose()
        # self.aspectRatioPolicy = Qt.KeepAspectRatio
        # self.aspectRatioPolicy = Qt.KeepAspectRatioByExpanding
        # self.aspectRatioPolicy = Qt.IgnoreAspectRatio
        self.aspectRatioPolicy = aspectRatioPolicy
        # self.setMinimumSize(minimum_size,minimum_size)
        # print()
        self.setSizePolicy(sizePolicy,sizePolicy)    
        # self.setSizePolicy(test_sizePolicy)    
        # print(self.name,f"{self.hasHeightForWidth() = }")
        # print(f"{self.sizePolicy()}")
        # print(self.parent())
        # print(self.name, self.maximumSize())
        self.setScaledContents(False)
        self.updateGeometry()

    # def sizeHint(self):
    #     limiting_size = self.scaledPixmap().size().toTuple()
    #     return min(limiting_size)
        
    def resizeEvent(self, event):
        # pixmap_length = min(self.size().toTuple())
        # pixmap_size = QSize(pixmap_length, pixmap_length)
        self.setPixmap(self._pixmap.scaled(
                            self.width(), self.height(),
                            # 10, 50,
                            # pixmap_length, pixmap_length,
                            self.aspectRatioPolicy
                            # Qt.IgnoreAspectRatio,
                            ))
        # self.setPixmap(self.scaledPixmap())
        # self.updateGeometry()
        # print(self.name)

    # def scaledPixmap(self):
    #     # print(f"About to scale {self.size().toTuple() = }")
    #     return self._pixmap.scaled(self.size(),
    #                                       Qt.KeepAspectRatio)

    # def heightForWidth(self, width: int):
    #     h =  0 if self._pixmap.isNull() else self._pixmap.height() * width / self._pixmap.width()
    #     return min(h,self._pixmap.height())

        # self.aspect_ratio = 1
        # self.adjusted_to_size = (-1,-1)
        # self.ratio=1

    # def resizeEvent(self, event):
    #     size = event.size()
    #     if size == self.adjusted_to_size:
    #         # Avoid infinite recursion. I suspect Qt does this for you,
    #         # but it's best to be safe.
    #         return
    #     self.adjusted_to_size = size

    #     # print(f"{self.name} {self.size() = }")
    #     scaled_pixmap = self._pixmap.scaled(
    #                         self.width(), self.height(),
    #                         # 10, 50,
    #                         self.aspectRatioPolicy
    #                         # Qt.IgnoreAspectRatio,
    #                         )
    #     self.setPixmap(scaled_pixmap)

    #     full_width = size.width()
    #     full_height = size.height()
    #     width = min(full_width, scaled_pixmap.size().width())
    #     height = min(full_height, scaled_pixmap.size().height())
    #     print(QSize(height,width))
    #     self.resize(QSize(height,width))

    # def resizeEvent(self, event):
    #     print(f"{self.name} {self.size() = }")
    #     scaled_pixmap = self._pixmap.scaled(
    #                         self.width(), self.height(),
    #                         # 10, 50,
    #                         self.aspectRatioPolicy
    #                         # Qt.IgnoreAspectRatio,
    #                         )
    #     self.setPixmap(scaled_pixmap)
        
    #     if self.size() == scaled_pixmap.size():
    #         return
    #     print(self.size() == scaled_pixmap.size())
    #     smallest_dim = min(scaled_pixmap.size().width(), scaled_pixmap.size().height())
    #     target_size = QSize(smallest_dim,smallest_dim)
    #     self.resize(target_size)
    #     # self.(scaled_pixmap.size())
    #     # self.setMaximumSize(QSize(16777215,16777215))
    
    # def sizeHint(self):
        # m_pixmap.size();
        # return QSize(5,5)

class MiniMosaics(QtWidgets.QLabel):
    clicked = Signal(str)
    "Widget to hold the image Qlabels"
    def __init__(self,
                    filepaths,
                    bands, lens_background_path,
                    interesting_background_path, deactivated_path, i,
                    status, activation, update_df_func,
                    image_width=None,
                    image_height=None,
                    parent=None):
        QtWidgets.QLabel.__init__(self, parent)
        self.filepaths = filepaths
        self.bands = bands
        self.n_bands = len(self.bands)
        self.is_activate = activation
        self.lens_background_path = lens_background_path
        self.interesting_background_path = interesting_background_path

        self.mini_layout = QtWidgets.QHBoxLayout(self)
        self.mini_layout.setSpacing(0) #TODO: FIND A GOOD VALUE/RECIPE
        self.mini_layout.setContentsMargins(0,0,0,0)

        self.deactivated_path = deactivated_path
        self.is_a_candidate = status
        self.update_df_func = update_df_func
        self.i = i

        
        # print(self.minimumSize())
        # print(self.hasHeightForWidth())

        self.target_width = 66 #At the very least, should be the initial size
        self.target_height = 66 #
        self.user_minimum_size = 66 if args.minimum_size is None else args.minimum_size
        if image_width is not None:
            self.target_width = image_width
        if image_height is not None:
            self.target_height = image_height

        self.aspectRatioPolicy = Qt.KeepAspectRatio
        # self.aspectRatioPolicy = Qt.KeepAspectRatioByExpanding
        # self.aspectRatioPolicy = Qt.IgnoreAspectRatio

        # sizePolicy = QtWidgets.QSizePolicy.MinimumExpanding
        # sizePolicy = QtWidgets.QSizePolicy.Expanding
        # sizePolicy = QtWidgets.QSizePolicy.Ignored
        # self.setMinimumSize(10,10)
        # self.setSizePolicy(sizePolicy,sizePolicy)

        # self.setScaledContents(args.resize)
        
        # qlabelSizePolicy = QtWidgets.QSizePolicy.MinimumExpanding    
        # qlabelSizePolicy = QtWidgets.QSizePolicy.Minimum
        # qlabelSizePolicy = QtWidgets.QSizePolicy.Expanding 
        qlabelSizePolicy = QtWidgets.QSizePolicy.Preferred
        # qlabelSizePolicy = QtWidgets.QSizePolicy.Maximum   
        # qlabelSizePolicy = QtWidgets.QSizePolicy.Ignored 
        # qlabelSizePolicy = QtWidgets.QSizePolicy.Fixed               
        # print(f"{self.sizeHint() = }")

        names = ['VIS','HYI','HJY','bak']
        self.qlabels = [MiniMosaicLabels(self.aspectRatioPolicy,
                                        self.user_minimum_size,
                                        qlabelSizePolicy,
                                        name = names[i],
                                        # parent = self
                                        ) for i,_ in enumerate(self.bands)]
        
        if self.is_activate:
            if self.is_a_candidate == C_UNINTERESTING:
                self.change_pixmaps(self.filepaths)
            elif self.is_a_candidate == C_LENS:
                self.change_pixmaps([self.lens_background_path]*self.n_bands)
            elif self.is_a_candidate == C_INTERESTING:
                self.change_pixmaps([self.interesting_background_path]*self.n_bands)
        else:
            self.change_pixmaps([self.self.deactivated_path]*self.n_bands)
            # print([self.self.deactivated_path]*self.n_bands)
        

        for qlabel in self.qlabels:   
            # qlabel.setMinimumSize(self.user_minimum_size,self.user_minimum_size)
            # qlabel.setSizePolicy(qlabelSizePolicy,qlabelSizePolicy)    
            # qlabel.setScaledContents(False)
            # qlabel.setScaledContents(True)

            qlabel.setPixmap(qlabel._pixmap.scaled(
                self.target_width, self.target_height,
                self.aspectRatioPolicy))
            # self.mini_layout.addWidget(qlabel, 1./len(self.qlabels))
            # self.mini_layout.addWidget(qlabel,Qt.AlignHCenter)
            self.mini_layout.addWidget(qlabel,Qt.AlignLeft)

        # self.black_rectangle = MiniMosaicLabels(Qt.IgnoreAspectRatio,
        #                                     99,
        #                                     QtWidgets.QSizePolicy.Ignored,
        #                                     'bak'
        #                                     )
        # self.black_rectangle._pixmap = QPixmap(self.deactivated_path)

        # spacer_scaling = QtWidgets.QSizePolicy.Minimum
        # spacer_scaling = QtWidgets.QSizePolicy.Ignored
        spacer_scaling = QtWidgets.QSizePolicy.Expanding

        spacer = QtWidgets.QSpacerItem(
                                        # 40,20,
                                        10,0,
                                        QtWidgets.QSizePolicy.Ignored,
                                        QtWidgets.QSizePolicy.Ignored
                                        )
                                        
        # self.mini_layout.addStretch()
        # self.mini_layout.addSpacerItem(spacer)

    # def sizeHint(self):
    #     return QSize(100,100)

    def change_pixmaps(self, paths_to_pixmap):
        # self._pixmaps = [QPixmap(path_to_pixmap) for path_to_pixmap in paths_to_pixmap]
        for qlabel, path_to_pixmap in zip(self.qlabels,paths_to_pixmap):
            qlabel._pixmap = QPixmap(path_to_pixmap)

    def activate(self):
        self.is_activate = True

    def change_filepath(self, filepaths):
        self.filepaths = filepaths
        self.change_pixmaps(self.filepaths)

    def change_and_paint_pixmap(self, filepaths):
        if self.is_activate:
            self.filepaths = filepaths
            self.change_pixmaps(self.filepaths)
            self.repaint_pixmaps()
            
    def repaint_pixmaps(self):
        for qlabel in self.qlabels:        
            qlabel.setPixmap(qlabel._pixmap.scaled( 
                qlabel.width(), qlabel.height(),
                self.aspectRatioPolicy))

    def deactivate(self): 
        self.change_and_paint_pixmap(self.deactivated_path) 
        self.is_activate = False 

    def set_candidate_status(self, status):
        if self.is_activate:
            self.is_a_candidate = status
            return True
        return False

    def toggle_candidate_status(self):
        if self.is_activate:
            self.is_a_candidate = not self.is_a_candidate


    def paint_pixmap(self):
        if self.is_activate:
            self.repaint_pixmaps()

    def paint_background_pixmap(self, background_path):
        if self.is_activate:
            self.change_pixmaps([background_path]*self.n_bands)
            self.repaint_pixmaps()


    def mousePressEvent(self, event):
        if self.is_activate:
            # self.is_a_candidate = not self.is_a_candidate
            modifiers = event.modifiers()
            # print('local:' ,modifiers)
            if self.is_a_candidate != C_UNINTERESTING:
                self.change_and_paint_pixmap(self.filepaths)
                new_class = C_UNINTERESTING
            else:
                if modifiers in [Qt.ControlModifier, Qt.ShiftModifier]:
                    self.paint_background_pixmap(self.interesting_background_path)
                    new_class = C_INTERESTING
                elif modifiers == Qt.NoModifier:
                    self.paint_background_pixmap(self.lens_background_path)
                    new_class = C_LENS

            self.update_df_func(event, self.i, new_class)
            self.is_a_candidate = new_class
        else:
            print('Inactive button')

    def change_nvisiblebands(self, nvisiblebands):
        # print(self.qlabels[:nvisiblebands])
        # print(self.qlabels[nvisiblebands:])
        for qlabel in self.qlabels[nvisiblebands:]:
            qlabel.hide()
        for qlabel in self.qlabels[:nvisiblebands]:
            # print("heh")
            qlabel.show()
        self.nvisiblebands = nvisiblebands

    # def resizeEvent(self, event):
    #     print(self.qlabels[0].width(), self.width())

    # def resizeEvent(self, event):
    #     for qlabel in self.qlabels[:self.nvisiblebands]:
    #         # print(qlabel.name, qlabel.size().toTuple())
    #         pixmap_length = min(qlabel.size().toTuple())
    #         # height = min(full_height, scaled_pixmap.size().height())
    #         qlabel.resize(QSize(pixmap_length,pixmap_length))
        

    # def resizeEvent(self, event):
        
    #     w = event.size().width()
    #     h = event.size().height()

    #     if w / h > self.aspect_ratio:  # too wide
    #         widget_stretch = h * self.aspect_ratio
    #         outer_stretch = (w - widget_stretch) / 1 + 0.5
    #         # outer_stretch = (w - widget_stretch) / 2 + 0.5
    #     # else:  # too tall
    #     #     self.layout().setDirection(QBoxLayout.TopToBottom)
    #     #     widget_stretch = w / self.aspect_ratio
    #     #     outer_stretch = (h - widget_stretch) / 2 + 0.5

    #     # self.layout().setStretch(0, outer_stretch)
    #     # self.layout().setStretch(1, widget_stretch)
    #     self.layout.setStretch(-1, outer_stretch)

    # def resizeEvent(self, event):
    #     # self.repaint_pixmaps()
    #     for qlabel in self.qlabels:
    #         qlabel.resizeEvent(event)
    #     # self.setPixmap(self._pixmap.scaled(
    #     #     self.width(), self.height(),
    #     #     self.aspectRatioPolicy))


class MosaicVisualizer(QtWidgets.QMainWindow):
    def __init__(self, path_to_the_stamps= args.path):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self._main.setStyleSheet('background-color: black')

        # self.setGeometry(800, 100, 100, 100)
        self.setCentralWidget(self._main)
        self.status = self.statusBar()
        self.random_seed = args.seed            
        self.stampspath = path_to_the_stamps

        self.main_band = args.main_band
        self.color_bands = args.color_bands.split(",")
        self.color_bands_vis = [_VIS_RESAMPLED_BAND,'Y','H']
        self.all_single_bands = (self.main_band, _VIS_RESAMPLED_BAND, *self.color_bands)
        self.composite_bands = [
                                "".join(self.color_bands_vis[::-1]),
                                "".join(self.color_bands[::-1]),
                                ]
        self.bands_to_plot = [self.main_band, *self.composite_bands]

        # print(f"{self.main_band}")
        # print(f"{self.color_bands}")
        # print(f"{self.color_bands_vis}")
        # print(f"{self.all_single_bands}")
        # print(f"{self.composite_bands}")
        self.scratchpath = './.temp'
        self.deactivated_path = './dark.png'
        os.makedirs(self.scratchpath, exist_ok=True)
        self.clean_dir(self.scratchpath)

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
            print("No suitable files were found in {self.base_band_path} or {color_bands_path}")
            sys.exit()

        if self.random_seed is not None:
            # 99 is always changed to this number when sorting to mantain compatibility with old classifications.
            # 128 bits, proton decay might be more likely than someone *randomly* using this number.
            # Please, do not use this number as your seed.
            seed_to_use = 120552782132343758881253061212639178445 if self.random_seed == 99 else self.random_seed
            # print("shuffling")
            rng = np.random.default_rng(seed_to_use)
            rng.shuffle(self.listimage) #inplace shuffling
        
        if len(self.listimage) == 0:
            print("WARNING: no images found in {}".format(self.stampspath))
        self.ncols = args.ncols
        if args.nrows is None:
            self.nrows = self.ncols
        else:
            self.nrows = args.nrows
        self.gridarea = self.nrows*self.ncols
        self.PAGE_MAX = int(np.ceil(len(self.listimage) / self.gridarea))
        self.scale2funct = {'linear': identity,
                            'sqrt': np.sqrt,
                            'cbrt': np.cbrt,
                            'log': log,
                            'asinh': asinh2}

        title_strings = ["Mosaic stamp visualizer"]
        if args.name is not None:
            self.name = args.name
            title_strings.append(self.name)
        else:
            self.name = ''
        self.setWindowTitle(' - '.join(title_strings))

        self.defaults = {
            # 'counter': 0,
            'page': 0, #Defaults to 0. Gets overwritten by --page argument.
            # 'total': -1,
            'colormap': 'gist_gray',
            'scale': 'asinh',
            'name': self.name,
            'ncols': self.ncols,
            'nrows': self.nrows,
            'nvisiblebands':'3',
            # 'gridsize': self.nrows, #Just for retrocompatibility.
        }
        self.config_dict = self.load_dict()
        
        self.interesting_background_path = '.background_interesting.png'
        self.lens_background_path = '.background.png'
        self.deactivated_path = '.backgrounddark.png'
        self.status2background_dict = {C_LENS:self.lens_background_path,
                                        C_INTERESTING:self.interesting_background_path}               

        self.bcounter = LabelledIntField('Page', self.config_dict['page'], self.PAGE_MAX)
        self.bcounter.setStyleSheet('background-color: black; color: gray')
        self.bcounter.lineEdit.returnPressed.connect(self.goto)
        self.bcounter.setInputText(self.config_dict['page'])

        self.buttons = []
        self.clean_dir(self.scratchpath)

        self.df = self.obtain_df()

        if self.config_dict['page'] >= self.PAGE_MAX:
            self.bcounter.setInputText(0)
            self.goto()
            
        self.prepare_pngs(self.gridarea)

        main_layout = QtWidgets.QVBoxLayout(self._main)
        stamp_grid_layout = QtWidgets.QGridLayout()
        # stamp_grid_layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        bottom_bar_layout = QtWidgets.QHBoxLayout()
        button_bar_layout = QtWidgets.QHBoxLayout()
        page_counter_layout = QtWidgets.QHBoxLayout()


        bottom_bar_layout.addLayout(button_bar_layout,10)
        bottom_bar_layout.addLayout(page_counter_layout,1)
        main_layout.addLayout(stamp_grid_layout, 8)
        main_layout.addLayout(bottom_bar_layout, )

        #### Buttons
        self.cbscale = QtWidgets.QComboBox()
        delegate = AlignDelegate(self.cbscale)
        self.cbscale.setItemDelegate(delegate)
        # self.cbscale.setEditable(True)
        self.cbscale.setFont(QFont("Arial",20))
        # Clickable(self.cbscale).connect(self.cbscale.showPopup)
        line_edit = self.cbscale.lineEdit()
        self.cbscale.addItems(self.scale2funct.keys())
        self.cbscale.setCurrentIndex(list(self.scale2funct.keys()).index(self.config_dict['scale']))
        self.cbscale.setStyleSheet('background-color: gray')
        self.cbscale.currentIndexChanged.connect(self.change_scale)


        self.cbcolormap = QtWidgets.QComboBox()
        delegate = AlignDelegate(self.cbcolormap)
        self.cbcolormap.setItemDelegate(delegate)
        # self.cbcolormap.setEditable(True)
        self.cbcolormap.setFont(QFont("Arial",20))
        line_edit = self.cbcolormap.lineEdit()
        # line_edit.setAlignment(Qt.AlignCenter)
        # line_edit.setReadOnly(True)
        self.listscales = ['gist_gray','viridis','gist_yarg','hot']
        self.cbcolormap.addItems(self.listscales)
        self.cbcolormap.setCurrentIndex(self.listscales.index(self.config_dict['colormap']))
        self.cbcolormap.setStyleSheet('background-color: gray')
        self.cbcolormap.currentIndexChanged.connect(self.change_colormap)

        self.cbnvisibleimages = QtWidgets.QComboBox()
        delegate = AlignDelegate(self.cbnvisibleimages)
        self.cbnvisibleimages.setItemDelegate(delegate)
        # self.cbcolormap.setEditable(True)
        self.cbnvisibleimages.setFont(QFont("Arial",20))
        line_edit = self.cbnvisibleimages.lineEdit()
        # line_edit.setAlignment(Qt.AlignCenter)
        # line_edit.setReadOnly(True)
        self.listscales = ["1","2","3"]
        self.cbnvisibleimages.addItems(self.listscales)
        self.cbnvisibleimages.setCurrentIndex(self.listscales.index(self.config_dict['nvisiblebands']))
        self.cbnvisibleimages.setStyleSheet('background-color: gray')
        self.cbnvisibleimages.currentIndexChanged.connect(self.change_nvisiblebands)


        self.bprev = QtWidgets.QPushButton('Prev')
        self.bprev.clicked.connect(self.prev)
        self.bprev.setStyleSheet('background-color: gray')
        self.bprev.setFont(QFont("Arial",20))

        self.bnext = QtWidgets.QPushButton('Next')
        self.bnext.clicked.connect(self.next)
        self.bnext.setStyleSheet('background-color: gray')
        self.bnext.setFont(QFont("Arial",20))

        self.bclickcounter = NamedLabel('Clicks', self.df['classification'].sum().astype(int))
        self.bclickcounter.setStyleSheet('background-color: black; color: gray')

        ##### Keyboard shortcuts
        self.knext = QShortcut(QKeySequence('f'), self)
        self.knext.activated.connect(self.next)

        self.kprev = QShortcut(QKeySequence('d'), self)
        self.kprev.activated.connect(self.prev)


        button_bar_layout.addWidget(self.cbscale)
        button_bar_layout.addWidget(self.cbcolormap)
        button_bar_layout.addWidget(self.cbnvisibleimages)
        button_bar_layout.addWidget(self.bprev)
        button_bar_layout.addWidget(self.bnext)
        page_counter_layout.addWidget(self.bclickcounter)
        page_counter_layout.addWidget(self.bcounter)

        self.total_n_frame = int(len(self.listimage)/(self.gridarea))
        start = self.config_dict['page']*self.gridarea

        for i in range(start,start+self.gridarea):
            try:
                classification = self.df.iloc[i,self.df.columns.get_loc('classification')]
                activation = True
            except IndexError:
                classification = False
                activation = False

            button = MiniMosaics(
                                    self.filepaths(i, self.config_dict['page']),
                                    self.bands_to_plot,
                                    # self.bands_to_plot[:int(self.config_dict['nvisiblebands'])],
                                    self.lens_background_path,
                                    self.interesting_background_path,
                                    self.deactivated_path,
                                    i-start, classification, activation,
                                    self.my_label_clicked,
                                    # image_width=66,
                                    # image_height=66,
                                    )
            button.change_nvisiblebands(int(self.config_dict['nvisiblebands']))
            stamp_grid_layout.addWidget(
                button, i % self.nrows, i // self.nrows)
            self.buttons.append(button)
            button.setAlignment(Qt.AlignCenter) #TODO CHECK HOW TO REACTIVATE THIS. (OR IF IT'S NEEDED)

            # button.adjustSize()
        if self.filetype != 'FITS':
            self.cbscale.setEnabled(False)
            self.cbcolormap.setEnabled(False)
        
        self.time_0 = time()

    def go_to_page(self, target_page):
        range_low = self.config_dict['page']*self.gridarea
        range_high = min(len(self.df),(self.config_dict['page']+1)*(self.gridarea))
        if hasattr(self, 'time_0'):
            self.df.iloc[range(range_low,range_high),
                        self.df.columns.get_loc('time')] += (time() - self.time_0)
            self.time_0 = time()
        else:
            True
        self.config_dict['page'] = target_page
        self.clean_dir(self.scratchpath)
        self.update_grid()
        self.bcounter.setInputText(self.config_dict['page'])
        self.save_dict()
        self.df.to_csv(
                self.df_name, index=False)

    @Slot()
    def goto(self):
        if self.bcounter.getValue()>self.PAGE_MAX:
            print("page: ",self.PAGE_MAX)
            self.status.showMessage('WARNING: There are only {} pages.'.format(
                self.PAGE_MAX+1),10000)
        elif self.bcounter.getValue()<0:
            self.status.showMessage('WARNING: Pages go from 1 to {}.'.format(
                self.PAGE_MAX+1),10000)
        else:
            self.go_to_page(self.bcounter.getValue())
    @Slot()
    def next(self):
        if self.config_dict['page']+1 >= self.PAGE_MAX:
            self.status.showMessage('You are already at the last page',10000)
        else:
            self.go_to_page(self.config_dict['page'] + 1)
    @Slot()
    def prev(self):
        if self.config_dict['page'] -1 < 0:
            self.status.showMessage('You are already at the first page',10000)
        else:
            self.go_to_page(self.config_dict['page'] - 1)

    def change_scale(self,i):
        self.config_dict['scale'] = self.cbscale.currentText()
        self.update_grid()
        self.save_dict()

    def change_colormap(self,i):
        self.config_dict['colormap'] = self.cbcolormap.currentText()
        self.update_grid(single_band_only=True)
        self.save_dict()

    def change_nvisiblebands(self,i):
        self.config_dict['nvisiblebands'] = self.cbnvisibleimages.currentText()
        self.update_grid(single_band_only=True,change_nvisiblebands=True)
        self.save_dict()

    def my_label_clicked(self, event, i, new_class):
        button = event.button()
        modifiers = event.modifiers()
        # if modifiers == Qt.NoModifier and button == Qt.LeftButton:
        if self.config_dict['page']*self.gridarea+i > len(self.listimage):
            print('Something is wrong. This condition should not be trigger.')
        else:
            object_index = self.gridarea*self.config_dict['page']+i
            print(self.df.iloc[object_index,
                        self.df.columns.get_loc('file_name')]) if args.printname else True
            self.df.iloc[object_index,
                        self.df.columns.get_loc('classification')] = new_class
            
            range_low = self.config_dict['page']*self.gridarea
            range_high = min(len(self.df),(self.config_dict['page']+1)*(self.gridarea))
            if hasattr(self, 'time_0'):
                self.df.iloc[range(range_low,range_high),
                        self.df.columns.get_loc('time')] += (time() - self.time_0)
                self.time_0 = time()

            self.bclickcounter.setText((self.df['classification'] == C_LENS).sum().astype(int))
            self.df.to_csv(
                self.df_name, index=False)

    def filepath(self, i, page, band = ''):
        colormap = self.config_dict['colormap'] if band == '' else ''
        return join(self.scratchpath, (str(i+1)+self.config_dict['scale']+
                                    #    self.config_dict['colormap']+
                                       colormap+
                                       str(page)+
                                       str(band)+
                                       '.png')
                                        )

    def filepaths(self, i, page,nvisiblebands = None):
        if nvisiblebands is None:
            nvisiblebands = len(self.bands_to_plot)
        return [self.filepath(i,page,band) 
                    for band in self.bands_to_plot[:nvisiblebands]]

    def save_dict(self):
        with open('.config_mosaic.json', 'w') as f:
            json.dump(self.config_dict, f, ensure_ascii=False, indent=4)

    def load_dict(self):
        try:
            with open('.config_mosaic.json', ) as f:
                temp_dict = json.load(f)
                if ((temp_dict['name'] != self.name) 
                    ):
                    temp_dict['name'] = self.name
                    temp_dict['page'] = 0
                if (('nrows' not in temp_dict) or
                  ('ncols' not in temp_dict) or
                  (temp_dict['nrows'] != self.nrows) or
                  (temp_dict['ncols'] != self.ncols)):

                    temp_dict['nrows'] = self.nrows
                    temp_dict['ncols'] = self.ncols
                    temp_dict['page'] = 0

                    # temp_dict['name'] = args.gridsize #I commented this on 06-02-2024.
                if args.page is not None:
                    temp_dict['page'] = min(args.page - 1, 0)
                if temp_dict['scale'] == 'log10':
                    temp_dict['scale'] = 'log'
                if temp_dict['colormap'] == 'gray':
                    temp_dict['colormap'] = 'gist_gray'
                for key in self.defaults.keys():
                    if key not in temp_dict.keys():
                        temp_dict[key] = self.defaults[key]
                return temp_dict
        except FileNotFoundError:
            print("Loaded default configuration.")
            return self.defaults

    def obtain_df(self):
        if self.random_seed is None:
            base_filename = 'classification_mosaic_autosave_{}_{}_{}_{}_99'.format(
                                    self.name,len(self.listimage),self.ncols,self.nrows)
            base_filename = 'classification_mosaic_autosave_{}_{}_{}_99'.format(
                                    self.name,len(self.listimage),self.ncols)
            string_to_glob = './Classifications/{}*.csv'.format(base_filename)
            # print("Globing for", string_to_glob)
            string_to_glob_for_files_with_seed = './Classifications/{}_*.csv'.format(base_filename)
            glob_results = set(glob.glob(string_to_glob)) - set(glob.glob(string_to_glob_for_files_with_seed))
        else:
            base_filename = base_filename = 'classification_mosaic_autosave_{}_{}_{}_{}_{}'.format(
                                    self.name,len(self.listimage),self.ncols,self.nrows,self.random_seed)
            string_to_glob = './Classifications/{}*.csv'.format(base_filename)
            glob_results = glob.glob(string_to_glob)

        # string_to_glob = './Classifications/classification_mosaic_autosave_{}_{}_{}_{}*.csv'.format(
        #                             self.name,len(self.listimage),self.gridsize, str(self.random_seed))
        class_file = natural_sort(glob.glob(
            string_to_glob)) #better to use natural sort.
        file_iteration = ""
        if len(class_file) >= 1:
            file_index = 0
            if len(class_file) > 1:
                file_index = -2
            self.df_name = class_file[file_index]
            print('Reading '+ self.df_name)
            df = pd.read_csv(self.df_name)
            if np.all(self.listimage == df['file_name'].values):
                if 'time' not in df.keys():
                    df['time'] = 0
                return df
            else:
                print("Classification file corresponds to a different dataset.")
                string_tested = os.path.basename(self.df_name).split(".csv")[0]
                file_iteration = find_filename_iteration(string_tested) if f'./Classifications/{base_filename}.csv' in class_file else ''

        
        self.dfc = ['file_name', 'classification', 'grid_pos','page']
        self.df_name = './Classifications/{}{}.csv'.format(base_filename,file_iteration)
        print('A new csv will be created', self.df_name)
        # self.config_dict['page'] = 1
        
        if file_iteration != "":
            print("To avoid this in the future use the argument `-N name` and give different names to different datasets.")
        df = pd.DataFrame(columns=self.dfc)
        df['file_name'] = self.listimage
        df['classification'] = np.zeros(np.shape(self.listimage))
        page,grid_pos = iloc_to_page_and_grid_pos(np.array(df.index) ,gridarea = self.gridarea)
        df['page'] = page
        df['grid_pos'] = grid_pos
        df['time'] = np.zeros(np.shape(self.listimage))
        return df

    def update_grid(self, single_band_only = False, change_nvisiblebands=False):
        start = self.config_dict['page']*self.gridarea
        n_images = self.gridarea
        self.prepare_pngs(n_images, single_band_only)
        i = start
        j = 0
        for button in self.buttons:
            try:
                object_index = self.gridarea*self.config_dict['page']+j
                status = self.df.iloc[object_index,self.df.columns.get_loc('classification')]
                if status == 0:
                    button.activate()
                    # print(self.filepaths(i,self.config_dict['page']))
                    button.change_and_paint_pixmap(self.filepaths(i,self.config_dict['page']))
                    button.set_candidate_status(status)
                else:
                    button.activate()
                    # print(self.filepaths(i,self.config_dict['page']))
                    button.change_filepath(self.filepaths(i,self.config_dict['page']))
                    button.paint_background_pixmap(self.status2background_dict[status])
                    button.set_candidate_status(status)

                if change_nvisiblebands:
                    button.change_nvisiblebands(int(self.config_dict['nvisiblebands']))
                self.df.iloc[object_index,
                             self.df.columns.get_loc('grid_pos')] = j

                self.df.iloc[object_index,
                             self.df.columns.get_loc('page')] = self.config_dict['page']

            except (KeyError,IndexError) as e:
                # print("Out of bounds in the dataframe.")
                button.deactivate()
                # button.change_and_paint_pixmap(self.filepath(i,self.config_dict['page']))
                # raise
            j = j+1
            i = i+1

    def prepare_pngs(self, number, single_band_only = False):
            "Generates the png files from the fits."
            start = self.config_dict['page']*self.gridarea
            for i in np.arange(start, start + number + 0): 
                if i < len(self.listimage):
                    self.prepare_png(i, single_band_only)
                else:
                    image = np.zeros((66, 66))# * 0.0000001
                    plt.imsave(self.filepath(i, self.config_dict['page']),
                        image, cmap=self.config_dict['colormap'], origin="lower")

    def prepare_png(self, i, single_band_only):
        if self.filetype == 'FITS':
            band_images = {band: self.read_fits(i,band) for band in self.all_single_bands}
            
            image = self.prepare_single_band(band_images[self.main_band])
            plt.imsave(self.filepath(i, self.config_dict['page'], band=self.main_band),
                    image, cmap=self.config_dict['colormap'], origin="lower")

            if not single_band_only:
                for composite_band in self.composite_bands:
                    bands = list(composite_band)
                    composite_image = self.prepare_composite_band(np.stack([band_images[band] for band in bands],axis=-1))
                    plt.imsave(self.filepath(i, self.config_dict['page'], band=composite_band),
                        composite_image, cmap=self.config_dict['colormap'], origin="lower")

    def prepare_single_band(self, image):
        scale_min, scale_max = self.scale_val(image)
        image = self.rescale_image(image, scale_min, scale_max)
        image[np.isnan(image)] = np.nanmin(image)
        return image
            

    def prepare_composite_band(self,images,
                                p_low=1, p_high=0.1,
                                value_at_min=0,
                                color_bkg_level=0.015,):
        
        composite_image = np.zeros_like(images,
                                    dtype=float)
        scale_min, scale_max = get_value_range_asymmetric(images,p_low,p_high)
        
        for i in range(images.shape[-1]):
            composite_image[:,:,i] = self.rescale_single_band(
                            images[:,:,i],
                            scale_min,
                            scale_max,
                            value_at_min,
                            color_bkg_level)
        return composite_image

    def rescale_single_band(self, image,
                            scale_min,
                            scale_max,
                            value_at_min=0,
                            color_bkg_level=-0.05):

        image = clip_normalize(image,scale_min,scale_max)
        image = self.scale2funct[self.config_dict['scale']](image)
        contrast, bias = get_contrast_bias_reasonable_assumptions(
                                                                    max(value_at_min,scale_min),
                                                                    color_bkg_level,
                                                                    scale_min,
                                                                    scale_max,
                                                                    self.scale2funct[self.config_dict['scale']])
        return contrast_bias_scale(image, contrast, bias)



    def heh(self):
            try:
                image = self.read_fits(i)
                scaling_factor = np.nanpercentile(image,q=90)
                if scaling_factor == 0:
                    # scaling_factor = np.nanpercentile(image,q=99)
                    scaling_factor = 1
                image = image / scaling_factor * 300 #Rescaling for better visualization.
                scale_min, scale_max = self.scale_val(image)
                image = self.rescale_image(image, scale_min, scale_max)
                image[np.isnan(image)] = np.nanmin(image)
                
                plt.imsave(self.filepath(i, self.config_dict['page']),
                        image, cmap=self.config_dict['colormap'], origin="lower")
            except Exception as E:
                # raise
                print(f"WARNING: exception saving file {E}")
                image = np.zeros((66, 66))# * 0.0000001
                plt.imsave(self.filepath(i, self.config_dict['page']),
                    image, cmap=self.config_dict['colormap'], origin="lower")
        # elif self.filetype == 'COMPRESSED':
            if i < len(self.listimage):
                try:
                    original_filepath = join(self.stampspath, self.listimage[i])
                    shutil.copyfile(original_filepath, self.filepath(i, self.config_dict['page']))
                except:
                    print('file not found: {}'.format(original_filepath))


    def clean_dir(self, path_dir):
        "Removes everything in the scratch folder."
        for f in os.listdir(path_dir):
            os.remove(join(path_dir, f))

    def read_fits(self, i, band = ''):
        file = join(self.stampspath, band, self.listimage[i])
        # Note : memmap=False is much faster when opening/closing many small files
        with fits.open(file, memmap=False) as hdu_list:
            image = hdu_list[0].data
        return image

    def rescale_image(self, image, scale_min, scale_max):
        factor = self.scale2funct[self.config_dict['scale']](scale_max - scale_min)#+2e-16)
        # factor = (self.scale2funct[self.config_dict['scale']](scale_max) -
        #          self.scale2funct[self.config_dict['scale']](scale_min))
        image = image.clip(min=scale_min, max=scale_max)

        #I'm gonna go with this one since it solves the bright noise problem and seems to not hurt anything else.
        indices0 = np.where(image < scale_min)
        indices1 = np.where((image >= scale_min) & (image < scale_max))
        indices2 = np.where(image >= scale_max)
        image[indices0] = 0.0
        image[indices2] = 1.0
        # image[indices1] = np.abs(self.scale2funct[self.config_dict['scale']](image[indices1]) / ((factor) * 1.0))
        image[indices1] = self.scale2funct[self.config_dict['scale']](image[indices1]) / ((factor) * 1.0)
        # image[indices1] /= image[indices1].max()
        return image

    def scale_val(self, image_array):
        if image_array.shape[0] > 173:
            box_size_vmin = np.round(np.sqrt(np.prod(image_array.shape) * 0.001)).astype(int)
            box_size_vmax = np.round(np.sqrt(np.prod(image_array.shape) * 0.01)).astype(int)
        else:
            box_size_vmin = 5
            box_size_vmax = 14
        vmin = np.nanmin(self.background_rms_image(box_size_vmin, image_array))
        if vmin == 0:
            vmin += 1e-3              
        
        xl, yl = np.shape(image_array)
        xmin = int((xl) / 2. - (box_size_vmax / 2.))
        xmax = int((xl) / 2. + (box_size_vmax / 2.))
        ymin = int((yl) / 2. - (box_size_vmax / 2.))
        ymax = int((yl) / 2. + (box_size_vmax / 2.))
        vmax = np.nanmax(image_array[xmin:xmax, ymin:ymax])
        return vmin*1.0, vmax*1.3 #vmin is 1 sigma.

    def scale_val_percentile(self,image_array,p_min=0.1,p_max=99.9):
        # image_to_plot = np.clip(image_array,np.percentile(p_min),np.percentile(p_max))
        # print(np.percentile(image_array,p_min),np.percentile(image_array,p_max))
        return np.nanpercentile(image_array,p_min),np.nanpercentile(image_array,p_max)

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

    def background_rms_image_old(self, cb, image):
        xg, yg = np.shape(image)
        cut0 = image[0:cb, 0:cb]
        cut1 = image[xg - cb:xg, 0:cb]
        cut2 = image[0:cb, yg - cb:yg]
        cut3 = image[xg - cb:xg, yg - cb:yg]
        l = [cut0, cut1, cut2, cut3]
        m = np.nanmean(np.nanmean(l, axis=1), axis=1)
        ml = min(m)
        mm = max(m)
        if mm > 5 * ml:
            s = np.sort(l, axis=0)
            nl = s[:-1]
            std = np.nanstd(nl)
        else:
            std = np.nanstd([cut0, cut1, cut2, cut3])
        return std


if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = MosaicVisualizer()
    app.show()
    app.activateWindow()
    # app.raise_()
    qapp.exec()