# This Python file uses the following encoding: utf-8
import argparse

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
parser.add_argument('-l',"--ncols","--gridsize", help="Number of columns per page.",type=int,
                    default=10)
parser.add_argument('-m',"--nrows", 
                    help="Number of rows per page. Leave unset if you want a squared grid",type=int,
                    default=None)
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

def identity(x):
    return x

def log_0(x):
    "Simple log base 1000 function that ignores numbers less than 0"
    return np.log(x, out=np.zeros_like(x), where=(x>0)) / np.log(1000)

def log(x,a=100):
    "Simple log base 1000 function that ignores numbers less than 0"
    return np.log(a*x+1) / np.log(a)

def asinh2(x):
    return np.arcsinh(x/2)

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
                parent=None):
        QtWidgets.QLabel.__init__(self, parent)
        self.aspectRatioPolicy = Qt.KeepAspectRatio
        # self.aspectRatioPolicy = Qt.IgnoreAspectRatio
        self.setMinimumSize(minimum_size,minimum_size)
        # print()
        self.setSizePolicy(sizePolicy,sizePolicy)    
        # print(f"{self.sizePolicy()}")
        # print(self.parent())
        self.setScaledContents(False)

    def resizeEvent(self, event):
        self.setPixmap(self._pixmap.scaled(
            self.width(), self.height(),
            # 10, 50,
            self.aspectRatioPolicy
            # Qt.IgnoreAspectRatio,
            ))

# class MiniMosaics(QtWidgets.QWidget):
class MiniMosaics(QtWidgets.QLabel):
    clicked = Signal(str)
    "Widget to hold the image Qlabels"
    def __init__(self, filepath, bands, lens_background_path,
                    interesting_background_path, deactivated_path, i,
                    status, activation, update_df_func,
                    image_width=None,
                    image_height=None,
                    parent=None):
        # QtWidgets.QWidget.__init__(self, parent)
        QtWidgets.QLabel.__init__(self, parent)
        self.filepath = filepath
        self.bands = bands
        self.n_bands = len(self.bands)
        self.is_activate = activation
        self.lens_background_path = lens_background_path
        self.interesting_background_path = interesting_background_path

        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setSpacing(1.5) #TODO: FIND A GOOD VALUE/RECIPE
        self.layout.setContentsMargins(0,0,0,0)

        self.deactivated_path = deactivated_path
        self.is_a_candidate = status
        self.update_df_func = update_df_func
        self.i = i

        sizePolicy = QtWidgets.QSizePolicy.MinimumExpanding
        sizePolicy = QtWidgets.QSizePolicy.Expanding
        # sizePolicy = QtWidgets.QSizePolicy.Ignored
        self.setMinimumSize(10,10)
        self.setSizePolicy(sizePolicy,sizePolicy)

        # self.setScaledContents(args.resize)
        

        self.target_width = 66 #At the very least, should be the initial size
        self.target_height = 66 #
        self.user_minimum_size = 66 if args.minimum_size is None else args.minimum_size
        if image_width is not None:
            self.target_width = image_width
        if image_height is not None:
            self.target_height = image_height

        self.aspectRatioPolicy = Qt.KeepAspectRatio
        qlabelSizePolicy = QtWidgets.QSizePolicy.MinimumExpanding    
        # qlabelSizePolicy = QtWidgets.QSizePolicy.Expanding         
        # qlabelSizePolicy = QtWidgets.QSizePolicy.Fixed               
        self.qlabels = [MiniMosaicLabels(self.aspectRatioPolicy,
                                        self.user_minimum_size,
                                        qlabelSizePolicy,
                                        # self,
                                        ) for _ in self.bands]
        if self.is_activate:
            
            if self.is_a_candidate == C_UNINTERESTING:
                self.change_pixmaps([self.filepath]*self.n_bands)
            elif self.is_a_candidate == C_LENS:
                self.change_pixmaps([self.lens_background_path]*self.n_bands)
            elif self.is_a_candidate == C_INTERESTING:
                self.change_pixmaps([self.interesting_background_path]*self.n_bands)
        else:
            self.change_pixmaps([self.self.deactivated_path]*self.n_bands)

        

        for qlabel in self.qlabels:   
            # qlabel.setMinimumSize(self.user_minimum_size,self.user_minimum_size)
            # qlabel.setSizePolicy(qlabelSizePolicy,qlabelSizePolicy)    
            # qlabel.setScaledContents(False)
            # qlabel.setScaledContents(True)

            qlabel.setPixmap(qlabel._pixmap.scaled(
                self.target_width, self.target_height,
                self.aspectRatioPolicy))
            self.layout.addWidget(qlabel)



    def change_pixmaps(self, paths_to_pixmap):
        # self._pixmaps = [QPixmap(path_to_pixmap) for path_to_pixmap in paths_to_pixmap]
        for qlabel, path_to_pixmap in zip(self.qlabels,paths_to_pixmap):
            qlabel._pixmap = QPixmap(path_to_pixmap)

    def activate(self):
        self.is_activate = True

    def change_filepath(self, filepath):
        self.filepath = filepath
        self.change_pixmaps([self.filepath]*self.n_bands)

    def change_and_paint_pixmap(self, filepath):
        if self.is_activate:
            self.filepath = filepath
            self.change_pixmaps([self.filepath]*self.n_bands)
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
                self.change_and_paint_pixmap(self.filepath)
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
        self.scratchpath = './.temp'
        self.deactivated_path = './dark.png'
        os.makedirs(self.scratchpath, exist_ok=True)
        self.clean_dir(self.scratchpath)
        if args.fits is None:
            print("No filetype was specified, defaulting to .fits")
            self.listimage = sorted([os.path.basename(x)
                                for x in glob.glob(join(self.stampspath, '*.fits'))])
            self.filetype='FITS'
            if len(self.listimage) == 0:
                print("No fits files were found, trying with .png, .jpg, and .jpeg")
                self.listimage = sorted([os.path.basename(x)
                                for x in (glob.glob(join(self.stampspath, '*.png')) +
                                          glob.glob(join(self.stampspath, '*.jpg')) +
                                          glob.glob(join(self.stampspath, '*.jpeg'))
                                         )])
                self.filetype='COMPRESSED'

        elif args.fits:
            self.listimage = sorted([os.path.basename(x)
                                for x in glob.glob(join(self.stampspath, '*.fits'))])
            self.filetype='FITS'
        else:
            self.listimage = sorted([os.path.basename(x)
                                for x in (glob.glob(join(self.stampspath, '*.png')) +
                                          glob.glob(join(self.stampspath, '*.jpg')) +
                                          glob.glob(join(self.stampspath, '*.jpeg'))
                                         )])
            self.filetype='COMPRESSED'


        if len(self.listimage) == 0:
            print("No suitable files were found in {self.stampspath}")

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
            # 'gridsize': self.nrows, #Just for retrocompatibility.
        }
        self.config_dict = self.load_dict()
        self.scale = self.scale2funct[self.config_dict['scale']]
        
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
            
        self.prepare_png(self.gridarea)

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
        button_bar_layout.addWidget(self.bprev)
        button_bar_layout.addWidget(self.bnext)
        page_counter_layout.addWidget(self.bclickcounter)
        page_counter_layout.addWidget(self.bcounter)

        self.total_n_frame = int(len(self.listimage)/(self.gridarea))
        start = self.config_dict['page']*self.gridarea

        for i in range(start,start+self.gridarea):
            filepath = self.filepath(i, self.config_dict['page'])
            try:
                classification = self.df.iloc[i,self.df.columns.get_loc('classification')]
                activation = True
            except IndexError:
                classification = False
                activation = False

            button = MiniMosaics(filepath, ['VIS','H'], self.lens_background_path,
                                    self.interesting_background_path,
                                    self.deactivated_path,
                                    i-start, classification, activation,
                                    self.my_label_clicked,
                                    # image_width=66,
                                    # image_height=66,
                                    )
            stamp_grid_layout.addWidget(
                button, i % self.nrows, i // self.nrows)
            self.buttons.append(button)
            button.setAlignment(Qt.AlignCenter) #TODO CHECK HOW TO REACTIVATE THIS. (OR IF IT'S NEEDED)
            # button.adjustSize()
        if self.filetype != 'FITS':
            self.cbscale.setEnabled(False)
            self.cbcolormap.setEnabled(False)
        
        self.time_0 = time()

    def go_to_counter_page(self, target_page):
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
            self.go_to_counter_page(self.bcounter.getValue())
    @Slot()
    def next(self):
        if self.config_dict['page']+1 >= self.PAGE_MAX:
            self.status.showMessage('You are already at the last page',10000)
        else:
            self.go_to_counter_page(self.config_dict['page'] + 1)
    @Slot()
    def prev(self):
        if self.config_dict['page'] -1 < 0:
            self.status.showMessage('You are already at the first page',10000)
        else:
            self.go_to_counter_page(self.config_dict['page'] - 1)

    def change_scale(self,i):
        self.config_dict['scale'] = self.cbscale.currentText()
        self.update_grid()
        self.save_dict()

    def change_colormap(self,i):
        self.config_dict['colormap'] = self.cbcolormap.currentText()
        self.update_grid()
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

    def filepath(self, i, page):
        return join(self.scratchpath, str(i+1)+self.config_dict['scale'] + self.config_dict['colormap'] + str(page)+'.png')

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
                if (('nrows' not in temp_dict) or
                  ('ncols' not in temp_dict) or
                  (temp_dict['nrows'] != self.nrows) or
                  (temp_dict['ncols'] != self.ncols)):

                    temp_dict['nrows'] = self.nrows
                    temp_dict['ncols'] = self.ncols
                    # temp_dict['name'] = args.gridsize #I commented this on 06-02-2024.
                if args.page is not None:
                    temp_dict['page'] = args.page
                if temp_dict['scale'] == 'log10':
                    temp_dict['scale'] = 'log'
                if temp_dict['colormap'] == 'gray':
                    temp_dict['colormap'] = 'gist_gray'
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

    def update_grid(self):
        start = self.config_dict['page']*self.gridarea
        n_images = self.gridarea
        self.prepare_png(n_images)
        i = start
        j = 0
        for button in self.buttons:
            try:
                object_index = self.gridarea*self.config_dict['page']+j
                status = self.df.iloc[object_index,self.df.columns.get_loc('classification')]
                if status == 0:
                    button.activate()
                    button.change_and_paint_pixmap(self.filepath(i,self.config_dict['page']))
                    button.set_candidate_status(status)
                else:
                    button.activate()
                    button.change_filepath(self.filepath(i,self.config_dict['page']))
                    button.paint_background_pixmap(self.status2background_dict[status])
                    button.set_candidate_status(status)

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

    def prepare_png(self, number):
            "Generates the png files from the fits."
            start = self.config_dict['page']*self.gridarea
            for i in np.arange(start, start + number + 0): 
                if i < len(self.listimage):
                    if self.filetype == 'FITS':
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
                    elif self.filetype == 'COMPRESSED':
                        if i < len(self.listimage):
                            try:
                                original_filepath = join(self.stampspath, self.listimage[i])
                                shutil.copyfile(original_filepath, self.filepath(i, self.config_dict['page']))
                            except:
                                print('file not found: {}'.format(original_filepath))
                else:
                    image = np.zeros((66, 66))# * 0.0000001
                    plt.imsave(self.filepath(i, self.config_dict['page']),
                        image, cmap=self.config_dict['colormap'], origin="lower")
                

    def clean_dir(self, path_dir):
        "Removes everything in the scratch folder."
        for f in os.listdir(path_dir):
            os.remove(join(path_dir, f))

    def read_fits(self, i):
        file = join(self.stampspath, self.listimage[i])
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