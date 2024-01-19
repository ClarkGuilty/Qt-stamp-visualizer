# This Python file uses the following encoding: utf-8
import argparse

import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
import glob
import os
import pandas as pd
import subprocess
from PIL import Image


from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Slot, QObject, QThread, Signal, QEvent
from PySide6.QtGui import QPixmap, QFont, QKeySequence, QShortcut, QIntValidator

from matplotlib.figure import Figure
from matplotlib import image as mpimg
import matplotlib.pyplot as plt


import urllib

from functools import partial
import shutil

import webbrowser
import json
from os.path import join
import re
import sys

parser = argparse.ArgumentParser(description='Configure the parameters of the execution.')
parser.add_argument('-p',"--path", help="Path to the images to inspect",
                    default="Stamps_to_inspect")
parser.add_argument('-N',"--name", help="Name of the classifying session.",
                    default="")
parser.add_argument('-l',"--gridsize", help="Number of stamps per side.",type=int,
                    default=10)
parser.add_argument('-s',"--seed", help="Seed used to shuffle the images.",type=int,
                    default=None)
parser.add_argument("--printname", help="Whether to print the name when you click",
                    action=argparse.BooleanOptionalAction,
                    default=False)
parser.add_argument("--page", help="Initial page",type=int,
                    default=None)
parser.add_argument('--resize',
                    help="Set to allow the resizing of the stamps with the window.",
                    action=argparse.BooleanOptionalAction,
                    default=False)
parser.add_argument('--fits',
                    help="Specify whether the images to classify are fits or png/jpeg.",
                    action=argparse.BooleanOptionalAction,
                    default=True)


args = parser.parse_args()

C_INTERESTING = 2
C_LENS = 1
C_UNINTERESTING = 0

def identity(x):
    return x

def log(x):
    "Simple log base 1000 function that ignores numbers less than 0"
    # return np.emath.logn(1000,x) #base 1000 like ds9
    return np.log(x, out=np.zeros_like(x), where=(x>0)) / np.log(1000)

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
    # print("heh")
    # print(latest_filename)
    # print(re_search)
    # print(re_match)
    # print(int_match)
    # print(file_iteration)
    # print("heh")


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
        self.lineEdit.setValidator(QIntValidator(1,total_pages+1))
        self.lineEdit.setText(str(initial_value+1))
        self.lineEdit.setFont(QFont("Arial",20))
        self.lineEdit.setStyleSheet('background-color: black; color: gray')
        self.lineEdit.setAlignment(Qt.AlignRight)
        layout.addWidget(self.lineEdit)

        self.total_pages = QtWidgets.QLabel()
        self.total_pages.setText("/ "+str(total_pages+1))
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

        # self.total_pages = QtWidgets.QLabel()
        # self.total_pages.setText("/ "+str(total_pages+1))
        # self.total_pages.setFont(QFont("Arial",20))
        # layout.addWidget(self.total_pages)

        # layout.addStretch()

    def setText(self, input):
        self.label.setText(str(input))

        
    def getValue(self):
        return int(self.lineEdit.text())-1



def Clickable(widget):
        "https://wiki.python.org/moin/PyQt/Making%20non-clickable%20widgets%20clickable"
        class Filter(QObject):
            clicked = Signal()
            def eventFilter(self, obj, event):
                if obj == widget:
                    if event.type() == QEvent.MouseButtonRelease:
                        if obj.rect().contains(event.pos()):
                            self.clicked.emit()
                            # The developer can opt for .emit(obj) to get the object within the slot.
                            return True
                return False
        filter = Filter(widget)
        widget.installEventFilter(filter)
        return filter.clicked

class AlignDelegate(QtWidgets.QStyledItemDelegate):
    "https://stackoverflow.com/a/54262963/10555034"
    def initStyleOption(self, option, index):
        super(AlignDelegate, self).initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter

class ClickableLabel(QtWidgets.QLabel):
    clicked = Signal(str)
    def __init__(self, filepath, lens_background_path,
                    interesting_background_path, deactivated_path, i,
                    status, activation, update_df_func, parent=None):
        QtWidgets.QLabel.__init__(self, parent)
        self.filepath = filepath
        self.is_activate = activation
        self.lens_background_path = lens_background_path
        self.interesting_background_path = interesting_background_path

        self.deactivated_path = deactivated_path
        self.is_a_candidate = status
        self.update_df_func = update_df_func
        self.i = i
        self.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                           QtWidgets.QSizePolicy.MinimumExpanding)
        self.setScaledContents(args.resize)

        if self.is_activate:
            if self.is_a_candidate == C_UNINTERESTING:
                self._pixmap = QPixmap(self.filepath)
            elif self.is_a_candidate == C_LENS:
                self._pixmap = QPixmap(self.lens_background_path)
            elif self.is_a_candidate == C_INTERESTING:
                self._pixmap = QPixmap(self.interesting_background_path)
        else:
            self._pixmap = QPixmap(self.deactivated_path)
        
        # self.target_width = min(66,self.width())
        # self.target_height = min(66,self.height()) #TODO: add case where width != height
        
        self.target_width = 100
        self.target_height = 100

        self.setPixmap(self._pixmap.scaled(
            self.target_width, self.target_height,
            Qt.KeepAspectRatio))

    def activate(self):
        self.is_activate = True

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

    def change_and_paint_pixmap(self, filepath):
        if self.is_activate:
            self.filepath = filepath
            self._pixmap = QPixmap(self.filepath)
            self.setPixmap(self._pixmap.scaled(
                self.width(), self.height(),
                Qt.KeepAspectRatio))

    def paint_pixmap(self):
        if self.is_activate:
            # target_width = min(66,self.width())
            # target_height = min(66,self.height()) #TODO: add case where width != height
            self.setPixmap(self._pixmap.scaled(
                # self.target_width, self.target_height,
                self.width(), self.height(),
                Qt.KeepAspectRatio
                ))
            # self.setPixmap(self._pixmap)

    def paint_background_pixmap(self, background_path):
        if self.is_activate:
            self._pixmap = QPixmap(background_path)
            self.setPixmap(self._pixmap.scaled(
                # self.target_width, self.target_height,
                self.width(), self.height(),
                Qt.KeepAspectRatio))

    def change_pixmap(self, filepath):
        self.filepath = filepath
        self._pixmap = QPixmap(self.filepath)

    # def mousePressEvent(self, event):
    #     # print(self.is_activate)
    #     if self.is_activate:
    #         self.is_a_candidate = not self.is_a_candidate
    #         modifiers = event.modifiers()
    #         print('local:' ,modifiers)
    #         if self.is_a_candidate:
    #             self.paint_background_pixmap()
    #         else:
    #             self.change_and_paint_pixmap(self.filepath)

    #         self.update_df_func(event, self.i)
    #     else:
    #         print('Inactive button')
    
    def mousePressEvent(self, event):
        # print(self.is_activate)
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
                    # print('Interesting',self.interesting_background_path)
                    new_class = C_INTERESTING
                elif modifiers == Qt.NoModifier:
                    self.paint_background_pixmap(self.lens_background_path)
                    new_class = C_LENS

            self.update_df_func(event, self.i, new_class)
            self.is_a_candidate = new_class
        else:
            print('Inactive button')

    def resizeEvent(self, event):
        self.setPixmap(self._pixmap.scaled(
            # 66, 66,
            self.width(), self.height(),
            Qt.KeepAspectRatio))


class MosaicVisualizer(QtWidgets.QMainWindow):
    def __init__(self, path_to_the_stamps= args.path):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self._main.setStyleSheet('background-color: black')
        self.setWindowTitle("Mosaic Visualizer")
        self.setCentralWidget(self._main)
        self.status = self.statusBar()
        self.random_seed = args.seed            
        self.stampspath = path_to_the_stamps
        self.scratchpath = './.temp'
        self.deactivated_path = './dark.png'
        os.makedirs(self.scratchpath, exist_ok=True)
        self.clean_dir(self.scratchpath)
        if args.fits:
            self.listimage = sorted([os.path.basename(x)
                                for x in glob.glob(join(self.stampspath, '*.fits'))])
        else:
            self.listimage = sorted([os.path.basename(x)
                                for x in (glob.glob(join(self.stampspath, '*.png')) +
                                          glob.glob(join(self.stampspath, '*.jpg')) +
                                          glob.glob(join(self.stampspath, '*.jpeg'))
                                         )])

        if self.random_seed is not None:
            # 99 is always changed to this number when sorting to mantain compatibility with old classifications.
            # 128 bits, proton decay might be more likely than someone *randomly* using this number.
            # Please, do not use this number as seed.
            seed_to_use = 120552782132343758881253061212639178445 if self.random_seed == 99 else self.random_seed
            # print("shuffling")
            rng = np.random.default_rng(seed_to_use)
            rng.shuffle(self.listimage) #inplace shuffling
        
        if len(self.listimage) == 0:
            print("WARNING: no images found in {}".format(self.stampspath))
        # print(join(self.stampspath, '*.fits'))
        # print(glob.glob(self.stampspath + '*.fits'))
        self.gridsize = args.gridsize
        self.gridarea = self.gridsize**2
        self.PAGE_MAX = int(np.floor(len(self.listimage) / self.gridarea))

        self.scale2funct = {'linear': identity,
                            'sqrt': np.sqrt,
                            'cbrt': np.cbrt,
                            'log': log,
                            # 'log10': log, #just for retrocompatibility
                            'asinh': asinh2}

        self.defaults = {
            # 'counter': 0,
            'page': 0, #Defaults to 0. Gets overwritten by --page argument.
            # 'total': -1,
            'colormap': 'gray',
            'scale': 'asinh',
            'name': args.name,
            'gridsize': args.gridsize
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

        if self.config_dict['page'] > self.PAGE_MAX:
            self.bcounter.setInputText(0)
            self.goto()
            
        self.prepare_png(self.gridsize**2)

        main_layout = QtWidgets.QVBoxLayout(self._main)
        stamp_grid_layout = QtWidgets.QGridLayout()
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
        Clickable(self.cbscale).connect(self.cbscale.showPopup)
        line_edit = self.cbscale.lineEdit()
        # self.cbscale.addItems(['linear','sqrt','' 'log', 'asinh'])
        self.cbscale.addItems(self.scale2funct.keys())
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
        self.listscales = ['gray','viridis','gist_yarg','hot']
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

        # print(self.df['classification'].sum().astype(int))
        self.bclickcounter = NamedLabel('Clicks', self.df['classification'].sum().astype(int))
        self.bclickcounter.setStyleSheet('background-color: black; color: gray')

        # button.clicked.connect(line_edit.clear)


        ##### Keyboard shortcuts
        self.knext = QShortcut(QKeySequence('f'), self)
        self.knext.activated.connect(self.next)

        self.kprev = QShortcut(QKeySequence('d'), self)
        self.kprev.activated.connect(self.prev)


        # self.blinear.clicked.connect(self.set_scale_linear)

        button_bar_layout.addWidget(self.cbscale)
        button_bar_layout.addWidget(self.cbcolormap)
        button_bar_layout.addWidget(self.bprev)
        button_bar_layout.addWidget(self.bnext)
        page_counter_layout.addWidget(self.bclickcounter)
        page_counter_layout.addWidget(self.bcounter)


        self.total_n_frame = int(len(self.listimage)/(self.gridsize**2))
        start = self.config_dict['page']*self.gridarea
        # print("Page: ",self.config_dict['page'])
        # n_images = len(self.df) % self.gridarea if self.config_dict['page'] == self.PAGE_MAX else self.gridarea

        for i in range(start,start+self.gridarea):
            filepath = self.filepath(i, self.config_dict['page'])
            try:
                classification = self.df.iloc[i,self.df.columns.get_loc('classification')]
                activation = True
            except IndexError:
                classification = False
                activation = False

            button = ClickableLabel(filepath, self.lens_background_path,
                                    self.interesting_background_path,
                                    self.deactivated_path,
                                    i-start, classification, activation,
                                    self.my_label_clicked)
            stamp_grid_layout.addWidget(
                # button, i // self.gridsize, i % self.gridsize)
                button, i % self.gridsize, i // self.gridsize)
            self.buttons.append(button)
            button.setAlignment(Qt.AlignCenter)
            # button.adjustSize()

#

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
            self.config_dict['page'] = self.bcounter.getValue()
            self.update_grid()
            # self.bcounter.setInputText(self.config_dict['page'])
            self.save_dict()

    @Slot()
    def next(self):
        self.config_dict['page'] = self.config_dict['page'] + 1
        if self.config_dict['page']>self.PAGE_MAX:
            # self.config_dict['counter']=self.PAGE_MAX
            self.config_dict['page']=self.PAGE_MAX
            self.status.showMessage('You are already at the last page',10000)
        else:
            self.clean_dir(self.scratchpath)
            self.update_grid()
            self.bcounter.setInputText(self.config_dict['page'])
            self.save_dict()

    @Slot()
    def prev(self):
        self.config_dict['page'] = self.config_dict['page'] - 1
        if self.config_dict['page'] < 0:
            self.config_dict['page'] = 0
            self.status.showMessage('You are already at the first page',10000)
        else:
            self.clean_dir(self.scratchpath)
            self.update_grid()
            self.bcounter.setInputText(self.config_dict['page'])
            self.save_dict()

    def change_scale(self,i):
        self.config_dict['scale'] = self.cbscale.currentText()
        self.update_grid()
        self.save_dict()

    def change_colormap(self,i):
        self.config_dict['colormap'] = self.cbcolormap.currentText()
        self.update_grid()
        self.save_dict()

    # def my_label_clicked(self, event, i):
    #     button = event.button()
    #     modifiers = event.modifiers()
    #     print(f'official mod {modifiers}')
    #     print(Qt.NoModifier)
    #     print(modifiers == Qt.NoModifier)
    #     if modifiers == Qt.NoModifier and button == Qt.LeftButton:
    #         if self.config_dict['page']*self.gridarea+i > len(self.listimage):
    #             print('Not an image')
    #         else:
    #             self.df.iloc[self.gridarea*self.config_dict['page']+i,
    #                         self.df.columns.get_loc('grid_pos')] = i+1
    #             print(self.df.iloc[self.gridarea*self.config_dict['page']+i,
    #                         self.df.columns.get_loc('file_name')]) if args.printname else True
    #             self.df.iloc[self.gridarea*self.config_dict['page']+i,
    #                         self.df.columns.get_loc('classification')] = int(self.buttons[i].is_a_candidate)

    #             self.bclickcounter.setText(self.df['classification'].sum().astype(int))
    #             self.df.to_csv(
    #                 self.df_name, index=False)

    def my_label_clicked(self, event, i, new_class):
        button = event.button()
        modifiers = event.modifiers()
        # if modifiers == Qt.NoModifier and button == Qt.LeftButton:
        if self.config_dict['page']*self.gridarea+i > len(self.listimage):
            print('Something is wrong. This condition should not be trigger.')
        else:
            object_index = self.gridarea*self.config_dict['page']+i
            self.df.iloc[object_index,
                        self.df.columns.get_loc('grid_pos')] = i+1
            print(self.df.iloc[object_index,
                        self.df.columns.get_loc('file_name')]) if args.printname else True
            self.df.iloc[object_index,
                        self.df.columns.get_loc('classification')] = new_class

            self.bclickcounter.setText((self.df['classification'] == C_LENS).sum().astype(int))
            self.df.to_csv(
                self.df_name, index=False)

    def filepath(self, i, page):
        return join(self.scratchpath, str(i+1)+self.config_dict['scale'] + self.config_dict['colormap'] + str(page)+'.png')

    def save_dict(self):
        with open('.config_mosaic.json', 'w') as f:
            json.dump(self.config_dict, f, ensure_ascii=False, indent=4)

    # def load_dict(self):
    #     try:
    #         with open('.config_mosaic.json', ) as f:
    #             return json.load(f)
    #     except FileNotFoundError:
    #         print("Loaded default configuration.")
    #         return self.defaults

    def load_dict(self):
        try:
            with open('.config_mosaic.json', ) as f:
                temp_dict = json.load(f)
                if ((temp_dict['name'] != args.name) or
                    (temp_dict['gridsize'] != args.gridsize)):
                    temp_dict['name'] = args.name
                    temp_dict['name'] = args.gridsize
                if args.page is not None:
                    temp_dict['page'] = args.page
                if temp_dict['scale'] == 'log10':
                    temp_dict['scale'] = 'log'
                return temp_dict
        except FileNotFoundError:
            print("Loaded default configuration.")
            return self.defaults

    def obtain_df(self):
        if self.random_seed is None:
            base_filename = 'classification_mosaic_autosave_{}_{}_{}_99'.format(
                                    args.name,len(self.listimage),self.gridsize)
            string_to_glob = './Classifications/{}*.csv'.format(base_filename)
            print("Globing for", string_to_glob)
            string_to_glob_for_files_with_seed = './Classifications/{}_*.csv'.format(base_filename)
            glob_results = set(glob.glob(string_to_glob)) - set(glob.glob(string_to_glob_for_files_with_seed))
        else:
            base_filename = base_filename = 'classification_mosaic_autosave_{}_{}_{}_{}'.format(
                                    args.name,len(self.listimage),self.gridsize,self.random_seed)
            string_to_glob = './Classifications/{}*.csv'.format(base_filename)
            glob_results = glob.glob(string_to_glob)

        # string_to_glob = './Classifications/classification_mosaic_autosave_{}_{}_{}_{}*.csv'.format(
        #                             args.name,len(self.listimage),self.gridsize, str(self.random_seed))
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
                return df
            else:
                print("Classification file corresponds to a different dataset.")
                string_tested = os.path.basename(self.df_name).split(".csv")[0]
                file_iteration = find_filename_iteration(string_tested)
        
        self.dfc = ['file_name', 'classification', 'grid_pos','page']
        # self.df_name = './Classifications/classification_mosaic_autosave_{}_{}_{}_{}{}.csv'.format(
        #                             args.name,len(self.listimage),self.gridsize,str(self.random_seed),
        #                             file_iteration)
        self.df_name = './Classifications/{}{}.csv'.format(base_filename,file_iteration)
        print('A new csv will be created', self.df_name)
        if file_iteration != "":
            print("To avoid this in the future use the argument `-N name` and give different names to different datasets.")
        df = pd.DataFrame(columns=self.dfc)
        df['file_name'] = self.listimage
        df['classification'] = np.zeros(np.shape(self.listimage))
        df['page'] = np.zeros(np.shape(self.listimage))
        df['grid_pos'] = np.zeros(np.shape(self.listimage))
        return df

    def update_grid(self):
        start = self.config_dict['page']*self.gridarea
        n_images = self.gridarea
        self.prepare_png(n_images)

        i = start
        j = 0
        for button in self.buttons:
            try:
                status = self.df.iloc[self.gridarea*self.config_dict['page']+j,self.df.columns.get_loc('classification')]
                if status == 0:
                    button.activate()
                    button.change_and_paint_pixmap(self.filepath(i,self.config_dict['page']))
                    button.set_candidate_status(status)
                else:
                    button.activate()
                    button.change_pixmap(self.filepath(i,self.config_dict['page']))
                    button.paint_background_pixmap(self.status2background_dict[status])
                    button.set_candidate_status(status)

                self.df.iloc[self.gridarea*self.config_dict['page']+j,
                             self.df.columns.get_loc('grid_pos')] = j+1

                self.df.iloc[self.gridarea*self.config_dict['page']+j,
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
            for i in np.arange(start, start + number + 1):
                if args.fits:
                    try:
                        image = self.read_fits(i)
                        scaling_factor = np.percentile(image,q=90)
                        if scaling_factor == 0:
                            scaling_factor = np.percentile(image,q=99)
                            scaling_factor = 1
                        image = image / scaling_factor*300 #Rescaling for better visualization.
                        # if image.shape[0] in [200,334]: #Special casen for HST/HSC COSMOS stamps
                            # image -= np.min(image) 
                            # image *= 7000 #TODO Set all the magnitudes to CFIS's
                        scale_min, scale_max = self.scale_val(image)
                        image = self.rescale_image(image, scale_min, scale_max)
                        plt.imsave(self.filepath(i, self.config_dict['page']),
                                image, cmap=self.config_dict['colormap'], origin="lower")
                    except:
                        image = np.zeros((66, 66))# * 0.0000001
                        plt.imsave(self.filepath(i, self.config_dict['page']),
                            image, cmap=self.config_dict['colormap'], origin="lower")
                else:
                    if i < len(self.listimage):
                        try:
                            original_filepath = join(self.stampspath, self.listimage[i])
                            shutil.copyfile(original_filepath, self.filepath(i, self.config_dict['page']))
                        except:
                            print('file not found: {}'.format(original_filepath) )
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
        
        if image_array.shape[0] > 100:
            box_size_vmin = np.round(np.sqrt(np.prod(image_array.shape) * 0.001)).astype(int)
            box_size_vmax = np.round(np.sqrt(np.prod(image_array.shape) * 0.01)).astype(int)
        else:
            box_size_vmin = 5
            box_size_vmax = 14
        vmin = np.min(self.background_rms_image(box_size_vmin, image_array))
        # print(f'{self.background_rms_image(box_size_vmin, image_array) = }')
        if vmin == 0:
            vmin += 1e-3              
        
        
        xl, yl = np.shape(image_array)
        xmin = int((xl) / 2. - (box_size_vmax / 2.))
        xmax = int((xl) / 2. + (box_size_vmax / 2.))
        ymin = int((yl) / 2. - (box_size_vmax / 2.))
        ymax = int((yl) / 2. + (box_size_vmax / 2.))
        vmax = np.max(image_array[xmin:xmax, ymin:ymax])
        # print(vmin,vmax)
        # return max(0,vmin), vmax*1.5
        # print(vmin, vmax*1.5)
        return vmin, vmax*1.5
        


    def scale_val_percentile(self,image_array,p_min=0.1,p_max=99.9):
        # image_to_plot = np.clip(image_array,np.percentile(p_min),np.percentile(p_max))
        # print(np.percentile(image_array,p_min),np.percentile(image_array,p_max))
        return np.percentile(image_array,p_min),np.percentile(image_array,p_max)

    def background_rms_image(self, cb, image):
        xg, yg = np.shape(image)
        cut0 = image[0:cb, 0:cb]
        cut1 = image[xg - cb:xg, 0:cb]
        cut2 = image[0:cb, yg - cb:yg]
        cut3 = image[xg - cb:xg, yg - cb:yg]
        l = [cut0, cut1, cut2, cut3]
        # m = np.mean(np.mean(l, axis=1), axis=1)
        m = np.mean(np.mean(l, axis=1), axis=1)
        ml = min(m)
        mm = max(m)
        if mm > 5 * ml:
            s = np.sort(l, axis=0)
            nl = s[:-1]
            std = np.std(nl)
        else:
            std = np.std([cut0, cut1, cut2, cut3])
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
