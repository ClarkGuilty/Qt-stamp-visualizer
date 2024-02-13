# This Python file uses the following encoding: utf-8
import sys
#from PySide6.QtWidgets import QApplication, QWidget, QPushButton
import PySide6

#import time
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
import glob
import os
import pandas as pd
import subprocess
from PIL import Image

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Slot, QObject, QThread, Signal
from PySide6.QtGui import QPixmap, QKeySequence, QShortcut

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import image as mpimg

import urllib

from functools import partial

import webbrowser
import json
from os.path import join

import argparse
import re

parser = argparse.ArgumentParser(description='configure the parameters of the execution.')
parser.add_argument('-p',"--path", help="path to the images to inspect",
                    default="Stamps_to_inspect")
parser.add_argument('-N',"--name", help="name of the classifying session.",
                    default=None)
parser.add_argument("--reset-config", help="removes the configuration dictionary during startup.",
                    action="store_true", default=False)
parser.add_argument("--verbose", help="activates loging to terminal",
                    action="store_true", default=False)
parser.add_argument("--clean", help="cleans the legacy survey folder.",
                    action="store_true")
parser.add_argument('--fits',
                    help=("forces app to only use fits (--fits) or png/jp(e)g (--no-fits). "+
                    "If unset, the app searches for fits files in the path, but defaults to "+
                    "png/jp(e)g if no fits files are found."),
                    action=argparse.BooleanOptionalAction,
                    default=None)
parser.add_argument('-s',"--seed", help="seed used to shuffle the images.",type=int,
                    default=None)

args = parser.parse_args()

LEGACY_SURVEY_PATH = './Legacy_survey/'

if args.reset_config:
    os.remove('.config.json')

if args.clean:
    for f in glob.glob(join(LEGACY_SURVEY_PATH,"*.jpg")):
        if os.path.exists(f):
            os.remove(f)

def identity(x):
    return x

def log(x):
    return np.emath.logn(1000,x) #base 1000 like ds9

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
    
    return f"-({int_match+1})"

class SingleFetchWorker(QObject):
    successful_download = Signal()
    failed_download = Signal()
    has_finished = Signal()

    def __init__(self, url, savefile, title):
        # super.__init__(self)
        super(SingleFetchWorker, self).__init__()
        self.url = url
        self.savefile = savefile
        self.title = title
    
    @Slot()
    def run(self):
        # print(f'Running file worker: {self.savefile}')
        if self.url == '':
            # print('There is already a file')
            self.successful_download.emit()
        else:
            try:
                urllib.request.urlretrieve(self.url, self.savefile)
                # print('Success!!!!')
                self.successful_download.emit()
            except urllib.error.HTTPError:
                with open(self.savefile,'w') as f:
                    Image.fromarray(np.zeros((66,66),dtype=np.uint8)).save(f)
                # self.failed_download.emit('No Legacy Survey data available.')
                self.failed_download.emit()
        self.has_finished.emit()
        # self.deleteLater()

class FetchThread(QThread):
    def __init__(self, df, initial_counter, parent=None):
#            super().__init__(parent)
            QThread.__init__(self, parent)

            self.df = df
            self.initial_counter = initial_counter
            self.legacy_survey_path = LEGACY_SURVEY_PATH
            self.stampspath = args.path
            self.listimage = sorted([os.path.basename(x) for x in glob.glob(join(self.stampspath,'*.fits'))])
            self.im = Image.fromarray(np.zeros((66,66),dtype=np.uint8))
    def download_legacy_survey(self, ra, dec, size=47,residual=False): #pixscale = 0.04787578125 for 66 pixels in CFIS.
        pixscale = '0.262'
        residual = (residual and size == 47)
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
        return sky[0][0], sky[1][0]

    def interrupt(self):
        self._active = False

    def run(self):
        index = self.initial_counter
        self._active = True
        while self._active and index < len(self.df): 
            stamp = self.df.iloc[index]
            if np.isnan(stamp['ra']) or np.isnan(stamp['dec']): #TODO: add smt for when there is no RADec.
                f = join(self.stampspath,self.listimage[index])
                ra,dec = self.get_ra_dec(fits.getheader(f,memmap=False))
            else:
                ra,dec = stamp[['ra','dec']]
            self.download_legacy_survey(ra,dec,size=47)
            self.download_legacy_survey(ra,dec,size=47,residual=True)
            # self.download_legacy_survey(ra,dec,300)
            self.download_legacy_survey(ra,dec,size=488)
            # self.download_legacy_survey(ra,dec,pixscale='0.048',residual=True)
            index+=1
        return 0

class ApplicationWindow(QtWidgets.QMainWindow):
    # workerThread = QThread()
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.status = self.statusBar()
        self.defaults = {
                    'counter':0,
                    'legacysurvey':False,
                    'legacybigarea':False,
                    'legacyresiduals':False,
                    'prefetch':False,
                    'autonext':True,
                    'prefetch':False,
                    'colormap':'gray',
                    'scale':'log',
                    'keyboardshortcuts':False,
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
        self.scale2funct = {'identity':identity,
                            'sqrt':np.sqrt,
                            'log':log,
                            'log10':log,
                            'cbrt':np.cbrt,
                            'asinh2':asinh2}
        self.scale = self.scale2funct[self.config_dict['scale']]

        title_strings = ["Sequential stamp visualizer"]
        if args.name is not None:
            self.name = args.name
            title_strings.append(self.name)
        else:
            self.name = ''
        self.setWindowTitle(' - '.join(title_strings))

        self.stampspath = args.path
        self.legacy_survey_path = LEGACY_SURVEY_PATH
        # self.listimage = sorted([os.path.basename(x) for x in glob.glob(join(self.stampspath,'*.fits'))])
        self.random_seed = args.seed

        
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


        if len(self.listimage) < 1:
            sys.exit()
        if self.config_dict['counter'] > len(self.listimage):
            self.config_dict['counter'] = 0
        
        if self.random_seed is not None:
            # print("shuffling")
            rng = np.random.default_rng(self.random_seed)
            rng.shuffle(self.listimage) #inplace shuffling
        

        self.df = self.obtain_df()

        self.number_graded = 0
        self.COUNTER_MIN =0
        self.COUNTER_MAX = len(self.listimage)
        self.filename = join(self.stampspath, self.listimage[self.config_dict['counter']])
        # self.status.showMessage(self.listimage[self.config_dict['counter']],)


        self.legacy_survey_qlabel = QtWidgets.QLabel(alignment=Qt.AlignCenter)
#        pixmap = QPixmap()
        #self.figure.gca().set_facecolor('black')

        main_layout = QtWidgets.QVBoxLayout(self._main)
        self.label_layout = QtWidgets.QHBoxLayout()
        self.plot_layout = QtWidgets.QHBoxLayout()
        button_layout = QtWidgets.QVBoxLayout()
        button_row0_layout = QtWidgets.QHBoxLayout()
        button_row10_layout = QtWidgets.QHBoxLayout()
        button_row11_layout = QtWidgets.QHBoxLayout()
        button_row2_layout = QtWidgets.QHBoxLayout()
        button_row3_layout = QtWidgets.QHBoxLayout()

        self.counter_widget = QtWidgets.QLabel("{}/{}".format(self.config_dict['counter']+1,self.COUNTER_MAX))
        self.counter_widget.setStyleSheet("font-size: 14px")
#        self.status.addPermanentWidget(self.counter_widget)


        self.label_plot = [QtWidgets.QLabel(self.listimage[self.config_dict['counter']], alignment=Qt.AlignCenter),
                            QtWidgets.QLabel("Legacy Survey", alignment=Qt.AlignCenter)]
        font = [x.font() for x in self.label_plot]
        for i in range(len(font)):
            font[i].setPointSize(16)
            self.label_plot[i].setFont(font[i])
        self.label_layout.addWidget(self.label_plot[0])

        self.plot_layout.setSpacing(0)
        self.plot_layout.setContentsMargins(0,0,0,0)

        self.figure = [Figure(figsize=(5,3),layout="constrained",facecolor='black'),
            Figure(figsize=(5,3),layout="constrained",facecolor='black')]
        #self.figure = [Figure(),Figure(figsize=(5,3))]
#        self.figure[0].tight_layout()

        self.canvas = [FigureCanvas(self.figure[0]),FigureCanvas(self.figure[1])]
        self.canvas[0].setStyleSheet('background-color: blue')
        self.plot_layout.addWidget(self.canvas[0])

        self.ax = [self.figure[0].subplots(),self.figure[1].subplots()]


        self.label_layout.addWidget(self.label_plot[1])
        self.plot_layout.addWidget(self.canvas[1])

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

        self.bviewls = QtWidgets.QPushButton('View LS')
        self.bviewls.clicked.connect(self.viewls)
        if self.filetype != 'FITS':
            self.bviewls.setEnabled(False)
        list_button_row0_layout.append(self.bviewls)

        self.blegsur = QtWidgets.QCheckBox('Legacy Survey (LS)')
        self.blegsur.clicked.connect(self.checkbox_legacy_survey)
        if self.filetype == 'FITS':
            if not self.config_dict['legacysurvey']:
                    self.label_plot[1].hide()
                    self.canvas[1].hide()
            else:
                    self.label_plot[1].show()
                    self.canvas[1].show()
                    self.blegsur.toggle()
                    self.set_legacy_survey()
        else:
            self.config_dict['legacysurvey'] = False
            self.blegsur.setEnabled(False)
            self.label_plot[1].hide()
            self.canvas[1].hide()
        list_button_row0_layout.append(self.blegsur)


        self.blsarea = QtWidgets.QCheckBox("Large FoV")
        self.blsarea.clicked.connect(self.checkbox_ls_change_area)
        if self.filetype == 'FITS':
            if self.config_dict['legacybigarea']:
                self.blsarea.toggle()
                if self.config_dict['legacysurvey']:
                    self.set_legacy_survey()
        #            self.checkbox_ls_change_area()
        else:
            self.blsarea.setEnabled(False)
            self.config_dict['legacybigarea'] = False
        list_button_row0_layout.append(self.blsarea)

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

        self.bprefetch = QtWidgets.QCheckBox("Pre-fetch")
        self.bprefetch.clicked.connect(self.prefetch_legacysurvey)
        if self.filetype == 'FITS':
            if self.config_dict['prefetch']:
                self.config_dict['prefetch'] = False
                self.prefetch_legacysurvey()
                self.bprefetch.toggle()
    #            self.checkbox_ls_change_area()
        else:
            self.bprefetch.setEnabled(False)
            self.config_dict['prefetch'] = False
        list_button_row0_layout.append(self.bprefetch)


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

        list_subclassifications = []
        self.bMerger = QtWidgets.QPushButton('Merger')
        self.bMerger.clicked.connect(partial(self.classify, 'X','Merger') )
        list_subclassifications.append(self.bMerger)

        self.bSpiral = QtWidgets.QPushButton('Spiral')
        self.bSpiral.clicked.connect(partial(self.classify, 'X','Spiral'))
        list_subclassifications.append(self.bSpiral)

        self.bRing = QtWidgets.QPushButton('Ring')
        self.bRing.clicked.connect(partial(self.classify, 'X','Ring'))
        list_subclassifications.append(self.bRing)

        self.bElliptical = QtWidgets.QPushButton('Elliptical')
        self.bElliptical.clicked.connect(partial(self.classify, 'X','Elliptical'))
        list_subclassifications.append(self.bElliptical)

        self.bDisc = QtWidgets.QPushButton('Disc')
        self.bDisc.clicked.connect(partial(self.classify, 'X','Disc'))
        list_subclassifications.append(self.bDisc)

        self.bEdgeon = QtWidgets.QPushButton('Edge-on')
        self.bEdgeon.clicked.connect(partial(self.classify, 'X','Edge-on'))
        list_subclassifications.append(self.bEdgeon)

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

        self.dict_subclass2button = {'Merger':self.bMerger,
                                  'Spiral':self.bSpiral,
                                  'Ring':self.bRing,
                                  'Elliptical':self.bElliptical,
                                  'Disc':self.bDisc,
                                  'Edge-on':self.bEdgeon,
                                  'A':None,
                                  'B':None,
                                  'C':None,
                                  'X':None,
                                  'SL':None,
                                  'ML':None,
                                  'FL':None,
                                  'NL':None,
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

        # self.basinh = QtWidgets.QPushButton('Asinh')
        # self.basinh.clicked.connect(self.set_scale_asinh)
        # self.basinh.clicked.connect(partial(self.set_scale,self.basinh,'asinh2'))
        # list_scales_buttons.append(self.basinh)

        list_colormap_buttons = []
        self.bInverted = QtWidgets.QPushButton('Inverted')
        self.bInverted.clicked.connect(partial(self.set_colormap,self.bInverted,'gist_yarg'))
        list_colormap_buttons.append(self.bInverted)

        self.bBb8 = QtWidgets.QPushButton('Bb8')
        self.bBb8.clicked.connect(partial(self.set_colormap,self.bBb8,'hot'))
        list_colormap_buttons.append(self.bBb8)

        self.bGray = QtWidgets.QPushButton('Gray')
        self.bGray.clicked.connect(partial(self.set_colormap,self.bGray,'gray'))
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
                                'gray':self.bGray,
                                'viridis': self.bViridis}

        self.bactivatedclassification = None
        self.bactivatedsubclassification = None
        self.bactivatedscale = self.scale2button[self.config_dict['scale']]
        self.bactivatedcolormap = self.colormap2button[self.config_dict['colormap']]

        grade = self.df.at[self.config_dict['counter'],'classification']
        if grade is not None and grade != 'None' and grade != 'Empty':
            self.bactivatedclassification = self.dict_class2button[grade]
            self.bactivatedclassification.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))
 
        subgrade = self.df.at[self.config_dict['counter'],'subclassification']
        if subgrade is not None and subgrade != 'None' and grade != 'Empty':
            self.bactivatedsubclassification = self.dict_subclass2button[subgrade]
            if self.bactivatedsubclassification is not None:
                self.bactivatedsubclassification.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))


        self.bactivatedscale.setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))
        self.bactivatedcolormap.setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))

#        self.bactivatedscale.setStyleSheet("background-color : white;color : black;".format(self.buttoncolor))
#        self.bactivatedscale = self.sender()

        #Keyboard shortcuts
        self.ksurelens = QShortcut(QKeySequence('q'), self)
        self.ksurelens.activated.connect(partial(self.keyClassify, 'A','A'))

        self.kmaybelens = QShortcut(QKeySequence('w'), self)
        self.kmaybelens.activated.connect(partial(self.keyClassify, 'B','B'))

        self.kflexion = QShortcut(QKeySequence('e'), self)
        self.kflexion.activated.connect(partial(self.keyClassify, 'C','C'))

        self.knonlens = QShortcut(QKeySequence('r'), self)
        self.knonlens.activated.connect(partial(self.keyClassify, 'X','X'))

        self.kMerger = QShortcut(QKeySequence('a'), self)
        self.kMerger.activated.connect(partial(self.keyClassify, 'X','Merger'))

        self.kSpiral = QShortcut(QKeySequence('s'), self)
        self.kSpiral.activated.connect(partial(self.keyClassify, 'X','Spiral'))

        self.kRing = QShortcut(QKeySequence('d'), self)
        self.kRing.activated.connect(partial(self.keyClassify, 'X','Ring'))

        self.kElliptical = QShortcut(QKeySequence('f'), self)
        self.kElliptical.activated.connect(partial(self.keyClassify, 'X','Elliptical'))

        self.kDisc = QShortcut(QKeySequence('g'), self)
        self.kDisc.activated.connect(partial(self.keyClassify, 'X','Disc'))

        self.kEdgeon = QShortcut(QKeySequence('h'), self)
        self.kEdgeon.activated.connect(partial(self.keyClassify, 'X','Edge-on'))

        for button in list_button_row0_layout:
            button_row0_layout.addWidget(button)

        button_row0_layout.addWidget(self.counter_widget,alignment=Qt.AlignRight)

        for button in list_classifications:
            button_row10_layout.addWidget(button)

        for button in list_subclassifications:
            button_row11_layout.addWidget(button)

        for button in list_scales_buttons:
            button_row2_layout.addWidget(button)

        for button in list_colormap_buttons:
            button_row3_layout.addWidget(button)

        # button_row3_layout.addWidget(self.bInverted)
        # button_row3_layout.addWidget(self.bBb8)
        # button_row3_layout.addWidget(self.bGray)
        # button_row3_layout.addWidget(self.bViridis)

        button_layout.addLayout(button_row0_layout, 25)
        button_layout.addLayout(button_row10_layout, 25)
        button_layout.addLayout(button_row11_layout, 25)
        if self.filetype == 'FITS':
            button_layout.addLayout(button_row2_layout, 25)
            button_layout.addLayout(button_row3_layout, 25)
        else:
            print("Use fits images to change colormap and colorscale.")


        main_layout.addLayout(self.label_layout, 2)
        main_layout.addLayout(self.plot_layout, 88)
        main_layout.addLayout(button_layout, 10)


    @Slot()
    def prefetch_legacysurvey(self):
        if self.config_dict['prefetch']:
            # self.fetchthread.quit()
            self.fetchthread.terminate()
            # self.fetchthread.interrupt()
            # print(self.fetchthread._active)
            self.config_dict['prefetch'] = False
        else:
            self.fetchthread = FetchThread(self.df,self.config_dict['counter'],) #Always store in an object.
            self.fetchthread.finished.connect(self.fetchthread.deleteLater)
            self.fetchthread.setTerminationEnabled(True)
            self.fetchthread.start()
            self.config_dict['prefetch'] = True


    @Slot()
    def goto(self):
        i, ok = QtWidgets.QInputDialog.getInt(self, 'Visual inspection', '',self.config_dict['counter']+1,1,self.COUNTER_MAX+1)
        if ok:
            self.config_dict['counter'] = i-1
            self.filename = join(self.stampspath,self.listimage[self.config_dict['counter']])
            # if self.singlefetchthread_active:
            #     self.singlefetchthread.terminate()
            self.plot()
            if self.config_dict['legacysurvey']:
            # if self.blegsur.isChecked():
                self.set_legacy_survey()

            self.update_classification_buttoms()
            self.update_counter()
            self.save_dict()


    def save_dict(self):
        with open('.config.json', 'w') as f:
        # Write the dictionary to the file in JSON format
#            json.dump(self.config_dict , f)
            json.dump(self.config_dict, f, ensure_ascii=False, indent=4)

    def load_dict(self):
        try:
            with open('.config.json', ) as f:
                return json.load(f)
        except FileNotFoundError:
            return self.defaults


    def update_counter(self):
#        self.config_dict['counter']+1
        self.counter_widget.setText("{}/{}".format(self.config_dict['counter']+1,self.COUNTER_MAX))

    def keyClassify(self, grade, subgrade):
        if self.config_dict['keyboardshortcuts'] == True:
            self.classify(grade, subgrade)
        
    @Slot()
    def classify(self, grade, subgrade):
        cnt = self.config_dict['counter']# - 1
#        self.df.at[cnt,'file_name'] = self.filename
        assert self.df.at[cnt,'file_name'] == self.listimage[self.config_dict['counter']] #TODO handling this possibility better.
        self.df.at[cnt,'classification'] = grade
        self.df.at[cnt,'subclassification'] = subgrade
        if self.filetype == 'FITS':
            self.df.at[cnt,'ra'] = self.ra
            self.df.at[cnt,'dec'] = self.dec
        self.df.at[cnt,'comment'] = grade
#        print('updating '+'classification_autosave'+str(self.nf)+'.csv file')
        # self.df.to_csv(self.df_name, index=False)
        self.df.to_csv(self.df_name)

        self.update_classification_buttoms()
        if self.config_dict['autonext']:
            self.next()
        

    def generate_legacy_survey_filename_url(self,ra,dec,pixscale='0.048',residual=False,size=47):
        # residual = (residual and pixscale == '0.048')
        pixscale = '0.262'
        residual = (residual and size == 47)
        res = '-resid' if residual else '-grz'
        savename = 'N' + '_' + str(ra) + '_' + str(dec) +f"_{size}" + f'ls-dr10{res}.jpg'
        savefile = os.path.join(self.legacy_survey_path, savename)        
        if os.path.exists(savefile):
            # print(savefile)
            return savefile, ''
        self.status.showMessage("Downloading legacy survey jpeg.")
        # print(url)
        url = (f'http://legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}'+
         f'&layer=ls-dr10{res}&size={size}&pixscale={pixscale}')
        return savefile, url

    def generate_title(self, residuals=False, bigarea=False):
        if residuals:
            return "Residuals, {0} arcsec x {0} arcsec".format(12.56)
        if bigarea:
            return '{0} arcmin x {0} arcmin'.format(2.13)
        return "{0}''x{0}''".format(12.56)

    def plot_legacy_survey(self, savefile, title, canvas_id = 1):
        self.label_plot[canvas_id].setText(title)
        self.ax[canvas_id].cla()
        if savefile != self.legacy_filename:
            return
        self.ax[canvas_id].imshow(mpimg.imread(savefile))
        self.ax[canvas_id].set_axis_off()
        self.canvas[canvas_id].draw()

    def plot_no_legacy_survey(self, title='Waiting for data',
                            canvas_id = 1, colormap='Greys_r'):
        self.label_plot[canvas_id].setText(title)
        self.ax[canvas_id].cla()
        self.ax[canvas_id].imshow(np.zeros((66,66)), cmap=colormap)
        self.ax[canvas_id].set_axis_off()
        self.canvas[canvas_id].draw()

    @Slot()
    def set_legacy_survey(self):
        pixscale = '0.5' if self.config_dict['legacybigarea'] else '0.048'
        size = 488 if self.config_dict['legacybigarea'] else 47
        try:
            savefile, url = self.generate_legacy_survey_filename_url(self.ra,self.dec,
                                        pixscale=pixscale,
                                        residual=self.config_dict['legacyresiduals'],
                                        size=size) #TODO Check for bugs

            title = self.generate_title(residuals=self.config_dict['legacyresiduals'],
                                        bigarea=self.config_dict['legacybigarea'])
            # print('setting ls')
            if url == '':
                # print('There is already a file')
                self.legacy_filename = savefile
                # self.plot_legacy_survey(title = 'There is already an image')
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
                            canvas_id = 1, colormap='viridis'))
            # self.singleFetchWorker.failed_download.connect(self.plot_no_legacy_survey)
            self.workerThread.finished.connect(self.workerThread.deleteLater)
            self.workerThread.setTerminationEnabled(True)

            self.workerThread.start()
            self.workerThread.quit()
            # self.singleFetchWorker.has_finished.connect(self.singleFetchWorker.deleteLater)
            # self.singleFetchWorker.has_finished.connect(self.workerThread.deleteLater)
        
        except FileNotFoundError as E:
            # print("File not found during seting_legacy_survey()")
            self.plot_no_legacy_survey()
            # print(E.args)
            # print(type(E))
            # raise
        except Exception as E:
            print("Exception while setting up the Legacy Survey image:")
            print(E.args)
            print(type(E))
            # raise


    @Slot()
    def checkbox_legacy_survey(self):
        if self.config_dict['legacysurvey']:
        # if self.blegsur.isChecked():
                self.label_plot[1].hide()
                self.canvas[1].hide()
        else:
                self.label_plot[1].show()
                self.canvas[1].show()
                self.set_legacy_survey()

        self.config_dict['legacysurvey'] = not self.config_dict['legacysurvey']

    @Slot()
    def checkbox_ls_change_area(self):
        self.config_dict['legacybigarea'] = not self.config_dict['legacybigarea']
        if self.config_dict['legacysurvey']:
        # if self.blegsur.isChecked():
                    self.set_legacy_survey()

    @Slot()
    def checkbox_ls_use_residuals(self):
        self.config_dict['legacyresiduals'] = not self.config_dict['legacyresiduals']
        if self.config_dict['legacysurvey']:
        # if self.blegsur.isChecked():
                    self.set_legacy_survey()


    @Slot()
    def checkbox_auto_next(self):
        self.config_dict['autonext'] = not self.config_dict['autonext']

    @Slot()
    def checkbox_keyboard_shortcuts(self):
        self.config_dict['keyboardshortcuts'] = not self.config_dict['keyboardshortcuts']


    @Slot()
    def open_ds9(self):
        subprocess.Popen(["ds9", '-fits',self.filename, '-zoom','8'  ])

    @Slot()
    def viewls(self):
        webbrowser.open("https://www.legacysurvey.org/viewer?ra={}&dec={}&layer=ls-dr10-grz&zoom=16&spectra".format(self.ra,self.dec))
        #subprocess.Popen(["ds9", '-fits',self.filename, '-zoom','8'  ])

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

    def background_rms_image(self,cb, image):
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
        if image_array.shape[0] > 170:
            box_size_vmin = np.round(np.sqrt(np.prod(image_array.shape) * 0.001)).astype(int)
            box_size_vmax = np.round(np.sqrt(np.prod(image_array.shape) * 0.01)).astype(int)
        else:
            #Sensible default values
            box_size_vmin = 5
            box_size_vmax = 14

        if len(np.shape(image_array)) == 2:
            image_array = [image_array]
        vmin = np.nanmin([self.background_rms_image(box_size_vmin, image_array[i]) for i in range(len(image_array))])
        
        xl, yl = np.shape(image_array[0])
        xmin = int((xl) / 2. - (box_size_vmax / 2.))
        xmax = int((xl) / 2. + (box_size_vmax / 2.))
        ymin = int((yl) / 2. - (box_size_vmax / 2.))
        ymax = int((yl) / 2. + (box_size_vmax / 2.))
        vmax = np.nanmax([image_array[i][xmin:xmax, ymin:ymax] for i in range(len(image_array))])
        return vmin*1.0, vmax*1.3 #vmin is 1 sigma of noise.

    def rescale_image(self, image):
            factor = self.scale(self.scale_max - self.scale_min)
            image = image.clip(min=self.scale_min, max=self.scale_max)
            #image = (image - self.scale_min) / factor
            indices0 = np.where(image < self.scale_min)
            indices1 = np.where((image >= self.scale_min) & (image <= self.scale_max))
            indices2 = np.where(image > self.scale_max)
            image[indices0] = 0.0
            image[indices2] = 1.0
            image[indices1] = self.scale(image[indices1]) / (factor * 1.0)
            return image

    def load_fits(self,filepath):
        opened_fits = fits.open(filepath)
        self.ra,self.dec = self.get_ra_dec(opened_fits[0].header)
        return opened_fits[0].data

    def get_ra_dec(self,header):
        w = WCS(header,fix=False)
        sky = w.pixel_to_world_values([w.array_shape[0]//2], [w.array_shape[1]//2])
        return sky[0][0], sky[1][0]

    def plot(self, scale_min = None, scale_max = None, canvas_id = 0):
        self.label_plot[canvas_id].setText(self.listimage[self.config_dict['counter']])
        self.ax[canvas_id].cla()
        if self.filetype == 'FITS':
            image = self.load_fits(self.filename)
            scaling_factor = np.nanpercentile(image,q=90)
            if scaling_factor == 0:
                # scaling_factor = np.nanpercentile(image,q=99)
                scaling_factor = 1
            image = image / scaling_factor*300 #Rescaling for better visualization.
            self.image = np.copy(image)
            if scale_min is not None and scale_max is not None:
                self.scale_min = scale_min
                self.scale_max = scale_max
            else:
                self.scale_min, self.scale_max = self.scale_val(image)
                # print(f"{self.scale_min = } {self.scale_max = }")
            image = self.rescale_image(image)
            self.ax[canvas_id].imshow(image,cmap=self.config_dict['colormap'], origin='lower')
        else:
            image = np.asarray(Image.open(self.filename))
            self.image = np.copy(image)
            self.ax[canvas_id].imshow(image, origin='upper') #For pngs this is best.
        self.ax[canvas_id].set_axis_off() #Always before .draw()!
        self.canvas[canvas_id].draw()

    def replot(self, scale_min = None, scale_max = None,canvas_id = 0):
        self.label_plot[canvas_id].setText(self.listimage[self.config_dict['counter']])
        self.ax[canvas_id].cla()
        image = np.copy(self.image)
        if self.filetype == 'FITS':
            image = self.rescale_image(image)
            self.ax[canvas_id].imshow(image,cmap=self.config_dict['colormap'], origin='lower')
        else:
            self.ax[canvas_id].imshow(image, origin='lower')
        self.ax[canvas_id].set_axis_off()
        self.canvas[canvas_id].draw()


    def obtain_df(self):
        if self.random_seed is None:
            base_filename = f'classification_single_{self.name}_{len(self.listimage)}'
            string_to_glob = f'./Classifications/{base_filename}*.csv'
            # print("Globing for", string_to_glob)
            string_to_glob_for_files_with_seed = f'./Classifications/{base_filename}_*.csv'
            glob_results = set(glob.glob(string_to_glob)) - set(glob.glob(string_to_glob_for_files_with_seed))
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
            if np.all(self.listimage == df['file_name'].values):
                return df
            else:
                print("Classification file corresponds to a different dataset.")
                string_tested = os.path.basename(self.df_name).split(".csv")[0]
                file_iteration = find_filename_iteration(string_tested) if f'./Classifications/{base_filename}.csv' in class_file else ''


        self.dfc = ['file_name', 'classification', 'grid_pos','page']
        self.df_name = f'./Classifications/{base_filename}{file_iteration}.csv'
        print('A new csv will be created', self.df_name)
        if file_iteration != "":
            print("To avoid this in the future use the argument `-N name` and give different names to different datasets.")
        self.df_name = f'./Classifications/{base_filename}{file_iteration}.csv'
        self.config_dict['counter'] = 0
        dfc = ['file_name', 'classification', 'subclassification',
                'ra','dec','comment','legacy_survey_data']
        df = pd.DataFrame(columns=dfc)
        df['file_name'] = self.listimage
        df['classification'] = ['Empty'] * len(self.listimage)
        df['subclassification'] = ['Empty'] * len(self.listimage)
        df['ra'] = np.full(len(self.listimage),np.nan)
        df['dec'] = np.full(len(self.listimage),np.nan)
        df['comment'] = ['Empty'] * len(self.listimage)
        df['legacy_survey_data'] = ['Empty'] * len(self.listimage)
        return df

    @Slot()
    def next(self):
        self.config_dict['counter'] = self.config_dict['counter'] + 1

        if self.config_dict['counter']>self.COUNTER_MAX-1:
            self.config_dict['counter']=self.COUNTER_MAX-1
            self.status.showMessage('Last image')

        else:
            self.filename = join(self.stampspath, self.listimage[self.config_dict['counter']])
            self.save_dict()
            self.plot()
            if self.config_dict['legacysurvey']:
            # if self.blegsur.isChecked():
                self.set_legacy_survey()
            
            
            self.update_classification_buttoms()
            self.update_counter()

    @Slot()
    def prev(self):
        self.config_dict['counter'] = self.config_dict['counter'] - 1

        if self.config_dict['counter']<self.COUNTER_MIN:
            self.config_dict['counter']=self.COUNTER_MIN
            self.status.showMessage('First image')

        else:
            self.filename = join(self.stampspath, self.listimage[self.config_dict['counter']])
            self.plot()
            if self.config_dict['legacysurvey']:
            # if self.blegsur.isChecked():
                self.set_legacy_survey()
            self.update_counter()
            self.save_dict()
            
            self.update_classification_buttoms()


    def update_classification_buttoms(self):
        grade = self.df.at[self.config_dict['counter'],'classification']


        # print(grade)
        if self.bactivatedclassification is not None:
            self.bactivatedclassification.setStyleSheet("background-color : white;color : black;")

        #if grade is not None and not np.isnan(float(grade)) and grade != 'None':
        if grade is not None and grade != 'None' and grade != 'Empty':
            # print(grade)
            button = self.dict_class2button[grade]
            if button is not None:
                # return
                button.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))
                self.bactivatedclassification = button



        subgrade = self.df.at[self.config_dict['counter'],'subclassification']
        # print(subgrade)
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
