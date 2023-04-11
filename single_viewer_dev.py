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


parser = argparse.ArgumentParser(description='Configure the parameters of the execution.')
parser.add_argument('-p',"--path", help="Path to the images to inspect",
                    default="Stamps_to_inspect")
parser.add_argument('-N',"--name", help="Name of the classifying session.",
                    default="")
parser.add_argument("--reset-config", help="Removes the configuration dictionary during startup.",
                    action="store_true", default=False)
parser.add_argument("--clean", help="Cleans the legacy survey folder.",
                    action="store_true")


args = parser.parse_args()

LEGACY_SURVEY_PATH = './Legacy_survey/'

if args.reset_config:
    os.remove('.config.json')

if args.clean:
    for f in glob.glob(join(LEGACY_SURVEY_PATH,"*.jpg")):
        os.remove(f)

def identity(x):
    return x

def log(x):
    return np.emath.logn(1000,x) #base 1000 like ds9

def asinh2(x):
    return np.arcsinh(x/2)


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
            print('File already exists:', savefile)
            return True
        url = (f'http://legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}'+
         f'&layer=ls-dr10{res}&size={size}&pixscale={pixscale}')
        print(url)
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
        self.setWindowTitle("Stamp Visualizer")
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
#        print(self.config_dict)

        self.ds9_comm_backend = "xpa"
        self.is_ds9_open = False
        self.singlefetchthread_active = False
        self.background_downloading = self.config_dict['prefetch']
        self.colormap = self.config_dict['colormap']
        self.buttoncolor = "darkRed"
        self.buttonclasscolor = "darkRed"
        self.scale2funct = {'identity':identity,'sqrt':np.sqrt,'log':log, 'log10':log, 'asinh2':asinh2}
        self.scale = self.scale2funct[self.config_dict['scale']]


        self.stampspath = args.path
        self.legacy_survey_path = LEGACY_SURVEY_PATH
        self.listimage = sorted([os.path.basename(x) for x in glob.glob(join(self.stampspath,'*.fits'))])
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



        self.bgoto = QtWidgets.QPushButton('Go to')
        self.bgoto.clicked.connect(self.goto)

        self.bnext = QtWidgets.QPushButton('Next')
        self.bnext.clicked.connect(self.next)

        self.bprev = QtWidgets.QPushButton('Prev')
        self.bprev.clicked.connect(self.prev)

        self.bds9 = QtWidgets.QPushButton('ds9')
        self.bds9.clicked.connect(self.open_ds9)

        self.bviewls = QtWidgets.QPushButton('View LS')
        self.bviewls.clicked.connect(self.viewls)

        self.blegsur = QtWidgets.QCheckBox('Legacy Survey (LS)')
        self.blegsur.clicked.connect(self.checkbox_legacy_survey)
        if not self.config_dict['legacysurvey']:
                self.label_plot[1].hide()
                self.canvas[1].hide()
        else:
                self.label_plot[1].show()
                self.canvas[1].show()
                self.blegsur.toggle()
                self.set_legacy_survey()


        self.blsarea = QtWidgets.QCheckBox("1 arcmin²")
        self.blsarea.clicked.connect(self.checkbox_ls_change_area)
        if self.config_dict['legacybigarea']:
            self.blsarea.toggle()
            if self.config_dict['legacysurvey']:
                self.set_legacy_survey()
#            self.checkbox_ls_change_area()

        self.blsresidual = QtWidgets.QCheckBox("Residuals")
        self.blsresidual.clicked.connect(self.checkbox_ls_use_residuals)
        if self.config_dict['legacyresiduals']:
            self.blsresidual.toggle()
            if self.config_dict['legacysurvey']:
                self.set_legacy_survey()

        self.bprefetch = QtWidgets.QCheckBox("Pre-fetch")
        self.bprefetch.clicked.connect(self.prefetch_legacysurvey)
        if self.config_dict['prefetch']:
            self.config_dict['prefetch'] = False
            self.prefetch_legacysurvey()
            self.bprefetch.toggle()
#            self.checkbox_ls_change_area()

        self.bautopass = QtWidgets.QCheckBox("Auto-next")
        self.bautopass.clicked.connect(self.checkbox_auto_next)
        if self.config_dict['autonext']:
            self.bautopass.toggle()

        self.bkeyboardshortcuts = QtWidgets.QCheckBox("Keyboard shortcuts")
        self.bkeyboardshortcuts.clicked.connect(self.checkbox_keyboard_shortcuts)
        if self.config_dict['keyboardshortcuts']:
            self.bkeyboardshortcuts.toggle()

        self.bsurelens = QtWidgets.QPushButton('Sure Lens')
        self.bsurelens.clicked.connect(partial(self.classify, 'SL','SL') )

        self.bmaybelens = QtWidgets.QPushButton('Maybe Lens')
        self.bmaybelens.clicked.connect(partial(self.classify, 'ML','ML'))

        self.bflexion = QtWidgets.QPushButton('Flexion')
        self.bflexion.clicked.connect(partial(self.classify, 'FL','FL'))

        self.bnonlens = QtWidgets.QPushButton('Non Lens')
        self.bnonlens.clicked.connect(partial(self.classify, 'NL','NL'))

        self.bMerger = QtWidgets.QPushButton('Merger')
        self.bMerger.clicked.connect(partial(self.classify, 'NL','Merger') )

        self.bSpiral = QtWidgets.QPushButton('Spiral')
        self.bSpiral.clicked.connect(partial(self.classify, 'NL','Spiral'))

        self.bRing = QtWidgets.QPushButton('Ring')
        self.bRing.clicked.connect(partial(self.classify, 'NL','Ring'))

        self.bElliptical = QtWidgets.QPushButton('Elliptical')
        self.bElliptical.clicked.connect(partial(self.classify, 'NL','Elliptical'))

        self.bDisc = QtWidgets.QPushButton('Disc')
        self.bDisc.clicked.connect(partial(self.classify, 'NL','Disc'))

        self.bEdgeon = QtWidgets.QPushButton('Edge-on')
        self.bEdgeon.clicked.connect(partial(self.classify, 'NL','Edge-on'))

        self.dict_class2button = {'SL':self.bsurelens,
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
                                  'SL':None,
                                  'ML':None,
                                  'FL':None,
                                  'NL':None,
                                  'None':None}

        self.blinear = QtWidgets.QPushButton('Linear')
        self.blinear.clicked.connect(self.set_scale_linear)

        self.bsqrt = QtWidgets.QPushButton('Sqrt')
        self.bsqrt.clicked.connect(self.set_scale_sqrt)

        self.blog = QtWidgets.QPushButton('Log')
        self.blog.clicked.connect(self.set_scale_log)

        self.basinh = QtWidgets.QPushButton('Asinh')
        self.basinh.clicked.connect(self.set_scale_asinh)

        self.bInverted = QtWidgets.QPushButton('Inverted')
        self.bInverted.clicked.connect(self.set_colormap_Inverted)

        self.bBb8 = QtWidgets.QPushButton('Bb8')
        self.bBb8.clicked.connect(self.set_colormap_Bb8)

        self.bGray = QtWidgets.QPushButton('Gray')
        self.bGray.clicked.connect(self.set_colormap_Gray)

        self.bViridis = QtWidgets.QPushButton('Viridis')
        self.bViridis.clicked.connect(self.set_colormap_Viridis)

        self.scale2button = {'identity':self.blinear,'sqrt':self.bsqrt,'log':self.blog,'log10':self.blog,
                            'asinh2': self.basinh}
        self.colormap2button = {'Inverted':self.bInverted,'Bb8':self.bBb8,'Gray':self.bGray,
                            'Viridis': self.bViridis}

        self.bactivatedclassification = None
        self.bactivatedsubclassification = None
        self.bactivatedscale = self.scale2button[self.config_dict['scale']]
        self.bactivatedcolormap = self.bGray

        grade = self.df.at[self.config_dict['counter'],'classification']
        if grade != 'None':
            self.bactivatedclassification = self.dict_class2button[grade]
            self.bactivatedclassification.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))
 
        subgrade = self.df.at[self.config_dict['counter'],'subclassification']
        if subgrade != 'None':
            self.bactivatedsubclassification = self.dict_subclass2button[subgrade]
            if self.bactivatedsubclassification is not None:
                self.bactivatedsubclassification.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))


        self.bactivatedscale.setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))
        self.bactivatedcolormap.setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))

#        self.bactivatedscale.setStyleSheet("background-color : white;color : black;".format(self.buttoncolor))
#        self.bactivatedscale = self.sender()

        #Keyboard shortcuts
        self.ksurelens = QShortcut(QKeySequence('q'), self)
        self.ksurelens.activated.connect(partial(self.keyClassify, 'SL','SL'))

        self.kmaybelens = QShortcut(QKeySequence('w'), self)
        self.kmaybelens.activated.connect(partial(self.keyClassify, 'ML','ML'))

        self.kflexion = QShortcut(QKeySequence('e'), self)
        self.kflexion.activated.connect(partial(self.keyClassify, 'FL','FL'))

        self.knonlens = QShortcut(QKeySequence('r'), self)
        self.knonlens.activated.connect(partial(self.keyClassify, 'NL','NL'))

        self.kMerger = QShortcut(QKeySequence('a'), self)
        self.kMerger.activated.connect(partial(self.keyClassify, 'NL','Merger'))

        self.kSpiral = QShortcut(QKeySequence('s'), self)
        self.kSpiral.activated.connect(partial(self.keyClassify, 'NL','Spiral'))

        self.kRing = QShortcut(QKeySequence('d'), self)
        self.kRing.activated.connect(partial(self.keyClassify, 'NL','Ring'))

        self.kElliptical = QShortcut(QKeySequence('f'), self)
        self.kElliptical.activated.connect(partial(self.keyClassify, 'NL','Elliptical'))

        self.kDisc = QShortcut(QKeySequence('g'), self)
        self.kDisc.activated.connect(partial(self.keyClassify, 'NL','Disc'))

        self.kEdgeon = QShortcut(QKeySequence('h'), self)
        self.kEdgeon.activated.connect(partial(self.keyClassify, 'NL','Edge-on'))


        #Buttons
        button_row0_layout.addWidget(self.bgoto)
        button_row0_layout.addWidget(self.bprev)
        button_row0_layout.addWidget(self.bnext)
        button_row0_layout.addWidget(self.bds9)
        button_row0_layout.addWidget(self.bviewls)
        button_row0_layout.addWidget(self.blegsur)
        button_row0_layout.addWidget(self.blsarea)
        button_row0_layout.addWidget(self.blsresidual)
        button_row0_layout.addWidget(self.bprefetch)
        button_row0_layout.addWidget(self.bautopass)
        button_row0_layout.addWidget(self.bkeyboardshortcuts)
        button_row0_layout.addWidget(self.counter_widget,alignment=Qt.AlignRight)

        button_row10_layout.addWidget(self.bsurelens)
        button_row10_layout.addWidget(self.bmaybelens)
        button_row10_layout.addWidget(self.bflexion)
        button_row10_layout.addWidget(self.bnonlens)


        button_row11_layout.addWidget(self.bMerger)
        button_row11_layout.addWidget(self.bSpiral)
        button_row11_layout.addWidget(self.bRing)
        button_row11_layout.addWidget(self.bElliptical)
        button_row11_layout.addWidget(self.bDisc)
        button_row11_layout.addWidget(self.bEdgeon)

        button_row2_layout.addWidget(self.blinear)
        button_row2_layout.addWidget(self.bsqrt)
        button_row2_layout.addWidget(self.blog)
        button_row2_layout.addWidget(self.basinh)
        button_row3_layout.addWidget(self.bInverted)
        button_row3_layout.addWidget(self.bBb8)
        button_row3_layout.addWidget(self.bGray)
        button_row3_layout.addWidget(self.bViridis)

        button_layout.addLayout(button_row0_layout, 25)
        button_layout.addLayout(button_row10_layout, 25)
        button_layout.addLayout(button_row11_layout, 25)
        button_layout.addLayout(button_row2_layout, 25)
        button_layout.addLayout(button_row3_layout, 25)


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
        self.df.at[cnt,'ra'] = self.ra
        self.df.at[cnt,'dec'] = self.dec
        self.df.at[cnt,'comment'] = grade
#        print('updating '+'classification_autosave'+str(self.nf)+'.csv file')
        self.df.to_csv(self.df_name)

        # if self.bactivatedclassification is not None:
        #     self.bactivatedclassification.setStyleSheet("background-color : white;color : black;")
        # buttom = self.dict_class2button[grade]
        # buttom.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))
        # self.bactivatedclassification = buttom


        # if self.dict_subclass2button[subgrade] is not None:
        #     if self.bactivatedsubclassification is not None:
        #         self.bactivatedsubclassification.setStyleSheet("background-color : white;color : black;")
        #     buttom = self.dict_subclass2button[grade]
        #     buttom.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))
        #     self.bactivatedsubclassification = buttom        
        # else:
        #     if self.bactivatedsubclassification is not None:
        #         self.bactivatedsubclassification.setStyleSheet("background-color : white;color : black;")

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

    def plot_no_legacy_survey(self, title='Waiting for data', canvas_id = 1):
        self.label_plot[canvas_id].setText(title)
        self.ax[canvas_id].cla()
        self.ax[canvas_id].imshow(np.zeros((66,66)), cmap='Greys_r')
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
                                        size=size) #TODO FIX BUG HERE.

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
            self.singleFetchWorker.failed_download.connect(self.plot_no_legacy_survey)
            self.workerThread.finished.connect(self.workerThread.deleteLater)
            self.workerThread.setTerminationEnabled(True)

            self.workerThread.start()
            self.workerThread.quit()
            # self.singleFetchWorker.has_finished.connect(self.singleFetchWorker.deleteLater)
            # self.singleFetchWorker.has_finished.connect(self.workerThread.deleteLater)
        
        except FileNotFoundError as E:
            self.plot_no_legacy_survey()
            # print(E.args)
            # print(type(E))
            # raise
        except Exception as E:
            print(E.args)
            print(type(E))


    @Slot()
    def checkbox_legacy_survey(self):
        if self.config_dict['legacysurvey']:
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
        subprocess.Popen(["ds9", '-fits',self.filename, '-zoom','8'  ])

    @Slot()
    def viewls(self):
        webbrowser.open("https://www.legacysurvey.org/viewer?ra={}&dec={}&layer=ls-dr10-grz&zoom=16&spectra".format(self.ra,self.dec))
        #subprocess.Popen(["ds9", '-fits',self.filename, '-zoom','8'  ])

    @Slot()
    def set_scale_linear(self):
        if self.sender() != self.bactivatedscale:
            self.scale = identity
            self.replot()
            self.sender().setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))
            self.bactivatedscale.setStyleSheet("background-color : white;color : black;")
            self.bactivatedscale = self.sender()
            self.config_dict['scale']='identity'
            self.save_dict()

    @Slot()
    def set_scale_sqrt(self):
        if self.sender() != self.bactivatedscale:
            self.scale = np.sqrt
            self.replot()
            self.sender().setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))
            self.bactivatedscale.setStyleSheet("background-color : white;color : black;")
            self.bactivatedscale = self.sender()
            self.config_dict['scale']='sqrt'
            self.save_dict()

    @Slot()
    def set_scale_log(self):
        if self.sender() != self.bactivatedscale and self.sender() is not None: #TODO test if this works/
            self.scale = log
            self.replot()
            self.sender().setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))
            self.bactivatedscale.setStyleSheet("background-color : white;color : black;")
            self.bactivatedscale = self.sender()
            self.config_dict['scale']='log'
            self.save_dict()

    @Slot()
    def set_scale_asinh(self):
        if self.sender() != self.bactivatedscale:
            self.scale = asinh2
            self.replot()
            self.sender().setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))
            self.bactivatedscale.setStyleSheet("background-color : white;color : black;")
            self.bactivatedscale = self.sender()
            self.config_dict['scale']='asinh2'
            self.save_dict()

    @Slot()
    def set_colormap_Inverted(self):
        if self.sender() != self.bactivatedcolormap:
            self.config_dict['colormap'] = "gist_yarg"
            self.replot()
            self.sender().setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))
            self.bactivatedcolormap.setStyleSheet("background-color : white;color : black;")
            self.bactivatedcolormap = self.sender()
            self.save_dict()

    @Slot()
    def set_colormap_Bb8(self):
        if self.sender() != self.bactivatedcolormap:
            self.config_dict['colormap'] = "hot"
            self.replot()
            self.sender().setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))
            self.bactivatedcolormap.setStyleSheet("background-color : white;color : black;")
            self.bactivatedcolormap = self.sender()
            self.save_dict()

    @Slot()
    def set_colormap_Gray(self):
        if self.sender() != self.bactivatedcolormap:
            self.config_dict['colormap'] = "gray"
            self.replot()
            self.sender().setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))
            self.bactivatedcolormap.setStyleSheet("background-color : white;color : black;")
            self.bactivatedcolormap = self.sender()
            self.save_dict()

    @Slot()
    def set_colormap_Viridis(self):
        if self.sender() != self.bactivatedcolormap:
            self.config_dict['colormap'] = "viridis"
            self.replot()
            self.sender().setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))
            self.bactivatedcolormap.setStyleSheet("background-color : white;color : black;")
            self.bactivatedcolormap = self.sender()
            self.save_dict()

    def background_rms_image(self,cb, image):
        xg, yg = np.shape(image)
        cut0 = image[0:cb, 0:cb]
        cut1 = image[xg - cb:xg, 0:cb]
        cut2 = image[0:cb, yg - cb:yg]
        cut3 = image[xg - cb:xg, yg - cb:yg]
        l = [cut0, cut1, cut2, cut3]
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

    def scale_val(self,image_array):
        if len(np.shape(image_array)) == 2:
            image_array = [image_array]
        vmin = np.min([self.background_rms_image(5, image_array[i]) for i in range(len(image_array))])
        xl, yl = np.shape(image_array[0])
        box_size = 14  # in pixel
        xmin = int((xl) / 2. - (box_size / 2.))
        xmax = int((xl) / 2. + (box_size / 2.))
        vmax = np.max([image_array[i][xmin:xmax, xmin:xmax] for i in range(len(image_array))])
        return vmin, vmax*1.3

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
        image = self.load_fits(self.filename)
        self.image = np.copy(image)
        if scale_min is not None and scale_max is not None:
            self.scale_min = scale_min
            self.scale_max = scale_max
        else:
            self.scale_min, self.scale_max = self.scale_val(image)

        image = self.rescale_image(image)

        self.ax[canvas_id].imshow(image,cmap=self.config_dict['colormap'], origin='lower')
        self.ax[canvas_id].set_axis_off()
        self.canvas[canvas_id].draw()

    def replot(self, scale_min = None, scale_max = None,canvas_id = 0):
        self.label_plot[canvas_id].setText(self.listimage[self.config_dict['counter']])

        self.ax[canvas_id].cla()

        image = np.copy(self.image)
        image = self.rescale_image(image)

        self.ax[canvas_id].imshow(image,cmap=self.config_dict['colormap'], origin='lower')
        self.ax[canvas_id].set_axis_off()
        self.canvas[canvas_id].draw()

    def obtain_df(self):
        string_to_glob = './Classifications/classification_single_{}_{}*.csv'.format(
                                    args.name,len(self.listimage))
        class_file = np.sort(glob.glob(string_to_glob))
        print("Globing for", string_to_glob)
        if len(class_file) >=1:
            self.df_name = class_file[len(class_file)-1]
            print('reading',self.df_name)
            df = pd.read_csv(self.df_name,index_col=0)
            if len(df) == len(self.listimage):
                self.listimage = df['file_name'].values
                return df
            else:
                print("The number of rows in the csv and the number of images must be equal.")

        self.df_name = './Classifications/classification_single_{}_{}.csv'.format(
                                    args.name,len(self.listimage))
        print('Creating dataframe', self.df_name)        
        self.config_dict['counter'] = 0
        dfc = ['file_name', 'classification', 'subclassification',
                'ra','dec','comment','legacy_survey_data']
        df = pd.DataFrame(columns=dfc)
        df['file_name'] = self.listimage
        df['classification'] = ['None'] * len(self.listimage)
        df['subclassification'] = ['None'] * len(self.listimage)
        df['ra'] = np.full(len(self.listimage),np.nan)
        df['dec'] = np.full(len(self.listimage),np.nan)
        df['comment'] = ['None'] * len(self.listimage)
        df['legacy_survey_data'] = ['None'] * len(self.listimage)
        return df

    @Slot()
    def next(self):
        self.config_dict['counter'] = self.config_dict['counter'] + 1

        if self.config_dict['counter']>self.COUNTER_MAX-1:
            self.config_dict['counter']=self.COUNTER_MAX-1
            self.status.showMessage('Last image')

        else:
            self.filename = join(self.stampspath, self.listimage[self.config_dict['counter']])
            # if self.singlefetchthread_active:
            #     self.singlefetchthread.terminate()
            self.plot()
            if self.config_dict['legacysurvey']:
                self.set_legacy_survey()
            
            self.save_dict()
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
                self.set_legacy_survey()
            self.update_counter()
            self.save_dict()
            self.update_classification_buttoms()


    def update_classification_buttoms(self):
        grade = self.df.at[self.config_dict['counter'],'classification']
        button = self.dict_class2button[grade]

        # print(grade)
        if self.bactivatedclassification is not None:
            self.bactivatedclassification.setStyleSheet("background-color : white;color : black;")

        if button is not None:
            # return
            button.setStyleSheet("background-color : {};color : white;".format(self.buttonclasscolor))
            self.bactivatedclassification = button



        subgrade = self.df.at[self.config_dict['counter'],'subclassification']
        # print(subgrade)
        if self.bactivatedsubclassification is not None:
            self.bactivatedsubclassification.setStyleSheet("background-color : white;color : black;")

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
