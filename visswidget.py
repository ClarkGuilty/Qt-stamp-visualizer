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
from PySide6.QtCore import Qt, Slot, QObject, QThread
#from PySide6.QtGui import QPixmap

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import image as mpimg

import urllib

from functools import partial

import webbrowser
import json
from os.path import join

def identity(x):
    return x

def asinh2(x):
    return np.arcsinh(x/2)


class FetchThread(QThread):
    def __init__(self, df, initial_counter, parent=None):
#            super().__init__(parent)
            QThread.__init__(self, parent)

            self.df = df
            self.initial_counter = initial_counter
            self.legacy_survey_path = './Legacy_survey/'
            self.stampspath = './Stamps_to_inspect/'
            self.listimage = sorted([os.path.basename(x) for x in glob.glob(self.stampspath+ '*.fits')])
            self.im = Image.fromarray(np.zeros((66,66),dtype=np.uint8))
    def download_legacy_survey(self, ra, dec, pixscale): #pixscale = 0.04787578125 is 66 pixels in CFIS.

        savename = 'N' + '_' + str(ra) + '_' + str(dec) +"_"+ pixscale + 'dr8.jpg'
        savefile = os.path.join(self.legacy_survey_path, savename)
        if os.path.exists(savefile):
            return True
        url = 'http://legacysurvey.org/viewer/cutout.jpg?ra=' + str(ra) + '&dec=' + str(
            dec) + '&layer=dr8&pixscale='+pixscale
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

    def run(self):
        for index, stamp in self.df.iterrows():
            if np.isnan(stamp['ra']) or np.isnan(stamp['dec']):
                f = join(self.stampspath,self.listimage[index])
                ra,dec = self.get_ra_dec(fits.getheader(f))

            else:
                ra,dec = stamp[['ra','dec']]
            self.download_legacy_survey(ra,dec,'0.048')
            self.download_legacy_survey(ra,dec,pixscale='0.5')
        return

class ApplicationWindow(QtWidgets.QMainWindow):
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
                    'prefetch':False,
                    'autonext':False,
                    'prefetch':False,
                    'colormap':'gray',
                    'scale':'log10',
                        }
        self.config_dict = self.load_dict()
        self.im = Image.fromarray(np.zeros((66,66),dtype=np.uint8))
#        print(self.config_dict)


        self.ds9_comm_backend = "xpa"
        self.is_ds9_open = False
        self.background_downloading = self.config_dict['prefetch']
        self.colormap = self.config_dict['colormap']
        self.buttoncolor = "darkRed"
        self.scale2funct = {'identity':identity,'sqrt':np.sqrt,'log10':np.log10, 'asinh2':asinh2}
        self.scale = self.scale2funct[self.config_dict['scale']]


        self.stampspath = './Stamps_to_inspect/'
        self.legacy_survey_path = './Legacy_survey/'
        self.listimage = sorted([os.path.basename(x) for x in glob.glob(self.stampspath+ '*.fits')])

        self.df = self.obtain_df()

        self.number_graded = 0
        self.COUNTER_MIN =0
        self.COUNTER_MAX = len(self.listimage)
        self.filename = join(self.stampspath, self.listimage[self.config_dict['counter']])
        self.status.showMessage(self.listimage[self.config_dict['counter']],)


        self.legacy_survey_qlabel = QtWidgets.QLabel(alignment=Qt.AlignCenter)
#        pixmap = QPixmap()
        #self.figure.gca().set_facecolor('black')

        main_layout = QtWidgets.QVBoxLayout(self._main)
        self.label_layout = QtWidgets.QHBoxLayout()
        self.plot_layout = QtWidgets.QHBoxLayout()
        button_layout = QtWidgets.QVBoxLayout()
        button_row0_layout = QtWidgets.QHBoxLayout()
        button_row1_layout = QtWidgets.QHBoxLayout()
        button_row2_layout = QtWidgets.QHBoxLayout()
        button_row3_layout = QtWidgets.QHBoxLayout()

        self.counter_widget = QtWidgets.QLabel("{}/{}".format(self.config_dict['counter']+1,self.COUNTER_MAX+1))
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

#        self.blegsur = QtWidgets.QPushButton('Legacy Survey')
#        self.blegsur.clicked.connect(self.set_legacy_survey)
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


        self.blsarea = QtWidgets.QCheckBox("1 arcmin??")
        self.blsarea.clicked.connect(self.checkbox_ls_change_area)

        if self.config_dict['legacybigarea']:
            self.blsarea.toggle()
            if self.config_dict['legacysurvey']:
                self.set_legacy_survey()
#            self.checkbox_ls_change_area()

        self.bprefetch = QtWidgets.QCheckBox("Pre-fetch")
        self.bprefetch.clicked.connect(self.prefetch_legacysurvey)

        if self.config_dict['prefetch']:
            self.bprefetch.toggle()
#            self.checkbox_ls_change_area()

        self.bautopass = QtWidgets.QCheckBox("Auto-next")
        self.bautopass.clicked.connect(self.checkbox_auto_next)

        if self.config_dict['autonext']:
            self.bautopass.toggle()

        self.bsurelens = QtWidgets.QPushButton('Sure Lens')
        self.bsurelens.clicked.connect(partial(self.classify, 'SL','SL',1) )

        self.bmaybelens = QtWidgets.QPushButton('Maybe Lens')
        self.bmaybelens.clicked.connect(partial(self.classify, 'ML','ML',1))

        self.bflexion = QtWidgets.QPushButton('Flexion')
        self.bflexion.clicked.connect(partial(self.classify, 'FL','FL',1))

        self.bnonlens = QtWidgets.QPushButton('Non Lens')
        self.bnonlens.clicked.connect(partial(self.classify, 'NL','NL',1))

        self.blinear = QtWidgets.QPushButton('Linear')
        self.blinear.clicked.connect(self.set_scale_linear)

        self.bsqrt = QtWidgets.QPushButton('Sqrt')
        self.bsqrt.clicked.connect(self.set_scale_sqrt)

        self.blog = QtWidgets.QPushButton('Log10')
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

        self.scale2button = {'identity':self.blinear,'sqrt':self.bsqrt,'log10':self.blog,
                            'asinh2': self.basinh}
        self.colormap2button = {'Inverted':self.bInverted,'Bb8':self.bBb8,'Gray':self.bGray,
                            'Viridis': self.bViridis}

        self.bactivatedscale = self.scale2button[self.config_dict['scale']]
        self.bactivatedcolormap = self.bGray

        self.bactivatedscale.setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))
        self.bactivatedcolormap.setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))

#        self.bactivatedscale.setStyleSheet("background-color : white;color : black;".format(self.buttoncolor))
#        self.bactivatedscale = self.sender()

        button_row1_layout.addWidget(self.bsurelens)
        button_row1_layout.addWidget(self.bmaybelens)
        button_row1_layout.addWidget(self.bflexion)
        button_row1_layout.addWidget(self.bnonlens)


        button_row2_layout.addWidget(self.blinear)
        button_row2_layout.addWidget(self.bsqrt)
        button_row2_layout.addWidget(self.blog)
        button_row2_layout.addWidget(self.basinh)
        button_row3_layout.addWidget(self.bInverted)
        button_row3_layout.addWidget(self.bBb8)
        button_row3_layout.addWidget(self.bGray)
        button_row3_layout.addWidget(self.bViridis)

        button_row0_layout.addWidget(self.bgoto)
        button_row0_layout.addWidget(self.bprev)
        button_row0_layout.addWidget(self.bnext)
        button_row0_layout.addWidget(self.bds9)
        button_row0_layout.addWidget(self.bviewls)
        button_row0_layout.addWidget(self.blegsur)
        button_row0_layout.addWidget(self.blsarea)
        button_row0_layout.addWidget(self.bprefetch)
        button_row0_layout.addWidget(self.bautopass)
        button_row0_layout.addWidget(self.counter_widget,alignment=Qt.AlignRight)

        button_layout.addLayout(button_row0_layout, 25)
        button_layout.addLayout(button_row1_layout, 25)
        button_layout.addLayout(button_row2_layout, 25)
        button_layout.addLayout(button_row3_layout, 25)


        main_layout.addLayout(self.label_layout, 2)
        main_layout.addLayout(self.plot_layout, 88)
        main_layout.addLayout(button_layout, 10)

    @Slot()
    def prefetch_legacysurvey(self):
        if not self.config_dict['prefetch']:
            self.fetchthread = FetchThread(self.df,self.config_dict['counter']) #Always store in an object.
            self.fetchthread.finished.connect(self.fetchthread.deleteLater)
            self.fetchthread.start()
        else:
            self.config_dict['prefetch'] = True

    @Slot()
    def goto(self):
        i, ok = QtWidgets.QInputDialog.getInt(self, 'Visual inspection', '',self.config_dict['counter']+1,1,self.COUNTER_MAX+1)
        if ok:
            self.config_dict['counter'] = i-1
            self.filename = join(self.stampspath,self.listimage[self.config_dict['counter']])
            self.plot()
            if self.config_dict['legacysurvey']:
                self.set_legacy_survey()
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
        self.counter_widget.setText("{}/{}".format(self.config_dict['counter']+1,self.COUNTER_MAX+1))

    def classify(self, grade, subgrade, col):
        cnt = self.config_dict['counter']# - 1
#        self.df.at[cnt,'file_name'] = self.filename
        assert self.df.at[cnt,'file_name'] == self.listimage[self.config_dict['counter']] #TODO handling this possibility better.
        self.df.at[cnt,'classification'] = grade
        self.df.at[cnt,'subclassification'] = subgrade
        self.df.at[cnt,'ra'] = self.ra
        self.df.at[cnt,'dec'] = self.dec
        self.df.at[cnt,'comment'] = grade
#        print('updating '+'classification_autosave'+str(self.nf)+'.csv file')
        self.df.to_csv(os.path.join("Classifications",'classification_autosave'+str(self.nf)+'.csv'), index=False)
        if self.config_dict['autonext']:
            self.next()


    def get_legacy_survey(self,ra,dec,pixscale = '0.048'): #pixscale = 0.04787578125 is 66 pixels in CFIS.
#        savename = 'N' + str(self.config_dict['counter'])+ '_' + str(self.ra) + '_' + str(self.dec) +"_"+pixscale + 'dr8.jpg'

        savename = 'N' + '_' + str(ra) + '_' + str(dec) +"_"+pixscale + 'dr8.jpg'
        savefile = os.path.join(self.legacy_survey_path, savename)
        if os.path.exists(savefile):
            return savefile
        self.status.showMessage("Downloading legacy survey jpeg.")
        url = 'http://legacysurvey.org/viewer/cutout.jpg?ra=' + str(ra) + '&dec=' + str(
            dec) + '&layer=dr8&pixscale='+str(pixscale)
        try:
            urllib.request.urlretrieve(url, savefile)
        except urllib.error.HTTPError:
            self.status.showMessage("No data for RA: {}, DEC: {}.".format(ra,dec))
            raise
#        url = 'http://legacysurvey.org/viewer/cutout.jpg?ra=' + str(self.ra) + '&dec=' + str(
#            self.dec) + '&layer=dr8-resid&pixscale=0.06'
#        savename = 'N' + str(self.config_dict['counter'])+ '_' + str(self.ra) + '_' + str(self.dec) + 'dr8-resid.jpg'
#        urllib.request.urlretrieve(url, os.path.join(self.legacy_survey_path, savename))
        return savefile

    def plot_legacy_survey(self, title='12x12', canvas_id = 1):
        self.label_plot[canvas_id].setText(title)
        self.ax[canvas_id].cla()
        self.ax[canvas_id].imshow(mpimg.imread(self.legacy_filename))
        self.ax[canvas_id].set_axis_off()
        self.canvas[canvas_id].draw()

    def plot_no_legacy_survey(self, title='No Legacy Survey data available', canvas_id = 1):
        self.label_plot[canvas_id].setText(title)
        self.ax[canvas_id].cla()
        self.ax[canvas_id].imshow(np.zeros((66,66)))
        self.ax[canvas_id].set_axis_off()
        self.canvas[canvas_id].draw()

    @Slot()
    def set_legacy_survey(self):
        if self.config_dict['legacybigarea'] == True:
            try:
                self.legacy_filename = self.get_legacy_survey(self.ra,self.dec,pixscale='0.5')
                self.plot_legacy_survey(title='{0}x{0}'.format(0.5))
            except urllib.error.HTTPError:
                self.plot_no_legacy_survey()
        else:
            try:
                self.legacy_filename = self.get_legacy_survey(self.ra,self.dec)
                self.plot_legacy_survey(title='{0}x{0}'.format(12.56))
            except urllib.error.HTTPError:
                self.plot_no_legacy_survey()

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
    def checkbox_auto_next(self):
        self.config_dict['autonext'] = not self.config_dict['autonext']



    @Slot()
    def open_ds9(self):
        subprocess.Popen(["ds9", '-fits',self.filename, '-zoom','8'  ])

    @Slot()
    def viewls(self):
        webbrowser.open("https://www.legacysurvey.org/viewer?ra={}&dec={}&layer=ls-dr10&zoom=16&spectra".format(self.ra,self.dec))
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
        if self.sender() != self.bactivatedscale:
            self.scale = np.log10
            self.replot()
            self.sender().setStyleSheet("background-color : {};color : white;".format(self.buttoncolor))
            self.bactivatedscale.setStyleSheet("background-color : white;color : black;")
            self.bactivatedscale = self.sender()
            self.config_dict['scale']='log10'
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
        class_file = np.sort(glob.glob('./Classifications/classification_autosave*.csv'))
        print (class_file, len(class_file))
        if len(class_file) >=1:
            print ('loop')
            print('reading '+str(class_file[len(class_file)-1]))
            df = pd.read_csv(class_file[len(class_file)-1])
            self.nf = len(class_file)
#            firstnone=df['classification'].tolist().index('None')
#            self.config_dict['counter'] =firstnone #Remembering last position

        else:
            df=[]
        if len(df) != len(self.listimage):
            print('creating classification_autosave'+str(len(class_file)+1)+'.csv')
            dfc = ['file_name', 'classification', 'subclassification','ra','dec','comment','legacy_survey_data']
            self.nf = len(class_file) + 1
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
            self.plot()
            if self.config_dict['legacysurvey']:
                self.set_legacy_survey()
            self.update_counter()
            self.save_dict()

    @Slot()
    def prev(self):
        self.config_dict['counter'] = self.config_dict['counter'] - 1

        if self.config_dict['counter']<self.COUNTER_MIN:
            self.config_dict['counter']=self.COUNTER_MIN
            self.status.showMessage('First image')

        else:
            #self.status.showMessage(self.listimage[self.config_dict['counter']])
            self.filename = self.stampspath + self.listimage[self.config_dict['counter']]
            self.plot()
            if self.config_dict['legacysurvey']:
                self.set_legacy_survey()
            self.update_counter()
            self.save_dict()


            
            
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
