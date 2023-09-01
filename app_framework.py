from typing import List, Tuple

from PyQt5.QtWidgets import QMainWindow, QFileDialog, QTableWidgetItem, QPushButton, QApplication, QCheckBox, QGraphicsScene, QGraphicsView, QDialog, QLabel,QLineEdit, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import QAbstractTableModel, Qt

import input_diag_design
import main_window_design

import os 
import time 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mne
from mne_qt_browser.figure import MNEQtBrowser
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from utils import elec_phys_signal



class PeriodicInput(QDialog, input_diag_design.Ui_Dialog):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)

    def get_info(self):
        freq = self.freq_SpinBox.value()
        amp = self.amp_spinBox.value()
        width = self.freq_width_spinBox.value()
        return freq, amp, width


class MyApp(QMainWindow, main_window_design.Ui_dialog):
    def __init__(self, dpi):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        
        self.dpi = dpi
        self.periodic_params = [(15.0, 4.0, 3.0)]

        self.ap_psd_fig = None 
        self.per_psd_fig = None 

        # Passing a value for compatibility
        self.update_ap_view(0)

    # Removes selected row from per_tableview
    def remove_per_signal(self):
        select = self.per_table.selectionModel()
        rows = select.selectedRows()

        for idx in rows:
            self.per_table.removeRow(idx.row())
            del self.periodic_params[idx.row()]

        if len(rows)>0:
            self.update_ap_view(0)

    # Adds a new row to per_tableview
    def add_per_signal(self):
        dialog = PeriodicInput()
        result = dialog.exec_()

        if result == QDialog.Accepted:
            freq, amp, width = dialog.get_info()
            row = len(self.periodic_params)
            self.per_table.insertRow(row)
            self.per_table.setItem(row, 0, QTableWidgetItem(str(freq)))
            self.per_table.setItem(row, 1, QTableWidgetItem(str(amp)))
            self.per_table.setItem(row, 2, QTableWidgetItem(str(width)))

            self.periodic_params.append((freq,amp,width))
    
            self.update_ap_view(0)

    def update_time_plot(self):

        self.ap_browser = self.ap_obj.plot(show=False, 
                                block=False, 
                                #show_scrollbars=False, 
                                #show_scalebars=False,
                                overview_mode='hidden')
        self.ap_browser.setFixedSize(710, 440)
        # Create a QGraphicsScene to hold the QtBrowser
        self.ap_browser_scene = QGraphicsScene()
        self.ap_browser_scene.addWidget(self.ap_browser)
        # Set the QGraphicsScene for your QGraphicsView
        self.ap_time_widget.setScene(self.ap_browser_scene)

        self.per_browser = self.per_obj.plot(show=False, 
                                block=False, 
                                #show_scrollbars=False, 
                                #show_scalebars=False,
                                overview_mode='hidden')

        self.per_browser.setFixedSize(710, 440)
        self.per_browser_scene = QGraphicsScene()
        self.per_browser_scene.addWidget(self.per_browser)
        self.per_time_widget.setScene(self.per_browser_scene)


    def update_psd_plot(self, value):
        
        xscale = 'log' if self.actionPSD_X_axis_log_scale.isChecked() else 'linear'
        dB = True if self.actionPSD_Y_axis_log_scale.isChecked() else False

        self.ap_psd_scene = QGraphicsScene()
        self.ap_psd_fig = self.ap_psd.plot(picks='Aperiodic', 
                                            dB=dB, 
                                            amplitude='auto', 
                                            xscale=xscale, 
                                            spatial_colors=False,
                                            show=False)
        self.ap_psd_fig.set_dpi(self.dpi)
        self.ap_psd_fig.set_size_inches(420/self.dpi, 420/self.dpi, forward=True)
        self.ap_psd_canvas = FigureCanvas(self.ap_psd_fig)
        self.ap_psd_scene.addWidget(self.ap_psd_canvas)
        self.ap_power_widget.setScene(self.ap_psd_scene)

        self.per_psd_scene = QGraphicsScene()
        self.per_psd_fig = self.per_psd.plot(picks='Periodic', 
                                            dB=dB, 
                                            amplitude='auto', 
                                            xscale=xscale, 
                                            spatial_colors=False,
                                            show=False)
        self.per_psd_fig.set_dpi(self.dpi)
        self.per_psd_fig.set_size_inches(420/self.dpi, 420/self.dpi, forward=True)
        self.per_psd_canvas = FigureCanvas(self.per_psd_fig)
        self.per_psd_scene.addWidget(self.per_psd_canvas)
        self.per_power_widget.setScene(self.per_psd_scene)

    def update_ap_view(self, value):

        ap_model = self.ap_model_comboBox.currentText()

        ap_inter = self.ap_fixed_intercept_spinBox.value() 
        ap_exp = self.ap_fixed_exponent_spinBox.value()
        ap_knee = self.ap_knee_spinBox.value()

        sample_rate = self.fs_spinBox.value()
        duration = self.signal_duration_spinBox.value()
        
        if self.white_noise_checkbox.isChecked():
            nlv = self.ap_gaussian_spinBox.value()
        else:
            nlv = None

        # TODO: extract periodic params
        for row, l in enumerate(self.periodic_params):
            # L has the form of (freq, aplitude, width)
            self.per_table.insertRow(row)
            self.per_table.setItem(row, 0, QTableWidgetItem(str(l[0])))
            self.per_table.setItem(row, 1, QTableWidgetItem(str(l[1])))
            self.per_table.setItem(row, 2, QTableWidgetItem(str(l[2])))
        self.per_table.setRowCount(len(self.periodic_params))

        # TODO: add epoching
        epoch_duration = self.epoch_duration_spinBox.value()
        epoch_overlap = self.epoch_overlap_spinBox.value()

        # Generate signals
        ap_signal, full_signal, per_signal = elec_phys_signal(
                                                exponent=-ap_exp, 
                                                knee = ap_knee,
                                                intercept = ap_inter,
                                                sample_rate=sample_rate, 
                                                duration=duration, 
                                                nlv=nlv, 
                                                periodic_params=self.periodic_params)

        info = mne.create_info(ch_names=["Aperiodic"], ch_types=["misc"], sfreq=sample_rate)
        self.ap_obj = mne.io.RawArray(ap_signal.reshape(1,-1), info)

        info = mne.create_info(ch_names=["Periodic"], ch_types=["misc"], sfreq=sample_rate)
        self.per_obj = mne.io.RawArray(per_signal.reshape(1,-1), info)

        info = mne.create_info(ch_names=["Full Signal"], ch_types=["misc"], sfreq=sample_rate)
        self.full_obj = mne.io.RawArray(full_signal.reshape(1,-1), info)

        # Calculate PSDs
        self.ap_psd = self.ap_obj.compute_psd(picks='Aperiodic', fmin=1, fmax=100, n_fft=2048, average='median')
        self.per_psd = self.per_obj.compute_psd(picks='Periodic', fmin=1,  fmax=100, n_fft=2048, average='median')

        # plot signals 
        self.update_time_plot()

        # Plot PSDs
        self.update_psd_plot(0)