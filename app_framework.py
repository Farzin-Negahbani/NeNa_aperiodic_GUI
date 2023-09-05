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

import fooof
from utils import elec_phys_signal, irasa
from yasa import sliding_window


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
        self.periodic_params = [(15.0, 3.0, 4.5)]
        self.irasa_hset_textEdit.setText("1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9")

        # Passing a value for compatibility
        self.update_ap_view(0)

    def run_fooof(self):

        peak_width_limits = self.fooof_peak_width_min_spinBox.value(), self.fooof_peak_width_max_spinBox.value()
        max_n_peaks = self.fooof_max_no_peaks_spinBox.value()
        min_peak_height = self.fooof_peak_height_min_spinBox.value()
        peak_threshold = self.fooof_peak_th_spinBox.value()
        freq_band = self.fooof_min_freq_spinBox.value(), self.fooof_max_freq_spinBox.value()

        aperiodic_mode = 'fixed' if self.fooof_ap_mode_spinBox.currentText() =='Fixed (knee=0)' else 'knee'

        start_time = time.time()

        self.fm = fooof.FOOOF(peak_width_limits=peak_width_limits, 
                    max_n_peaks=max_n_peaks, 
                    min_peak_height=min_peak_height, 
                    peak_threshold=peak_threshold, 
                    aperiodic_mode=aperiodic_mode, 
                    verbose=True)
        
        self.fm.fit(self.full_psd.freqs, np.median(self.full_psd._data.squeeze(),axis=0), freq_band)

        end_time = time.time()

        # Plot results
        self.fooof_psd_scene = QGraphicsScene()
        self.fooof_psd_fig = plt.figure() 
        plt.tight_layout()
        plt.yscale('log')
        plt.plot(self.fm.freqs, np.exp(self.fm.power_spectrum), c='k', label="Power spectrum", lw=2)
        plt.plot(self.fm.freqs, np.exp(self.fm._ap_fit), c='b',linestyle='--', label="Aperiodic fit", lw=2)
        plt.plot(self.fm.freqs, np.exp(self.fm.fooofed_spectrum_), c='r', label="FoooF model fit", lw=2)
        self.fooof_psd_fig.set_dpi(self.dpi)
        self.fooof_psd_fig.set_size_inches(670/self.dpi, 370/self.dpi, forward=True)
        self.fooof_psd_canvas = FigureCanvas(self.fooof_psd_fig)
        self.fooof_psd_scene.addWidget(self.fooof_psd_canvas)
        self.fooof_res_psd_widget.setScene(self.fooof_psd_scene)
        self.fooof_psd_fig.suptitle('Full signal PSD with FoooF result', fontsize=8)
        self.fooof_psd_fig.axes[0].set_xlabel("Frequency (Hz)", fontsize=8)
        self.fooof_psd_fig.axes[0].set_ylabel("PSD log($V^2$/Hz)", fontsize=8)
        self.fooof_psd_fig.axes[0].legend()

        self.fooof_res_r2_textEdit.setText(str(round(self.fm.r_squared_,3)))
        self.fooof_res_error_textEdit.setText(str(round(self.fm.error_,3)))
        self.fooof_res_runtime_testEdit.setText(str(round(end_time-start_time,3)))
        self.fooof_res_inter_textEdit.setText(str(round(self.fm.aperiodic_params_[0],3)))
        self.fooof_res_std_textEdit.setText(str(round(np.std(self.fm.power_spectrum-self.fm._ap_fit,axis=-1, ddof=1),3)))

        if aperiodic_mode == 'fixed':
            self.fooof_res_knee_textEdit.setText(str(0))
            self.fooof_res_exp_textEdit.setText(str(round(-self.fm.aperiodic_params_[1],3)))
        else:
            self.fooof_res_knee_textEdit.setText(str(round(self.fm.aperiodic_params_[1],3)))
            self.fooof_res_exp_textEdit.setText(str(round(-self.fm.aperiodic_params_[2],3)))
        
        self.fooof_res_peak_params_table.clearContents()
        
        for r, l  in enumerate(self.fm.peak_params_):
            self.fooof_res_peak_params_table.insertRow(r)
            self.fooof_res_peak_params_table.setItem(r, 0, QTableWidgetItem(str(round(l[0],1))))
            self.fooof_res_peak_params_table.setItem(r, 1, QTableWidgetItem(str(round(l[1],1))))
            self.fooof_res_peak_params_table.setItem(r, 2, QTableWidgetItem(str(round(l[2],1))))
        self.fooof_res_peak_params_table.setRowCount(len(self.fm.peak_params_))


    def run_irasa(self):
        freq_band = self.irasa_freq_band_min_spinBox.value(), self.irasa_freq_band_max_spinBox.value()
        window_len = self.irasa_sliding_window_spinBox.value()
        hset = [float(f) for f in self.irasa_hset_textEdit.toPlainText().split(",")]

        start_time = time.time()
        freqs, psd_ap, psd_osc, fit_params = irasa(self.full_obj_cont, 
                            band=freq_band,
                            hset=hset, 
                            return_fit=True, 
                            win_sec=window_len,
                            kwargs_welch=dict(average='median', window='hann'))
        end_time = time.time()

        # Plot results
        self.irasa_psd_scene = QGraphicsScene()
        self.irasa_psd_fig = plt.figure() 
        plt.tight_layout()
        plt.yscale('log')
        plt.plot(freqs, (psd_ap+psd_osc).ravel(), c='k', label="Power spectrum", lw=2)
        plt.plot(freqs, psd_ap.ravel(), c='b',linestyle='--', label="Aperiodic fit", lw=2)
        #plt.plot(freqs, psd_osc.ravel(), c='cyan',linestyle='--', label="Periodic components", lw=2)
        self.irasa_psd_fig.set_dpi(self.dpi)
        self.irasa_psd_fig.set_size_inches(670/self.dpi, 370/self.dpi, forward=True)
        self.irasa_psd_canvas = FigureCanvas(self.irasa_psd_fig)
        self.irasa_psd_scene.addWidget(self.irasa_psd_canvas)
        self.irasa_res_psd_widget.setScene(self.irasa_psd_scene)
        self.irasa_psd_fig.suptitle('Full signal PSD with IRASA result', fontsize=8)
        self.irasa_psd_fig.axes[0].set_xlabel("Frequency (Hz)", fontsize=8)
        self.irasa_psd_fig.axes[0].set_ylabel("PSD log($V^2$/Hz)", fontsize=8)
        self.irasa_psd_fig.axes[0].legend()

        self.irasa_res_exp_textEdit.setText(str(round(fit_params['Slope'].values[0],3)))
        self.irasa_res_inter_textEdit.setText(str(round(fit_params['Intercept'].values[0],3)))
        self.irasa_res_r2_textEdit.setText(str(round(fit_params['R^2'].values[0],3)))
        self.irasa_res_std_textEdit.setText(str(round(np.log10(fit_params['std(osc)'].values[0]),3)))
        self.irasa_rest_runtime_textEdit.setText(str(round(end_time-start_time,3)))

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

        self.full_browser = self.full_obj.plot(picks='all',
                                show=False, 
                                block=False, 
                                #show_scrollbars=False, 
                                #show_scalebars=False,
                                overview_mode='hidden')

        self.full_browser.setFixedSize(710, 440)
        self.full_browser_scene = QGraphicsScene()
        self.full_browser_scene.addWidget(self.full_browser)
        self.full_signal_time_widget.setScene(self.full_browser_scene)


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

        self.full_psd_scene = QGraphicsScene()
        self.full_psd_fig = self.full_psd.plot(picks='Full Signal', 
                                            dB=dB, 
                                            amplitude='auto', 
                                            xscale=xscale, 
                                            spatial_colors=False,
                                            show=False)
        self.full_psd_fig.set_dpi(self.dpi)
        self.full_psd_fig.set_size_inches(420/self.dpi, 420/self.dpi, forward=True)
        self.full_psd_canvas = FigureCanvas(self.full_psd_fig)
        self.full_psd_scene.addWidget(self.full_psd_canvas)
        self.full_signal_power_widget.setScene(self.full_psd_scene)

    def update_ap_view(self, value):

        ap_model = self.ap_model_comboBox.currentText()

        ap_inter = self.ap_fixed_intercept_spinBox.value() 
        ap_exp = self.ap_fixed_exponent_spinBox.value()
        ap_knee = self.ap_knee_spinBox.value()
        random_seed = self.rand_seed_spinBox.value()

        sample_rate = self.fs_spinBox.value()
        duration = self.signal_duration_spinBox.value()
        
        epoch_duration = self.epoch_duration_spinBox.value()
        epoch_step = self.epoch_step_spinBox.value()

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

        # Generate signals
        ap_signal, full_signal, per_signal = elec_phys_signal(
                                                exponent=-ap_exp, 
                                                knee = ap_knee,
                                                intercept = ap_inter,
                                                sample_rate=sample_rate, 
                                                duration=duration, 
                                                nlv=nlv, 
                                                periodic_params=self.periodic_params,
                                                seed=random_seed)

        info = mne.create_info(ch_names=["Aperiodic"], ch_types=["misc"], sfreq=sample_rate)
        self.ap_obj = mne.io.RawArray(ap_signal.reshape(1,-1), info)

        #info = mne.create_info(ch_names=["Periodic"], ch_types=["misc"], sfreq=sample_rate)
        #self.per_obj = mne.io.RawArray(per_signal.reshape(1,-1), info)

        info = mne.create_info(ch_names=["Full Signal"], ch_types=["misc"], sfreq=sample_rate)
        self.full_obj_cont = mne.io.RawArray(full_signal.reshape(1,-1), info)

        _, ep_data = sliding_window(full_signal.reshape(1,-1),
                                    window=epoch_duration,
                                    sf=sample_rate,
                                    step=epoch_step)

        self.full_obj = mne.EpochsArray(ep_data, info.copy())

        # Decide on welch params
        n_fft = int(sample_rate * epoch_duration)

        # Calculate PSDs
        self.ap_psd = self.ap_obj.compute_psd(method='welch', 
                                            picks='all', 
                                            fmin=1, 
                                            fmax=100, 
                                            n_fft=n_fft, 
                                            average='median')

        self.full_psd = self.full_obj.compute_psd(method='welch', 
                                                picks='all', 
                                                fmin=1,  
                                                fmax=100, 
                                                n_fft=n_fft, 
                                                average='median')

        # plot signals 
        self.update_time_plot()

        # Plot PSDs
        self.update_psd_plot(0)