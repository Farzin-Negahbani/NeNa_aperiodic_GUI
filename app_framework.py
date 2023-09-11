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
from utils import periodic_signal, irasa, sim_peak_oscillation
from yasa import sliding_window
from scipy.stats import zscore

from neurodsp.sim import (sim_powerlaw, sim_random_walk, sim_synaptic_current,
                          sim_knee, sim_frac_gaussian_noise, sim_frac_brownian_motion)



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
        self.irasa_hset_textEdit.setText("1.1, 1.15, 1.2, 1.25")

        self.xscale = 'linear'
        self.yscale = 'log'

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
        plt.plot(self.fm.freqs, 10**(self.fm.power_spectrum)    , c='k', label="Power spectrum", lw=2)
        plt.plot(self.fm.freqs, 10**(self.fm._ap_fit)           , c='b',linestyle='--', label="Aperiodic fit", lw=2)
        plt.plot(self.fm.freqs, 10**(self.fm.fooofed_spectrum_) , c='r', label="FOOOF model fit", lw=2)
        plt.yscale(self.yscale)
        plt.xscale(self.xscale)
        self.fooof_psd_fig.set_dpi(self.dpi)
        self.fooof_psd_fig.set_size_inches(690/self.dpi, 370/self.dpi, forward=True)
        self.fooof_psd_canvas = FigureCanvas(self.fooof_psd_fig)
        self.fooof_psd_scene.addWidget(self.fooof_psd_canvas)
        self.fooof_res_psd_widget.setScene(self.fooof_psd_scene)
        self.fooof_psd_fig.suptitle('Full signal PSD with FOOOF result', fontsize=8)
        self.fooof_psd_fig.axes[0].set_xlabel("Frequency (Hz)", fontsize=8)
        self.fooof_psd_fig.axes[0].set_ylabel("PSD log($V^2$/Hz)", fontsize=8)
        self.fooof_psd_fig.axes[0].legend()

        self.fooof_res_r2_textEdit.setText(str(round(self.fm.r_squared_,3)))
        #self.fooof_res_error_textEdit.setText(str(round(self.fm.error_,3)))
        self.fooof_res_runtime_testEdit.setText(str(round(end_time-start_time,3)))
        self.fooof_res_inter_textEdit.setText(str(round(self.fm.aperiodic_params_[0],3)))

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
        plt.plot(freqs, (psd_ap+psd_osc).ravel(), c='k', label="Power spectrum", lw=2)
        plt.plot(freqs, psd_ap.ravel(), c='b',linestyle='--', label="Aperiodic fit", lw=2)
        plt.yscale(self.yscale)
        plt.xscale(self.xscale)
        #plt.plot(freqs, psd_osc.ravel(), c='cyan',linestyle='--', label="Periodic components", lw=2)
        self.irasa_psd_fig.set_dpi(self.dpi)
        self.irasa_psd_fig.set_size_inches(690/self.dpi, 370/self.dpi, forward=True)
        self.irasa_psd_canvas = FigureCanvas(self.irasa_psd_fig)
        self.irasa_psd_scene.addWidget(self.irasa_psd_canvas)
        self.irasa_res_psd_widget.setScene(self.irasa_psd_scene)
        self.irasa_psd_fig.suptitle('Full signal PSD with IRASA result', fontsize=8)
        #TODO Adjust labels based on scale
        self.irasa_psd_fig.axes[0].set_xlabel("Frequency (Hz)", fontsize=8)
        self.irasa_psd_fig.axes[0].set_ylabel("PSD log($V^2$/Hz)", fontsize=8)
        self.irasa_psd_fig.axes[0].legend()

        self.irasa_res_exp_textEdit.setText(str(round(fit_params['Slope'].values[0],3)))
        self.irasa_res_inter_textEdit.setText(str(round(fit_params['Intercept'].values[0],3)))
        self.irasa_res_r2_textEdit.setText(str(round(fit_params['R^2'].values[0],3)))
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
                                theme='light',
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
                                theme='light',
                                overview_mode='hidden')

        self.full_browser.setFixedSize(710, 440)
        self.full_browser_scene = QGraphicsScene()
        self.full_browser_scene.addWidget(self.full_browser)
        self.full_signal_time_widget.setScene(self.full_browser_scene)


    def update_psd_plot(self, value):
        
        self.xscale = 'log' if self.actionPSD_X_axis_log_scale.isChecked() else 'linear'
        self.yscale = 'log' if self.actionPSD_Y_axis_log_scale.isChecked() else 'linear'
        dB = True if self.yscale == 'log' else False

        # Calculate PSDs
        self.ap_psd = self.ap_obj.compute_psd(method='welch', 
                                            picks='all', 
                                            fmin=1, 
                                            fmax=100, 
                                            n_fft=self.n_fft, 
                                            average='median')

        self.full_psd = self.full_obj.compute_psd(method='welch', 
                                                picks='all', 
                                                fmin=1,  
                                                fmax=100, 
                                                n_fft=self.n_fft, 
                                                average='median')
                                                
        self.ap_psd_scene = QGraphicsScene()
        self.ap_psd_fig = self.ap_psd.plot(picks='Aperiodic', 
                                            dB=dB, 
                                            amplitude='auto', 
                                            xscale=self.xscale, 
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
                                            xscale=self.xscale, 
                                            spatial_colors=False,
                                            show=False)
        self.full_psd_fig.set_dpi(self.dpi)
        self.full_psd_fig.set_size_inches(420/self.dpi, 420/self.dpi, forward=True)
        self.full_psd_canvas = FigureCanvas(self.full_psd_fig)
        self.full_psd_scene.addWidget(self.full_psd_canvas)
        self.full_signal_power_widget.setScene(self.full_psd_scene)

    def update_ap_view(self, value):

        random_seed = self.rand_seed_spinBox.value()
        np.random.seed(random_seed)

        duration = self.signal_duration_spinBox.value()
        epoch_step = self.epoch_step_spinBox.value()
        self.sample_rate = self.fs_spinBox.value()
        self.epoch_duration = self.epoch_duration_spinBox.value()

        self.n_fft = min(self.nfft_spinBox.value(), int(self.sample_rate*self.epoch_duration))

        if self.n_fft != self.nfft_spinBox.value():
            self.nfft_spinBox.setValue(self.n_fft)

        for row, l in enumerate(self.periodic_params):
            # L has the form of (freq, aplitude, width)
            self.per_table.insertRow(row)
            self.per_table.setItem(row, 0, QTableWidgetItem(str(l[0])))
            self.per_table.setItem(row, 1, QTableWidgetItem(str(l[1])))
            self.per_table.setItem(row, 2, QTableWidgetItem(str(l[2])))
        self.per_table.setRowCount(len(self.periodic_params))

        self.ap_model = self.ap_model_comboBox.currentText()

        if self.ap_model == '1/f Power Law':
            # Power law parameters
            ap_exp = self.ap_fixed_exponent_spinBox.value()

            model_signal = sim_powerlaw(duration, self.sample_rate, ap_exp, f_range=(1, None))

        elif self.ap_model == '1/f Knee':
            # TODO increase signal duration limit
            # Knee parameters
            knee_knee = self.ap_knee_knee_spinBox.value()
            knee_exp1 = self.ap_knee_exponent1_spinBox.value()
            knee_exp2 = self.ap_knee_exponent2_spinBox.value()

            model_signal = sim_knee(duration, self.sample_rate, knee_exp1, knee_exp2, knee_knee)
            
        elif self.ap_model == 'Synaptic Activity Model':
            n_neurons = self.ap_synaptic_neurons_spinBox.value()
            firing_rate = self.ap_synaptic_firing_rate_spinBox.value()
            t_rise = self.ap_synaptic_trise_spinBox.value()
            t_decay = self.ap_synaptic_tdecay_spinBox.value()
            #t_ker = self.ap_synaptic_tkernel_spinBox.value()

            # model_signal = sim_synaptic_current(duration, self.sample_rate, n_neurons=n_neurons, firing_rate=firing_rate,
            #              tau_r=t_rise, tau_d=t_decay, t_ker=t_ker)
            model_signal = sim_synaptic_current(duration, self.sample_rate, n_neurons=n_neurons, firing_rate=firing_rate,
                         tau_r=t_rise, tau_d=t_decay, t_ker=None)
        elif self.ap_model == 'Wilson Cowan Model':
            print()
        elif ap_model == 'Hodgkin–Huxley Model':
            print()

        per_signal = np.zeros(int(self.sample_rate*duration))
        for osc_params in self.periodic_params:
            freq_osc, amp_osc, width = osc_params
            per_signal += sim_peak_oscillation(model_signal, 
                                                fs=self.sample_rate, 
                                                freq=freq_osc, 
                                                bw=width, 
                                                height=amp_osc)
        full_signal = per_signal + model_signal

        if self.white_noise_checkbox.isChecked():
            nlv = self.ap_gaussian_spinBox.value()

            full_signal += np.random.normal(scale=nlv, size=full_signal.size)

                            


        info = mne.create_info(ch_names=["Aperiodic"], ch_types=["misc"], sfreq=self.sample_rate)
        self.ap_obj = mne.io.RawArray(model_signal.reshape(1,-1), info)

        info = mne.create_info(ch_names=["Full Signal"], ch_types=["misc"], sfreq=self.sample_rate)
        self.full_obj_cont = mne.io.RawArray(full_signal.reshape(1,-1), info)

        _, ep_data = sliding_window(full_signal.reshape(1,-1),
                                    window=self.epoch_duration,
                                    sf=self.sample_rate,
                                    step=epoch_step)

        self.full_obj = mne.EpochsArray(ep_data, info.copy())

        # plot signals 
        self.update_time_plot()

        # Plot PSDs
        self.update_psd_plot(0)