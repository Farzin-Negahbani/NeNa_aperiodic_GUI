from typing import List, Tuple

from PyQt5.QtWidgets import QMainWindow, QFileDialog, QTableWidgetItem, QPushButton, QApplication, QCheckBox, QGraphicsScene, QGraphicsView
from PyQt5.QtCore import QAbstractTableModel, Qt

import os 
import time 
import design
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import mne
from mne_qt_browser.figure import MNEQtBrowser

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from numpy.fft import irfft, rfftfreq
import scipy as sp
import scipy.signal as sig


def elec_phys_signal(exponent: float,
                     periodic_params: List[Tuple[float, float, float]] = None,
                     nlv: float = None,
                     highpass: bool = False,
                     sample_rate: float = 2400,
                     duration: float = 180,
                     seed: int = 1):
    """
    Generate 1/f noise with optionally added oscillations.

    Parameters
    ----------
    exponent : float
        Aperiodic 1/f exponent.
    periodic_params : list of tuples
        Oscillations parameters as list of tuples in form of
                [(center_frequency1, peak_amplitude1, peak_width1),
                (center_frequency2, peak_amplitude2, peak_width2)]
        for two oscillations.
    nlv : float, optional
        Level of white noise. The default is None.
    highpass : bool, optional
        Whether to apply a 4th order butterworth highpass filter at 1Hz.
        The default is False.
    sample_rate : float, optional
        Sample rate of the signal. The default is 2400Hz.
    duration : float, optional
        Duration of the signal in seconds. The default is 180s.
    seed : int, optional
        Seed for reproducability. The default is 1.

    Returns
    -------
    aperiodic_signal : ndarray
        Aperiodic 1/f activitiy without oscillations.
    full_signal : ndarray
        Aperiodic 1/f activitiy with added oscillations.
    """
    if seed:
        np.random.seed(seed)
    # Initialize
    n_samples = int(duration * sample_rate)
    amps = np.ones(n_samples//2, complex)
    freqs = rfftfreq(n_samples, d=1/sample_rate)
    freqs = freqs[1:]  # avoid divison by 0

    # Create random phases
    rand_dist = np.random.uniform(0, 2*np.pi, size=amps.shape)
    rand_phases = np.exp(1j * rand_dist)

    # Multiply phases to amplitudes and create power law
    amps *= rand_phases
    amps /= freqs ** (exponent / 2)

    # Add oscillations
    amps_osc = amps.copy()
    amps_per = np.zeros_like(amps)
    if periodic_params:
        for osc_params in periodic_params:
            freq_osc, amp_osc, width = osc_params

            amp_dist = sp.stats.norm(freq_osc, width).pdf(freqs)
            # add same random phases
            amp_dist = amp_dist * rand_phases   

            tmp_amps = amp_osc * amp_dist

            amps_osc += tmp_amps
            
            amps_per += tmp_amps

    # Create colored noise time series from amplitudes
    aperiodic_signal = irfft(amps)
    periodic_signal =  irfft(amps_per)
    full_signal = irfft(amps_osc)

    # Add white noise
    if nlv:
        w_noise = np.random.normal(scale=nlv, size=n_samples-2)
        aperiodic_signal += w_noise
        full_signal += w_noise
        periodic_signal += w_noise

    # Highpass filter
    if highpass:
        sos = sig.butter(4, 1, btype="hp", fs=sample_rate, output='sos')
        aperiodic_signal = sig.sosfilt(sos, aperiodic_signal)
        full_signal = sig.sosfilt(sos, full_signal)
        periodic_signal = sig.sosfilt(sos, periodic_signal)

    return aperiodic_signal, full_signal, periodic_signal


class MyApp(QMainWindow, design.Ui_dialog):
    def __init__(self, dpi):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.dpi = dpi
        # Just ignoring the value 
        self.update_ap_view(0)

    # Removes selected row from per_tableview
    def remove_per_signal(self):
        print("Not implemented")
    
    # Adds a new row to per_tableview
    def add_per_signal(self):
        print("Not implemented")

    def update_ap_view(self, value):
        print(f"value {value}")

        ap_model = self.ap_model_comboBox.currentText()

        ap_fixed_inter = self.ap_fixed_intercept_spinBox.value() 
        ap_fixed_exp = self.ap_fixed_exponent_spinBox.value()
        sample_rate = 2400 

        ap_signal, full_signal, per_signal = elec_phys_signal(
                                                exponent=-ap_fixed_exp, 
                                                sample_rate=sample_rate, 
                                                duration=10, 
                                                nlv=0.001, 
                                                periodic_params=[(15, 2, 2), (40,10,2)])

        info = mne.create_info(ch_names=["Aperiodic"], ch_types=["misc"], sfreq=sample_rate)
        ap_obj = mne.io.RawArray(ap_signal.reshape(1,-1), info)

        info = mne.create_info(ch_names=["Periodic"], ch_types=["misc"], sfreq=sample_rate)
        per_obj = mne.io.RawArray(per_signal.reshape(1,-1), info)

        info = mne.create_info(ch_names=["Full Signal"], ch_types=["misc"], sfreq=sample_rate)
        full_obj = mne.io.RawArray(full_signal.reshape(1,-1), info)

        ap_browser = ap_obj.plot(show=False, 
                                block=False, 
                                show_scrollbars=False, 
                                show_scalebars=False,
                                overview_mode='hidden')
        ap_browser.setFixedSize(826, 406)
        # Create a QGraphicsScene to hold the QtBrowser
        ap_browser_scene = QGraphicsScene()
        ap_browser_scene.addWidget(ap_browser)
        # Set the QGraphicsScene for your QGraphicsView
        self.ap_time_widget.setScene(ap_browser_scene)

        ap_psd_scene = QGraphicsScene()
        ap_psd = ap_obj.compute_psd(picks='Aperiodic',fmax=100, n_fft=2048, average='median')
        self.ap_psd_fig = ap_psd.plot(picks='Aperiodic', dB=True, amplitude='auto', xscale='log', show=False)
        self.ap_psd_fig.set_dpi(self.dpi)
        self.ap_psd_fig.set_size_inches(411/self.dpi, 411/self.dpi, forward=True)
        ap_psd_canvas = FigureCanvas(self.ap_psd_fig)
        ap_psd_scene.addWidget(ap_psd_canvas)
        self.ap_power_widget.setScene(ap_psd_scene)
        #ap_psd_canvas.draw()

        per_browser = per_obj.plot(show=False, 
                                block=False, 
                                show_scrollbars=False, 
                                show_scalebars=False,
                                overview_mode='hidden')
        
        per_browser.setFixedSize(826, 406)
        per_browser_scene = QGraphicsScene()
        per_browser_scene.addWidget(per_browser)
        self.per_time_widget.setScene(per_browser_scene)


        per_psd_scene = QGraphicsScene()
        per_psd = per_obj.compute_psd(picks='Periodic',fmax=100, n_fft=2048, average='median')
        self.per_psd_fig = per_psd.plot(picks='Periodic', dB=True, amplitude='auto', xscale='log', show=False)
        self.per_psd_fig.set_dpi(self.dpi)
        self.per_psd_fig.set_size_inches(411/self.dpi, 411/self.dpi, forward=True)
        per_psd_canvas = FigureCanvas(self.per_psd_fig)
        per_psd_scene.addWidget(per_psd_canvas)
        self.per_power_widget.setScene(per_psd_scene)
        #per_psd_canvas.draw()