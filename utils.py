from typing import List, Tuple

import numpy as np
from numpy.fft import irfft, rfftfreq
import scipy as sp
import scipy.signal as sig

def elec_phys_signal(exponent: float,
                     knee: float = 0,
                     intercept: float = 0,
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
    amps *= rand_phases * 10**intercept
    
    # TODO checkk this! why divide by 2?
    #amps /= (knee + freqs ** (exponent / 2))
    amps /= (knee + freqs ** exponent)

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
        #periodic_signal += w_noise

    # Highpass filter
    if highpass:
        sos = sig.butter(4, 1, btype="hp", fs=sample_rate, output='sos')
        aperiodic_signal = sig.sosfilt(sos, aperiodic_signal)
        full_signal = sig.sosfilt(sos, full_signal)
        periodic_signal = sig.sosfilt(sos, periodic_signal)

    return aperiodic_signal, full_signal, periodic_signal

