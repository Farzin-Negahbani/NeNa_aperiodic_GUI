from typing import List, Tuple

import numpy as np
from numpy.fft import irfft, rfftfreq
import scipy as sp
import scipy.signal as signal
import mne
import fractions
import pandas as pd 
from neurodsp.sim import (sim_powerlaw, sim_random_walk, sim_synaptic_current,
                          sim_knee, sim_frac_gaussian_noise, sim_frac_brownian_motion)
from scipy.linalg import norm

from scipy.stats import zscore
from neurodsp.utils.data import create_times

import numba as nb
from numba import jit, prange

# Simulate the generative model.
def wilson_cowan_with_noise(noise):
    
    c1=15;   c2=15;     c3=15;  c4=7;         # Parameters from Fig 10 of [Wilson & Cowan, 1972]
    ae=1;    thetaE=2;  ai=2;  thetaI=2.5;
    tauE=50; tauI=50;
    RE=1;    RI=1;
    ke=1;    ki=1;
    P=1.25;  Q=0;
    
    N = 60000;
    dt = 0.1;
    
    E = np.zeros([N])                         # Output variables
    I = np.zeros([N])
    t = np.arange(N)*dt/1000                  # Time in [ms]
    
    E[0] = 0.22                               # Initial conditions
    I[0] = 0.22
    
    for n in np.arange(N-1):                  # Simulate the model
        x      = c1*E[n] -c2*I[n] + P
        Se     = 1/(1+np.exp(-ae*(x-thetaE))) - 1/(1+np.exp(ae*thetaE))
        E[n+1] = E[n] + dt*1/tauE*( -E[n] + (ke-RE*E[n])*Se + noise["E"]*np.random.randn() )
        x      = c3*E[n] -c4*I[n] + Q;
        Si     = 1/(1+np.exp(-ai*(x-thetaI))) - 1/(1+np.exp(ai*thetaI));
        I[n+1] = I[n] + dt*1/tauI*( -I[n] + (ki-RI*I[n])*Si + noise["I"]*np.random.randn() )
        
    E = E[10000:]; I = I[10000:]              # Drop initial transient.
    t = t[10000:]-t[10000]                    # start time at 0 s.
    
    return E,I,t


@jit(nopython=True, parallel=True)
def _fft_sim_peak_loop(freqs, freq, sig_ap_hat, sig_periodic, times, height, bw):
    
    #for f_val, fft in zip(freqs, sig_ap_hat):

    for i in prange(len(freqs)):
        f_val = freqs[i]
        fft = sig_ap_hat[i]
        # Compute the sum of squares of the cosines
        cos_times = 2 * np.pi * f_val * times
        cos_norm = np.linalg.norm(np.cos(cos_times), 2) ** 2

        # Compute random phase shift
        pha = np.cos(cos_times + 2 * np.pi * np.random.rand())

        # Define relative height above the aperiodic power spectrum
        hgt = height * np.exp(-(f_val - freq) ** 2 / (2 * bw ** 2))

        sig_periodic += (-np.real(fft) + np.sqrt(np.real(fft) ** 2 + \
            (10 ** hgt - 1) * np.abs(fft) ** 2)) / cos_norm * pha

    return sig_periodic

def sim_peak_oscillation(sig_ap, fs, freq, bw, height):
    """Simulate a signal with an aperiodic component and a specific oscillation peak.

    Parameters
    ----------
    sig_ap : 1d array
        The timeseries of the aperiodic component.
    fs : float
        Sampling rate of ``sig_ap``.
    freq : float
        Central frequency for the gaussian peak in Hz.
    bw : float
        Bandwidth, or standard deviation, of gaussian peak in Hz.
    height : float
        Relative height of the gaussian peak at the central frequency ``freq``.
        Units of log10(power), over the aperiodic component.

    Returns
    -------
    sig : 1d array
        Time series with desired power spectrum.

    Notes
    -----
    - This function creates a time series whose power spectrum consists of an aperiodic component
    and a gaussian peak at ``freq`` with standard deviation ``bw`` and relative ``height``.
    - The periodic component of the signal will be sinusoidal.

    Examples
    --------
    Simulate a signal with aperiodic exponent of -2 & oscillation central frequency of 20 Hz:

    >>> from neurodsp.sim import sim_powerlaw
    >>> fs = 500
    >>> sig_ap = sim_powerlaw(n_seconds=10, fs=fs, exponent=-2.0)
    >>> sig = sim_peak_oscillation(sig_ap, fs=fs, freq=20, bw=5, height=7)
    """

    sig_len = len(sig_ap)

    #times = create_times(sig_len / fs, fs)
    times = np.arange(0, sig_len / fs, 1/fs)

    # Compute the Fourier transform of the aperiodic signal
    #   We extract the first half of the frequencies from the FFT, since the signal is real
    sig_ap_hat = np.fft.fft(sig_ap)[0:(sig_len // 2 + 1)]

    # Create the corresponding frequency vector, which is used to create the cosines to sum
    freqs = np.linspace(0, fs / 2, num=sig_len // 2 + 1, endpoint=True)

    # Compute the periodic signal
    sig_periodic = np.zeros(sig_len)

    sig_periodic = _fft_sim_peak_loop(freqs, freq, sig_ap_hat, sig_periodic, times, height, bw)

    return  sig_periodic


def periodic_signal(
                     periodic_params: List[Tuple[float, float, float]] = None,
                     nlv: float = None,
                     highpass: bool = True,
                     sample_rate: float = 2400,
                     duration: float = 180,
                     seed: int = 1):
    """
    Generate 1/f noise with optionally added oscillations.

    Parameters
    ----------
   
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

    # Add oscillations
    amps_per = np.zeros_like(rand_phases)
    
    if periodic_params:
        for osc_params in periodic_params:
            freq_osc, amp_osc, width = osc_params

            amp_dist = sp.stats.norm(freq_osc, width).pdf(freqs)
            # add same random phases
            amp_dist = amp_dist * rand_phases   

            amps_per += amp_osc * amp_dist

    # Create colored noise time series from amplitudes
    periodic_signal =  irfft(amps_per, n=n_samples)

    # Add white noise
    if nlv:
        w_noise = np.random.normal(scale=nlv, size=n_samples)
        periodic_signal += w_noise

    # Highpass filter
    if highpass:
        sos = signal.butter(4, 1, btype="hp", fs=sample_rate, output='sos')
        periodic_signal = signal.sosfilt(sos, periodic_signal)

    return zscore(periodic_signal)


def irasa(data, sf=None, ch_names=None, band=(1, 45),
          hset=[1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,
                1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9],
          return_fit=True,
          win_sec=4,
          kwargs_welch=dict(average="median", window="hamming"),
          verbose=True,
          ):
    """

    Separate the aperiodic (= fractal, or 1/f) and oscillatory component
    of the power spectra of EEG data using the IRASA method.
    .. versionadded:: 0.1.7
    Parameters
    ----------
    data : :py:class:`numpy.ndarray` or :py:class:`mne.io.BaseRaw`
        1D or 2D EEG data. Can also be a :py:class:`mne.io.BaseRaw`, in which
        case ``data``, ``sf``, and ``ch_names`` will be automatically
        extracted, and ``data`` will also be converted from Volts (MNE default)
        to micro-Volts (YASA).
    sf : float
        The sampling frequency of data AND the hypnogram.
        Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw`.
    ch_names : list
        List of channel names, e.g. ['Cz', 'F3', 'F4', ...]. If None,
        channels will be labelled ['CHAN000', 'CHAN001', ...].
        Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw`.
    band : tuple or None
        Broad band frequency range.
        Default is 1 to 30 Hz.
    hset : list or :py:class:`numpy.ndarray`
        Resampling factors used in IRASA calculation. Default is to use a range
        of values from 1.1 to 1.9 with an increment of 0.05.
    return_fit : boolean
        If True (default), fit an exponential function to the aperiodic PSD
        and return the fit parameters (intercept, slope) and :math:`R^2` of
        the fit.

        The aperiodic signal, :math:`L`, is modeled using an exponential
        function in semilog-power space (linear frequencies and log PSD) as:
        .. math:: L = a + \text{log}(F^b)

        where :math:`a` is the intercept, :math:`b` is the slope, and
        :math:`F` the vector of input frequencies.
    win_sec : int or float
        The length of the sliding window, in seconds, used for the Welch PSD
        calculation. Ideally, this should be at least two times the inverse of
        the lower frequency of interest (e.g. for a lower frequency of interest
        of 0.5 Hz, the window length should be at least 2 * 1 / 0.5 =
        4 seconds).
    kwargs_welch : dict
        Optional keywords arguments that are passed to the
        :py:func:`scipy.signal.welch` function.
    verbose : bool or str
        Verbose level. Default (False) will only print warning and error
        messages. The logging levels are 'debug', 'info', 'warning', 'error',
        and 'critical'. For most users the choice is between 'info'
        (or ``verbose=True``) and warning (``verbose=False``).

    Returns
    -------
    freqs : :py:class:`numpy.ndarray`
        Frequency vector.
    psd_aperiodic : :py:class:`numpy.ndarray`
        The fractal (= aperiodic) component of the PSD.
    psd_oscillatory : :py:class:`numpy.ndarray`
        The oscillatory (= periodic) component of the PSD.
    fit_params : :py:class:`pandas.DataFrame` (optional)
        Dataframe of fit parameters. Only if ``return_fit=True``.

    Notes
    -----
    The Irregular-Resampling Auto-Spectral Analysis (IRASA) method is
    described in Wen & Liu (2016). In a nutshell, the goal is to separate the
    fractal and oscillatory components in the power spectrum of EEG signals.

    The steps are:

    1. Compute the original power spectral density (PSD) using Welch's method.
    2. Resample the EEG data by multiple non-integer factors and their
       reciprocals (:math:`h` and :math:`1/h`).
    3. For every pair of resampled signals, calculate the PSD and take the
       geometric mean of both. In the resulting PSD, the power associated with
       the oscillatory component is redistributed away from its original
       (fundamental and harmonic) frequencies by a frequency offset that varies
       with the resampling factor, whereas the power solely attributed to the
       fractal component remains the same power-law statistical distribution
       independent of the resampling factor.
    4. It follows that taking the median of the PSD of the variously
       resampled signals can extract the power spectrum of the fractal
       component, and the difference between the original power spectrum and
       the extracted fractal spectrum offers an approximate estimate of the
       power spectrum of the oscillatory component.

    Note that an estimate of the original PSD can be calculated by simply
    adding ``psd = psd_aperiodic + psd_oscillatory``.

    For an example of how to use this function, please refer to 
    https://github.com/raphaelvallat/yasa/blob/master/notebooks/09_IRASA.ipynb
    
    For an article discussing the challenges of using IRASA (or fooof) see [5].
    
    References
    ----------
    [1] Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory
        Components in the Power Spectrum of Neurophysiological Signal.
        Brain Topography, 29(1), 13–26. https://doi.org/10.1007/s10548-015-0448-0
    [2] https://github.com/fieldtrip/fieldtrip/blob/master/specest/
    [3] https://github.com/fooof-tools/fooof
    [4] https://www.biorxiv.org/content/10.1101/299859v1
    [5] https://doi.org/10.1101/2021.10.15.464483
    """
    import fractions
    from yasa.io import set_log_level
    import logging

    set_log_level(verbose)
    # Check if input data is a MNE Raw object
    if isinstance(data, mne.io.BaseRaw):
        sf = data.info["sfreq"]  # Extract sampling frequency
        ch_names = data.ch_names  # Extract channel names
        hp = data.info["highpass"]  # Extract highpass filter
        lp = data.info["lowpass"]  # Extract lowpass filter
        data = data.get_data(units=dict(eeg="uV", emg="uV", eog="uV", ecg="uV"))
    else:
        # Safety checks
        assert isinstance(data, np.ndarray), "Data must be a numpy array."
        data = np.atleast_2d(data)
        assert data.ndim == 2, "Data must be of shape (nchan, n_samples)."
        nchan, npts = data.shape
        assert nchan < npts, "Data must be of shape (nchan, n_samples)."
        assert sf is not None, "sf must be specified if passing a numpy array."
        assert isinstance(sf, (int, float))
        if ch_names is None:
            ch_names = ["CHAN" + str(i).zfill(3) for i in range(nchan)]
        else:
            ch_names = np.atleast_1d(np.asarray(ch_names, dtype=str))
            assert ch_names.ndim == 1, "ch_names must be 1D."
            assert len(ch_names) == nchan, "ch_names must match data.shape[0]."
        hp = 0  # Highpass filter unknown -> set to 0 Hz
        lp = sf / 2  # Lowpass filter unknown -> set to Nyquist

    # Check the other arguments
    hset = np.asarray(hset)
    assert hset.ndim == 1, "hset must be 1D."
    assert hset.size > 1, "2 or more resampling fators are required."
    hset = np.round(hset, 4)  # avoid float precision error with np.arange.
    band = sorted(band)
    assert band[0] > 0, "first element of band must be > 0."
    assert band[1] < (sf / 2), "second element of band must be < (sf / 2)."
    win = int(win_sec * sf)  # nperseg

    # Inform about maximum resampled fitting range
    h_max = np.max(hset)
    band_evaluated = (band[0] / h_max, band[1] * h_max)
    freq_Nyq = sf / 2  # Nyquist frequency
    freq_Nyq_res = freq_Nyq / h_max  # minimum resampled Nyquist frequency
    logging.info(f"Fitting range: {band[0]:.2f}Hz-{band[1]:.2f}Hz")
    logging.info(f"Evaluated frequency range: {band_evaluated[0]:.2f}Hz-{band_evaluated[1]:.2f}Hz")
    if band_evaluated[0] < hp:
        logging.warning(
            "The evaluated frequency range starts below the "
            f"highpass filter ({hp:.2f}Hz). Increase the lower band"
            f" ({band[0]:.2f}Hz) or decrease the maximum value of "
            f"the hset ({h_max:.2f})."
        )
    if band_evaluated[1] > lp and lp < freq_Nyq_res:
        logging.warning(
            "The evaluated frequency range ends after the "
            f"lowpass filter ({lp:.2f}Hz). Decrease the upper band"
            f" ({band[1]:.2f}Hz) or decrease the maximum value of "
            f"the hset ({h_max:.2f})."
        )
    if band_evaluated[1] > freq_Nyq_res:
        logging.warning(
            "The evaluated frequency range ends after the "
            "resampled Nyquist frequency "
            f"({freq_Nyq_res:.2f}Hz). Decrease the upper band "
            f"({band[1]:.2f}Hz) or decrease the maximum value "
            f"of the hset ({h_max:.2f})."
        )

    # Calculate the original PSD over the whole data
    freqs, psd = signal.welch(data, sf, nperseg=win, **kwargs_welch)

    # Start the IRASA procedure
    psds = np.zeros((len(hset), *psd.shape))

    for i, h in enumerate(hset):
        # Get the upsampling/downsampling (h, 1/h) factors as integer
        rat = fractions.Fraction(str(h))
        up, down = rat.numerator, rat.denominator
        # Much faster than FFT-based resampling
        data_up = signal.resample_poly(data, up, down, axis=-1)
        data_down = signal.resample_poly(data, down, up, axis=-1)
        # Calculate the PSD using same params as original
        freqs_up, psd_up = signal.welch(data_up, h * sf, nperseg=win, **kwargs_welch)
        freqs_dw, psd_dw = signal.welch(data_down, sf / h, nperseg=win, **kwargs_welch)
        # Geometric mean of h and 1/h
        psds[i, :] = np.sqrt(psd_up * psd_dw)

    # Now we take the median PSD of all the resampling factors, which gives
    # a good estimate of the aperiodic component of the PSD.
    psd_aperiodic = np.median(psds, axis=0)

    # We can now calculate the oscillations (= periodic) component.
    psd_osc = psd - psd_aperiodic

    # Let's crop to the frequencies defined in band
    mask_freqs = np.ma.masked_outside(freqs, *band).mask
    freqs = freqs[~mask_freqs]
    psd_aperiodic = np.compress(~mask_freqs, psd_aperiodic, axis=-1)
    psd_osc = np.compress(~mask_freqs, psd_osc, axis=-1)

    if return_fit:
        # Aperiodic fit in semilog space for each channel
        from scipy.optimize import curve_fit
        intercepts, slopes, r_squared = [], [], []

        def func(t, a, b):
            # See https://github.com/fooof-tools/fooof
            return a + np.log10(t**b)

        for y in np.atleast_2d(psd_aperiodic):
            y_log = np.log10(y)
            # Note that here we define bounds for the slope but not for the
            # intercept.
            popt, pcov = curve_fit(
                func, freqs, y_log, p0=(2, -1), bounds=((-np.inf, -10), (np.inf, 2))
            )
            intercepts.append(popt[0])
            slopes.append(popt[1])
            # Calculate R^2: https://stackoverflow.com/q/19189362/10581531
            residuals = y_log - func(freqs, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_log - np.mean(y_log)) ** 2)
            r_squared.append(1 - (ss_res / ss_tot))

        # Create fit parameters dataframe
        fit_params = {
            "Chan": ch_names,
            "Intercept": intercepts,
            "Slope": slopes,
            "R^2": r_squared,
            "std(osc)": np.std(psd_osc, axis=-1, ddof=1),
        }
        return freqs, psd_aperiodic, psd_osc, pd.DataFrame(fit_params)
    else:
        return freqs, psd_aperiodic, psd_osc
