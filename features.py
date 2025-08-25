"""
Script for feature calculations:
- Statistical
- Autocorrelation
- Frequency
- Entropy
- Hjorth parameters
- Fractal dimension
- Bispectrum
- Wavelet
- Functional connectivity
"""


# ---------- Imports ----------
import antropy as ent
from mne_connectivity import envelope_correlation, EpochTemporalConnectivity
import mne_features.univariate as mne_feat_uni
import numpy as np
import pywt
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy
from tqdm import tqdm

from constants import *


# ---------- Feature dimension reduction ----------
def get_features(
        metrics: dict[str, list[np.ndarray | list[float]]],
        dim_reduction: str,
        feature_type: str | None = None
    ) -> dict[str, np.ndarray]:
    """
    Performs the feature dimension reduction to mean, std channel (if
    desired = if dim_reduction == "average"), returning a dictionary of arrays.

    If dim_reduction == "average", for each epoch, mean and std are
    calculated, resulting in two features per epoch. 

    If dim_reduction == "multichannel", all channels' data are saved,
    resulting in n_channels features per epoch.

    Functional connectivity features (feature_type == "FC") do not need the
    dimension reduction, since they are global features and they are not
    calculated by channel.

    Parameters
    ----------
        metrics: dict[str, list[np.ndarray | list[float]]]
            Dictionary containing the data, where the keys are the metric name 
            and the values are the values of that metric by epoch and channel.
        dim_reduction: str
            Type of dimension reduction to be applied: "average" (the epoch mean
            and std over the channels) or "multichannel" (no dimension
            reduction).
        feature_type: str | None, optional
            Feature type: 
            - "Wavelet", which has each feature has 12
            sub-features depending on the scale (e.g.
            Wavelet_coefs_mean_scale_4);
            - "FC", which does not need dimension reduction
            - None, which are the other feature types, which do not need
            any particular treatment.
    
    Returns
    -------
        features: dict[str, np.ndarray]
            Dictionary where de keys are the name of the metrics (e.g.
            mean_mean or mean_ch_7) and the values are the values of that
            metric by epoch in an array.
    """
    features: dict[str, np.ndarray] = {}
    match dim_reduction:
        case "average":
            if feature_type is None:
                for metric, values in metrics.items():
                    values: np.ndarray = np.array(values)
                    mean: np.ndarray = values.mean(axis=1)
                    std: np.ndarray = values.std(axis=1)
                    
                    features[f"{metric}_mean"] = mean
                    features[f"{metric}_std"] = std
                
                return features
            
            if feature_type == "Wavelet":
                for metric, values in metrics.items():
                    values: np.ndarray = np.array(values)
                    mean: np.ndarray = values.mean(axis=1)
                    std: np.ndarray = values.std(axis=1)
                    for i in range(12):
                        features[f"{metric}_scale_{i+1}_mean"] = mean[:,i]
                        features[f"{metric}_scale_{i+1}_std"] = std[:,i]
                
                return features
            
        case "multichannel":
            if feature_type is None:
                for metric, values in metrics.items():
                    values: np.ndarray = np.array(values)
                    for ch in range(values.shape[1]):
                        features[f"{metric}_ch_{ch+1}"] = values[:,ch]
                
                return features
            
            if feature_type == "Wavelet":
                for metric, values in metrics.items():
                    values: np.ndarray = np.array(values)
                    for i in range(values.shape[2]):
                        for ch in range(values.shape[1]):
                            features[f"{metric}_scale_{i+1}_ch_{ch+1}"] = \
                                values[:,ch,i]
                
                return features

        
    if feature_type == "FC":
        for metric, values in metrics.items():
            values: np.ndarray = np.array(values)
            features[metric] = values
        
        return features
    

# ---------- Statistical features ----------

def calc_statistical(data: np.ndarray) -> dict[str, list[np.ndarray]]:
    """
    Calculates statistical features:
    - "Mean": Mean value of each channel.
    - "Median": Central value of each channel.
    - "Std": Standard deviation of each channel.
    - "Skewness": Asymmetry of each channel distribution.
    - "Kurtosis": Tailedness of each channel distribution.
    - "Mean Absolute Values of 1st Differences": Mean absolute 1st order
    differences along time dimension.
    - "Mean Absolute Values of 2nd Differences": Mean absolute 2nd order
    differences along time dimension.
    - "Normalized Mean Absolute Values of 1st Differences": First order MAV
    normalized by the standard deviation.
    - "Normalized Mean Absolute Values of 2nd Differences": Second order MAV
    normalized by the standard deviation.
    - "RMS": Root Mean Square of the signal in each channel

    Parameters
    ----------
        data: np.ndarray
            EEG data of (shape n_epochs, n_channels, n_times)
    
    Returns
    -------
        statistical_metrics: dict[str, list[np.ndarray]]
            Dictionary containing the extracted features by epochs.
    """
    statistical_metrics : dict[str, np.ndarray] = {
        "Mean": [],
        "Median": [],
        "Std": [],
        "Skewness": [],
        "Kurtosis": [],
        "Mean Absolute Values of 1st Differences": [],
        "Mean Absolute Values of 2nd Differences": [],
        "Normalized Mean Absolute Values of 1st Differences": [],
        "Normalized Mean Absolute Values of 2nd Differences": [],
        "RMS": []
    }
    for ep in tqdm(
        data,
        desc=f"{CYAN}Calculating statistical features{RESET}",
        colour="cyan"
    ):
        statistical_metrics["Mean"].append(np.mean(ep, axis=1))
        statistical_metrics["Median"].append(np.median(ep, axis=1))
        statistical_metrics["Std"].append(std:=np.std(ep, axis=1))
        statistical_metrics["Skewness"].append(skew(ep, axis=1))
        statistical_metrics["Kurtosis"].append(kurtosis(ep, axis=1))
        statistical_metrics["Mean Absolute Values of 1st Differences"].append(
            mav1d:=np.mean(np.abs(np.diff(ep, axis=1)), axis=1)
        )
        statistical_metrics["Mean Absolute Values of 2nd Differences"].append(
            mav2d:=np.mean(np.abs(np.diff(ep, n=2, axis=1)), axis=1)
        )
        statistical_metrics[
            "Normalized Mean Absolute Values of 1st Differences"].append(
                mav1d/std
            )
        statistical_metrics[
            "Normalized Mean Absolute Values of 2nd Differences"].append(
                mav2d/std
            )
        statistical_metrics["RMS"].append(np.sqrt(np.mean(ep**2, axis=-1)))
    
    return statistical_metrics


# ---------- Autocorrelation metrics ----------

def calc_autocorr(x: np.ndarray) -> np.ndarray:
    """
    Computes the normalized autocorrelation of a 1D signal.

    Parameters
    ----------
        x: np.ndarray
            1D input signal
    
    Returns
    -------
        autocorr: np.ndarray
            Normalized autocorrelation of the signal, starting from lag 0. The
            result is scaled so that the maximum value equals 1.
    """
    autocorr: np.ndarray = np.correlate(x, x, mode="full")
    autocorr: np.ndarray = autocorr / autocorr.max()
    autocorr: np.ndarray = autocorr[autocorr.size//2:]

    return autocorr


def AUautocorr(autocorr: np.ndarray, fs: float = 500.0) -> float:
    """
    Computes the area under the autocorrelation curve.

    Parameters
    ----------
        autocorr: np.ndarray
            Normalized autocorrelation of the signal, starting from lag 0. The
            result is scaled so that the maximum value equals 1.
        
        fs: float, optional, default = 500.0
            Sample frequency in Hz.

    Returns
    -------
        area: float
            Area under the absolute autocorrelation curve, computed using the 
            trapezoidal rule
    """
    area: float = np.trapezoid(np.abs(autocorr), dx=1/fs)

    return area


def AACW(
        autocorr: np.ndarray,
        win: float=1/np.e,
        fs: float = 500
    )-> tuple[float, float]:
    """
    Computes the Autocorrelation Window (ACW) and its area. The ACW is defined
    as the first time lag where the autocorrelation falls below a a given
    threshold (`win`).

    Parameters
    ----------
        autocorr: np.ndarray
            Normalized autocorrelation of the signal, starting from lag 0. The
            result is scaled so that the maximum value equals 1.
        win: float, optional, default = 1/np.e
            Threshold for determining the window length.
        fs: float, optional, default = 500.0
            Sample frequency in Hz.

    Returns
    -------
    acw_time: float
        Duration of the autocorrelation window in seconds
    area: float
        Area under the autocorrelation curve up to the ACW
    """
    acw: int | None = None
    for i, cor in enumerate(autocorr):
        if cor <= win:
            acw: int = i
            break
    if acw is None:
        acw: int = len(autocorr)
    
    area: float = np.trapezoid(np.abs(autocorr[:acw]), dx=1/fs)
    return acw/fs, area


def calc_autocorrelation(
        data: np.ndarray, fs: float = 500.0) ->  dict[str, list[list[float]]]:
    """
    Calculates autocorrelation features:
    - "Autocorr_area": Area under the entire autocorrelation curve.
    - "ACW": Autocorrelation Window (time until correlation <= 1/e).
    - "ACW_area": Area under autocorrelation until ACW.
    - "ACW0": Time until first zero-crossing of autocorrelation.
    - "ACW0_area": Area under autocorrelation until zero-crossing.

    Parameters
    ----------
        data: np.ndarray
            3D array of shape (n_epochs, n_channels, n_samples).
        fs: int, optional, default = 500
            Sampling frequency in Hz.

    Returns
    -------
        autocorrelation_metrics: dict[str, list[list[float]]]
                Dictionary containing the extracted features by epochs.
    """
    autocorrelation_metrics: dict[str, list[list[float]]] = {
        "Autocorr_area": [],
        "ACW": [],
        "ACW_area": [],
        "ACW0": [],
        "ACW0_area": []}
    for ep in tqdm(
        data,
        desc=f"{CYAN}Calculating correlation features{RESET}",
        colour="cyan"
    ):
        ep_results: dict[str, list[float]] = {
            metric: [] for metric in autocorrelation_metrics
        }
        for ch in ep:
            x: np.ndarray = ch[:]
            autocorr = calc_autocorr(x)
            
            ep_results["Autocorr_area"].append(AUautocorr(autocorr, fs))
            ep_results["ACW"].append(AACW(autocorr, fs=fs)[0])
            ep_results["ACW_area"].append(AACW(autocorr, fs=fs)[1])
            ep_results["ACW0"].append(AACW(autocorr, win=0, fs=fs)[0])
            ep_results["ACW0_area"].append(AACW(autocorr, win=0, fs=fs)[1])
        
        for metric in autocorrelation_metrics:
            autocorrelation_metrics[metric].append(ep_results[metric])

    return autocorrelation_metrics


# ---------- Frequency features ----------

def bandpower(
        data: np.ndarray,
        band: tuple[float, float],
        fs: float = 500.0
        ) -> float:
    """
    Computes the relative band power (RBP) of a 1D signal.

    Parameters
    ----------
        data: np.ndarray
            1D input signal (time series)
        band: tuple[float, float]
            Frequency band of interest as (fmin, fmax) in Hz.
        fs: int, optional, default = 500
            Sampling frequency in Hz.
    
    Returns
    -------
    rbp: float
        Relative band power: ratio between the spectral power in the given
        frequency band and the total spectral power of the signal.
    """
    fmin: float; fmax: float
    fmin, fmax = band
    freqs: np.ndarray; psd: np.ndarray
    freqs, psd = welch(data, fs=fs)
    idx_band: np.ndarray = np.logical_and(freqs >= fmin, freqs <= fmax)

    area_band: np.float = np.trapezoid(psd[idx_band], freqs[idx_band])
    area_total: np.float = np.trapezoid(psd, freqs)

    return area_band / area_total


def calc_band_power(
        data: np.ndarray, fs: float = 500.0) -> dict[str, list[list[float]]]:
    """
    Calculates RBP features:
    - "Delta RBP": Relative power in the delta band (0.5 - 4 Hz).
    - "Theta RBP": Relative power in the theta band (4 - 8 Hz).
    - "Alpha RBP": Relative power in the alpha band (8 - 13 Hz).
    - "Beta RBP": Relative power in the beta band (13 - 30 Hz).
    - "Gamma RBP": Relative power in the gamma band (30 - 70 Hz).
    - "Alpha/Delta Ratio"
    - "Alpha/Theta Ratio"
    - "Delta/Theta Ratio"
    - "Delta/Beta Ratio"
    - "Theta/Beta Ratio"

    Parameters
    ----------
        data: np.ndarray
            3D array of shape (n_epochs, n_channels, n_samples).
        fs: int, optional, default = 500
            Sampling frequency in Hz.

    Returns
    -------
        band_power_metrics: dict[str, list[list[float]]]
            Dictionary containing the extracted features by epochs.
    """
    band_power_metrics: dict[str, list[list[float]]] = {
        "Delta RBP": [],
        "Theta RBP": [],
        "Alpha RBP": [],
        "Beta RBP": [],
        "Gamma RBP": [],
        "Alpha/Delta Ratio": [],
        "Alpha/Theta Ratio": [],
        "Delta/Theta Ratio": [],
        "Delta/Beta Ratio": [],
        "Theta/Beta Ratio": []}
    for ep in tqdm(
        data,
        desc=f"{CYAN}Calculating band power features{RESET}",
        colour="cyan"
    ):
        ep_results: dict[str, list[float]] = {
            metric: [] for metric in band_power_metrics
        }
        for ch in ep:
            x: np.ndarray = ch[:]

            delta_rbp: float
            theta_rbp: float
            alpha_rbp: float
            beta_rbp: float
            ep_results["Delta RBP"].append(
                delta_rbp:=bandpower(x, BANDS["delta"], fs)
            )
            ep_results["Theta RBP"].append(
                theta_rbp:=bandpower(x, BANDS["theta"], fs)
            )
            ep_results["Alpha RBP"].append(
                alpha_rbp:=bandpower(x, BANDS["alpha"], fs)
            )
            ep_results["Beta RBP"].append(
                beta_rbp:=bandpower(x, BANDS["beta"], fs)
            )
            ep_results["Gamma RBP"].append(bandpower(x, BANDS["gamma"], fs))
            ep_results["Alpha/Delta Ratio"].append(alpha_rbp/delta_rbp)
            ep_results["Alpha/Theta Ratio"].append(alpha_rbp/theta_rbp)
            ep_results["Delta/Theta Ratio"].append(delta_rbp/theta_rbp)
            ep_results["Delta/Beta Ratio"].append(delta_rbp/beta_rbp)
            ep_results["Theta/Beta Ratio"].append(theta_rbp/beta_rbp)

        for metric in band_power_metrics:
            band_power_metrics[metric].append(ep_results[metric])

    return band_power_metrics


# ---------- Entropy features ----------

def calc_entropy(
        data: np.ndarray, fs: float = 500.0) -> dict[str, list[np.ndarray]]:
    """
    Computes entropy-based features that quantify the complexity,
    irregularity, and unpredictability of signals:
    - "Approximate Entropy": Quantifies regularity and predictability of time
    series; lower values = more regular.
    - "Sample Entropy": Improved version of approximate entropy, less
    dependent on data length.
    "Spectral Entropy": Shannon entropy of the normalized power spectral
    density, measures signal flatness/uniformity in frequency domain.
    "SVD Entropy" : Entropy of singular value decomposition, reflects
    complexity and dimensionality of the signal.

    Parameters:
    -----------
        data: np.ndarray
            3D array of shape (n_epochs, n_channels, n_samples).
        fs: int, optional, default = 500
            Sampling frequency in Hz.
    
    Returns:
    --------
        entropy_features: dict[str, list[np.ndarray]]
            Dictionary containing the extracted features by epochs.
    """
    entropy_features: dict[str, list[np.ndarray]] = {
        "Approximate Entropy": [],
        "Sample Entropy": [],
        "Spectral Entropy": [],
        "SVD Entropy": []
    }
    for ep in tqdm(
        data,
        desc=f"{CYAN}Calculating entropy features{RESET}",
        colour="cyan"
    ):
        entropy_features["Approximate Entropy"].append(
            mne_feat_uni.compute_app_entropy(ep)
        )
        entropy_features["Sample Entropy"].append(
            mne_feat_uni.compute_samp_entropy(ep)
        )
        entropy_features["Spectral Entropy"].append(
            mne_feat_uni.compute_spect_entropy(fs, ep)
        )
        entropy_features["SVD Entropy"].append(
            mne_feat_uni.compute_svd_entropy(ep)
        )
    
    return entropy_features


# ---------- Hjorth features ----------

def calc_hjorth(data: np.ndarray) -> dict[str, list[np.ndarray]]:
    """
    Computes Hjorth parameters:
    - "Hjorth_Activity": Variance of the signal, representing the signal
    power.
    - "Hjorth_Mobility" : Square root of variance of first derivative divided
    by variance of the signal. Reflects mean frequency.
    - "Hjorth_Complexity": Ratio of the mobility of the first derivative 
    to the mobility of the signal itself. Reflects similarity of the 
    signal to a pure sine wave (higher = more complex).

    Parameters
    ----------
        data: np.ndarray
            3D array of shape (n_epochs, n_channels, n_samples).
    
    Returns
    -------
        hjorth_metrics: dict[str, list[np.ndarray]]
            Dictionary containing the extracted features by epochs.
    """
    hjorth_metrics: dict[str, list[np.ndarray]] = {
        "Hjorth_Activity": [], "Hjorth_Mobility": [], "Hjorth_Complexity": []
    }
    for ep in tqdm(
        data, desc=f"{CYAN}Calculating Hjorth features{RESET}", colour="cyan"
    ):
        hjorth_metrics["Hjorth_Activity"].append(np.var(ep, axis=1))
        hjorth_metrics["Hjorth_Mobility"].append(
            mne_feat_uni.compute_hjorth_mobility(ep)
        )
        hjorth_metrics["Hjorth_Complexity"].append(
            mne_feat_uni.compute_hjorth_complexity(ep)
        )
    
    return hjorth_metrics


## Fractal dimension features
def calc_fd(data: np.ndarray) -> dict[str, list[np.ndarray]]:
    """
    Computes Fractal Dimension features which provide measures of signal
    complexity and self-similarity:
    - "Katz_FD": Katz's fractal dimension, measures the complexity of the
    waveform based on its length and diameter.
    - "Higuchi_FD": Higuchi's fractal dimension, estimates signal
    self-similarity across multiple time scales.
    - "Petrosian_FD": Petrosian's fractal dimension, relates to the number of
    sign changes in the derivative, providing a fast measure of complexity.

    Parameters
    ----------
        data: np.ndarray
            3D array of shape (n_epochs, n_channels, n_samples).
    
    Returns
    -------
        fractal_metrics: dict[str, list[np.ndarray]]
            Dictionary containing the extracted features by epochs.
    """
    fractal_metrics: dict[str, list[np.ndarray]] = {
        "Katz_FD": [], "Higuchi_FD": [], "Petrosian_FD": []
    }
    for ep in tqdm(
        data, desc=f"{CYAN}Calculating Fractal features{RESET}", colour="cyan"
    ):
        fractal_metrics["Katz_FD"].append(mne_feat_uni.compute_katz_fd(ep))
        fractal_metrics["Higuchi_FD"].append(
            mne_feat_uni.compute_higuchi_fd(ep)
        )
        fractal_metrics["Petrosian_FD"].append(ent.petrosian_fd(ep))
    
    return fractal_metrics


# ---------- Bispectrum features ----------

def bispectrum(x: np.ndarray) -> np.ndarray:
    """
    Computes the bispectrum of a 1D signal. The bispectrum is a third-order
    spectrum that characterizes nonlinear phase coupling between different
    frequency components. 
    It is defined as:

        B(f1, f2) = X(f1) * X(f2) * X*(f1+f2)

    where X(f) is the Fourier transform of the signal.

    Parameters
    ----------
        x: np.ndarray
            1D input signal.
    
    Returns
    -------
        B: np.ndarray
            Complex bispectrum matrix, where each element corresponds to
            the bispectral value at frequency pair (f1, f2).
    """
    N: int = len(x)
    X: np.ndarray = np.fft.fft(x)
    B: np.ndarray = np.zeros((N, N), dtype=complex)

    X_f1: np.ndarray = X[:, None]
    X_f2: np.ndarray = X[None, :]
    product: np.ndarray = X_f1 * X_f2

    indices: np.ndarray = (np.add.outer(np.arange(N), np.arange(N))) % N
    X_conj_sum: np.ndarray = np.conj(X[indices])
    B: np.numpy = product * X_conj_sum
    
    return B


def bispectrum_features(
        x: np.ndarray,
        eps: float = 1e-7
    ) -> tuple[float, float, float, float]:
    """
    Extracts scalar features from the bispectrum of a 1D signal.
    
    Parameters
    ----------
        x : np.ndarray
            1D input signal.
        eps : float, optional, default = 1e-7
            Small constant to avoid log(0).

    Returns
    -------
        Bis_mag : float
            Sum of bispectrum magnitudes (overall bispectral energy).
        sum_log_amps : float
            Sum of log amplitudes of the bispectrum.
        sum_log_diag : float
            Sum of log amplitudes along the bispectrum diagonal.
            (diagonal corresponds to interactions f1=f2).
        _1madB : float
            First-order spectral moment of the diagonal amplitudes, i.e.
            a weighted frequency mean along the diagonal.
    """
    B: np.ndarray = bispectrum(x)
    mag_B: np.ndarray = np.abs(B)

    # Bispectrum magnitude
    Bis_mag: float = np.sum(mag_B)

    # Sum of log amplitudes of Bispectrum
    sum_log_amps: float = np.sum(np.log(mag_B + eps))

    # Sum of log amps of diagonal of Bispectrum
    D: np.ndarray = np.diag(mag_B)
    sum_log_diag: float = np.sum(np.log(D + eps))

    # 1st order spectral moment of amplitudes of diagonal elements of the
    # Bispectrum
    freqs: np.ndarray = np.arange(len(D))
    _1madB: np.float = np.sum(freqs * D) / np.sum(D)

    return Bis_mag, sum_log_amps, sum_log_diag, _1madB


def calc_bispectrum(data: np.ndarray) -> dict[str, np.ndarray]:
    """
    Computes bispectrum-based features:
    - "Bispectrum_magnitude": Sum of magnitudes of the bispectrum.
    - "Bispectrum_Sum of log amplitudes": Sum of log amplitudes of all
    bispectrum elements.
    - "Bispectrum_Sum of log amps of diagonal": Sum of log amplitudes along
    the diagonal.
    - "Bispectrum_1st spectral moment log amps": First-order spectral moment 
    of diagonal amplitudes (frequency-weighted mean).

    Parameters
    ----------
        data: np.ndarray
            3D array of shape (n_epochs, n_channels, n_samples).
    
    Returns
    -------
        bispectrum_metrics: dict[str, np.ndarray]
            Dictionary containing the extracted features by epochs.
    """
    bispectrum_metrics: dict[str, np.ndarray] = {
        "Bispectrum_magnitude": [],
        "Bispectrum_Sum of log amplitudes": [],
        "Bispectrum_Sum of log amps of diagonal": [],
        "Bispectrum_1st spectral moment log amps": []
    }
    for ep in tqdm(
        data,
        desc=f"{CYAN}Calculating bispectrum features{RESET}",
        colour="cyan"
    ):
        ep_results: dict[str, list[float]] = {metric: [] for metric in bispectrum_metrics}
        for ch in ep:
            x: np.ndarray = ch[:]
            Bis_mag: float; H1: float; H2: float; H3: float
            Bis_mag, H1, H2, H3 = bispectrum_features(x)
            ep_results["Bispectrum_magnitude"].append(Bis_mag)
            ep_results["Bispectrum_Sum of log amplitudes"].append(H1)
            ep_results["Bispectrum_Sum of log amps of diagonal"].append(H2)
            ep_results["Bispectrum_1st spectral moment log amps"].append(H3)
    
        for metric in bispectrum_metrics:
            bispectrum_metrics[metric].append(ep_results[metric])
    
    return bispectrum_metrics


# ---------- Wavelet Features ----------

def calc_wavelets(data: np.ndarray, fs: float = 500) -> dict[str, np.ndarray]:
    """
    Extracts continuous wavelet transform (CWT) features using a predefined
    wavelet and scales, and extracts summary statistics (mean and standard
    deviation) of the absolute wavelet coefficients for each scale.

    Parameters
    ----------
        data: np.ndarray
            3D array of shape (n_epochs, n_channels, n_samples).
        fs: int, optional, default = 500.0
            Sampling frequency in Hz.
    
    Returns
    -------
        wavelet_metrics: dict[str, np.ndarray]
            Dictionary containing the extracted features by epochs.
    """
    wavelet_metrics: dict[str, list[np.ndarray]] = {
        "Wavelet_coefs_mean": [], "Wavelet_coefs_std": []
    }
    for ep in tqdm(
        data, desc=f"{CYAN}Calculating wavelet features{RESET}", colour="cyan"
    ):
        ep_results: dict[list[np.ndarray]] = {
            metric: [] for metric in wavelet_metrics
        }
        for ch in ep:
            x: np.ndarray = ch[:]

            WAVELET_SCALES: list[float] = get_wavelet_scales(fs)

            coefs: np.ndarray
            coefs, _ = pywt.cwt(
                x, WAVELET_SCALES, WAVELET, sampling_period=1/fs
            )
            ep_results["Wavelet_coefs_mean"].append(
                np.mean(np.abs(coefs), axis=1)
            )
            ep_results["Wavelet_coefs_std"].append(
                np.std(np.abs(coefs), axis=1)
            )

        for metric in wavelet_metrics:
            wavelet_metrics[metric].append(ep_results[metric])
    
    return wavelet_metrics


# ---------- Functional connectivity features ----------

def get_region_indices(
        region_list: list[str], channel_names: list[str]) -> list[int]:
    """
    Returns the indices of channels that belong to a specified brain region.

    Parameters
    ----------
    region_list: list[str]
        List of channel names representing the brain region.
    channel_names: list[str]
        List of all channel names in the dataset.

    Returns
    -------
    indices: list[int]
        Indices of channels in `channel_names` that belong to `region_list`.
    """
    return [i for i, ch in enumerate(channel_names) if ch in region_list]


def mean_connectivity_between(
        region1: list[str],
        region2: list[str],
        mat: np.ndarray,
        channel_names: list[str]
    ) -> float:
    """
    Computes the mean connectivity between two sets of channels.

    Parameters
    ----------
    region1 : list[str]
        List of channel names for the first region.
    region2 : list[str]
        List of channel names for the second region.
    mat : np.ndarray
        2D connectivity matrix (n_channels x n_channels).
    channel_names : list[str]
        List of all channel names in the dataset.

    Returns
    -------
    mean_conn : float
        Mean of connectivity values between the two regions, excluding
        self-connections.
    """
    idx1: list[int] = get_region_indices(region1, channel_names)
    idx2: list[int] = get_region_indices(region2, channel_names)
    values: list[float] = []

    for i in idx1:
        for j in idx2:
            if i != j:
                values.append(mat[i, j])
    return np.mean(values)


def calc_functional_connectivity(
        data: np.ndarray, ch_names: list[str]) -> dict[str, list[float]]:
    """
    Extracts functional connectivity features from amplitude envelope
    correlation and extracts region-based summary statistics and ratios.
    - "FC_mean": Mean connectivity across all channel pairs.
        - "FC_std": Standard deviation of connectivity values.
        - "FC_entropy": Entropy of connectivity values.
        - "FC_Left/Right Ratio"
        - "FC_Frontal/Posterior Ratio"
        - "FC_Frontal/Occipital Ratio"
        - "FC_Left-Right/Right Ratio"
        - "FC_Frontal-Posterior/Frontal Ratio"
        - "FC_Frontal-Posterior/Posterior Ratio"
        - "FC_Frontal-Occipital/Frontal Ratio"
        - "FC_Frontal-Occipital/Occipital Ratio"
        - "FC_Frontal-Temporal/Frontal Ratio"
        - "FC_Frontal-Temporal/Temporal Ratio"
    """
    fc_metrics: dict[str, list[float]] = {
        "FC_mean": [],
            "FC_std": [],
            "FC_enntropy": [],
            "FC_Left/Right Ratio": [],
            "FC_Frontal/Posterior Ratio": [],
            "FC_Frontal/Occipital Ratio": [],
            "FC_Frontal/Temporal Ratio": [],
            "FC_Left-Right/Left Ratio": [],
            "FC_Left-Right/Right Ratio": [],
            "FC_Frontal-Posterior/Frontal Ratio": [],
            "FC_Frontal-Posterior/Posterior Ratio": [],
            "FC_Frontal-Occipital/Frontal Ratio": [],
            "FC_Frontal-Occipital/Occipital Ratio": [],
            "FC_Frontal-Temporal/Frontal Ratio": [],
            "FC_Frontal-Temporal/Temporal Ratio": []}
    aec: EpochTemporalConnectivity = envelope_correlation(
        data, orthogonalize=False
    )
    fc: np.ndarray = np.abs(aec.get_data()[:,:,:,0])
    for ep in tqdm(
        fc,
        desc=f"{CYAN}Calculating Functional Connectivity features{RESET}",
        colour="cyan"
    ):
        x: np.ndarray = ep[:]

        left: float = mean_connectivity_between(
            LEFT_CHANNELS, LEFT_CHANNELS, x, ch_names
        )
        right: float = mean_connectivity_between(
            RIGHT_CHANNELS, RIGHT_CHANNELS, x, ch_names
        )
        frontal: float = mean_connectivity_between(
            FRONTAL_CHANNELS, FRONTAL_CHANNELS, x, ch_names
        )
        posterior: float = mean_connectivity_between(
            POSTERIOR_CHANNELS, POSTERIOR_CHANNELS, x, ch_names
        )
        occipital: float = mean_connectivity_between(
            OCCIPITAL_CHANNELS, OCCIPITAL_CHANNELS, x, ch_names
        )
        temporal: float = mean_connectivity_between(
            TEMPORAL_CHANNELS, TEMPORAL_CHANNELS, x, ch_names
        )
        left_right: float = mean_connectivity_between(
            LEFT_CHANNELS, RIGHT_CHANNELS, x, ch_names
        )
        frontal_posterior: float = mean_connectivity_between(
            FRONTAL_CHANNELS, POSTERIOR_CHANNELS, x, ch_names
        )
        frontal_occipital: float = mean_connectivity_between(
            FRONTAL_CHANNELS, OCCIPITAL_CHANNELS, x, ch_names
        )
        frontal_temporal: float = mean_connectivity_between(
            FRONTAL_CHANNELS, TEMPORAL_CHANNELS, x, ch_names
        )
        
        fc_metrics["FC_mean"].append(np.mean(x))
        fc_metrics["FC_std"].append(np.std(x))
        fc_metrics["FC_enntropy"].append(entropy(x, axis=(0,1)))
        fc_metrics["FC_Left/Right Ratio"].append(left/right)
        fc_metrics["FC_Frontal/Posterior Ratio"].append(frontal/posterior)
        fc_metrics["FC_Frontal/Occipital Ratio"].append(frontal/occipital)
        fc_metrics["FC_Frontal/Temporal Ratio"].append(frontal/temporal)
        fc_metrics["FC_Left-Right/Left Ratio"].append(left_right/left)
        fc_metrics["FC_Left-Right/Right Ratio"].append(left_right/right)
        fc_metrics["FC_Frontal-Posterior/Frontal Ratio"].append(
            frontal_posterior/frontal
        )
        fc_metrics["FC_Frontal-Posterior/Posterior Ratio"].append(
            frontal_posterior/posterior
        )
        fc_metrics["FC_Frontal-Occipital/Frontal Ratio"].append(
            frontal_occipital/frontal
        )
        fc_metrics["FC_Frontal-Occipital/Occipital Ratio"].append(
            frontal_occipital/occipital
        )
        fc_metrics["FC_Frontal-Temporal/Frontal Ratio"].append(
            frontal_temporal/frontal
        )
        fc_metrics["FC_Frontal-Temporal/Temporal Ratio"].append(
            frontal_temporal/temporal
        )

    return fc_metrics


# ---------- Constants ----------
# Pairs (function, list of parameters needed)
FEATURES_LIST = [
    (calc_statistical, []),
    (calc_autocorrelation, ["fs"]),
    (calc_band_power, ["fs"]),
    (calc_entropy, ["fs"]),
    (calc_hjorth, []),
    (calc_fd, []),
    (calc_bispectrum, []),
    (calc_wavelets, []),
    (calc_functional_connectivity, ["ch_names"])
]

# Type of feature, following the sorting in FEATURES_LIST
FEATURE_TYPES = [None, None, None, None, None, None, None, "Wavelet", "FC"]

# Dict "type of feature": index in FEATURES_LIST
FEATURE_MAP_NAMES = {
    "statistical": 0,
    "autocorrelation": 1,
    "band_power": 2,
    "entropy": 3,
    "hjorth": 4,
    "fractal": 5,
    "bispectrum": 6,
    "wavelets": 7,
    "functional_connectivity": 8
}
