from tqdm import tqdm
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch
import antropy as ent
import pywt
import mne_features.univariate as mne_feat_uni
from mne_connectivity import envelope_correlation

from constants import *


## Get Features
def get_features(metrics, dim_reduction, feature_type=None):
    features_ = {}
    match dim_reduction:
        case "average":
            if feature_type is None:
                for metric, values in metrics.items():
                    values = np.array(values)
                    mean = values.mean(axis=1)
                    std = values.std(axis=1)
                    
                    features_[f"{metric}_mean"] = mean
                    features_[f"{metric}_std"] = std
                
                return features_
            
            if feature_type == "Wavelet":
                for metric, values in metrics.items():
                    values = np.array(values)
                    mean = values.mean(axis=1)
                    std = values.std(axis=1)
                    for i in range(12):
                        features_[f"{metric}_scale_{i+1}_mean"] = mean[:,i]
                        features_[f"{metric}_scale_{i+1}_std"] = std[:,i]
                
                return features_
            
        case "multichannel":
            if feature_type is None:
                for metric, values in metrics.items():
                    values = np.array(values)
                    for ch in range(values.shape[1]):
                        features_[f"{metric}_ch_{ch+1}"] = values[:,ch]
                
                return features_
            
            if feature_type == "Wavelet":
                for metric, values in metrics.items():
                    values = np.array(values)
                    for i in range(values.shape[2]):
                        for ch in range(values.shape[1]):
                            features_[f"{metric}_scale_{i+1}_ch_{ch+1}"] = values[:,ch,i]
                
                return features_

        
    if feature_type == "FC":
        for metric, values in metrics.items():
            values = np.array(values)
            features_[metric] = values
        
        return features_

## Statistical Features
def calc_statistical(data):
    statistical_metrics = {"Mean": [],
                           "Median": [],
                           "Std": [],
                           "Skewness": [],
                           "Kurtosis": [],
                           "Mean Absolute Values of 1st Differences": [],
                           "Mean Absolute Values of 2nd Differences": [],
                           "Normalized Mean Absolute Values of 1st Differences": [],
                           "Normalized Mean Absolute Values of 2nd Differences": [],
                           "RMS": []}
    for ep in tqdm(data, desc=f"{CYAN}Calculating statistical features{RESET}", colour="cyan"):
        statistical_metrics["Mean"].append(np.mean(ep, axis=1))
        statistical_metrics["Median"].append(np.median(ep, axis=1))
        statistical_metrics["Std"].append(std:=np.std(ep, axis=1))
        statistical_metrics["Skewness"].append(skew(ep, axis=1))
        statistical_metrics["Kurtosis"].append(kurtosis(ep, axis=1))
        statistical_metrics["Mean Absolute Values of 1st Differences"].append(mav1d:=np.mean(np.abs(np.diff(ep, axis=1)), axis=1))
        statistical_metrics["Mean Absolute Values of 2nd Differences"].append(mav2d:=np.mean(np.abs(np.diff(ep, n=2, axis=1)), axis=1))
        statistical_metrics["Normalized Mean Absolute Values of 1st Differences"].append(mav1d/std)
        statistical_metrics["Normalized Mean Absolute Values of 2nd Differences"].append(mav2d/std)
        statistical_metrics["RMS"].append(np.sqrt(np.mean(ep**2, axis=-1)))
    
    return statistical_metrics


## Autocorrelation metrics
def calc_autocorr(x):
    autocorr = np.correlate(x, x, mode="full")
    autocorr = autocorr / autocorr.max()
    autocorr = autocorr[autocorr.size//2:]

    return autocorr

def AUautocorr(autocorr, fs=500):
    area = np.trapezoid(np.abs(autocorr), dx=1/fs)
    return area

def AACW(autocorr, win=1/np.e, fs=500):
    acw = None
    for i, cor in enumerate(autocorr):
        if cor <= win:
            acw = i
            break
    if acw is None:
        acw =  len(autocorr)
    
    area = np.trapezoid(np.abs(autocorr[:acw]), dx=1/fs)
    return acw/fs, area

def calc_autocorrelation(data, fs=500):
    autocorrelation_metrics = {"Autocorr_area": [],
                               "ACW": [],
                               "ACW_area": [],
                               "ACW0": [],
                               "ACW0_area": []}
    for ep in tqdm(data, desc=f"{CYAN}Calculating correlation features{RESET}", colour="cyan"):
        ep_results = {metric: [] for metric in autocorrelation_metrics}
        for ch in ep:
            x = ch[:]
            autocorr = calc_autocorr(x)
            
            ep_results["Autocorr_area"].append(AUautocorr(autocorr, fs))
            ep_results["ACW"].append(AACW(autocorr, fs=fs)[0])
            ep_results["ACW_area"].append(AACW(autocorr, fs=fs)[1])
            ep_results["ACW0"].append(AACW(autocorr, win=0, fs=fs)[0])
            ep_results["ACW0_area"].append(AACW(autocorr, win=0, fs=fs)[1])
        
        for metric in autocorrelation_metrics:
            autocorrelation_metrics[metric].append(ep_results[metric])

    return autocorrelation_metrics


## Frequency Features
def bandpower(data, band, fs=500):
    fmin, fmax = band
    freqs, psd = welch(data, fs=fs)
    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)

    area_band = np.trapezoid(psd[idx_band], freqs[idx_band])
    area_total = np.trapezoid(psd, freqs)
    return area_band / area_total

def calc_band_power(data, fs=500):
    band_power_metrics = {"Delta RBP": [],
                          "Theta RBP": [],
                          "Alpha RBP": [],
                          "Beta RBP": [],
                          "Gamma RBP": [],
                          "Alpha/Delta Ratio": [],
                          "Alpha/Theta Ratio": [],
                          "Delta/Theta Ratio": [],
                          "Delta/Beta Ratio": [],
                          "Theta/Beta Ratio": []}
    for ep in tqdm(data, desc=f"{CYAN}Calculating band power features{RESET}", colour="cyan"):
        ep_results = {metric: [] for metric in band_power_metrics}
        for ch in ep:
            x = ch[:]

            ep_results["Delta RBP"].append(delta_rbp:=bandpower(x, BANDS["delta"], fs))
            ep_results["Theta RBP"].append(theta_rbp:=bandpower(x, BANDS["theta"], fs))
            ep_results["Alpha RBP"].append(alpha_rbp:=bandpower(x, BANDS["alpha"], fs))
            ep_results["Beta RBP"].append(beta_rbp:=bandpower(x, BANDS["beta"], fs))
            ep_results["Gamma RBP"].append(bandpower(x, BANDS["gamma"], fs))
            ep_results["Alpha/Delta Ratio"].append(alpha_rbp/delta_rbp)
            ep_results["Alpha/Theta Ratio"].append(alpha_rbp/theta_rbp)
            ep_results["Delta/Theta Ratio"].append(delta_rbp/theta_rbp)
            ep_results["Delta/Beta Ratio"].append(delta_rbp/beta_rbp)
            ep_results["Theta/Beta Ratio"].append(theta_rbp/beta_rbp)

        for metric in band_power_metrics:
            band_power_metrics[metric].append(ep_results[metric])

    return band_power_metrics


## Entropy features
def calc_entropy(data, fs=500):
    entropy_features = {"Approximate Entropy": [],
                        "Sample Entropy": [],
                        "Spectral Entropy": [],
                        "SVD Entropy": []}
    for ep in tqdm(data, desc=f"{CYAN}Calculating entropy features{RESET}", colour="cyan"):
        entropy_features["Approximate Entropy"].append(mne_feat_uni.compute_app_entropy(ep))
        entropy_features["Sample Entropy"].append(mne_feat_uni.compute_samp_entropy(ep))
        entropy_features["Spectral Entropy"].append(mne_feat_uni.compute_spect_entropy(fs, ep))
        entropy_features["SVD Entropy"].append(mne_feat_uni.compute_svd_entropy(ep))
    
    return entropy_features


## Hjorth features
def calc_hjorth(data):
    hjorth_metrics = {"Hjorth_Activity": [],
                      "Hjorth_Mobility": [],
                      "Hjorth_Complexity": []}
    for ep in tqdm(data, desc=f"{CYAN}Calculating Hjorth features{RESET}", colour="cyan"):
        hjorth_metrics["Hjorth_Activity"].append(np.var(ep, axis=1))
        hjorth_metrics["Hjorth_Mobility"].append(mne_feat_uni.compute_hjorth_mobility(ep))
        hjorth_metrics["Hjorth_Complexity"].append(mne_feat_uni.compute_hjorth_complexity(ep))
    
    return hjorth_metrics


## Fractal dimension features
def calc_fd(data):
    fractal_metrics = {"Katz_FD": [],
                       "Higuchi_FD": [],
                       "Petrosian_FD": []}
    for ep in tqdm(data, desc=f"{CYAN}Calculating Fractal features{RESET}", colour="cyan"):
        fractal_metrics["Katz_FD"].append(mne_feat_uni.compute_katz_fd(ep))
        fractal_metrics["Higuchi_FD"].append(mne_feat_uni.compute_higuchi_fd(ep))
        fractal_metrics["Petrosian_FD"].append(ent.petrosian_fd(ep))
    
    return fractal_metrics


## Bispectrum features
def bispectrum(x):
    N = len(x)
    X = np.fft.fft(x)
    B = np.zeros((N, N), dtype=complex)

    X_f1 = X[:, None]
    X_f2 = X[None, :]
    product = X_f1 * X_f2

    indices = (np.add.outer(np.arange(N), np.arange(N))) % N
    X_conj_sum = np.conj(X[indices])
    B = product * X_conj_sum
    
    return B

def bispectrum_features(x, eps=1e-7):
    B = bispectrum(x)
    mag_B = np.abs(B)

    # Bispectrum magnitude
    Bis_mag = np.sum(mag_B)

    # Sum of log amplitudes of Bispectrum
    sum_log_amps = np.sum(np.log(mag_B + eps))

    # Sum of log amps of diagonal of Bispectrum
    D = np.diag(mag_B)
    sum_log_diag = np.sum(np.log(D))

    # 1st order spectral moment of amplitudes of diagonal elements of the Bispectrum
    freqs = np.arange(len(D))
    _1madB = np.sum(freqs * D) / np.sum(D)

    return Bis_mag, sum_log_amps, sum_log_diag, _1madB

def calc_bispectrum(data):
    bispectrum_metrics = {"Bispectrum_magnitude": [],
                          "Bispectrum_Sum of log amplitudes": [],
                          "Bispectrum_Sum of log amps of diagonal": [],
                          "Bispectrum_1st spectral moment log amps": []}
    for ep in tqdm(data, desc=f"{CYAN}Calculating bispectrum features{RESET}", colour="cyan"):
        ep_results = {metric: [] for metric in bispectrum_metrics}
        for ch in ep:
            x = ch[:]
            Bis_mag, H1, H2, H3 = bispectrum_features(x)
            ep_results["Bispectrum_magnitude"].append(Bis_mag)
            ep_results["Bispectrum_Sum of log amplitudes"].append(H1)
            ep_results["Bispectrum_Sum of log amps of diagonal"].append(H2)
            ep_results["Bispectrum_1st spectral moment log amps"].append(H3)
    
        for metric in bispectrum_metrics:
            bispectrum_metrics[metric].append(ep_results[metric])
    
    return bispectrum_metrics


## Wavelet Features
def calc_wavelets(data, fs=500):
    wavelet_metrics = {"Wavelet_coefs_mean": [],
                       "Wavelet_coefs_std": []}
    for ep in tqdm(data, desc=f"{CYAN}Calculating wavelet features{RESET}", colour="cyan"):
        ep_results = {metric: [] for metric in wavelet_metrics}
        for ch in ep:
            x = ch[:]

            coefs, freqs = pywt.cwt(x, WAVELET_SCALES, WAVELET, sampling_period=1/fs)
            ep_results["Wavelet_coefs_mean"].append(np.mean(np.abs(coefs), axis=1))
            ep_results["Wavelet_coefs_std"].append(np.std(np.abs(coefs), axis=1))
        
        for metric in wavelet_metrics:
            wavelet_metrics[metric].append(ep_results[metric])
    
    return wavelet_metrics


## Functional connectivity features
def get_region_indices(region_list, channel_names):
    return [i for i, ch in enumerate(channel_names) if ch in region_list]

def mean_connectivity_between(region1, region2, mat, channel_names):
    idx1 = get_region_indices(region1, channel_names)
    idx2 = get_region_indices(region2, channel_names)
    values = []

    for i in idx1:
        for j in idx2:
            if i != j:
                values.append(mat[i, j])
    return np.mean(values)

def calc_functional_connectivity(data, ch_names):
    fc_metrics = {"FC_mean": [],
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
    aec = envelope_correlation(data, orthogonalize=False)
    fc = np.abs(aec.get_data()[:,:,:,0])
    for ep in tqdm(fc, desc=f"{CYAN}Calculating Functional Connectivity features{RESET}", colour="cyan"):
        x = ep[:]

        left = mean_connectivity_between(LEFT_CHANNELS, LEFT_CHANNELS, x, ch_names)
        right = mean_connectivity_between(RIGHT_CHANNELS, RIGHT_CHANNELS, x, ch_names)
        frontal = mean_connectivity_between(FRONTAL_CHANNELS, FRONTAL_CHANNELS, x, ch_names)
        posterior = mean_connectivity_between(POSTERIOR_CHANNELS, POSTERIOR_CHANNELS, x, ch_names)
        occipital = mean_connectivity_between(OCCIPITAL_CHANNELS, OCCIPITAL_CHANNELS, x, ch_names)
        temporal = mean_connectivity_between(TEMPORAL_CHANNELS, TEMPORAL_CHANNELS, x, ch_names)
        left_right = mean_connectivity_between(LEFT_CHANNELS, RIGHT_CHANNELS, x, ch_names)
        frontal_posterior = mean_connectivity_between(FRONTAL_CHANNELS, POSTERIOR_CHANNELS, x, ch_names)
        frontal_occipital = mean_connectivity_between(FRONTAL_CHANNELS, OCCIPITAL_CHANNELS, x, ch_names)
        frontal_temporal = mean_connectivity_between(FRONTAL_CHANNELS, TEMPORAL_CHANNELS, x, ch_names)
        
        fc_metrics["FC_mean"].append(np.mean(x))
        fc_metrics["FC_std"].append(np.std(x))
        fc_metrics["FC_enntropy"].append(entropy(x, axis=(0,1)))
        fc_metrics["FC_Left/Right Ratio"].append(left/right)
        fc_metrics["FC_Frontal/Posterior Ratio"].append(frontal/posterior)
        fc_metrics["FC_Frontal/Occipital Ratio"].append(frontal/occipital)
        fc_metrics["FC_Frontal/Temporal Ratio"].append(frontal/temporal)
        fc_metrics["FC_Left-Right/Left Ratio"].append(left_right/left)
        fc_metrics["FC_Left-Right/Right Ratio"].append(left_right/right)
        fc_metrics["FC_Frontal-Posterior/Frontal Ratio"].append(frontal_posterior/frontal)
        fc_metrics["FC_Frontal-Posterior/Posterior Ratio"].append(frontal_posterior/posterior)
        fc_metrics["FC_Frontal-Occipital/Frontal Ratio"].append(frontal_occipital/frontal)
        fc_metrics["FC_Frontal-Occipital/Occipital Ratio"].append(frontal_occipital/occipital)
        fc_metrics["FC_Frontal-Temporal/Frontal Ratio"].append(frontal_temporal/frontal)
        fc_metrics["FC_Frontal-Temporal/Temporal Ratio"].append(frontal_temporal/temporal)

    return fc_metrics


FEATURES_LIST = [(calc_statistical, []),
                 (calc_autocorrelation, ["fs"]),
                 (calc_band_power, ["fs"]),
                 (calc_entropy, ["fs"]),
                 (calc_hjorth, []),
                 (calc_fd, []),
                 (calc_bispectrum, []),
                 (calc_wavelets, []),
                 (calc_functional_connectivity, ["ch_names"])]

FEATURE_TYPES = [None, None, None, None, None, None, None, "Wavelet", "FC"]

FEATURE_MAP_NAMES = {"statistical": 0,
                     "autocorrelation": 1,
                     "band_power": 2,
                     "entropy": 3,
                     "hjorth": 4,
                     "fractal": 5,
                     "bispectrum": 6,
                     "wavelets": 7,
                     "functional_connectivity": 8}
