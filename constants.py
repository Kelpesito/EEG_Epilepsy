from colorama import Fore, Style
import pywt
import numpy as np

"""
Guía de colores:
----------------
Blanco: Por defecto
Verde: Información (prints)
Amarillo: Información interpolada (f-strings)
Cian: Progresos
"""
GREEN = Fore.LIGHTGREEN_EX
YELLOW = Fore.YELLOW
CYAN = Fore.LIGHTCYAN_EX
RESET = Style.RESET_ALL

GOOD_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
    "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz", "EKG"
    ]
RENAME_DICT_BIPOLAR = {
    'EEG Fp1-Ref': "Fp1", 'EEG Fp2-Ref': "Fp2", 'EEG F3-Ref': "F3",
    'EEG F4-Ref': "F4", 'EEG C3-Ref': "C3", 'EEG C4-Ref': "C4",
    'EEG P3-Ref': "P3", 'EEG P4-Ref': "P4", 'EEG O1-Ref': "O1",
    'EEG O2-Ref': "O2", 'EEG F7-Ref': "F7", 'EEG F8-Ref': "F8",
    'EEG T7-Ref': "T3", 'EEG T8-Ref': "T4", 'EEG P7-Ref': "T5",
    'EEG P8-Ref': "T6", 'EEG Fz-Ref': "Fz", 'EEG Cz-Ref': "Cz",
    'EEG Pz-Ref': "Pz", 'ECG LA-RA': "EKG"
    }
RENAME_DICT_EXTENDED = {
    "FP1": "Fp1", "FP2": "Fp2", "FZ": "Fz", "CZ": "Cz", "PZ": "Pz",
    "T7": "T3", "T8": "T4", "T9": "T5", "T10": "T6", "PO1": "O1", "PO2": "O2"
    }

BIPOLAR_PAIRS = [("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),  # Left Temporal Chain
                 ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T4", "O2"),  # Right Temporal Chain
                 ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),  # Left Parasagital Chain
                 ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),  # Right Parasagital Chain
                 ("Fz", "Cz"), ("Cz", "Pz"),  # Central chain
                 ]
ANODES = [a for a, _ in BIPOLAR_PAIRS]
CATHODES = [c for c, _ in BIPOLAR_PAIRS]
BANANA_NAMES = [f"{a}-{c}" for a, c in BIPOLAR_PAIRS]


BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 70)}


WAVELET = "morl"
fs = 500
fc = pywt.central_frequency(WAVELET)
WAVELET_FREQS = np.array([1.48, 2.08, 2.92, 4.09, 5.74, 8.04, 11.28, 15.81, 22.17, 31.09, 43.59, 61.115])
WAVELET_SCALES = fc*fs/WAVELET_FREQS


LEFT_CHANNELS  = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5']
RIGHT_CHANNELS = ['Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6']
CENTRAL_CHANNELS = ['Fz', 'Cz', 'Pz']
FRONTAL_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz']
POSTERIOR_CHANNELS = ['P3', 'P4', 'O1', 'O2', 'Pz', "T5", "T6"]
TEMPORAL_CHANNELS = ['T3', 'T4', 'T5', 'T6']
OCCIPITAL_CHANNELS = ["O1", "O2"]

DESIRED_ORDER = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
