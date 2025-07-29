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
