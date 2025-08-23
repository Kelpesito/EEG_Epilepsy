"""
EEG Processing Pipeline
Script for EEG processing:
- Pre-filtering
- Artifact correction
- Feature extraction
- Clustering
- Manual labeling
- Comparison between 2 EEGs
"""

# ---------- Imports ----------

import argparse
from functools import wraps
import json
import os
import re
import time
from tkinter import filedialog as fd
from typing import Any, Callable, Optional
import warnings

from autoreject import AutoReject, RejectLog
from colorama import init
import h5py
import hdbscan
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import mne
from mne_connectivity import (
    Connectivity, envelope_correlation, EpochTemporalConnectivity
)
from mne.preprocessing import compute_current_source_density, ICA
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, silhouette_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import umap

from constants import *
from features import *


# ---------- Warnings ----------

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*force_all_finite.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Graph is not fully connected.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Exited postprocessing.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Exited at iteration.*",
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*k >= N for N \* N square matrix.*"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*divide by zero.*"
)


# ---------- Color configuration ----------

init(autoreset=True)  # Automatically reset colors after each print


# ---------- Argument parser ----------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the EEG processing pipeline.

    Returns
    --------
        argparse.Namespace
            Parsed arguments with optins for visualization, montage type, etc.
    """
    parser = argparse.ArgumentParser(description="EEG Processing Pipeline")
    parser.add_argument(
        "--visualize", "-v",
        action='store_true',
        help='Show plots during preprocessing',
    )
    parser.add_argument(
        "--montage", "-m",
        type=str,
        default="average",
        choices=["bipolar", "average", "laplacian"],
        help='Type of montage to apply (default: Average)',
    )
    parser.add_argument(
        "--function", "-f",
        type=str,
        default='run',
        choices=[
            'run', 'inspect', 'correct', "features", "clustering", "labels",
            "versus",
            ],
        help=(
            "Execution mode: "
            "run (full pipeline), "
            "inspect (just load & filter & plot), " 
            "correct (Run the artifact correction algorithm), "
            "features (Calculate the features vector), "
            "clustering (Label epochs by unsupervised clustering), "
            "labels (Create ground-truth labels), "
            "versus (Comparative analysis). "
            "(default 'run')"
        )
    )
    parser.add_argument(
        "--sel_ecg",
        action="store_true",
        help="Option to select ECG for artifact correction",
    )
    parser.add_argument(
        "--plot_f",
        action="store_true",
        help="Plot final EEG after re-reference",
    )
    parser.add_argument(
        "--save", "-s",
        type=str,
        default=None,
        help="Name or route of files to save",
    )
    parser.add_argument(
        "--epoch_duration", "-d",
        type=float,
        default=10.,
        help="Epoch duration (default 10)"
    )
    parser.add_argument(
        "--dim_reduction", "-r",
        type=str,
        default="multichannel",
        choices=["multichannel", "average"],
        help=(
            "Type of dimensionality reduction after features calculation: " 
            "multichannel (each channel is passed as feature), " 
            "average (reduce channel information by mean and std). " 
            "(default 'multichannel')"
        )
    )
    parser.add_argument(
        "--var_selection",
        type=str,
        default=None,
        nargs="+",
        help=(
            "Name of the type features to be calculated: 'statistical', "
            "'autocorrelation', 'band_power', 'entropy', 'hjorth', 'fractal', "
            "'bispectrum', 'wavelets', 'functional_connectivity'. "
            "By default, all features are calculated."
        )
    )
    
    return parser.parse_args()


# ---------- Misc functions ----------

def measure_time(
        alias: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to measure the execution time of a function.

    Parameters
    -----------
        alias: str | None, optional
            Optional name to display instead of function's name.

    Returns
    --------
        decorator: function
            The wrapped function with execution time printing.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            ini: float = time.time()  # Current time at start
            x: Any = func(*args, **kwargs)  # Function execution
            fin: float = time.time()  # Current time at finishing
            name: str = alias if alias else func.__name__
            # Conversion of duration to mins, secs
            mins: float; segs: float
            mins, segs = divmod(fin-ini, 60)  
            print(
                f"{GREEN}Execution time of {YELLOW}{name}{GREEN}: "
                f"{YELLOW}{int(mins)} min {segs:.2f} s\n{RESET}"
            )
            return x
        return wrapper
    return decorator


# ---------- EEG processor ----------

class EEGProcessor:
    """
    Class for processing EEG signals.

    This class allows  to load, filter, display and process EEG signals in a
    structured manner.

    Attributes
    -----------
        visualize: bool, optional, default = False
            Whether to visualize all the plots or not.
        montage: str, optional, default = "average"
            Montage to re-reference signal ("average", "bipolar" or
            "laplacian").
        select_ecg: bool, optional, default = False
            Whether to select ECG when artifact correction.
        plot_final: bool, optional, default = False
            Whether to visualize the important plots or not.
        save: str | None, optional
            Name of the destination path if saving is desired.
            Defaults to None.
        epoch_duration: float, optional, default = 10.0
            Duration of epoch in seconds.
        dim_reduction: str, optional, default = "multichannel"
            Type of dimension reduction applied ("multichannel" or "average")
        var_selection: list[str], optional, \
            default = `list(FEATURE_MAP_NAMES.keys())`
            List of the names of the type features wanted to calculate.
        epoch_overlap: float
            Epoch overlap.
            epoch_overlap = epoch_duration / 2
        ch_names: list[str] | None, default = None
            EEG channel names.
        fs: float | None, default = None
            Sample frequency of EEG.
    
    Methods
    -------
        inspect(): 
            Plot raw and filtered EEG. Useful to select ECG for artifact
            correction.
        correct_reference():
            Load raw EEG and pre-process it (pre-filter, artifact removal,
            re-reference and normalize).
        get_features():
            Feature extraction
        clustering():
            Clustering of epochs in a EEG.
        run():
            Main execution progeam: load, correct_reference, geat_features and
            clustering.
        versus():
            Comparative analysis.
        labels():
            Manual labeling of epochs.
    """
    def __init__(
            self,
            visualize: bool = False,
            montage: str = "average",
            select_ecg: bool = False,
            plot_final: bool = False,
            save: str | None = None,
            epoch_duration: float = 10.,
            dim_reduction: str = "multichannel",
            var_selection: list[str] | None = None
        ) -> None:
        """
        Class constructor. Sets all class' attributes

        Parameters
        -----------
            visualize: bool, optional, default = False
                Whether to visualize all the plots or not.
            montage: str, optional, default = "average"
                Montage to re-reference signal ("average", "bipolar" or
                "laplacian").
            select_ecg: bool, optional, default = False
                Whether to select ECG when artifact correction.
            plot_final: bool, optional, default = False
                Whether to visualize the important plots or not.
            save: str | None, optional
                Name of the destination path if saving is desired.
                Defaults to None.
            epoch_duration: float, optional, default = 10.0
                Duration of epoch in seconds.
            dim_reduction: str, optional, default = "multichannel"
                Type of dimension reduction applied ("multichannel" or
                "average")
            var_selection: list[str] | None, optional, default = None
                List of the names of the type features wanted to calculate.
        """
        
        self.visualize: bool = visualize
        self.montage: str = montage
        self.select_ecg: bool = select_ecg
        self.plot_final: bool = plot_final
        self.save: str | None = save
        self.epoch_duration: float = epoch_duration
        self.dim_reduction: str = dim_reduction
        self.var_selection: list[str] = (
            var_selection if var_selection is not None
            else list(FEATURE_MAP_NAMES.keys())
        )

        self.epoch_overlap: float = self.epoch_duration/2  
        self.ch_names: list[str] | None = None
        self.fs: float | None = None
        

    # ---------- Helpers ----------

    def load_eeg(self) -> mne.io.Raw:
        """
        Loads an EEG, drops useless channels (references, empty...), and shows
        the plot (if self.visualize is True).

        If channels are not named the same as in `GOOD_CHANNELS`, the names
        have to be renamed first.

        Returns
        --------
            raw: mne.io.Raw
                Raw EEG.
        """
        # Ask for a EDF file to load
        filename: str = fd.askopenfilename(
            title="Select a file", filetypes=[("EDF files", "*.edf")]
        )  
        print(f"{GREEN}Loaded file: {YELLOW}{filename}\n{RESET}")

        # Read file
        raw: mne.io.Raw = mne.io.read_raw_edf(
            filename, preload=True, encoding="latin1", verbose=0
        )  

        # Pick channels
        if "EKG" in raw.ch_names:  # If all channels are nice named
            raw.pick_channels(GOOD_CHANNELS)  # Pick channels
        elif "ECG LA-RA" in raw.ch_names:  # If channels are named as "X-Ref"
            rename_safe: dict = {
                old: new
                for old, new in RENAME_DICT_BIPOLAR.items()
                if old in raw.ch_names and new not in raw.ch_names
            }
            raw.rename_channels(rename_safe)
            raw.pick_channels(GOOD_CHANNELS)  # Pick channels
        else:  # Extended channels
            # Combine ECG data
            ecg_data: np.ndarray = raw.copy().pick_channels(
                ["ECG1", "ECG2"]).get_data()
            ecg: np.ndarray = ecg_data.mean(axis=0, keepdims=True)

            # Create new raw ecg
            info: mne.Info = mne.create_info(
                ch_names=["EKG"], sfreq=raw.info['sfreq'], ch_types=['ecg']
            )
            ecg_raw: mne.io.RawArray = mne.io.RawArray(ecg, info)

            # Replace ECG channels
            raw.drop_channels(["ECG1", "ECG2"])
            raw.add_channels([ecg_raw])

            # Rename channels
            rename_safe: dict = {
                old: new
                for old, new in RENAME_DICT_EXTENDED.items()
                if old in raw.ch_names and new not in raw.ch_names
            }
            raw.rename_channels(rename_safe)

            raw.pick_channels(GOOD_CHANNELS)  # Pick channels

        raw.reorder_channels(sorted(raw.ch_names))
        raw.set_channel_types({'EKG': 'ecg'})  # Change "EKG" channel to "ecg"
        
        # Set the standard 10-20 configuration
        raw.set_montage("standard_1020")  
        
        print(f"{GREEN}EEG information:{RESET}")
        print(raw.info)
        print()

        # Plot EEG
        if self.visualize:
            raw.plot(
                scalings="auto", duration=10, start=0, show=False)
            plt.show()

        return raw
    

    def pre_filter_raw(
            self,
            raw: mne.io.Raw,
            hpf: float = 0.5, 
            lpf: float = 70.0,
            notch: float = 50.0
            ) -> mne.io.Raw:
        """
        Apply filtering to raw EEG.

        If self.visualize is True, plot the new EEG.

        Parameters
        -----------
            raw: mne.io.Raw
                Raw EEG.
            hpf: float, optional, default = 0.5
                High Pass Frequency, in Hz.
            lpf: float, optional, default = 70.0
                Low Pass Frequency, in Hz.
            notch: float, optional, default = 50.0
                Cut-off frequency, Notch filter.
        
        Returns
        --------
            raw: mne.io.Raw
                Filtered EEG
        """

        print(
            f"{GREEN}Filtering EEG:\n" + \
            f"- HPF = {YELLOW}{hpf}{GREEN} Hz\n" + \
            f"- LPF = {YELLOW}{lpf}{GREEN} Hz\n" + \
            f"- Notch = {YELLOW}{notch}{GREEN} Hz\n{RESET}"
        )
        
        picks = mne.pick_types(raw.info, ecg=True, eeg=True)
        raw.filter(l_freq=hpf, h_freq=lpf, picks=picks)
        raw.notch_filter(freqs=notch, picks=picks)

        # Plot EEG
        if self.visualize:
            raw.plot(
                scalings="auto", duration=10, start=0, show=False
            )
            plt.show()

        return raw
    

    def artifacts_autoreject(
            self, epochs: mne.Epochs) -> tuple[mne.Epochs, RejectLog]:
        """
        Perform the AutoReject steps of artifact correction.

        Parameters
        ----------
            epochs: mne.Epochs
                Epochs object to correct artifacts.
        Returns
        -------
            reject_log: RejectLog
                Object containing the indexes of the bad epochs.

        """
        # Create AutoReject instance
        ar: AutoReject = AutoReject(
            n_interpolate=np.array(range(1, 4)),
            consensus=np.arange(0.1, 0.3, 0.05),
            random_state=999,
            n_jobs=-1
        )
        reject_log: RejectLog
        epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

        # Plot of rejection map
        if self.visualize:
            reject_log.plot('horizontal', aspect="auto")

        return epochs_clean, reject_log
    

    def artifacts_ica(
            self, epochs: mne.Epochs, reject_log: RejectLog) -> mne.Epochs:
        """
        Perform the ICA step of artifact correction.

        Parameters
        ----------
            epochs: mne.Epochs
                Epochs object to correct artifacts.
            reject_log: RejectLog
                Object containing the indexes of the bad epochs, obtained in
                the first step of AutoReject.
        
        Returns
        -------
            epochs: mne.Epochs
                Epochs object with ICA applied.
        """
        # Create instance of ICA
        ica: mne.preprocessing.ICA  = mne.preprocessing.ICA(random_state=25)
        ica.fit(epochs[~reject_log.bad_epochs])
        
        # Plot ICA components
        if self.visualize:
            ica.plot_sources(epochs[~reject_log.bad_epochs])
            ica.plot_components()

        # Find EMG components
        muscle_idx: list[int]; muscle_scores: np.ndarray
        muscle_idx, muscle_scores = ica.find_bads_muscle(
            epochs[~reject_log.bad_epochs]
        )
        print((
            f"{GREEN}ICA components corresponding to muscle: "
            f"{YELLOW}{', '.join(map(str, muscle_idx))}\n{RESET}"
        ))
        
        # Plot EMG scores
        if self.visualize:
            ica.plot_scores(muscle_scores, exclude=muscle_idx)
        
        # Find ECG components
        if self.select_ecg:
            ecg_idx: list[int]; ecg_scores: np.ndarray
            ecg_idx, ecg_scores = ica.find_bads_ecg(
                epochs[~reject_log.bad_epochs]
            )
            print((
                f"{GREEN}ICA components corresponding to ECG: "
                f"{YELLOW}{', '.join(map(str, ecg_idx))}\n{RESET}"
            ))
            
            # Plot ECG scores
            if self.visualize:
                ica.plot_scores(ecg_scores, exclude=ecg_idx)
        else:
            ecg_idx: list[int] = []
        
        ica.apply(epochs, exclude=list(set(muscle_idx) | set(ecg_idx)))

        return epochs


    @measure_time(alias="Artifact correction")
    def artifact_correction(self, raw: mne.io.Raw) -> mne.Epochs:
        """
        Perform the artifact correction of a filtered EEG in three steps:
            1. AutoReject before ICA
            2. ICA (Independent Component Analysis) to remove bad components 
            (e.g. muscle, or ECG if self.select_ecg is True)
            3. AutoReject to drop bad epochs
        
        If self.visualize is True, plot the rejection plot after AutoReject, 
        the topomaps of the ICA, the scores of the components and the plot of
        the corrected EEG in epochs.

        If self.save is not None, the object Epochs is saved into
        `self.save`-epo.fif.

        Parameters
        -----------
            raw: mne.io.Raw
                Filtered EEG
        
        Returns
        --------
            epochs_clean: mne.Epochs
                Fixed EEG divided in epochs
        """
        print(f"{GREEN}Removing Artifacts:\n{RESET}")

        # Divide Raw EEG in epochs (50% overlap)
        epochs: mne.Epochs = mne.make_fixed_length_epochs(
            raw.pick(["eeg"] + (["ecg"] if self.select_ecg else [])),
            duration=self.epoch_duration,
            overlap=self.epoch_overlap,
            preload=True
        )
        
        ## 1. AutoReject
        print(
            f"{GREEN}Phase 1: {YELLOW}AutoReject{GREEN} before ICA\n{RESET}"
        )
        reject_log: RejectLog = self.artifacts_autoreject(epochs)[1]

        ## 2. ICA
        print(f"{GREEN}Phase 2: {YELLOW}ICA\n{RESET}")
        epochs: mne.Epochs = self.artifacts_ica(epochs, reject_log)

        ## 3. AutoReject
        print(f"{GREEN}Phase 3: {YELLOW}Autoreject\n{RESET}")
        epochs_clean: mne.Epochs = self.artifacts_autoreject(epochs)[0]

        # Plot of fixed EEG
        if self.visualize:
            epochs_clean.plot(n_epochs=1, block=True)
        
        # Drop ECG channel
        if 'EKG' in epochs_clean.ch_names:
            epochs_clean.drop_channels(['EKG'])
        
        # Save Epochs object
        if self.save:
            print((
                f"{GREEN}Saving corrected EEG into "
                f"{YELLOW}{self.save}-epo.fif\n{RESET}"
            ))
            epochs_clean.save(f"{self.save}-epo.fif", overwrite=True)

        return epochs_clean
    

    def set_reference(self, epochs: mne.Epochs) -> mne.Epochs:
        """
        Re-references the EEG following a monatage (average, bipolar or
        laplacian).

        Parameters
        ----------
            epochs: mne.Epochs
                Fixed EEG divided in epochs.
        
        Returns
        -------
            epochs_ref: mne.Epochs
                The re-referenced EEG.
        """
        print((
            f"{GREEN}Setting reference montage: "
            f"{YELLOW}{self.montage}\n{RESET}"
        ))
        match self.montage:
            case "bipolar":
                epochs_ref: mne.Epochs = mne.set_bipolar_reference(
                    epochs, ANODES, CATHODES, ch_name=BANANA_NAMES
                )
                epochs_ref.pick_channels(BANANA_NAMES)
            case "average":
                epochs_ref: mne.Epochs
                epochs_ref, _ = mne.set_eeg_reference(
                    epochs, ref_channels="average"
                )
            case "laplacian":
                epochs_ref: mne.Epochs = compute_current_source_density(
                    epochs
                )
        
        # Plot after re-reference
        if (self.visualize or self.plot_final) and self.montage == "bipolar":
            epochs_ref.plot(n_epochs=1, block=True, picks=BANANA_NAMES)
        elif self.visualize or self.plot_final:
            epochs_ref.plot(n_epochs=1, block=True)
        
        return epochs_ref
    

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalizes an array:
        x_norm = (x - mean) / std

        Parameters
        -----------
            data: np.ndarray
                Array to normalize.
        
        Returns
        -------
            np.ndarray
                Normalized array
        """
        print(f"{GREEN}Standardizing data{RESET}")
        mean: np.ndarray = data.mean(axis=(1, 2), keepdims=True)
        std: np.ndarray = data.std(axis=(1, 2), keepdims=True)
        return (data - mean)/std
    

    def save_correct(
            self, data_norm: np.ndarray, epochs_ref: mne.Epochs) -> None:
        """
        Save the corrected and normalized signal in a .h5 file and the
        metadata into a .json.

        Parameters
        ----------
            data_norm: np.ndarray
                Corrected and normalized signal.
                Shape (n_epochs, n_channels, n_times)
            epochs_ref: mne.Epochs
                Epochs object to save its metadata
        """
        print((
            f"{GREEN}Saving pre-processed EEG into "
            f"{YELLOW}{self.save}.h5\n{RESET}"
        ))
        # Save signal into .h5
        with h5py.File(f"{self.save}.h5", "w") as f:
            f.create_dataset("data", data=data_norm)
        
        print((
            f"{GREEN}Saving EEG metadata into "
            f"{YELLOW}{self.save}_metadata.json\n{RESET}"
        ))
        # Save metadata into .json 
        with open(f"{self.save}_metadata.json", "w") as f:
            metadata = {
                "ch_names": epochs_ref.ch_names,
                "fs": epochs_ref.info["sfreq"]
            }
            json.dump(metadata, f)

    
    def pre_process(self, epochs: mne.Epochs) -> np.ndarray:
        """
        Perform the pre-processing of a fixed Epochs object:
        1. Set reference
        2. Save channel names and sample frequency
        3. Normalize data

        Parameters
        ----------
            epochs: mne.Epochs
                Fixed Epochs Object
        
        Returns
        -------
            data_norm: np.ndarray 
                Array of shape (n_epochs, n_channels, n_times) normalized by
                epoch.
        """
        epochs_ref: mne.Epochs = self.set_reference(epochs)  # Set reference
        self.ch_names: list[str] = epochs_ref.ch_names
        self.fs: float = epochs_ref.info["sfreq"]
        data: np.ndarray = epochs_ref.get_data()
        data_norm: np.ndarray = self.normalize(data)  # Data normalization

        return data_norm

    
    def open_signal_fif_h5(self) -> np.ndarray:
        """
        If there is no corrected, normalized signal in memory (because the
        program is not running at run mode), the signal has to be loaded.

        There are two options for loading the signal:
        - .fif: corrected EEG Epochs object, needs to be pre-processed
        (re-referenced and normalized).
        - .h5: Corrected and normalized signal, needs to import the metadata
        (.json) too.

        Returns
        -------
            data_norm: np.ndarray
                Corrected, re-referenced and normalized signal.
        """
        filename: str = fd.askopenfilename(
            title="Select a file",
            filetypes=[("FIF files", "*.fif"), ("H5 files", "*.h5")]
        )
        print(f"{GREEN}Loaded file: {YELLOW}{filename}\n{RESET}")
        if ".fif" in filename:
            epochs: mne.Epochs = mne.read_epochs(filename, preload=True)
            data_norm: np.ndarray = self.pre_process(epochs)
        else:
            with h5py.File(filename, "r") as f:
                data_norm: np.ndarray = f["data"][:]
            filename: str = fd.askopenfilename(
                title="Select a file (metadata)",
                filetypes=[("JSON files", "*.json")]
            )
            print(f"{GREEN}Loaded file: {YELLOW}{filename}\n{RESET}")
            with open(filename, "r") as f:
                metadata: dict = json.load(f)
                self.ch_names: list[str] = metadata["ch_names"]
                self.fs: float = metadata["fs"]
        
        return data_norm
    

    def open_csv(self) -> pd.DataFrame:
        """
        Import a .csv file into a DataFrame.

        Returns
        -------
            X: pd.DataFrame
                Loaded DataFrame from .csv file
        """
        filename: str = fd.askopenfilename(
            title="Select a file",
            filetypes=[("CSV files", "*.csv")]
        )
        print(f"{GREEN}Loaded file: {YELLOW}{filename}\n{RESET}")
        X: pd.DataFrame = pd.read_csv(filename, index_col=0)

        return X
    

    def clustering_grid_search(
            self,
            X_scaled: np.ndarray,
            random_state: int = 28
        ) -> tuple[
            np.ndarray | None, np.ndarray | None, hdbscan.HDBSCAN | None
        ]:
        """
        Perform a grid search calculation of UMAP + HDBSCAN parameters with
        silhouette score as objective metric. The parameters tested in the
        grid search are:
        - UMAP:
            - n_neighbours: [2, 3, 5, 10, 20, 50]
            - min_dist: [0, 0.1, 0.2, 0.5, 1, 2]
            - n_components: [2, 5, 10, 30, 70, 100]
        - HDBSCAN:
            - min_cluster_size: [5, 10, 20, 50, 75, 100]
            - cluster_selection_epsilon: [0, 0.02, 0.05, 0.1, 0.2, 0.5]

        Parameters
        ----------
            X_scaled: np.ndarray
                Scaled feature matrix
            random_state: np.int
                Seed for random processes
        
        Returns
        -------
            best_labels: np.ndarray | None
                Array with the best predicted labels
            best_embedding: np.ndarray | None
                Array with the best embedded input matrix
            best_clusterer: hdbscan.HDBSCAN | None
                Best HDBSCAN clusterer object
        """
        param_grid: dict[str, list[int | float]] = {
            "umap__n_neighbours": [2, 3, 5, 10, 20, 50],
            "umap__min_dist": [0, 0.1, 0.2, 0.3, 0.5, 0.7],
            "umap__n_components": [2, 5, 10, 30, 70, 100],
            "hdbscan__min_cluster_size": [5, 10, 20, 50, 75, 100],
            "hdbscan__cluster_selection_epsilon": [
                0, 0.02, 0.05, 0.1, 0.2, 0.5
            ]
        }
        param_grid: ParameterGrid = ParameterGrid(param_grid)

        best_score: float = -1.
        best_config: dict[str, int | float] | None = None
        best_labels: np.ndarray | None = None
        best_embedding: np.ndarray | None = None
        best_clusterer: hdbscan.HDBSCAN | None = None
        for params in tqdm(
            param_grid,
            desc=f"{CYAN}UMAP + HDBCAN grid search{RESET}",
            colour="cyan"
        ):
            try:
                # UMAP
                reducer: umap.UMAP = umap.UMAP(
                    n_neighbors=params["umap__n_neighbours"],
                    min_dist=params["umap__min_dist"],
                    n_components=params["umap__n_components"],
                )
                embedding: np.ndarray = reducer.fit_transform(X_scaled)

                # HDBSCAN
                clusterer: hdbscan.HDBSCAN = hdbscan.HDBSCAN(
                    min_cluster_size=params["hdbscan__min_cluster_size"],
                    cluster_selection_epsilon=params[
                        "hdbscan__cluster_selection_epsilon"
                    ],
                )

                # Labels
                labels: np.ndarray = clusterer.fit_predict(embedding)

                # Discard only one cluster
                if len(set(labels)) <= 1 or (labels == -1).all():
                    continue

                # Calculate score
                score: float = silhouette_score(embedding, labels)

                if score > best_score:
                    best_score: float = score
                    best_config: dict[str, int | float] = params
                    best_labels: np.ndarray = labels
                    best_embedding: np.ndarray = embedding
                    best_clusterer: hdbscan.HDBSCAN = clusterer

                    # Print new bes score
                    print("\n New best score")
                    colored_items: list[str] = []
                    for k, v in best_config.items():
                        colored_items.append(f"{k} = {v}")
                    colored_dict_str: str = ", ".join(colored_items)
                    print(f"{colored_dict_str}:")
                    print(f"Silhouette score = {best_score}")
            
            except Exception as e:
                continue
        
        return best_labels, best_embedding, best_clusterer
    

    def plot_classified_epochs(self, best_labels: np.ndarray) -> None:
        """
        Plot in different figures, the Epochs for each classification.
        First, the EEG data must be loaded and converted into a EpochsArray
        object (if a .fif file is loaded, it is converted into a np.ndarray
        first, and then converted into the EpochsArray).

        Parameters
        ----------
            best_labels: np.ndarray
                Array with the predicted labels after clustering
        """
        ## Get Epochs object from data
        # Load data
        data: np.ndarray = self.open_signal_fif_h5()
        
        info: mne.Info = mne.create_info(
            ch_names=self.ch_names, sfreq=self.fs, ch_types="eeg"
        )
        
        # Convert data into Epochs
        epochs: mne.EpochsArray = mne.EpochsArray(data, info)

        unique_labels: np.ndarray = np.unique(best_labels)
        figures: list[mne.viz._mpl_figure.MNEBrowseFigure] = []
        for label_value in unique_labels:
            mask: np.ndarray = best_labels == label_value
            epochs_subset: mne.EpochsArray = epochs[mask]

            fig: mne.viz._mpl_figure.MNEBrowseFigure = epochs_subset.plot(
                title=f'Label= {label_value}', show=False, scalings="auto"
            )
            figures.append(fig)

        plt.show(block=True)


    def versus_pca(
            self, features_scaled: np.ndarray, y_true: np.ndarray) -> None:
        """
        Compare EEG pre vs post features performing dimensional reduction and
        k-Means clustering.

        If self.save is not None, the resulting figure is saved in .png.

        Parameters
        ----------
            features_scaled: np.ndarray
                Input scaled feature matrix
            y_true: np.ndarray
                Labels corresponding to EEG pre or post
        """

        # Reduction of Dimensionality
        pca: PCA = PCA(n_components=0.9)
        X_pca: np.ndarray = pca.fit_transform(features_scaled)

        # K-Means clustering
        kmeans: KMeans = KMeans(n_clusters=2, random_state=198)
        y_kmeans: np.ndarray = kmeans.fit_predict(X_pca)

        # Plot of clustering:
        plt.figure(figsize=(8,6))
        markers: list[str] = ["o", "x"]
        colors: list[str] = ["red", "blue"]
        for cluster in np.unique(y_kmeans):
            for label in np.unique(y_true):
                idx: np.ndarray = (y_kmeans == cluster) & (y_true == label)
                plt.scatter(
                    X_pca[idx, 0],
                    X_pca[idx, 1],
                    c=colors[label],
                    marker=markers[cluster],
                    label=f"Cluster {cluster}, EEG{label+1}"
                )

        plt.title("EEG1 vs EEG2: K-means")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend()
        plt.tight_layout()
        sns.despine()

        # Save figure
        if self.save:
            plt.savefig(f"{self.save}_EEG1_vs_EEG2.png", dpi=300)
        plt.show()


    def versus_features_importance(
            self, features: pd.DataFrame, y_true: np.ndarray) -> pd.DataFrame:
        """
        Calculate the importance of the features by n = 50 of evaluating
        Random Forest Classifiers with 1000 estimators.

        1. Divide data into train and test
        2. Train n Random Forest Classifiers
        3. Save the importances of each model
        4. For each Random Forest, calculate the accuracy of the test set
        5. Reduce the information of all models to mean and std.

        If self.save is not None, the DataFrame of the model importances is
        saved into a .csv.

        Parameters
        ----------
            features: pd.DataFrame
                Input feature matrix
            y_true: np.ndarray
                Labels corresponding to EEG pre or post
        
        Returns
        -------
            importance_df: pd.DataFrame
                DataFrame containing the importances of the features
        """
        
        #  Divide data into train and test
        X_train: np.ndarray
        X_test: np.ndarray
        y_train: np.ndarray
        y_test: np.ndarray
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            y_true,
            test_size=0.2,
            random_state=198,
            stratify=y_true
        )

        # Simulations for obtain feature importances
        n_runs: int = 50
        n_stimators: int = 1000
        importances_all: list[np.ndarray] = []
        accuracies: list[float] = []
        for rs in tqdm(
            range(n_runs),
            desc=f"{CYAN}Training random forest for feature importances",
            colour="cyan"
        ):
            # Create and train a Random Forest Classifier
            rf: RandomForestClassifier = RandomForestClassifier(
                n_estimators=n_stimators, random_state=rs
            )
            rf.fit(X_train, y_train)

            # Save importances
            importances_all.append(rf.feature_importances_)

            # Calculate predictions and scores
            y_pred: np.ndarray = rf.predict(X_test)
            accuracies.append(balanced_accuracy_score(y_test, y_pred))

        print((
            f"{GREEN}Mean accuracy along {n_runs} runs: "
            f"{YELLOW}{np.mean(accuracies):.3f} ± "
            f"{np.std(accuracies):.3f}{RESET}"
        ))
        
        # Calculate mean and std of importances into a DataFrame
        importances_all: np.ndarray = np.array(importances_all)
        importances_mean: np.ndarray = importances_all.mean(axis=0)
        importances_std: np.ndarray = importances_all.std(axis=0)
        importances_df: pd.DataFrame = pd.DataFrame({
            "feature": features.columns,
            "importance_mean": importances_mean,
            "importance_std": importances_std}
        ).sort_values("importance_mean", ascending=False)

        if self.save:
            importances_df.to_csv(f"{self.save}_importances.csv")

        return importances_df


    def versus_plot_top_n_features(
            self,
            features: pd.DataFrame,
            label: np.ndarray,
            importances_df: pd.DataFrame,
            n: int = 20
            ) -> None:
        """
        Plot a histogram of the top n features and the violin plot
        distribution of the top 10 features by channel and by label.

        If self.save is not None, the generated figures are saved into .png.

        Parameters
        ----------
            features: pd.DataFrame
                Input feature matrix
            label: np.ndarray
                Labels corresponding to EEG pre or post
            importances_df: pd.DataFrame
                DataFrame containing the importances of the features
            n: int, optional, default = 20
                Number of features to show in the horizontal histogram
        """
        # Create figure: horizontal histogram
        plt.figure(figsize=(8,5))
        plt.barh(
            importances_df.head(n)["feature"],
            importances_df.head(n)["importance_mean"],
            xerr=importances_df.head(n)["importance_std"],
            color="skyblue"
        )
        plt.gca().invert_yaxis()
        plt.xlabel("Mean importance")
        plt.title("Top features (Random Forest)")
        plt.tight_layout()
        sns.despine()
        
        # Save figure
        if self.save:
            plt.savefig(f"{self.save}_top_features.png", dpi=300)
        plt.show()

        ## Distribution by channel
        top_n: 10
        unique_features: list[str] = []
        # get top_n features by channel
        for f in importances_df["feature"]:
            feat_base: str = re.sub(r'_ch_\d+$', '', f)  # drop _ch_X
            if feat_base not in unique_features:
                unique_features.append(feat_base)
            if len(unique_features) >= top_n:
                break
        
        # Get features
        for i, feat_base in enumerate(unique_features):
            cols = [
                c for c in features.columns
                if c == feat_base or c.startswith(feat_base + "_ch_")
            ]
            if not cols:
                continue

            # Dataframe for feature, with all channels
            df_plot: pd.DataFrame = features[cols].copy()
            df_plot['label'] = label
            df_plot: pd.DataFrame = df_plot.melt(
                id_vars='label', var_name='channel', value_name='value'
            )

            # Plot: violin plot for channel
            plt.figure(figsize=(8, len(cols)*1.5))
            sns.violinplot(
                data=df_plot,
                y='channel',
                x='value',
                hue='label',
                split=True,
                palette=["red", "blue"],
                orient='h'
            )
            plt.xlabel(feat_base)
            plt.ylabel('Channel')
            plt.title(f'{feat_base} distribution by channel')
            plt.legend(title='Clase')
            sns.despine()

            # Save figure
            if self.save:
                plt.savefig(f"{self.save}_top_features_{i+1}.png", dpi=300)
            plt.show()

    
    def open_fif(self) -> mne.Epochs:
        """
        Open a .fif file into Epochs object.

        Returns
        --------
            epochs: mne.Epochs
                Epochs object containing EEG divided in epochs
        """
        filename: str = fd.askopenfilename(
            title="Select a file", filetypes=[("FIF files", "*.fif")]
        )
        epochs: mne.Epochs = mne.read_epochs(filename)
        print(f"{GREEN}Loaded file: {YELLOW}{filename}{RESET}")

        return epochs
    

    def calc_psd(self, epochs: mne.Epochs, n: int | str) -> dict[str, float]:
        """
        1. Calculate Power Spectral Density
        2. Calculate mean Relative Band Powers
        3. Save into a DataFrame RBP with mean total power

        If self.save is not None, saves the PSD figure. 

        Parameters
        ----------
            epochs: mne.Epochs
                Epochs object containing EEG divided in Epochs
            n: int | str
                Identifier for the generated figure
        
        Returns
        -------
            rbp: dict[str, float]
                Dictionary containing in the key the name of the band
                frequency, and in the values, the Relative Band Power.
        """

        # Calculate PSD
        psd: mne.time_frequency.EpochsSpectrum = epochs.compute_psd(
            fmin=0.5, fmax=70
        )
        psd.plot(average=False, amplitude=False, picks="data", exclude="bads")

        # Save PSD figure
        if self.save:
            plt.savefig(f"{self.save}_psd_{n}.png", dpi=300)
        plt.show()
        
        
        psd_data: np.ndarray; freqs: np.ndarray
        psd_data, freqs = psd.get_data(return_freqs=True)
        total_power = psd_data.sum(axis=2, keepdims=True)  # Total power
        
        rbp: dict[str, float] = {}
        # Calculate relative band powers
        for band, (fmin, fmax) in BANDS.items():
            idx_band: np.ndarray = (freqs >= fmin) & (freqs < fmax)
            band_power: np.ndarray = psd_data[:, :, idx_band].sum(axis=2)
            rbp[band] = (band_power / total_power.squeeze()).mean()

        rbp["total"] = total_power.squeeze().mean()
        
        return rbp
    

    def calc_functional_connectivity(
            self, epochs: mne.Epochs, fmin: float, fmax: float) -> np.ndarray:
        """
        Calculate the functional connectivity matrix over a specific band
        frequency by the calculation of the Amplitude Envelope Correlation.

        Parameters
        ----------
            epochs: mne.Epochs
                Epochs object containing EEG divided in Epochs
            fmin: float
                Lower frequency to filter the EEG
            fmax: float
                Higher frequency to filter the EEG
        
        Returns
        -------
            conn_matrix_ordered: np.ndarray
                Connectivity matrix ordered by channel.
        """
        # Filter epoch in the band (fmin, fmax)
        epochs_band: mne.Epochs = epochs.copy().filter(
            l_freq=fmin, h_freq=fmax
        )
        # Calculate Envelope Correlation
        conn_epochs: EpochTemporalConnectivity = envelope_correlation(
            data=epochs_band.get_data(),
            names=epochs_band.info["ch_names"],
            orthogonalize=False
        )
        # Combine epochs by median
        conn_med: Connectivity = conn_epochs.combine(
            lambda x: np.median(x, axis=0)
        )
        # Get data and re-order matrix
        conn_matrix: np.ndarray = conn_med.get_data(output="dense")[:,:,0]
        indices: list[int] = [
            epochs_band.info['ch_names'].index(ch) for ch in DESIRED_ORDER
        ]
        conn_matrix_ordered: np.ndarray = conn_matrix[
            np.ix_(indices, indices)
        ]

        return conn_matrix_ordered
    

    def plot_functional_connectivity(
            self, 
            conn_matrices: dict[str, np.ndarray],
            n: int | str,
            cmap: str | Colormap = "viridis"
            ) -> None:
        """
        Plot the functional connectivity matrices.

        Parameters:
        -----------
            conn_matrices: dict[str, np.ndarray]
                Dictionary, where the keys are the name of the power band and
                the value is the connectivity matrix.
            n: int | str
                Identifier for the generated figure
            cmap: str | mpl.colors.Colormap, optional, default = "viridis"
                Colormap to map scalar data to colors.
        """

        # Plot functional connectivity matrices
        fig: Figure; axes: np.ndarray
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        axes = axes.flatten()
        for i, (band, conn_matrix) in enumerate(conn_matrices.items()):
            ax: Axes = axes[i]
            im: AxesImage = ax.imshow(conn_matrix, cmap=cmap)
            fig.colorbar(
                im, ax=ax, fraction=0.046, pad=0.04, label='Connectivity'
            )

            ax.set_xticks(np.arange(len(DESIRED_ORDER)))
            ax.set_yticks(np.arange(len(DESIRED_ORDER)))
            ax.set_xticklabels(DESIRED_ORDER, rotation=90)
            ax.set_yticklabels(DESIRED_ORDER, rotation=0)

            ax.set_title(f"{band} ({BANDS[band][0]} - {BANDS[band][1]}Hz)")

        # Unshow empty axes
        if len(BANDS) < len(axes):
            for j in range(len(BANDS), len(axes)):
                axes[j].axis('off')

        plt.tight_layout()

        # Save figure to .png
        if self.save:
            plt.savefig(f"{self.save}_FC_{n}.png", dpi=300)
        plt.show()
    

    def versus_figs(self, epochs1, epochs2):
        """
        For EEG pre and post:

        1. Calculate PSD and plot spectra
        2. From PSD, calculate RBP
        3. Calculate and plot topomap
        4. Calculate functional connectivity matrices
        5. Plot functional connectivity matrices

        If self.save is not None, all the figures are saved into .png.

        Parameters
        ----------
            epochs1: mne.Epochs
                Epochs object containing EEG pre divided in Epochs
            epochs2: mne.Epochs
                Epochs object containing EEG post divided in Epochs
        """
        ## PSD
        rbp1: dict[str, float] = self.calc_psd(epochs1, 1)
        rbp2: dict[str, float] = self.calc_psd(epochs2, 2)

        # Print RBP results
        print((
            f"{GREEN}\nRelative Band Power ({YELLOW}EEG1{GREEN} / "
            f"{YELLOW}EEG2{GREEN}){RESET}"
        ))
        for band in rbp1:
            print((
                f"{GREEN}{band}:\t{YELLOW}{rbp1[band]:.3f}{GREEN} / "
                f"{YELLOW}{rbp2[band]:.3f}{RESET}"
            ))

        ## Topomap
        epochs1.compute_psd().plot_topomap(
            ch_type="eeg", normalize=False, contours=0, cmap="viridis"
        )
        # Save figure
        if self.save:
            plt.savefig(f"{self.save}_topomap_1.png", dpi=300)
        epochs2.compute_psd().plot_topomap(
            ch_type="eeg", normalize=False, contours=0, cmap="viridis"
        )
        # Save figure
        if self.save:
            plt.savefig(f"{self.save}_topomap_2.png", dpi=300)

        ## Functional connectivity
        conn_matrix_1: dict[str, np.ndarray]
        conn_matrix_2: dict[str, np.ndarray]
        conn_matrix_1, conn_matrix_2 = {}, {}
        for band, (fmin, fmax) in BANDS.items():
            conn_matrix_1[band] = self.calc_functional_connectivity(
                epochs1, fmin, fmax
            )
            conn_matrix_2[band] = self.calc_functional_connectivity(
                epochs2, fmin, fmax
            )
        
        self.plot_functional_connectivity(conn_matrix_1, 1)
        self.plot_functional_connectivity(conn_matrix_2, 2)

        # Difference
        delta_matrix: dict[str, np.ndarray] = {}
        for band in conn_matrix_1:
            delta_matrix[band] = conn_matrix_2[band] - conn_matrix_1[band]
        
        self.plot_functional_connectivity(
            delta_matrix, "delta", cmap="coolwarm"
        )
        
    
    def open_h5(self):
        """
        Open a .h5 file.

        Returns
        --------
            data: np.ndarray
                Array containing EEG signal with shape
                (n_epochs, n_channels, n_times).
        """
        filename: str = fd.askopenfilename(
            title="Select a file", filetypes=[("H5 files", "*.h5")]
        )
        print(f"{GREEN}Loaded file: {YELLOW}{filename}\n{RESET}")
        with h5py.File(filename, "r") as f:
            data: np.ndarray = f["data"][:]
        
        filename: str = fd.askopenfilename(
            title="Select a file (metadata)",
            filetypes=[("JSON files", "*.json")]
        )
        print(f"{GREEN}Loaded file: {YELLOW}{filename}\n{RESET}")
        with open(filename, "r") as f:
            metadata: dict[str, list[str] | float] = json.load(f)
            
        self.ch_names: list[str] = metadata["ch_names"]
        self.fs: float = metadata["fs"]
        
        return data


    # ---------- Main functions ----------

    def inspect(self) -> None:
        """
        1. Load an EEG and visualize it
        2. Prefilter the raw EEG and visualize it

        Useful to know whether select ECG or not in artifact correction
        """
        self.visualize = True
        raw: mne.io.Raw = self.load_eeg()  # Load raw EEG
        raw: mne.io.Raw = self.pre_filter_raw(raw)  # Prefilter EEG

    
    def correct_reference(self) -> None:
        """
        1. Load EEG and prefilter it
        2. Perform the artifact correction
        3. Re-reference data and normalize
        4. If self.save is not None, the normalized data is saved into a .h5
        file and the metadata into a .json file.
        """
        raw: mne.io.Raw = self.load_eeg()  # Load raw EEG
        raw: mne.io.Raw = self.pre_filter_raw(raw)  # Prefilter EEG
        # Artifact correction  
        epochs_clean: mne.Epochs = self.artifact_correction(raw)
        # Re-reference channels
        epochs_ref: mne.Epochs = self.set_reference(epochs_clean)  
        data: np.ndarray = epochs_ref.get_data()
        data_norm: np.ndarray = self.normalize(data)  # Normalize data

        # Save data
        if self.save:
            self.save_correct(data_norm, epochs_ref)

    
    @measure_time(alias="Feature Extraction")
    def get_features(
        self, data_norm: np.ndarray | None = None) -> pd.DataFrame:
        """
        Perform the feature extraction of the EEG by epoch.
        The different types of possible features are:

        - Statistical
        - Autocorrelation
        - Band Power
        - Entropy
        - Hjorth features
        - Fractal dimension
        - Bispectrum features
        - Wavelet features
        - Functional connectivity features

        Each feature is calculated by epoch and channel (except if
        self.dim_reduction is set to "average", which is feature is reduced to
        mean and standard deviation of the channels in the epoch).

        If self.save is not None, the features matrix is saved into a .csv.

        Parameters
        ----------
            data_norm: np.ndarray | None, optional
                Corrected, re-referenced and normalized signal to do the
                feature extraction.

        Returns
        -------
            features_df: pd.DataFrame
                Feature matrix of the EEG divided in epochs.
        """
        print(f"{GREEN}Calculating Features:\n{RESET}")
        # If there is no data_norm in memory, load it.
        if data_norm is None:
            data_norm: np.ndarray = self.open_signal_fif_h5()
        
        # Feature extraction
        features: dict[str, np.ndarray] = {}  # dict to save the features
        param_context: dict[str, float | list[str]] = {
            "fs": self.fs, "ch_names": self.ch_names
        }
        for feature in self.var_selection:
            if feature in FEATURE_MAP_NAMES:
                # Get parameters for features calculation (fs, ch_names) if
                #  needed
                calc_feature: Callable[..., dict[str, np.ndarray]]
                params: list[str]
                calc_feature, params = FEATURES_LIST[
                    FEATURE_MAP_NAMES[feature]
                ]
                feature_type: str | None = FEATURE_TYPES[
                    FEATURE_MAP_NAMES[feature]
                ]
                kwargs: dict[str, float | list[str]] = {
                    name: param_context[name] for name in params
                }
                
                # Calculate features
                feature_metric: dict[str, np.ndarray] = calc_feature(
                    data_norm, **kwargs
                )
                
                # Update dictionary
                features.update(get_features(
                    feature_metric,
                    self.dim_reduction,
                    feature_type=feature_type
                ))
        
        # Convert to DataFrame
        features_df: pd.DataFrame = pd.DataFrame(features)
        
        # Save dataframe into csv
        if self.save:
            print((
                f"{GREEN}Saving Feature Vectors into "
                f"{YELLOW}{self.save}_features.csv\n{RESET}"
            ))
            features_df.to_csv(f"{self.save}_features.csv")

        return features_df
    

    @measure_time(alias="Clustering")
    def clustering(
        self,
        X: np.ndarray | pd.DataFrame | None = None,
        random_state: int = 28) -> None:
        """
        Perform the clustering algorithm to classify epochs in a EEG.
        1. Load .csv with feature matrix (if not loaded yet)
        2. Standardize data
        3. Run grid search + clustering algorithm with UMAP & HDBSCAN
        
        If self.visualize is True, plot the dendogram.

        If sel.visualize or self.plot_final are True, plot the EEG
        visualization of the resulting clusters.

        If self.save is not None, save the predicted labels into a csv file.

        Parameters
        ----------
            X: np.array | pd.DataFrame | None, optional
                A matrix or DataFrame defining the feature matrix
            random_state: int, optional, default = 28
                Seed for random processes
        """
        print(f"{GREEN}Clustering Features:\n{RESET}")
        # If there is no X in memory, load it.
        if X is None:
            X = self.open_csv()

        # Standardize
        scaler: StandardScaler = StandardScaler()
        X_scaled: np.ndarray = scaler.fit_transform(X)

        # Grid search and clustering
        best_labels: np.ndarray
        best_embedding: np.ndarray
        best_clusterer: hdbscan.HDBSCAN
        (best_labels,
        best_embedding,
        best_clusterer) = self.clustering_grid_search(X_scaled, random_state)

        # Plot dendogram
        if self.visualize:
            plt.figure(figsize=(12, 8))
            best_clusterer.condensed_tree_.plot(
                select_clusters=True,
                selection_palette=sns.color_palette('tab10')
            )
            plt.title("Condensed dendorgam of HDBSCAN (best model)")
            plt.show()

        if self.visualize or self.plot_final:
            # Cluster visualization
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                x=best_embedding[:, 0],
                y=best_embedding[:, 1],
                hue=best_labels,
                palette='tab10'
            )
            plt.title("Cluster visualization (best configuration)")
            plt.legend(title="Cluster")
            plt.show()

            # Classified epochs
            self.plot_classified_epochs(best_labels)

        # Save labels into a .csv
        if self.save:
            print((
                f"{GREEN}Saving Labels into "
                f"{YELLOW}{self.save}_cluster_labels.csv\n{RESET}"
            ))
            best_labels_df = pd.DataFrame({"labels": best_labels})
            best_labels_df.to_csv(f"{self.save}_cluster_labels.csv")

        return best_labels
    
    
    def run(self) -> None:
        """
        Main pipeline for EEG analysis:
        1. Load EEG
        2. Pre-filter EEG
        3. Artifact correction
        4. Re-reference and normalize
        5. Feature extraction
        6. Clustering
        """
        raw = self.load_eeg()
        raw = self.pre_filter_raw(raw)
        epochs_clean = self.artifact_correction(raw)
        data_norm = self.pre_process(epochs_clean)
        features_df = self.get_features(data_norm)
        self.clustering(features_df)
    

    @measure_time(alias="Pre vs post")
    def versus(self) -> None:
        """
        Performs the comparative analysis between two EEGs:
        - Visual separation (pre vs post) with PCA and K-means
        - Top important features
        - Plot comparisons (spectra, topomap, functional matrix)
        """
        # Load features matrix (pre and post)
        X1: pd.DataFrame = self.open_csv()
        X2: pd.DataFrame = self.open_csv()

        # Create label column and join
        X1["label"] = "EEG1"
        X2["label"] = "EEG2"
        X = pd.concat([X1, X2], ignore_index=True)

        # Standardize data
        features: pd.DataFrame = X.drop(columns="label")
        scaler: StandardScaler = StandardScaler()
        features_scaled: np.ndarray = scaler.fit_transform(features)

        le: LabelEncoder = LabelEncoder()
        y_true: np.ndarray = le.fit_transform(X["label"])

        # Visual separation and K-means clustering
        self.versus_pca(features_scaled, y_true)
        # Features importance
        importances_df: pd.DataFrame = self.versus_features_importance(
            features, y_true
        )
        # Plot Top 20 features
        self.versus_plot_top_features(features, X["label"], importances_df)

        # Load EEG epochs
        epochs1: mne.Epochs = self.open_fif()
        epochs2: mne.Epochs = self.open_fif()

        # Plot comparisons (spectra, topomap, functional matrix)
        self.versus_figs(epochs1, epochs2)
        
    
    def labels(self) -> None:
        """
        Performs data binary labeling of epochs. This feature can be done
        thanks to the manual rejection of epochs by mne's plot method, which
        allows to get the list of epochs manually selected.
        
        If self.save is not None, the ground-truth labels are saved into a
        .npy file.
        """
        # Load data
        data: np.ndarray = self.open_h5()
        
        # Convert array to Epochs object
        info: mne.Info = mne.create_info(
            ch_names=self.ch_names, sfreq=self.fs, ch_types="eeg"
        )
        epochs: mne.EpochsArray = mne.EpochsArray(data, info)

        epochs_copy: mne.EpochsArray = epochs.copy()
        
        # Plot EEG to select epochs
        epochs_copy.plot(scalings="auto", block=True)

        # Get bad indexes
        bad_idx: list[int] = [
            i for i, log in enumerate(epochs_copy.drop_log) if log
        ]

        labels = np.zeros(len(epochs), dtype=int)
        labels[bad_idx] = 1

        # Save indexes
        if self.save:
            print((
                f"{GREEN}Saving pre-processed EEG into "
                f"{YELLOW}{self.save}_labels.npy\n{RESET}"
            ))
            np.save(f"{self.save}_labels.npy", labels)


# ---------- main() ----------

def main() -> None:
    """
    Main program execution.
    """
    os.system("cls")
    args: argparse.Namespace = parse_args()
    processor: EEGProcessor = EEGProcessor(
        visualize=args.visualize,
        montage=args.montage,
        select_ecg=args.sel_ecg,
        plot_final=args.plot_f,
        save=args.save,
        epoch_duration=args.epoch_duration,
        dim_reduction=args.dim_reduction,
        var_selection=args.var_selection
    )
    match args.function:
        case "run":
            processor.run()
        case "inspect":
            processor.inspect()
        case "correct":
            processor.correct_reference()
        case "features":
            processor.get_features()
        case "clustering":
            processor.clustering()
        case "labels":
            processor.labels()
        case "versus":
            processor.versus()


# ---------- Entry point ----------

if __name__ == "__main__":
    main()
