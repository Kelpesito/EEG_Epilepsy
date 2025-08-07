from tkinter import filedialog as fd
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from autoreject import AutoReject
import argparse
import os
from colorama import init
import time
from functools import wraps
import json
import h5py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap
import hdbscan

from constants import *
from features import *

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Graph is not fully connected.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Exited postprocessing.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Exited at iteration.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Failed at iteration.*")


init(autoreset=True)


def parse_args():
    parser = argparse.ArgumentParser(description="EEG Processing Pipeline")
    parser.add_argument("--visualize", "-v",
                        action='store_true',
                        help='Show plots during preprocessing')
    parser.add_argument("--montage", "-m",
                        type=str, default="average", choices=["bipolar", "average", "laplacian"],
                        help='Type of montage to apply (default: Average)')
    parser.add_argument("--function", "-f",  type=str, default='run', choices=['run', 'inspect', 'correct', "features", "clustering"],
                        help="Execution mode: run (full pipeline), inspect (just load & filter & plot), " \
                        "correct (Run the artifact correction algorithm), features (Calculate the features vector), " \
                        "clustering (Label epochs by unsupervised clustering) (default 'run')")
    parser.add_argument("--sel_ecg",
                        action="store_true",
                        help="Option to select ECG for artifact correction")
    parser.add_argument("--plot_f",
                        action="store_true",
                        help="Plot final EEG after re-reference")
    parser.add_argument("--save", "-s",
                        type=str, default=None,
                        help="Name or route of files to save")
    parser.add_argument("--epoch_duration", "-d",
                        type=float, default=10.,
                        help="Epoch duration (default 10)")
    parser.add_argument("--dim_reduction", "-r",
                        type=str, default="multichannel", choices=["multichannel", "average"],
                        help="Type of dimensionality reduction after features calculation: multichannel " \
                        "(don't apply, each channel is passed as feature), average (reduce channel information by mean and std) " \
                        "(default 'multichannel')")
    parser.add_argument("--var_selection",
                        type=str, default=None, nargs="+",
                        help="Name of the type features to be calculated: 'statistical', 'autocorrelation', 'band_power', 'entropy', " \
                        "'hjorth', 'fractal', 'bispectrum', 'wavelets', 'functional_connectivity'. By default, all features are calculated.")
    
    # statistical autocorrelation band_power hjorth fractal wavelets functional_connectivity
    
    return parser.parse_args()


def measure_time(alias=None):
    def measure_time(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ini = time.time()
            x = func(*args, **kwargs)
            fin = time.time()
            name = alias if alias else func.__name__
            mins, segs = (fin-ini)//60, (fin-ini)%60
            print(f"{GREEN}Execution time of {YELLOW}{name}{GREEN}: {YELLOW}{mins} min {segs:.2f} s\n{RESET}")
            return x
        return wrapper
    return measure_time


class EEGProcessor:
    def __init__(
            self, visualize=False, montage="average", select_ecg=False, plot_final=False, save=None, epoch_duration=10, dim_reduction="average",
            var_selection=None):
        self.visualize = visualize
        self.montage = montage
        self.select_ecg = select_ecg
        self.plot_final = plot_final
        self.save = save
        self.epoch_duration = epoch_duration
        self.dim_reduction = dim_reduction
        self.var_selection = var_selection if var_selection is not None else list(FEATURE_MAP_NAMES.keys())

        self.epoch_overlap = self.epoch_duration/2
        self.ch_names = None
        self.fs = None


    def load_eeg(self):
        # Ask for a file
        filename = fd.askopenfilename(
            title="Select a file", filetypes=[("EDF files", "*.edf")])  
        
        print(f"{GREEN}Loaded file: {YELLOW}{filename}\n{RESET}")

        raw = mne.io.read_raw_edf(filename, preload=True, verbose=0)  # Read file
        raw.pick_channels(
            ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7",
            "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz", "EKG"])  # Pick channels
        raw.reorder_channels(sorted(raw.ch_names))
        raw.set_channel_types({'EKG': 'ecg'})  # Change "EKG" channel to "ecg"
        raw.set_montage("standard_1020")
        
        print(f"{GREEN}EEG information:{RESET}")
        print(raw.info)
        print()

        if self.visualize:
            raw.plot(scalings="auto", duration=10, start=0, remove_dc=True, show=False)
            plt.show()

        return raw


    def pre_filter_raw(self, raw, hpf=0.5, lpf=70, notch=50):
        print(
            f"{GREEN}Filtering EEG:\n" + \
            f"- HPF = {YELLOW}{hpf}{GREEN} Hz\n" + \
            f"- LPF = {YELLOW}{lpf}{GREEN} Hz\n" + \
            f"- Notch = {YELLOW}{notch}{GREEN} Hz\n{RESET}")
        raw.filter(l_freq=hpf, h_freq=lpf)
        raw.notch_filter(freqs=notch)

        if self.visualize:
            raw.plot(scalings="auto", duration=10, start=0, remove_dc=True, show=False)
            plt.show()

        return raw

    @measure_time(alias="Artifact correction")
    def artifact_correction(self, raw):
        print(f"{GREEN}Removing Artifacts:\n{RESET}")
        epochs = mne.make_fixed_length_epochs(
            raw.pick(["eeg"] + (["ecg"] if self.select_ecg else [])),
            duration=self.epoch_duration, overlap=self.epoch_overlap, preload=True)
        
        # 1. Autoreject
        print(f"{GREEN}Phase 1: {YELLOW}Autoreject{GREEN} before ICA\n{RESET}")
        ar = AutoReject(n_interpolate=np.array(range(1,4)), consensus=np.arange(0, 0.2, 0.05), random_state=999, n_jobs=-1)
        epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
        if self.visualize:
            reject_log.plot('horizontal', aspect="auto")

        # 2. ICA
        print(f"{GREEN}Phase 2: {YELLOW}ICA\n{RESET}")
        ica = mne.preprocessing.ICA(random_state=25)
        ica.fit(epochs[~reject_log.bad_epochs])
        if self.visualize:
            ica.plot_sources(epochs[~reject_log.bad_epochs])
            ica.plot_components()

        muscle_idx, muscle_scores = ica.find_bads_muscle(epochs[~reject_log.bad_epochs])
        print(f"{GREEN}ICA components corresponding to muscle: {YELLOW}{', '.join(map(str, muscle_idx))}\n{RESET}")
        if self.visualize:
            ica.plot_scores(muscle_scores, exclude=muscle_idx)
        

        if self.select_ecg:
            ecg_idx, ecg_scores = ica.find_bads_ecg(epochs[~reject_log.bad_epochs])
            print(f"{GREEN}ICA components corresponding to ECG: {YELLOW}{', '.join(map(str, ecg_idx))}\n{RESET}")
            if self.visualize:
                ica.plot_scores(ecg_scores, exclude=ecg_idx)
        else:
            ecg_idx = []
        
        ica.apply(epochs, exclude=list(set(muscle_idx) | set(ecg_idx)))

        # 3. Autoreject
        print(f"{GREEN}Phase 3: {YELLOW}Autoreject\n{RESET}")
        ar = AutoReject(n_interpolate=np.array(range(1,4)), consensus=np.arange(0, 0.2, 0.05), random_state=999, n_jobs=-1)
        epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
        if self.visualize:
            reject_log.plot('horizontal', aspect="auto")
            epochs_clean.plot(n_epochs=1, block=True)
        
        if 'EKG' in epochs_clean.ch_names:
            epochs_clean.drop_channels(['EKG'])
        if self.save:
            print(f"{GREEN}Saving corrected EEG into {YELLOW}{self.save}-epo.fif\n{RESET}")
            epochs_clean.save(f"{self.save}-epo.fif", overwrite=True)

        return epochs_clean
    

    def set_reference(self, epochs):
        print(f"{GREEN}Setting reference montage: {YELLOW}{self.montage}\n{RESET}")
        match self.montage:
            case "bipolar":
                epochs_ref = mne.set_bipolar_reference(epochs, ANODES, CATHODES, ch_name=BANANA_NAMES, copy=True, on_bad='warn', verbose=None)
                epochs_ref.pick_channels(BANANA_NAMES)
            case "average":
                epochs_ref, _ = mne.set_eeg_reference(epochs, ref_channels="average")
            case "laplacian":
                epochs_ref = mne.preprocessing.compute_current_source_density(epochs)
            case _:
                raise Exception("Invalid montage. Select between 'bipolar', 'average' and 'laplacian'")
        
        if (self.visualize or self.plot_final) and self.montage == "bipolar":
            epochs_ref.plot(n_epochs=1, block=True, picks=BANANA_NAMES)
        elif self.visualize or self.plot_final:
            epochs_ref.plot(n_epochs=1, block=True)
        
        return epochs_ref
    

    def normalize(self, data):
        print(f"{GREEN}Standardizing data{RESET}")
        print(data.shape)
        mean = data.mean(axis=(1, 2), keepdims=True)
        std = data.std(axis=(1, 2), keepdims=True)
        print(mean.shape, std.shape)
        return (data - mean)/std
    

    def pre_process(self, epochs):
        epochs_ref = self.set_reference(epochs)
        self.ch_names = epochs_ref.ch_names
        self.fs = epochs_ref.info["sfreq"]
        data = epochs_ref.get_data()
        data_norm = self.normalize(data)

        return data_norm


    @measure_time(alias="Feature Extraction")
    def get_features(self, data_norm=None):
        print(f"{GREEN}Calculating Features:\n{RESET}")
        if data_norm is None:
            filename = fd.askopenfilename(
                title="Select a file",
                filetypes=[("FIF files", "*.fif"), ("H5 files", "*.h5")])
            print(f"{GREEN}Loaded file: {YELLOW}{filename}\n{RESET}")
            if ".fif" in filename:
                epochs = mne.read_epochs(filename, preload=True)
                data_norm = self.pre_process(epochs)
            else:
                with h5py.File(filename, "r") as f:
                    data_norm = f["data"][:]
                filename = fd.askopenfilename(
                    title="Select a file (metadata)",
                    filetypes=[("JSON files", "*.json")])
                print(f"{GREEN}Loaded file: {YELLOW}{filename}\n{RESET}")
                with open(filename, "r") as f:
                    metadata = json.load(f)
                    self.ch_names = metadata["ch_names"]
                    self.fs = metadata["fs"]

        features = {}
        param_context = {"fs": self.fs, "ch_names": self.ch_names}
        for feature in self.var_selection:
            if feature in FEATURE_MAP_NAMES:
                calc_feature, params = FEATURES_LIST[FEATURE_MAP_NAMES[feature]]
                feature_type = FEATURE_TYPES[FEATURE_MAP_NAMES[feature]]

                kwargs = {name: param_context[name] for name in params}
                feature_metric = calc_feature(data_norm, **kwargs)
                features.update(get_features(feature_metric, self.dim_reduction, feature_type=feature_type))
            
        features_df = pd.DataFrame(features)
        if self.save:
            print(f"{GREEN}Saving Feature Vectors into {YELLOW}{self.save}_features.csv\n{RESET}")
            features_df.to_csv(f"{self.save}_features.csv")

        return features_df
    

    @measure_time(alias="Clustering")
    def clustering(self, X=None, random_state=28):
        if X is None:
            filename = fd.askopenfilename(
                title="Select a file",
                filetypes=[("CSV files", "*.csv")])
            print(f"{GREEN}Loaded file: {YELLOW}{filename}\n{RESET}")
            X = pd.read_csv(filename, index_col=0)

        umap_n_neighbours_list = [2, 3, 5, 7, 10, 20]
        umap_min_dist_list = [0, 0.1, 0.2, 0.3, 0.5, 0.7]
        umap_n_components_list = [2, 5, 10, 30, 70, 100]
        hdbscan_min_cluster_size_list = [5, 10, 20, 50, 75, 100]
        hdbscan_cluster_selection_epsilon_list = [0, 0.02, 0.05, 0.1, 0.2, 0.5]

        # Estandarizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        best_score = -1
        best_config = None
        best_labels = None
        best_umap_embedding = None
        best_clusterer = None

        for n_neighbours in tqdm(umap_n_neighbours_list, desc=f"{CYAN}Obtaining best parameters for clustering{RESET}", colour="cyan"):
            for min_dist in umap_min_dist_list:
                for n_components in umap_n_components_list:
                    # UMAP. 
                    reducer = umap.UMAP(n_jobs=1, n_neighbors=n_neighbours, min_dist=min_dist, n_components=n_components, random_state=random_state)
                    X_umap = reducer.fit_transform(X_scaled)
                    for min_cluster_size in hdbscan_min_cluster_size_list:
                        for cluster_selection_epsilon in hdbscan_cluster_selection_epsilon_list:
                            # HDBSCAN
                            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_selection_epsilon)
                            labels = clusterer.fit_predict(X_umap)

                            # Avoid all noise
                            if len(set(labels)) <= 1 or (labels == -1).all():
                                continue

                            # Silhouette score
                            score = silhouette_score(X_umap, labels)

                            if score > best_score:
                                best_score = score
                                best_config = dict(n_neighbours=n_neighbours,
                                                min_dist=min_dist,
                                                n_components=n_components,
                                                min_cluster_size=min_cluster_size,
                                                cluster_selection_epsilon=cluster_selection_epsilon)
                                best_labels = labels
                                best_umap_embedding = X_umap
                                best_clusterer = clusterer
                                
                                print(f"\nNew best score")
                                colored_items = []
                                for k, v in best_config.items():
                                    colored_items.append(f"{k} = {v}")
                                colored_dict_str = ", ".join(colored_items)
                                print(f"{colored_dict_str}:")
                                print(f"Silhouette score = {best_score}")
        
        # Dendogram
        if self.visualize:
            plt.figure(figsize=(12, 8))
            best_clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette('tab10'))
            plt.title("Condensed dendorgam of HDBSCAN (best model)")
            plt.show()
        
        if self.visualize or self.plot_final:
            # Cluster visualization
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=best_umap_embedding[:, 0], y=best_umap_embedding[:, 1], hue=best_labels, palette='tab10')
            plt.title("Cluster visualization (best configuration)")
            plt.legend(title="Cluster")
            plt.show()

            # Classified epochs
            # Get Epochs object from data
            if self.ch_names is None or self.fs is None:
                filename = fd.askopenfilename(
                    title="Select a file",
                    filetypes=[("FIF files", "*.fif"), ("H5 files", "*.h5")])
                print(f"{GREEN}Loaded file: {YELLOW}{filename}\n{RESET}")
                if ".fif" in filename:
                    epochs = mne.read_epochs(filename, preload=True)
                    data = self.pre_process(epochs)
                else:
                    with h5py.File(filename, "r") as f:
                        data = f["data"][:]
                    filename = fd.askopenfilename(title="Select a file (metadata)",
                                                filetypes=[("JSON files", "*.json")])
                    print(f"{GREEN}Loaded file: {YELLOW}{filename}\n{RESET}")
                    with open(filename, "r") as f:
                        metadata = json.load(f)
                        self.ch_names = metadata["ch_names"]
                        self.fs = metadata["fs"]
            
            info = mne.create_info(ch_names=self.ch_names, sfreq=self.fs, ch_types="eeg")
            epochs = mne.EpochsArray(data, info)

            unique_labels = np.unique(best_labels)
            figures = []
            for label_value in unique_labels:
                mask = best_labels == label_value
                epochs_subset = epochs[mask]

                fig = epochs_subset.plot(title=f'Label= {label_value}', show=False, scalings="auto")
                figures.append(fig)

            plt.show(block=True)

        if self.save:
            print(f"{GREEN}Saving Labels into {YELLOW}{self.save}_cluster_labels.csv\n{RESET}")
            best_labels_df = pd.DataFrame({"labels": best_labels})
            best_labels_df.to_csv(f"{self.save}_cluster_labels.csv")

        return best_labels


    def run(self):
        raw = self.load_eeg()
        raw = self.pre_filter_raw(raw)
        epochs_clean = self.artifact_correction(raw)
        data_norm = self.pre_process(epochs_clean)

        features_df = self.get_features(data_norm=data_norm)

        best_labels = self.clustering(features_df)

    
    def inspect(self):
        self.visualize = True
        raw = self.load_eeg()
        raw = self.pre_filter_raw(raw)


    def correct_reference(self):
        raw = self.load_eeg()
        raw = self.pre_filter_raw(raw)
        epochs_clean = self.artifact_correction(raw)
        epochs_ref = self.set_reference(epochs_clean)
        data = epochs_ref.get_data()
        data_norm = self.normalize(data)

        if self.save:
            print(f"{GREEN}Saving pre-processed EEG into {YELLOW}{self.save}.h5\n{RESET}")
            with h5py.File(f"{self.save}.h5", "w") as f:
                f.create_dataset("data", data=data_norm, compression="gzip")
            print(f"{GREEN}Saving EEG metadata into {YELLOW}{self.save}_metadata.json\n{RESET}")
            with open (f"{self.save}_metadata.json", "w") as f:
                metadata = {"ch_names": epochs_ref.ch_names,
                            "fs": epochs_ref.info["sfreq"]}
                json.dump(metadata, f)


if __name__ == "__main__":
    os.system("cls")
    args = parse_args()
    processor = EEGProcessor(
        visualize=args.visualize, montage=args.montage, select_ecg=args.sel_ecg, plot_final=args.plot_f, save=args.save,
        epoch_duration=args.epoch_duration, dim_reduction=args.dim_reduction, var_selection=args.var_selection)

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
