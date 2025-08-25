"""
Miscelaneous functions
"""

# ---------- Imports ----------

from functools import wraps
import time
from tkinter import filedialog as fd
from typing import Any, Callable

import hdbscan
import mne
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score, davies_bouldin_score, silhouette_score
)
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import umap

from constants import *


# ---------- Decorators ----------

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


# ---------- Helpers ----------
# ---------- Data loading ----------

def open_csv() -> pd.DataFrame:
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


def open_fif() -> mne.Epochs:
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


# ---------- Pre-processing ----------

def normalize(data: np.ndarray) -> np.ndarray:
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
    return (data - mean) / std


# ---------- Clustering ----------

def clustering_grid_search(
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
                random_state=random_state,
                n_jobs=1
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


def clustering_scores(X: np.ndarray, y: np.ndarray) -> None:
    """
    Compute and display clustering quality metrics: Silhouette, 
    Calinski-Harabasz, and Davies-Bouldin scores.

    This function evaluates how well the clustering assignment `y`
    fits the embedded or feature representation `X`. It reports 
    three complementary internal validation metrics:

    - Silhouette score:
        Ranges from -1 to 1.
        Higher values indicate better-defined clusters, where points
        are close to their own cluster and far from other clusters.

    - Calinski-Harabasz score (Variance Ratio Criterion):
        Higher values indicate better separation between clusters 
        relative to their compactness. This score increases when
        clusters are dense and well separated.

    - Davies-Bouldin score:
        Lower values indicate better clustering. It measures the 
        average similarity between each cluster and its most similar
        one, based on the ratio of intra-cluster dispersion to 
        inter-cluster separation.

    Parameters
    ----------
        X : np.ndarray
            Feature matrix or low-dimensional embedding of the samples.
        y : np.ndarray
            Cluster labels assigned to the samples. Noise points (-1) 
            are included if present.
    """
    sil_score: float = silhouette_score(X, y)
    ch_score: float = calinski_harabasz_score(X, y)
    db_score: float = davies_bouldin_score(X, y)

    print(f"{GREEN}Best clustering scores:")
    print(f"{GREEN}- Silhouette score = {YELLOW}{sil_score:.3f}{RESET}")
    print((
        f"{GREEN}- Calinski-Harabasz score = "
        f"{YELLOW}{ch_score:.3f}{RESET}"
    ))
    print(f"{GREEN}- Davies-Bouldin score = {YELLOW}{db_score:.3f}{RESET}")
