import pandas as pd
import numpy as np
import shutil
from datetime import timedelta
import itertools
import logging


def print_centre(s):
    print(s.center(shutil.get_terminal_size().columns))


def clusters_filtering(
    cluster_info: dict,
    min_sample_in_cluster: int = 4,
    min_time_span: timedelta = timedelta(minutes=6),
    min_variance: float = 10,
) -> dict:
    k = 0
    _filter = {}
    for i, cluster_num in enumerate(cluster_info["labels"]):
        if cluster_num != -1:
            cluster_size_ok = cluster_info["cluster_sizes"][i] >= min_sample_in_cluster
            time_ok = cluster_info["cluster_time_span"][i] >= min_time_span
            variance_ok = cluster_info["cluster_variance_from_mc"][i] >= min_variance

            if cluster_size_ok & time_ok & variance_ok:
                _filter[cluster_num] = k
                k += 1
            else:
                logging.info(
                    f"Cluster {cluster_num} rejected => [size = {cluster_info['cluster_sizes'][i]}] [TimeSpan = {cluster_info['cluster_time_span'][i]}] [Variance = {cluster_info['cluster_variance_from_mc'][i]}]".center(
                        shutil.get_terminal_size().columns
                    )
                )
                _filter[cluster_num] = -1
        else:
            _filter[cluster_num] = -1

    return _filter


def get_mass_centers(y: pd.Series) -> np.ndarray:

    count, divisions = np.histogram(y, bins=20)
    sample_mean = np.mean(count)
    frequencies = [
        (((b - a) * (c - b) < 0) * (b > sample_mean), division)
        for (a, b, c), division in zip(
            zip(count, count[1:], count[2:]), itertools.pairwise(divisions[1:])
        )
    ]
    mass_centers = np.array(
        [el for is_mass_center, el in frequencies if is_mass_center]
    )
    mass_centers = np.mean(mass_centers, axis=1)
    return np.array([np.argmin(np.abs(mass_centers - x)) for x in y])


def get_mass_centers_dist(y: pd.Series | np.ndarray) -> np.ndarray:

    count, divisions = np.histogram(y, bins=20)
    sample_mean = np.mean(count)
    frequencies = [
        (((b - a) * (c - b) < 0) * (b > sample_mean), division)
        for (a, b, c), division in zip(
            zip(count, count[1:], count[2:]), itertools.pairwise(divisions[1:])
        )
    ]
    mass_centers = np.array(
        [el for is_mass_center, el in frequencies if is_mass_center]
    )
    mass_centers = np.mean(mass_centers, axis=1)

    return np.array([np.min(np.abs(mass_centers - x)) for x in y])
