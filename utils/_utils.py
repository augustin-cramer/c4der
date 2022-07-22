import pandas as pd
import numpy as np
import shutil
from datetime import timedelta
import itertools


def print_centre(s):
    print(s.center(shutil.get_terminal_size().columns))


def standardize_cluster_format(
    df: pd.DataFrame | pd.Series, min_sample_in_cluster: int, min_time_span: timedelta
) -> dict:
    i = 0
    converter = {}
    for k, v in (df.groupby("labels").count().x > min_sample_in_cluster).items():
        start = min(df[df.labels == k]["timestamps"])
        end = max(df[df.labels == k]["timestamps"])
        time_ok = (end - start) >= min_time_span

        if v and time_ok and k != -1:
            converter[k] = i
            i += 1
        else:
            converter[k] = -1

    return converter


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
