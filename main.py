from ast import parse
import itertools
from datetime import datetime, timedelta
from time import perf_counter
from typing import Optional, Union, List

import numpy as np
import pandas as pd
import plotly.express as px

from c4der._c4der import c4der

import argparse
import shutil


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


def get_mass_centers_dist(y: pd.Series) -> pd.Series | pd.DataFrame:

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
    return y.apply(lambda x: np.argmin(np.abs(mass_centers - x)))


def main(
    file_path: str,
    min_samples_in_c: int = 10,
    min_timespan_minutes: int = 6,
    max_time: Optional[tuple[int, int]] = None,
    spatial_eps_1=20,
    n_jobs: int = -1,
    spatial_eps_2: int = 50,
    temporal_eps: int = 240,
    min_samples: int = 5,
    spatial_weight_on_x: float = 0.01,
    non_spatial_epss: Union[float | str, List[float | str], None] = "strict",
    include_mass_center: bool = True,
    clean_noise: bool = False,
    plot: bool = True,
    save_data: bool = False,
):

    min_timespan = timedelta(minutes=min_timespan_minutes)

    df = pd.read_csv(file_path, index_col=0)

    df["area"] = df.lengths * df.widths
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    first_day = df.loc[0, "timestamps"]
    df = df[
        df.timestamps
        <= datetime(
            day=first_day.day,
            month=first_day.month,
            year=first_day.year,
            hour=max_time[0],
            minute=max_time[1],
        )
    ]
    df["time_seconds"] = df["timestamps"].apply(
        lambda x: pd.Timedelta(x - df.loc[0, "timestamps"]).total_seconds()
    )

    df["mass_centers"] = get_mass_centers_dist(df.y)

    ########### c4der Time ###########
    c4der_scan = c4der(
        spatial_eps_1=spatial_eps_1,
        n_jobs=n_jobs,
        spatial_eps_2=spatial_eps_2,
        temporal_eps=temporal_eps,
        min_samples=min_samples,
        spatial_weight_on_x=spatial_weight_on_x,
        non_spatial_epss=non_spatial_epss,
    )
    if include_mass_center:
        c4der_scan.fit(
            df[["x", "y"]].to_numpy(),
            df["time_seconds"].to_numpy(),
            X_non_spatial=df["mass_centers"].to_numpy(),
        )
    else:
        c4der_scan.fit(
            df[["x", "y"]].to_numpy(),
            df["time_seconds"].to_numpy(),
        )

    df["labels"] = c4der_scan.labels_

    ###################################

    converter = standardize_cluster_format(
        df, min_sample_in_cluster=min_samples_in_c, min_time_span=min_timespan
    )
    df["labels"] = df["labels"].apply(lambda x: converter[x])

    if clean_noise:
        df = df[df.labels != -1]  # Get rid of the noise

    if plot:
        fig = px.scatter_3d(
            z=df["timestamps"],
            y=df["x"],
            x=df["y"],
            color=df["labels"],
        )

        fig.show()

    print_centre(
        f"Nombre de clients détectés :{sum((df.labels.unique()!=-1).astype(int))}"
    )
    for cluster in df.labels.unique():
        if cluster >= 0:
            start = min(df[df.labels == cluster]["timestamps"])
            end = max(df[df.labels == cluster]["timestamps"])
            print_centre(f'{start.strftime("%H:%M")} -->  {end.strftime("%H:%M")}')

    if save_data:
        df.to_csv(f"{file_path.split('.')[0]}_treated.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="test_set/20220628.csv")
    parser.add_argument("--h_max", type=int, default=14)
    parser.add_argument("--m_max", type=int, default=00)
    parser.add_argument("--plot_fig", type=bool, default=False)
    args = parser.parse_args()
    start = perf_counter()
    print_centre("################## c4der ################## \n")
    main(file_path=args.filepath, max_time=(args.h_max, args.m_max), plot=args.plot_fig)
    print_centre(f"Execution time : {str((perf_counter()-start))[2:5]} ms  \n")
    print_centre("###########################################")
