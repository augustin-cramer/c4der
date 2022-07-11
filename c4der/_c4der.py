
from datetime import datetime, timedelta
from lib2to3.pytree import convert
from time import time
from typing import List, Optional, Union
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

def kernel(Z, eps1, eps2, ar=0.6) -> np.ndarray:
    Z -= eps1
    sigma = ((eps1 - eps2) ** 2) / np.log(ar)
    return np.exp((Z**2) / sigma)


class c4der:
    def __init__(
        self,
        spatial_eps_1: float,
        n_jobs: int,
        spatial_eps_2: float,
        temporal_eps: float,
        min_samples: int,
        non_spatial_epss: Union[float, List[float], None] = None,
        spatial_weight_on_x: float = 0.5,
        algorithm="auto",
        leaf_size=30,
    ) -> None:

        if spatial_eps_1 > spatial_eps_2:
            self.spatial_eps_1 = spatial_eps_2
            self.spatial_eps_2 = spatial_eps_1
        else:
            self.spatial_eps_1 = spatial_eps_1
            self.spatial_eps_2 = spatial_eps_2

        if spatial_weight_on_x < 0 or spatial_weight_on_x > 1:
            raise ValueError(
                "Parameter spatial_weight_on_x is degenerated. Please provide a value in (0,1). Default is .5"
            )

        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.min_samples = min_samples
        self.spatial_weight_on_x = spatial_weight_on_x
        self.non_spatial_epss = non_spatial_epss
        self.temporal_eps = temporal_eps
        self.n_jobs = n_jobs
        self.labels: np.ndarray

    def fit(
        self,
        X_spatial: np.ndarray,
        X_temporal: np.ndarray,
        X_non_spatial: Optional[np.ndarray] = None,
    ):

        n_samples = X_spatial.shape[0]
        stack = []

        if n_samples != X_temporal.shape[0] or (
            X_non_spatial is not None and X_non_spatial.shape[0] != n_samples
        ):
            raise ValueError("Input arrays must have the same first dimension")

        if X_spatial.shape[1] != 2:
            raise ValueError("X_spatial must have two coordinates per row")

        time_dist = pdist(X_temporal.reshape(n_samples, 1), metric="euclidean")
        spatial_weighted_dist = np.sqrt(2) * pdist(
            X_spatial,
            metric="mahalanobis",
            VI=np.diag([self.spatial_weight_on_x, 1 - self.spatial_weight_on_x]),
        )

        filtered_dist = np.where(
            time_dist <= self.temporal_eps,
            spatial_weighted_dist,
            2 * self.spatial_eps_2,
        )

        if X_non_spatial is not None and type(self.non_spatial_epss) is not None:
            if (
                type(self.non_spatial_epss) is List
            ):  # Both shall be true simultuneously. For typing purposes we still let both as conditions
                for i, eps in enumerate(self.non_spatial_epss):
                    non_spatial_dist = pdist(X_non_spatial[:, i], metric="euclidean")

                    filtered_dist = np.where(
                        non_spatial_dist <= eps, filtered_dist, 2 * self.spatial_eps_2
                    )
            else:
                assert (
                    type(self.non_spatial_epss) is float
                ), "Number of non spatial eps and actual inputed non spatial variables do not match"
                non_spatial_dist = pdist(X_non_spatial.reshape(n_samples, 1), metric="braycurtis")
                filtered_dist = np.where(
                    non_spatial_dist <= self.non_spatial_epss,
                    filtered_dist,
                    2 * self.spatial_eps_2,
                )
        final_dist = squareform(filtered_dist)

        ##### It's Kernel time uWu

        neighbors_model = NearestNeighbors(
            radius=self.spatial_eps_2,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric="precomputed",
            n_jobs=self.n_jobs,
        )

        neighbors_model.fit(final_dist)

        neighborhoods = neighbors_model.radius_neighbors(
            final_dist, return_distance=False
        )
        for i, neighborhood in enumerate(neighborhoods):
            if len(neighborhood) > 0:
                ambiguous_points = neighborhood[
                    final_dist[i, neighborhood] > self.spatial_eps_1
                ]
                if len(ambiguous_points) > 0:
                    kernel_values = kernel(
                        final_dist[ambiguous_points, i],
                        eps1=self.spatial_eps_1,
                        eps2=self.spatial_eps_2,
                    )
                    rejected = ambiguous_points[
                        np.where(
                            np.random.uniform(size=len(kernel_values)) > kernel_values
                        )
                    ]
                    neighborhood = [el for el in neighborhood if el not in rejected]

        n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])
        labels = np.full(n_samples, -1, dtype=np.intp)
        core_samples = np.asarray(n_neighbors >= self.min_samples, dtype=bool)
        label_num = 0
        # Cluster diffusion
        for i in range(labels.shape[0]):
            if labels[i] != -1 or not core_samples[i]:
                continue
            # Depth-first search starting from i, ending at the non-core points.
            # This is very similar to the classic algorithm for computing connected
            # components, the difference being that we label non-core points as
            # part of a cluster (component), but don't expand their neighborhoods.
            while True:
                if labels[i] == -1:
                    labels[i] = label_num

                    if core_samples[i]:

                        neighb = neighborhoods[i]

                        for i in range(neighb.shape[0]):
                            v = neighb[i]

                            if labels[v] == -1:
                                stack.append(v)

                if not stack:
                    break
                i = stack.pop()

            label_num += 1
        self.core_sample_indices_ = np.where(core_samples)[0]
        self.labels_ = labels

        return self

    def get_params(self) -> dict :
        return vars(self)


if __name__ == "__main__":
    import pandas as pd
    import plotly.express as px

    df = pd.read_csv("../deep-hair/3D_500_area.csv", index_col=0)
    df.Area = np.sqrt(df.Area)
    #df = df[df["T"] < 3320]

    c4der_scan = c4der(
        50, -1, 70, 240, 10, spatial_weight_on_x=.2, 
        non_spatial_epss=0.1
    )
    c4der_scan.fit(
        df[["X", "Y"]].to_numpy(),
        df["T"].to_numpy(),
        X_non_spatial=df["Area"].to_numpy(),
    )
    start = datetime(year = 2022, month = 7,day=2, hour=9, minute=1)
    df.loc[:,"T"] = df['T'].apply(lambda x: start + timedelta(seconds=x) )
    df["labels"] = c4der_scan.labels_
    converter = (df.groupby('labels').count().X > 20)
    i=0
    for k,v in converter.items():
        if v and k!= -1 :
            converter[k] = i
            i+=1
        else :
            converter[k]= -1
    df['labels'] = df['labels'].apply(lambda x: converter[x])

    df = df[df.labels != -1]
    fig = px.scatter_3d(
        z=df["T"],
        y=df["X"],
        x=df["Y"],
        color=df["labels"],
    )
    #fig.show()
    print(f'Nombre de clients détectés :{len(df.labels.unique())}')
    #df.loc[:,'T'] = df.loc[:,'T'].astype(np.datetime64)
    for cluster in df.labels.unique():
        start = min(df[df.labels==cluster]['T'])
        end = max(df[df.labels==cluster]['T'])
        print(f'{start.strftime("%H:%M")} -->  {end.strftime("%H:%M")}')