from typing import List, Optional, Union
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from ._c4der_inner import c4der_inner


class c4der:
    def __init__(
        self,
        spatial_eps_1: float,
        n_jobs: int,
        spatial_eps_2: float,
        temporal_eps: float,
        non_spatial_epss: Union[float, List[float], None] = None,
        spatial_weight_on_x: float = 0.5,
        algorithm="auto",
        leaf_size = 30
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

        if n_samples != X_temporal.shape[0] or (
            X_non_spatial is not None and X_non_spatial.shape[0] != n_samples
        ):
            raise ValueError("Input arrays must have the same first dimension")

        if X_spatial.shape[1] != 2:
            raise ValueError("X_spatial must have two coordinates per row")

        n_features = X_non_spatial.shape[1] if X_non_spatial is not None else 0

        time_dist = pdist(X_temporal.reshape(n_samples, 1), metric="euclidean")
        spatial_weighted_dist = pdist(
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
                    filtered_dist = np.where(
                        X_non_spatial[:, i] <= eps, filtered_dist, 2 * eps
                    )
            else:
                assert (
                    type(self.non_spatial_epss) is float
                ), "Number of non spatial eps and actual inputed non spatial variables do not match"
                filtered_dist = np.where(
                    X_non_spatial <= self.non_spatial_epss,
                    filtered_dist,
                    2 * self.non_spatial_epss,
                )


        ##### It's Kernel time uWu

        neighbors_model = NearestNeighbors(
            radius=self.spatial_eps_2,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric='precomputed',
            n_jobs=self.n_jobs,
        )

        neighbors_model.fit(squareform(filtered_dist))

        neighborhoods = neighbors_model.radius_neighbors(squareform(filtered_dist), return_distance=False)
        n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])
        labels = np.full(n_samples, -1, dtype=np.intp)

        #Determines which datapoints are core sample 
        # ...

        #Cluster diffusion
        dbscan_inner(core_samples, neighborhoods, labels)



