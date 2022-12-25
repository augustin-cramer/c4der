from typing import List, Optional, Union
from datetime import timedelta
import logging

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

from utils._utils import get_mass_centers_dist


def kernel(
    Z: np.ndarray, eps1: float, eps2: float, ar: float = 0.6
) -> np.ndarray:
    """
    It takes a vector of values, and returns a vector of values, where each value is the result of
    applying the kernel function to the corresponding value in the input vector

    :param Z: The input data, a numpy array of shape (n_samples, n_features)
    :param eps1: the energy of the first state
    :param eps2: the lower bound of the kernel
    :param ar: the acceptance rate at eps2 of the kernel
    :return: The kernel function is being returned.
    """
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
        non_spatial_epss: Union[float | str, List[float | str], None] = None,
        spatial_weight_on_x: float = 0.5,
        algorithm="auto",
        leaf_size=30,
    ) -> None:

        # Must have spatial_eps_2 > spatial_eps_1
        if spatial_eps_1 > spatial_eps_2:
            self.spatial_eps_1 = spatial_eps_2
            self.spatial_eps_2 = spatial_eps_1
        else:
            self.spatial_eps_1 = spatial_eps_1
            self.spatial_eps_2 = spatial_eps_2

        # Verify spatial_weight_on_x in ]0,1[
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
        self.labels_: np.ndarray

    def _evaluate_border_points(self, neighborhoods: List[np.ndarray], final_dist: np.ndarray):
        """
        Randomly rejects points in the neighborhood further than `spatial_eps_1` using a
        kernel function

        :param neighborhoods: list of lists of points in the neighborhood of each point
        :param final_dist: the distance matrix between all points
        """

        for i, neighborhood in enumerate(neighborhoods):
            if len(neighborhood) > 0:
                # Take all ambiguous points apart
                ambiguous_points = neighborhood[
                    final_dist[i, neighborhood] > self.spatial_eps_1
                ]
                if len(ambiguous_points) > 0:
                    # Evaluate their kernel value
                    kernel_values = kernel(
                        final_dist[ambiguous_points, i],
                        eps1=self.spatial_eps_1,
                        eps2=self.spatial_eps_2,
                    )
                    # Create a random list of rejected points
                    rejected = ambiguous_points[
                        np.where(
                            np.random.uniform(size=len(kernel_values))
                            > kernel_values
                        )
                    ]

                    # Filter out the rejected points from neighborhood
                    neighborhood = [
                        el for el in neighborhood if el not in rejected
                    ]

    def _get_neighborhoods(self, final_dist):
        """
        > The function takes in a distance matrix and returns a list of lists, where each list contains
        the indices of the neighbors of each point

        :param final_dist: The distance matrix that we calculated earlier
        :return: The indices of the neighbors of each point.
        """

        neighbors_model = NearestNeighbors(
            radius=self.spatial_eps_2,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric="precomputed",
            n_jobs=self.n_jobs,
        )

        neighbors_model.fit(final_dist)

        return neighbors_model.radius_neighbors(
            final_dist, return_distance=False
        )

    def _compute_distances(
        self,
        n_samples: int,
        X_spatial: np.ndarray,
        X_temporal: np.ndarray,
        X_non_spatial: Optional[np.ndarray] = None,
    ):
        """
        The function takes in the number of samples, the spatial, temporal and non-spatial data and
        returns the distance matrix.

        The distance matrix is computed by first computing the time distance between the samples, then
        the spatial distance between the samples and then the non-spatial distance between the samples.

        The time distance is computed using the Euclidean distance. The spatial distance is computed
        using the Mahalanobis distance. The non-spatial distance is computed using the Bray-Curtis
        distance.

        The spatial distance is weighted by the spatial weight on x.

        :param n_samples: number of samples
        :type n_samples: int
        :param X_spatial: the spatial coordinates of the data points
        :type X_spatial: np.ndarray
        :param X_temporal: the temporal data, which is a 1D array of length n_samples
        :type X_temporal: np.ndarray
        :param X_non_spatial: Non Spatial cols
        :type X_non_spatial: Optional[np.ndarray]
        :return: The distance matrix
        """

        # Compute time distance
        time_dist = pdist(X_temporal.reshape(n_samples, 1), metric="cityblock")

        # Compute spatial distance
        spatial_weighted_dist = np.sqrt(2) * pdist(
            X_spatial,
            metric="mahalanobis",
            VI=np.diag(
                [self.spatial_weight_on_x, 1 - self.spatial_weight_on_x]
            ),
        )

        # Give a high spatial_dist value to the points being too far apart time wise
        filtered_dist = np.where(
            time_dist <= self.temporal_eps,
            spatial_weighted_dist,
            2 * self.spatial_eps_2,
        )

        # Compute  non_spatial_distance if necessary
        if (
            X_non_spatial is not None
            and type(self.non_spatial_epss) is not None
        ):
            if type(self.non_spatial_epss) is list:

                for i, eps in enumerate(self.non_spatial_epss):
                    logging.debug(filtered_dist)
                    if eps == "strict":
                        non_spatial_dist = pdist(
                            X_non_spatial[:, i].reshape(n_samples, 1),
                            metric="cityblock",
                        )
                        # Give a high spatial_dist value to the points being too
                        # far apart non spatial wise
                        filtered_dist = np.where(
                            non_spatial_dist <= 0.5,
                            filtered_dist,
                            2 * self.spatial_eps_2,
                        )

                    else:
                        non_spatial_dist = pdist(
                            X_non_spatial[:, i].reshape(n_samples, 1),
                            metric="euclidean",
                        )

                        # Give a high spatial_dist value to the points being too
                        # far apart non spatial wise
                        filtered_dist = np.where(
                            non_spatial_dist <= eps,
                            filtered_dist,
                            2 * self.spatial_eps_2,
                        )
            else:
                assert (
                    type(self.non_spatial_epss) is float
                    or type(self.non_spatial_epss) is str
                ), "Number of non spatial eps and actual inputed non spatial variables do not match"

                if type(self.non_spatial_epss) is float:
                    non_spatial_dist = pdist(
                        X_non_spatial.reshape(n_samples, 1),
                        metric="braycurtis",
                    )

                    filtered_dist = np.where(
                        non_spatial_dist <= self.non_spatial_epss,
                        filtered_dist,
                        2 * self.spatial_eps_2,
                    )

                else:
                    non_spatial_dist = pdist(
                        X_non_spatial.reshape(n_samples, 1),
                        metric="cityblock",
                    )

                    filtered_dist = np.where(
                        non_spatial_dist <= 0.5,
                        filtered_dist,
                        2 * self.spatial_eps_2,
                    )
        return squareform(filtered_dist)

    def _cluster_contamination(self, core_samples, neighborhoods, n_samples):
        """
        :param core_samples: A boolean array of shape [n_samples] that is True for core samples
        :param neighborhoods: a list of arrays, each array contains the indices of the neighbors of a
        point
        :param n_samples: The number of samples (or total weight) in the dataset
        """

        label_num = 0
        stack = []
        labels = np.full(n_samples, -1, dtype=np.intp)

        # Cluster diffusion
        for i in range(labels.shape[0]):
            if labels[i] != -1 or not core_samples[i]:
                continue
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

    def fit(
        self,
        X_spatial: np.ndarray,
        X_temporal: np.ndarray,
        X_non_spatial: Optional[np.ndarray] = None,
    ):

        """
        The function takes in the spatial and temporal data and computes the distance between each
        point. Then, it creates a neighborhood for each point based on the distance between the points.
        The function then evaluates if the points are core points or not. If they are core points, the
        function clusters the points together.

        :param X_spatial: spatial data (2D)
        :type X_spatial: np.ndarray
        :param X_temporal: the temporal data
        :type X_temporal: np.ndarray
        :param X_non_spatial: Non spatial considerations (Area of the bounding boxes, etc ...)
        :type X_non_spatial: Optional[np.ndarray]
        :return: The return is the self object.
        """

        # Check if the inputs are valid
        n_samples = X_spatial.shape[0]

        if n_samples != X_temporal.shape[0] or (
            X_non_spatial is not None and X_non_spatial.shape[0] != n_samples
        ):
            raise ValueError("Input arrays must have the same first dimension")

        if X_spatial.shape[1] != 2:
            raise ValueError("X_spatial must have two coordinates per row")

        # ""masked"" square matrix representing distance between point
        # discriminated by time and non-spatial consideration
        final_dist = self._compute_distances(
            n_samples, X_spatial, X_temporal, X_non_spatial
        )

        # Efficient Indexation of points by neighborhoods (see scikit-learn)
        neighborhoods = self._get_neighborhoods(final_dist=final_dist)

        # Evaluates if points between the two spatial bounds are added to the neighborhood or not
        self._evaluate_border_points(
            neighborhoods=neighborhoods, final_dist=final_dist
        )

        # Number of neighbors for each point
        n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])

        # Boolean array that determines if a point is a core point.
        core_samples = np.asarray(n_neighbors >= self.min_samples, dtype=bool)

        # Cluster diffusion
        self._cluster_contamination(
            core_samples=core_samples,
            neighborhoods=neighborhoods,
            n_samples=n_samples,
        )

        #### Get info on clusters
        cluster_sizes = []
        cluster_time_span = []
        cluster_variance_from_mc: None | list = []
        cluster_mean_point = []
        for cluster_num in np.unique(self.labels_):
            mask = np.where(self.labels_ == cluster_num)
            cluster_sizes.append(
                np.sum((self.labels_ == cluster_num).astype(int))
            )
            cluster_time_span.append(
                timedelta(
                    seconds=max(X_temporal[mask]) - min(X_temporal[mask])
                )
            )
            mass_centers_pos = get_mass_centers_dist(X_spatial[:, 1])
            cluster_variance_from_mc.append(
                np.sum(np.std(X_spatial[mask], axis=1))
            )
            cluster_mean_point.append(np.mean(X_spatial[mask], axis=0))
        self.cluster_info = {
            "cluster_sizes": cluster_sizes,
            "cluster_time_span": cluster_time_span,
            "cluster_variance_from_mc": cluster_variance_from_mc,
            "cluster_mean_point": cluster_mean_point,
            "labels": list(np.unique(self.labels_)),
        }
        return self

    def get_params(self) -> dict:
        return vars(self)

    def get_cluster_info(self) -> dict:
        return self.cluster_info
