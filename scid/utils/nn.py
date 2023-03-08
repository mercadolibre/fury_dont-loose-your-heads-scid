import scann
import numpy as np
from typing import Optional, Union


# https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md
from timeit import timeit


class FastNN:
    def __init__(self, distance_measure: str = "dot_product",
                 # Tree
                 # Partitioning: ScaNN partitions the dataset during training time, and at query time selects the top
                 # partitions to pass onto the scoring stage.
                 # The most important parameters are: num_leaves and num_leaves_to_search.
                 num_leaves: Optional[int] = None, num_leaves_scaling: Optional[int] = 1,
                 num_leaves_to_search: Optional[int] = None, spherical_partitioning_type: bool = False,
                 training_iterations: int = 30, min_cluster_size: int = 500,
                 training_sample_size: Union[int, str] = "auto", training_sample_scaling: float = 0.05,
                 # Score
                 # Scoring: ScaNN computes the distances from the query to all datapoints in the dataset (if partitioning isn't enabled)
                 # or all datapoints in a partition to search (if partitioning is enabled).
                 # These distances aren't necessarily exact.

                 score: str = "ah", dimensions_per_block: int = 2, anisotropic_quantization_threshold: float = 0.2,
                 # Reorder
                 # Rescoring: ScaNN takes the best k' distances from scoring and re-computes these distances more accurately.
                 # From these k' re-computed distances the top k are selected.
                 reordering_num_neighbors: int = 500, final_num_neighbors: int = 10,
                 # Dataset
                 # We have to normalize due to we are using dot_product.
                 normalize: bool = True):
        self.distance_measure = distance_measure
        self.num_leaves = num_leaves
        self.num_leaves_scaling = num_leaves_scaling
        self.num_leaves_to_search = num_leaves_to_search
        self.spherical_partitioning_type = spherical_partitioning_type
        self.training_iterations = training_iterations
        self.min_cluster_size = min_cluster_size
        self.training_sample_size = training_sample_size
        self.training_sample_scaling = training_sample_scaling
        self.score = score
        self.dimensions_per_block = dimensions_per_block
        self.anisotropic_quantization_threshold = anisotropic_quantization_threshold
        self.reordering_num_neighbors = reordering_num_neighbors
        self.final_num_neighbors = final_num_neighbors
        self.normalize = normalize

    def fit(self, X):
        with timeit(f'Fitting ScaNN of matrix of shape {X.shape}'):
            return self._fit(X)

    def _fit(self, X):
        if self.normalize:
            X[np.linalg.norm(X, axis=1) == 0] = 1.0 / np.sqrt(X.shape[1])
            X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]

        # If a dataset has n points, the number of partitions will be the same order of magnitude as sqrt(n)
        # for a good balance of partitioning quality and speed.
        if self.num_leaves is None:
            self.num_leaves = int(np.sqrt(X.shape[0]) * self.num_leaves_scaling)

        # During inference ScaNN will compare the query vector against all the partition centroids and select
        # the closest num_leaves_to_search ones to search in. Therefore, num_leaves_to_search / num_leaves
        # determines the proportion of the dataset that is pruned. Raising this proportion increases accuracy
        # but leads to more points being scored and therefore less speed (The more leaves to search,
        # the better the retrieval quality, and higher computational cost).
        if self.num_leaves_to_search is None:
            self.num_leaves_to_search = self.num_leaves // 2

        if isinstance(self.training_sample_size, str):
            if self.training_sample_size == "auto":
                self.training_sample_size = int(len(X) * self.training_sample_scaling)
            else:
                raise ValueError(f"training_sample_size must be str: auto or int")

        self.scann_searcher_ = (
            scann.scann_ops_pybind.builder(X, 10, self.distance_measure)
            .tree(
                num_leaves=self.num_leaves,
                num_leaves_to_search=self.num_leaves_to_search,
                spherical=self.spherical_partitioning_type,
                min_partition_size=self.min_cluster_size,
                training_iterations=self.training_iterations,
                training_sample_size=self.training_sample_size,
            )
            .score_ah(
                dimensions_per_block=self.dimensions_per_block,
                anisotropic_quantization_threshold=self.anisotropic_quantization_threshold,
            )
            .reorder(self.reordering_num_neighbors)
        ).build()
        return self

    def query(self, X, num_neigbours, return_distances=False):
        with timeit(f'perform {X.shape[0]} queries to ScaNN'):
            neighbors_idx, distances = self.scann_searcher_.search_batched(
                X, final_num_neighbors=num_neigbours
            )
        if return_distances:
            return neighbors_idx, distances
        else:
            return neighbors_idx
