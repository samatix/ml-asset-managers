from itertools import groupby
from operator import itemgetter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


class KMeansBase(KMeans):
    def __init__(self, max_n_clusters=None, n_init=10, **kwargs):
        """
        KMeans Base Clustering method
        This kmeans algorithm doesn't require a
        :param max_n_clusters:
        :type max_n_clusters:
        :param n_init:
        :type n_init:
        :param kwargs:
        :type kwargs:
        """
        self.max_n_clusters = None
        self.n_init = n_init
        self.quality = None
        self.silhouette = None
        super().__init__(n_init=n_init, **kwargs)

    def fit(self, corr, y=None, sample_weight=None):
        if self.max_n_clusters is None:
            self.max_n_clusters = corr.shape[1] - 1

        # Transform the correlation matrix to X
        X = ((1 - np.nan_to_num(corr)) / 2) ** 0.5
        # Init the best figures
        best_kmeans = None
        best_silhouette = np.array([])
        best_quality = -1

        for _ in range(self.n_init):
            for i in range(2, self.max_n_clusters + 1):
                self.n_clusters = i
                kmeans = super().fit(X, y=y, sample_weight=sample_weight)
                silhouette = silhouette_samples(X, self.labels_)
                quality = silhouette.mean() / silhouette.std()
                if best_silhouette.size > 0 or quality > best_quality:
                    best_silhouette = silhouette
                    best_kmeans = kmeans
                    best_quality = quality

        # We keep only the best kmeans data
        self.cluster_centers_ = best_kmeans.cluster_centers_
        self.labels_ = best_kmeans.labels_
        self.inertia_ = best_kmeans.inertia_
        self.n_iter_ = best_kmeans.n_iter_
        self.quality = best_quality
        self.silhouette = best_silhouette
        return self


class KmeansHL(KMeansBase):
    def eval_scores(self):
        clusters = groupby(
            sorted(zip(self.labels_, self.silhouette)),
            itemgetter(0)
        )

        for key, data in clusters:
            silhouettes = tuple(data)
            yield key, np.mean(silhouettes) / np.std(silhouettes)

    def fit(self, corr, y=None, sample_weight=None):
        super().fit(corr=corr, y=y, sample_weight=sample_weight)

        clusters_scores = tuple(self.eval_scores())
        clusters_scores_avg = np.mean(np.nan_to_num(clusters_scores))

        cluster_redo = (
            key for key, score in self.eval_scores()
            if score < clusters_scores_avg
        )


