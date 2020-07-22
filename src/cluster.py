import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


class KmeansBase(KMeans):
    def __init__(self, max_n_clusters=10, n_init=10, **kwargs):
        self.max_n_clusters = max_n_clusters
        self.n_init = n_init
        self.quality = None
        self.silhouette = None
        super().__init__(n_init=n_init, **kwargs)

    def fit(self, corr, y=None, sample_weight=None):
        # Transform the correlation matrix to X
        X = ((1 - np.nan_to_num(corr)) / 2) ** 0.5
        # Init the best figures
        best_kmeans = None
        best_silhouette = np.array([])
        best_quality = -1

        for _ in range(self.n_init):
            for i in range(2, self.max_n_clusters + 1):
                self.n_clusters = i
                kmeans = super().fit(X)
                silhouette = silhouette_samples(X, self.labels_)
                quality = silhouette.mean() / silhouette.std()
                if best_silhouette.size > 0 or quality > best_quality:
                    best_silhouette = silhouette
                    best_kmeans = kmeans
                    best_quality = quality

        self.cluster_centers_ = best_kmeans.cluster_centers_
        self.labels_ = best_kmeans.labels_
        self.inertia_ = best_kmeans.inertia_
        self.n_iter_ = best_kmeans.n_iter_
        self.quality = best_quality
        self.silhouette = best_silhouette
        return self
