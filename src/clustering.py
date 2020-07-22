import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


class KmeansBase(KMeans):
    def __init__(self, max_n_clusters=10, n_init=10, **kwargs):
        self.max_n_clusters = max_n_clusters
        self.n_init = n_init
        super().__init__(n_init=n_init, **kwargs)

    def fit(self, X, y=None, sample_weight=None):
        x, silh = ((1 - X.fillna(0)) / 2) ** 0.5, np.array([])
        for init in range(self.n_init):
            for i in range(2, self.max_n_clusters + 1):
                super().fit(x)
                silhouette = silhouette_samples(x, self.labels_)
                quality = silhouette.mean() / silhouette.std()
        new_idx = np.argsort(self.labels_)
        X1 = X.iloc[new_idx]

        X1 = X1.iloc[:, new_idx]

        self.clusters = {
            i: X.columns[np.where(self.labels_ == i)[0]].tolist()
            for i in np.unique(self.labels_)
        }

        self.silhouette = silhouette

        return self
