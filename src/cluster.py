from itertools import groupby
from operator import itemgetter
import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

logger = logging.Logger(__name__)


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
        self.max_n_clusters = max_n_clusters
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
        best_cluster_centers_ = None
        best_labels_ = None
        best_inertia_ = None
        best_n_iter_ = None
        best_silhouette = None
        best_quality = -1

        for init in range(self.n_init):
            for i in range(2, self.max_n_clusters + 1):
                logging.info(f"Clustering iteration: init: {init}, "
                             f"max_cluster: {i}")
                self.n_clusters = i
                super().fit(X, y=y, sample_weight=sample_weight)
                silhouette = silhouette_samples(X, self.labels_)
                quality = silhouette.mean() / silhouette.std()
                if best_silhouette is None or quality > best_quality:
                    best_silhouette = silhouette
                    best_cluster_centers_ = self.cluster_centers_
                    best_labels_ = self.labels_.copy()
                    best_inertia_ = self.inertia_
                    best_n_iter_ = self.n_iter_
                    best_quality = quality

        # We keep only the best kmeans data
        self.cluster_centers_ = best_cluster_centers_
        self.labels_ = best_labels_
        self.inertia_ = best_inertia_
        self.n_iter_ = best_n_iter_
        self.quality = best_quality
        self.silhouette = best_silhouette
        self.n_clusters = len(best_labels_)
        return self

    def get_cluster(self):
        cluster = {}
        for i in self.labels_:
            cluster[i] = np.where(self.labels_ == i)[0]


class KMeansHL(KMeansBase):
    @staticmethod
    def eval_scores(labels, silhouette):
        clusters = groupby(
            sorted(zip(labels, silhouette)),
            itemgetter(0)
        )

        for key, data in clusters:
            silhouettes = tuple(data)
            if silhouettes is not None:
                mean = np.mean(silhouettes, axis=0)[1]
                vol = np.std(silhouettes, axis=0)[1]
                if vol != 0.:
                    yield key, mean / vol
                else:
                    yield key, 0.
            else:
                yield key, None

    def merge(self, sub_other, corr, cluster_redo, clusters_scores_avg):
        rows_idx, = np.where(
            np.isin(self.labels_, cluster_redo)
        )
        new_labels = self.labels_.copy()
        new_labels[rows_idx] = sub_other.labels_ + max(self.labels_)
        X = ((1 - np.nan_to_num(corr)) / 2.) ** .5
        new_silhouette = silhouette_samples(X, new_labels)
        new_quality = new_silhouette.mean() / new_silhouette.std()
        clusters_scores = tuple(self.eval_scores(new_labels, new_silhouette))
        _, new_clusters_scores_avg = np.mean(np.nan_to_num(clusters_scores),
                                             axis=0)
        logger.info(f"A solution with score {new_clusters_scores_avg} using "
                    f"KMeans HL found compared to the original score  "
                    f"{clusters_scores_avg}")
        if new_clusters_scores_avg > clusters_scores_avg:
            # TODO: Trigger recalculation of kmeans attributes from new labels
            self.labels_ = new_labels
            self.quality = new_quality
            self.silhouette = new_silhouette

    def fit(self, corr, y=None, sample_weight=None):
        # Initial clustering
        super().fit(corr=corr, y=y, sample_weight=sample_weight)

        # Second clustering
        clusters_scores = tuple(self.eval_scores(self.labels_,
                                                 self.silhouette))
        _, clusters_scores_avg = np.mean(np.nan_to_num(clusters_scores),
                                         axis=0)
        cluster_redo = tuple(
            key for key, score in clusters_scores
            if score < clusters_scores_avg
        )

        if len(cluster_redo) > 1:
            # Redo elements in bad quality clusters
            rows_idx, = np.where(np.isin(self.labels_, cluster_redo))
            # Get the sub-correlation matrix with indexes in rows_idx
            corr_sub = corr[rows_idx[:, None], rows_idx]
            kmeans_sub = KMeansHL(
                max_n_clusters=min(
                    self.max_n_clusters,
                    corr_sub.shape[1] - 1),
                n_init=self.n_init,
                init=self.init,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose,
                random_state=self.random_state,
                copy_x=self.copy_x,
                algorithm=self.algorithm
            ).fit(corr_sub)
            self.merge(sub_other=kmeans_sub, corr=corr,
                       cluster_redo=cluster_redo,
                       clusters_scores_avg=clusters_scores_avg)
        return self
