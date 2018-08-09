#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2017 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

import itertools
import numpy as np
from sortedcollections import ValueSortedDict
from pyannote.algorithms.clustering.hac import \
    HierarchicalAgglomerativeClustering
from pyannote.algorithms.clustering.hac.model import HACModel
from pyannote.algorithms.clustering.hac.stop import DistanceThreshold
from scipy.spatial.distance import squareform
from pyannote.audio.embedding.utils import pdist, cdist, l2_normalize
from scipy.cluster.hierarchy import dendrogram, linkage


class EmbeddingModel(HACModel):
    """
    Parameters
    ----------
    distance : str
        Defaults to 'angular'.
    mode : {'loose', 'strict'}
    """

    def __init__(self, distance='angular', mode='strict'):
        super(EmbeddingModel, self).__init__(is_symmetric=True)
        self.distance = distance
        self.mode = mode

    def compute_model(self, cluster, parent=None):

        # extract all embeddings for requested cluster
        support = parent.current_state.label_timeline(cluster).support()
        X = parent.features.crop(support, mode=self.mode)

        # average them all
        x = np.average(X, axis=0)
        n = len(X)

        return (x, n)

    def compute_merged_model(self, clusters, parent=None):

        # merge all embeddings by computing their weighted average
        X, N = zip(*[self[cluster] for cluster in clusters])
        x = np.average(X, axis=0, weights=N)
        n = np.sum(N)

        return (x, n)

    def compute_similarity_matrix(self, parent=None):

        clusters = list(self._models)
        n_clusters = len(clusters)

        X = np.vstack([self[cluster][0] for cluster in clusters])

        nX = l2_normalize(X)
        similarities = -squareform(pdist(nX, metric=self.distance))

        matrix = ValueSortedDict()
        for i, j in itertools.combinations(range(n_clusters), 2):
            matrix[clusters[i], clusters[j]] = similarities[i, j]
            matrix[clusters[j], clusters[i]] = similarities[j, i]

        return matrix

    def compute_similarities(self, cluster, clusters, parent=None):

        x = self[cluster][0].reshape((1, -1))
        X = np.vstack([self[c][0] for c in clusters])

        # L2 normalization
        nx = l2_normalize(x)
        nX = l2_normalize(X)

        similarities = -cdist(nx, nX, metric=self.distance)

        matrix = ValueSortedDict()
        for i, cluster_ in enumerate(clusters):
            matrix[cluster, cluster_] = similarities[0, i]
            matrix[cluster_, cluster] = similarities[0, i]

        return matrix

    def compute_similarity(self, cluster1, cluster2, parent=None):

        x1, _ = self[cluster1]
        x2, _ = self[cluster2]

        nx1 = l2_normalize(x1)
        nx2 = l2_normalize(x2)

        similarities = -cdist([nx1], [nx2], metric=self.distance)
        return similarities[0, 0]


class EmbeddingClustering(HierarchicalAgglomerativeClustering):
    """Audio sequence clustering based on embeddings
    Parameters
    ----------
    distance : str, optional
        Defaults to 'angular'.
    threshold : float, optional
        Defaults to 1.0.
    Usage
    -----
    >>> embedding = Precomputed(...)
    >>> clustering = EmbeddingClustering()
    >>> result = clustering(starting_point, features=embedding)
    """

    def __init__(self, threshold=1.0, force=False, distance='cosine',
                 mode='loose', constraint=None, logger=None):
        model = EmbeddingModel(distance=distance, mode=mode)
        stopping_criterion = DistanceThreshold(threshold=threshold,
                                               force=force)
        super(EmbeddingClustering, self).__init__(
            model,
            stopping_criterion=stopping_criterion,
            constraint=constraint,
            logger=logger)


class Clustering(object):
    def __init__(self, min_cluster_size=5, min_samples=None,
                 metric='euclidean'):
        super(Clustering, self).__init__()

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric

    def apply(self, fX):
        from hdbscan import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size,
                            min_samples=self.min_samples,
                            metric='precomputed')
        distance_matrix = squareform(pdist(fX, metric=self.metric))

        # apply clustering
        cluster_labels = clusterer.fit_predict(distance_matrix)
        
        # cluster embedding
        n_clusters = np.max(cluster_labels) + 1
        fC = l2_normalize(
            np.vstack([np.sum(fX[cluster_labels == k, :], axis=0)
                       for k in range(n_clusters)]))

        # tag each undefined embedding to closest cluster
        undefined = cluster_labels == -1
        closest_cluster = np.argmin(
            cdist(fC, fX[undefined, :], metric=self.metric), axis=0)
        cluster_labels[undefined] = closest_cluster        

        return cluster_labels

class ClusteringAP(object):
    def __init__(self, damping=0.8, preference=-20,
                 metric='angular'):
        super(ClusteringAP, self).__init__()

        self.damping = damping
        self.preference = preference
        self.metric = metric

    def computeLogDistMat(self, X, metric='angular'):
        dist=pdist(X, metric=metric)
        distMat = squareform((dist))*(-1.0)
        return distMat

    def apply(self, fX):
        from sklearn.cluster import AffinityPropagation
        clusterer = AffinityPropagation(damping=self.damping, max_iter=100, 
                                        convergence_iter=15,
                                        preference=self.preference,
                                        affinity='precomputed')
        distance_matrix = self.computeLogDistMat(fX, metric=self.metric)

        # apply clustering
        cluster_labels = clusterer.fit_predict(distance_matrix)  

        return cluster_labels

class ClusteringHAC(object):
    def __init__(self, method='average', threshold=5,
                 metric='angular'):
        super(ClusteringHAC, self).__init__()

        self.method = method
        self.threshold = threshold
        self.metric = metric

    def computeDistMat(self, X, metric='angular'):
        dist=pdist(X, metric=metric)
        return dist

    def my_fcluster(self, z, length, threshold):
        labels = np.array([i for i in range(length)])
        label_num = length
        for c1, c2, d, _ in z:
            if d < threshold:
                labels[labels == c1] = label_num
                labels[labels == c2] = label_num
                label_num += 1
        return list(labels)

    def apply(self, fX):        
        distance_matrix = self.computeDistMat(fX, metric=self.metric)
        Z = linkage(distance_matrix, method=self.method)

        # apply clustering
        cluster_labels = self.my_fcluster(Z, len(fX), self.threshold) 

        return cluster_labels