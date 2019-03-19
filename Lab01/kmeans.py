from cluster import cluster
from copy import deepcopy
import numpy as np
from sklearn.datasets.samples_generator import make_blobs 

class KMeansCluster(cluster):

    def __init__(self, k = 5 , max_iterations = 100):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X):
        n = X.shape[0]
        c = X.shape[1]
        mean = np.mean(X)
        std = np.std(X)
        centroids = std * np.random.randn(self.k, c) + mean
        centroids_pre = np.zeros(centroids.shape)
        clusters = np.zeros(n)
        iter_Num = 0
        while iter_Num < self.max_iterations:
            for i in range(n):
                dist_min = np.linalg.norm(X[i] - centroids[0])
                for t in range(1, self.k):
                    dist = np.linalg.norm(X[i] - centroids[t])
                    if dist < dist_min:
                        dist_min = dist
                        clusters[i] = t
            centroids_pre = deepcopy(centroids)
            for k in range(self.k):
                centroids[k] = np.mean(X[clusters == k], axis=0)
            iter_Num += 1
            if np.linalg.norm(centroids - centroids_pre) == 0:
                break
        return clusters, centroids
        
X, cluster_assignments = make_blobs(n_samples=200, centers=4, cluster_std=0.60, random_state=0)
test = KMeansCluster(4)
clusters, centroids = test.fit(X)
print(clusters)
print(centroids)
print(cluster_assignments)