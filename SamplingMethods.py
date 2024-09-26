import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

#### Information about Methods chosen for Experiments:

# We have chosen deterministic methods, in which it is knwon how much instances are retained at the end,
# since we pass a desired subset size to the methods and not use methods that terminate based on a specific 
#stopping criterion.




# Implementation of our Methods scikit-learn conform

## Random Sampling Techniques
### - Stratified Random Sampler

class StratifiedRandomSampling(BaseEstimator, TransformerMixin):
    
    def __init__(self, sample_size, random_state=0):
        self.size = sample_size
        self.random_state = random_state
    
    def fit(self, X, y=None):
        return self
    
    def fit_transform(self, X, y=None):
        _, samples, _, sample_targets = train_test_split(X, y,
                                                         test_size=self.size,
                                                         stratify=y,
                                                         random_state=self.random_state)
        return samples, sample_targets

# - Clustering Sampling

class ClusteringSampling(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    
    def fit_transform(self):
        pass



## Clustering-based IS Techniques
### - Cluster Centroid Selection (CLC) [1]

class ClusterNearestCentroidSelection(BaseEstimator, TransformerMixin):
   # This version selects the instances nearest to the centroids and not the centroids (prototypes) themselves!
   def __init__(self):
      pass
      #super().__init__()

class ClusterCentroidsSelection(BaseEstimator, TransformerMixin):

  def __init__(self, sample_size, random_state):
    self.ssize = sample_size
    self.random_state = random_state

  def fit(self, X, y=None):
    return self

  def fit_transform(self, X, y):
    samples_per_class = self._det_clusters_per_class(y)

    X_ = np.empty(shape=(0, X.shape[1]), dtype=X.dtype)
    y_ = np.empty(shape=(0), dtype=np.int8)

    for class_, ssize_ in zip(np.unique(y), samples_per_class):
      # get current class
      X_k = X[y==class_]

      # cluster current class with respective sample size (ssize) as num of clusters
      kmeans = KMeans(n_clusters=ssize_,
                      init="k-means++",
                      n_init="auto",
                      max_iter=300,
                      algorithm="lloyd",
                      random_state=42)
      kmeans.fit(X_k)

      # add cluster centers to reduced dataset and label accordingly
      cluster_representatives = kmeans.cluster_centers_
      X_ = np.vstack((X_, cluster_representatives))
      y_ = np.hstack([y_, np.ones(len(cluster_representatives)) * class_])

    return X_, y_

  def _det_clusters_per_class(self, y):
    num_classes = np.unique(y, return_counts=True)[1]
    total_samples = sum(num_classes)

    # Check ob desired size < total samples
    assert self.ssize < total_samples, "Desired sample size has to be less than total samples in y."

    # Abrundung der sample size per class
    samples_per_class = np.floor(self.ssize * num_classes / total_samples)
    samples_per_class = samples_per_class.astype("int")

    return samples_per_class


## References
# [1] V. Toscano-Durán, J. Perera-Lago, E. Paluzo-Hidalgo, R. Gonzalez-Diaz, M. Á. Gutierrez-Naranjo, und M. Rucco, An In-Depth Analysis of Data Reduction Methods for Sustainable Deep Learning. 2024.
