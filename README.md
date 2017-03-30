# awc
Description 
-----------

Python implementation of Adaptive Weights Clustering algorithm.

AWC is a novel non-parametric clustering technique based on adaptive weights. Weights are recovered using an iterative procedure based on statistical test of "no gap". Method does not require specifying number of clusters. It applies equally well to clusters of convex structure and different density. The procedure is numerically feasible and applicable for large high dimensional datasets. The paper with detailed description of the algorithm will be available soon.

Installation
-----------

Build the latest development version from source:

    git clone https://github.com/larisahax/awc.git
    cd awc
    python setup.py install

Requirements
-----

numpy  
scipy  
matplotlib

Usage
------

AWC class provides the API for performing clustering. 

```python
AWC_object = AWC(n_neigh=-1, effective_dim=-1, n_outliers=0, discrete=False, speed=1.5)
```

        n_neigh : int, optional, default: -1
            The number of closest neighbors to be connected on the initialization step.
            If not specified, n_neigh = max(6, min(2 * effective_dim + 2, 0.1 * n_samples))

        effective_dim : int, optional, default: -1
            Effective dimension of data X. 
            If not specified, effective_dim = true dimension of the data, 
            in case the true dimension is less than 7, otherwise 2.

        n_outliers : int, optional, default: 0
            Minimum number of points each cluster must contain.
            Points from clusters with smaller size will be connected to the closest cluster.

        discrete : boolean, optional, default: False
            Specifies if data X consists of only discrete values.

        speed : int, optional, default: 1.5
            Controls the number of iterations.
            Increase of the speed parameter decreases the number of steps.
            
```python
AWC_object.awc(l, X, dmatrix=None)
```

        l : int
            The lambda parameter.
        
        X : array, shape (n_samples, n_features), optional, default: None 
            Input data. 
            awc works with distance matrix. 
            User must specify X or dmatrix.
            From X Euclidean distance matrix is computed and passed to awc.
            No need to specify X, if dmatrix is specified. 
        
        dmatrix : array, shape (n_samples, n_n_samples), optional, default: None
            Distance matrix.

Cluster structure found by AWC
```python 
clusters = AWC_object.get_clusters()
```
Cluster labels
```python 
labels = AWC_object.get_labels()
```
 To tune the parameter \lambda, plot sum of the weights for \lambda 's from some interval and take a value at the end of plateau or before a huge jump.
 ```python
 AWC_object.plot_sum_of_weights(lambda_interval, X=None, dmatrix=None)
 ```
    
      lambda_interval : list
            Lambda parameters for which sum of weights will be computed.
            
        X : array, shape (n_samples, n_features), optional, default: None 
            Input data. 
            No need to specify X, if dmatrix is specified. 
        
        dmatrix : array, shape (n_samples, n_n_samples), optional, default: None
            Distance matrix. If not specified, the Euclidean distance matrix is used from X.  

Example
------

See example.py for an example running awc on iris dataset.
