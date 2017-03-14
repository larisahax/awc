import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from awc.awc import AWC
from itertools import cycle
from sklearn import metrics

def draw(X, labels, name):
    n_clusters = len(set(labels))
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters), colors):
        class_members = labels == k
        plt.plot(X[class_members, 0], X[class_members, 1], col + 'o')
    
    plt.title(name)
    plt.show()
    
def run_iris():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    lambda_interval = np.linspace(0., 1., 11)
    AWC_object = AWC(speed=1.)
    # To tune parameter \lambda, plot sum of the weights for \lambda 's from some interval 
    #and take a value at the end of plateau or before huge jump.
    AWC_object.plot_sum_of_weights(X, lambda_interval)
    l = 0.6
    AWC_object.awc(X, l)
    clusters = AWC_object.get_clusters()
    labels = AWC_object.get_labels()
    draw(X, labels, 'Iris')
    
    print('Estimated number of clusters: %d' % len(set(labels)))
    print('cluster sizes: '),
    for c in clusters:
        print len(c),
    print("\nV-measure: %0.3f" % metrics.v_measure_score(y, labels))
    
    
if __name__ == '__main__':
    run_iris()