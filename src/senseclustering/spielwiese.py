from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pylab as plt

iris = load_iris()
X = iris["data"]
X2 = np.array([row / np.linalg.norm(row) for row in X])
y = iris["target"]
c = linkage(X2, 'cosine')

plt.figure()
dendrogram(c)
plt.show()

