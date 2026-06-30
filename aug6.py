import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

x,ytrue = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# cluster std = how spread out the clusters are
kmeans = KMeans(n_clusters=4)
kmeans.fit(x)

ymeans = kmeans.predict(x)
plt.scatter(x[:, 0], x[:, 1], c=ymeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')	
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()