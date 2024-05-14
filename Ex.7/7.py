import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

data = pd.read_csv('dataset1.csv')

X = data.values

num_clusters = 2

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
kmeans_silhouette_score = silhouette_score(X, kmeans_labels)

em = GaussianMixture(n_components=num_clusters, random_state=42)
em_labels = em.fit_predict(X)
em_silhouette_score = silhouette_score(X, em_labels)

print("Silhouette Score (k-Means):", kmeans_silhouette_score)
print("Silhouette Score (EM):", em_silhouette_score)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red', label='Centroids')
plt.title('k-Means Clustering')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=em_labels, cmap='viridis')
plt.scatter(em.means_[:, 0], em.means_[:, 1], marker='x', color='red', label='Centroids')
plt.title('EM Clustering')
plt.legend()

plt.show()
