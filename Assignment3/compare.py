import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

#function for plotting
def plot_clusters(X, labels_list, titles, centers_list=None, figsize=(18, 5)):
    n = len(labels_list)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for i, (labels, title) in enumerate(zip(labels_list, titles)):
        scatter = axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=40)
        if centers_list and centers_list[i] is not None:
            centers = centers_list[i]
            axes[i].scatter(
                centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X'
            )
        axes[i].set_title(title)
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
    plt.tight_layout()
    plt.show()

# Dataset 1: make_moons
X1, y1 = make_moons(n_samples=300, noise=0.05, random_state=42)

# k-Means
kmeans1 = KMeans(n_clusters=2, random_state=42)
labels_kmeans1 = kmeans1.fit_predict(X1)
centers_kmeans1 = kmeans1.cluster_centers_

# Agglomerative
agg1 = AgglomerativeClustering(n_clusters=2)
labels_agg1 = agg1.fit_predict(X1)

# DBSCAN
dbscan1 = DBSCAN(eps=0.3, min_samples=5)
labels_dbscan1 = dbscan1.fit_predict(X1)

# Plot
plot_clusters(
    X1,
    [labels_kmeans1, labels_agg1, labels_dbscan1],
    [
        f'k-Means (Silhouette: {silhouette_score(X1, labels_kmeans1):.2f})',
        f'Agglomerative (Silhouette: {silhouette_score(X1, labels_agg1):.2f})',
        f'DBSCAN (Silhouette: {silhouette_score(X1, labels_dbscan1):.2f})',
    ],
    centers_list=[centers_kmeans1, None, None],
)

# Dataset 2: make_blobs
X2, y2 = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=[1.0, 2.5, 0.5],
    random_state=42,
)

# k-Means
kmeans2 = KMeans(n_clusters=3, random_state=42)
labels_kmeans2 = kmeans2.fit_predict(X2)
centers_kmeans2 = kmeans2.cluster_centers_

# Agglomerative
agg2 = AgglomerativeClustering(n_clusters=3)
labels_agg2 = agg2.fit_predict(X2)

# DBSCAN (try to find reasonable eps for varying densities)
dbscan2 = DBSCAN(eps=0.8, min_samples=5)
labels_dbscan2 = dbscan2.fit_predict(X2)

# Plot
plot_clusters(
    X2,
    [labels_kmeans2, labels_agg2, labels_dbscan2],
    [
        f'k-Means (Silhouette: {silhouette_score(X2, labels_kmeans2):.2f})',
        f'Agglomerative (Silhouette: {silhouette_score(X2, labels_agg2):.2f})',
        f'DBSCAN (Silhouette: {silhouette_score(X2, labels_dbscan2):.2f})',
    ],
    centers_list=[centers_kmeans2, None, None],
)


# Dataset 3: make_circles
X3, y3 = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)

# k-Means
kmeans3 = KMeans(n_clusters=2, random_state=42)
labels_kmeans3 = kmeans3.fit_predict(X3)
centers_kmeans3 = kmeans3.cluster_centers_

# Agglomerative
agg3 = AgglomerativeClustering(n_clusters=2)
labels_agg3 = agg3.fit_predict(X3)

# DBSCAN
dbscan3 = DBSCAN(eps=0.2, min_samples=5)
labels_dbscan3 = dbscan3.fit_predict(X3)

# Plot
plot_clusters(
    X3,
    [labels_kmeans3, labels_agg3, labels_dbscan3],
    [
        f'k-Means (Silhouette: {silhouette_score(X3, labels_kmeans3):.2f})',
        f'Agglomerative (Silhouette: {silhouette_score(X3, labels_agg3):.2f})',
        f'DBSCAN (Silhouette: {silhouette_score(X3, labels_dbscan3):.2f})',
    ],
    centers_list=[centers_kmeans3, None, None],
)