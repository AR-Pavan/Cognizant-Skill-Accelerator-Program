import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load the Iris dataset
iris_data = load_iris()
df = pd.DataFrame(iris_data.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['actual_class'] = iris_data.target  # True class labels (for reference)

# Normalize the features to ensure fair distance calculations
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(columns=['actual_class']))

# Using the Elbow Method to find the optimal number of clusters for K-Means
inertia_values = []
cluster_range = range(1, 11)

for num_clusters in cluster_range:
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans_model.fit(scaled_features)
    inertia_values.append(kmeans_model.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, inertia_values, marker='o', linestyle='-')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia (Within-cluster sum of squares)")
plt.title("Elbow Method to Determine Optimal Clusters")
plt.show()

# Train K-Means with 3 clusters (based on the elbow method)
kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
df['kmeans_labels'] = kmeans_model.fit_predict(scaled_features)

# Perform Hierarchical Clustering and plot the Dendrogram
plt.figure(figsize=(10, 5))
linkage_matrix = linkage(scaled_features, method='ward')
dendrogram(linkage_matrix, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title("Dendrogram for Hierarchical Clustering")
plt.show()

# Train the Hierarchical Clustering model
hierarchical_model = AgglomerativeClustering(n_clusters=3)
df['hierarchical_labels'] = hierarchical_model.fit_predict(scaled_features)

# Calculate evaluation metrics
silhouette_kmeans = silhouette_score(scaled_features, df['kmeans_labels'])
silhouette_hierarchical = silhouette_score(scaled_features, df['hierarchical_labels'])
ari_kmeans = adjusted_rand_score(df['actual_class'], df['kmeans_labels'])
ari_hierarchical = adjusted_rand_score(df['actual_class'], df['hierarchical_labels'])

print(f"K-Means Silhouette Score: {silhouette_kmeans:.2f}")
print(f"Hierarchical Clustering Silhouette Score: {silhouette_hierarchical:.2f}")
print(f"K-Means Adjusted Rand Index (ARI): {ari_kmeans:.2f}")
print(f"Hierarchical Clustering Adjusted Rand Index (ARI): {ari_hierarchical:.2f}")

# Apply PCA for visualization of clusters
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)

# Plot the clusters identified by K-Means
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=df['kmeans_labels'], palette='viridis')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering (PCA View)")

# Plot the clusters identified by Hierarchical Clustering
plt.subplot(1, 2, 2)
sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=df['hierarchical_labels'], palette='viridis')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Hierarchical Clustering (PCA View)")

plt.show()
