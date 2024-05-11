from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Assuming `reduced_data` is a NumPy array containing the preprocessed, feature-extracted,
# and dimensionally reduced data for all spectra, where each row corresponds to one spectrum.

# Determine the number of clusters (k) using the elbow method
def determine_k(reduced_data, max_k=10):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(reduced_data)
        inertias.append(kmeans.inertia_)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.show()
    # You may add more sophisticated "elbow" detection logic here

# Call the function to plot the elbow curve
determine_k(reduced_data)

# Now choose k, either using the elbow method or other domain-specific heuristic
k = 3  # Replace this with the chosen k value

# Perform K-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(reduced_data)

# Evaluate clustering performance using silhouette score
silhouette_avg = silhouette_score(reduced_data, clusters)
print("Silhouette Score: ", silhouette_avg)

# Evaluate clustering performance using inertia
inertia = kmeans.inertia_
print("Inertia: ", inertia)

# Visualize the clusters in 2D
def visualize_clusters(data, clusters, centers):
    if data.shape[1] > 2:
        # Reduce dimensions for visualization purposes
        pca = PCA(n_components=2)
        reduced_data_vis = pca.fit_transform(data)
        print("Data reduced to 2D for visualization using PCA")
    else:
        reduced_data_vis = data
    
    plt.scatter(reduced_data_vis[:, 0], reduced_data_vis[:, 1], c=clusters, cmap='viridis', label='Data Points')
    plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', label='Centroids')
    plt.title('K-means Clustering')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.show()

# Visualize the clusters and centroids
visualize_clusters(reduced_data, clusters, kmeans.cluster_centers_)
