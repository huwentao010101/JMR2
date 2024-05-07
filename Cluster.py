from sklearn.cluster import KMeans
import numpy as np

# Assuming `reduced_data` is a NumPy array containing the preprocessed, feature-extracted,
# and dimensionally reduced data for all spectra, where each row corresponds to one spectrum.

# Number of clusters (k) should be chosen based on domain knowledge or by using methods like the elbow method.
k = 3  # This is just an example; the actual number of clusters should be determined appropriately.

# Perform K-means clustering
kmeans = KMeans(n_clusters=k, random_state=0)  # You can set a random_state for reproducibility
clusters = kmeans.fit_predict(reduced_data)

# Now `clusters` is an array where each element represents the cluster index that the corresponding spectrum belongs to.

# If you want to evaluate the clustering performance, you can use various metrics such as silhouette score
from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(reduced_data, clusters)
print("Silhouette Score: ", silhouette_avg)

# Optionally, you can also use the inertia (within-cluster sum-of-squares) to evaluate the clustering quality
inertia = kmeans.inertia_
print("Inertia: ", inertia)

# You can also inspect the cluster centers
cluster_centers = kmeans.cluster_centers_
print("Cluster Centers: \n", cluster_centers)

# If you want to visualize the clusters, you can plot the data points with different colors for different clusters
# (This example assumes the data is 2D for visualization purposes)
import matplotlib.pyplot as plt

if reduced_data.shape[1] == 2:  # Check if the data is 2-dimensional for easy visualization
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
