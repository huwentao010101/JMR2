# Preparing the complete code for user
import os
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import nmrglue as ng
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
from sklearn.metrics import pairwise_distances
from Bio import pairwise2
from sklearn.preprocessing import normalize
import concurrent.futures
from sklearn.metrics import classification_report, accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score

# Your data loading and preprocessing code
# (Make sure to change file paths if necessary)

# 1. Read all files

# Function to load the saved data
labels_save_path = "./dataset/generated_labels.npy"
data_save_path= "./dataset/generated_noisy_data.npy"

def load_noisy_data(path=data_save_path):
    return np.load(path)


def load_labels(path=labels_save_path):
    return np.load(path)

# Example usage:
data_matrix = load_noisy_data()
loaded_labels = load_labels()


# file_paths = [f"/Users/huwentao/Desktop/pythonProject/hc/标准品/{i}" for i in os.listdir('/Users/huwentao/Desktop/pythonProject/hc/标准品')]
# data_list = [ng.bruker.read_pdata(os.path.join(file_path, '1/pdata/1'))[1] for file_path in file_paths]

# 2. Data preprocessing
# data_processed = [(data.flatten() - np.min(data.flatten())) / (np.max(data.flatten()) - np.min(data.flatten())) for data in data_list]
# max_length = max([len(data) for data in data_processed])
# data_padded = [np.pad(data, (0, max_length - len(data))) for data in data_processed]
# data_matrix = np.array(data_padded)

# Similarity/Distance computation functions (as provided above)

# Function to compute Cosine Similarity matrix
def cosine_similarity_matrix(data):
    # We will compute cosine distance and then subtract from 1 to get similarity
    return 1 - pairwise_distances(data, metric="cosine")


# Function to compute LCS (Longest Common Subsequence) matrix

def lcs_length(X, Y):
    """Compute the length of the LCS of two sequences."""
    m, n = len(X), len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    return L[m][n]

def lcs_similarity_matrix_with_tqdm(data):
    num_samples = data.shape[0]
    lcs_matrix = np.zeros((num_samples, num_samples))

    for i in tqdm(range(num_samples), desc="Computing LCS similarities"):
        for j in range(num_samples):
            lcs_matrix[i][j] = lcs_length(data[i], data[j])
    return lcs_matrix

# Function to compute DTW (Dynamic Time Warping) matrix

def compute_dtw_distance(pair):
    i, j, data = pair
    distance, _ = fastdtw(data[i], data[j])
    return (i, j, distance)

def dtw_distance_matrix_parallel(data):
    num_samples = data.shape[0]
    dtw_matrix = np.zeros((num_samples, num_samples))

    # Create a list of all pairs of data points
    pairs = [(i, j, data) for i in range(num_samples) for j in range(num_samples)]

    # Using ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, j, distance in tqdm(executor.map(compute_dtw_distance, pairs), total=len(pairs), desc="Computing DTW distances"):
            dtw_matrix[i][j] = distance

    return dtw_matrix

def dtw_distance_matrix_with_tqdm(data):
    num_samples = data.shape[0]
    dtw_matrix = np.zeros((num_samples, num_samples))

    for i in tqdm(range(num_samples), desc="Computing DTW distances"):
        for j in range(num_samples):
            distance, _ = fastdtw(data[i], data[j])
            dtw_matrix[i][j] = distance
    return dtw_matrix


# Function to compute Smith-Waterman matrix
def edit_distance_ratio(vec1, vec2):
    """Compute the Edit Distance Ratio for two vectors."""
    # Calculate the L1 distance between the two vectors
    l1_distance = norm(vec1 - vec2, ord=1)
    max_distance = len(vec1)  # Assuming vectors are of same length
    return 1 - (l1_distance / max_distance)

def edr_matrix(data_matrix):
    """Compute the EDR matrix for the given data matrix."""
    num_samples = len(data_matrix)
    edr_mat = np.zeros((num_samples, num_samples))

    for i in range(num_samples):
        for j in range(num_samples):
            edr_mat[i][j] = edit_distance_ratio(data_matrix[i], data_matrix[j])

    return edr_mat


# Function to compute Needleman-Wunsch matrix
def hausdorff_distance(set1, set2):
    """Compute the Hausdorff distance between two sets of vectors."""

    def min_distance(point, points_set):
        return min(norm(point - other_point) for other_point in points_set)

    forward_distance = max(min_distance(point, set2) for point in set1)
    backward_distance = max(min_distance(point, set1) for point in set2)

    return max(forward_distance, backward_distance)


def hausdorff_distance_matrix(data_matrix):
    """Compute the Hausdorff distance matrix for the given data matrix."""
    num_samples = len(data_matrix)
    hausdorff_mat = np.zeros((num_samples, num_samples))

    for i in tqdm(range(num_samples), desc='Computing Hausdorff distances'):
        for j in range(num_samples):
            hausdorff_mat[i][j] = hausdorff_distance(data_matrix[i], data_matrix[j])

    return hausdorff_mat

def match_labels(true_labels, cluster_labels):
    # Compute the cost matrix
    cost_matrix = -1 * np.array([[np.sum(true_labels == i) - np.sum((cluster_labels == j) & (true_labels == i))
                                  for j in set(cluster_labels)] for i in set(true_labels)])
    # Use the linear sum assignment to find the best match
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    new_cluster_labels = np.zeros_like(cluster_labels)
    for i, j in zip(row_ind, col_ind):
        new_cluster_labels[cluster_labels == j] = i
    return new_cluster_labels
# Extracting the preprocessed data from the provided code
normalized_data = normalize(data_matrix)  # Normalizing data for cosine similarity

# Let's calculate similarity/distance matrices for each of the above methods
cosine_sim_matrix = cosine_similarity_matrix(normalized_data)
# lcs_sim_matrix = lcs_similarity_matrix_with_tqdm(data_matrix)
dtw_dist_matrix = dtw_distance_matrix_parallel(data_matrix)
edr_sim_matrix = edr_matrix(data_matrix)
# hausdorff_dist_matrix = hausdorff_distance_matrix(data_matrix)

# Storing matrices in a dictionary for easier access
matrices = {
    "Cosine Similarity": cosine_sim_matrix,
    # "LCS": lcs_sim_matrix,
    "DTW": dtw_dist_matrix,
    "EDR":edr_sim_matrix
    # "HAU":hausdorff_dist_matrix
    # "Needleman-Wunsch": nw_sim_matrix
}

matrices.keys()
# Clustering and evaluation

n_clusters = 3  # You can adjust this value if needed
results = {}

for method, matrix in matrices.items():
    # Using distance matrix for clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(matrix)
    np.fill_diagonal(matrix, 0)
    labels = match_labels(loaded_labels, labels)
    # Compute silhouette score
    matrix[matrix < 0] = 0
    silhouette_avg = silhouette_score(matrix, labels, metric="precomputed")
    # If ground truth labels are available, compute NMI
    ground_truth_labels = loaded_labels  # Load or define your ground truth labels here

    nmi = normalized_mutual_info_score(ground_truth_labels, labels)
    f1 = f1_score(ground_truth_labels, labels, average='macro')

    results[method] = {
        "Silhouette Score": silhouette_avg,
        "NMI": nmi,  # Uncomment this if ground truth labels are available
        'accuracy': accuracy_score(loaded_labels, labels),
        'f1-score':f1
    }

# Print results
for method, scores in results.items():
    print(f"Method: {method}")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")
    print()




# Assuming `clustered_labels` is the variable containing labels assigned by your clustering algorithm
# and `loaded_labels` is the variable containing the true labels

# Match the cluster labels with true labels


# Print the classification report
a = classification_report(loaded_labels, labels)
print(a)

# Print the accuracy
print("Accuracy:", accuracy_score(loaded_labels, labels))


