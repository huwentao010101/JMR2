
import numpy as np
import nmrglue as ng
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tensorflow as tf
from Attention_Based_BiLSTM_Autoencoder import train_model
from Feature_Extraction import extract_features
from Cluster import cluster_data
from compare import compare_results

# 1. Data Generation
def generate_data(num_samples=1000, dimensions=128):
    file_paths = [f"/Users/huwentao/Desktop/pythonProject/hc/sample/{i}" for i in
                  os.listdir('/Users/huwentao/Desktop/pythonProject/hc/sample')]
    data_list = [ng.bruker.read_pdata(os.path.join(file_path, '1/pdata/1'))[1] for file_path in file_paths]
    data_processed = [(data.flatten() - np.min(data.flatten())) / (np.max(data.flatten()) - np.min(data.flatten())) for data in data_list]
    return np.array(data_processed[:num_samples])  # Limiting to num_samples

# Generate synthetic data
data = generate_data(num_samples=1000, dimensions=128)

# 2. Preprocessing (None in this case)
# data = preprocess_data(data)  # This line is commented out as preprocessing is not required.

# 3. Feature Extraction
features = extract_features(data)

# 4. Model Training
embedding_dim = 64
lstm_units = 64
sequence_length = 100
epochs = 32
batch_size = 8
model = train_model(features, embedding_dim=embedding_dim, lstm_units=lstm_units, sequence_length=sequence_length, epochs=epochs, batch_size=batch_size)

# 5. Clustering
num_clusters = 18
clusters = cluster_data(features, num_clusters=num_clusters)

# 6. Comparison of Results
comparison_metrics = compare_results(clusters, model)

# Output results
print("Clustering and model training completed.")
print("Comparison metrics:", comparison_metrics)
