import nmrglue as ng
import numpy as np
import os

# Your method to load NMR spectra data
file_paths = [f"/Users/huwentao/Desktop/pythonProject/hc/sample/{i}" for i in
              os.listdir('/Users/huwentao/Desktop/pythonProject/hc/sample')]
data_list = [ng.bruker.read_pdata(os.path.join(file_path, '1/pdata/1'))[1] for file_path in file_paths]
data_processed = [(data.flatten() - np.min(data.flatten())) / (np.max(data.flatten()) - np.min(data.flatten())) for data
                  in data_list]
max_length = max([len(data) for data in data_processed])
data_padded = [np.pad(data, (0, max_length - len(data))) for data in data_processed]
data_matrix = np.array(data_padded)


# Function to add Gaussian noise
def add_gaussian_noise(data, mean=0, std_dev=0.05):
    noise = np.random.normal(mean, std_dev, data.shape)
    return data + noise


# Function to generate 1000 noisy versions for each spectra data
def generate_noisy_data(data, num_samples=30):
    noisy_data = []
    for _ in range(num_samples):
        noisy_sample = add_gaussian_noise(data)
        noisy_data.append(noisy_sample)
    return np.array(noisy_data)


# Generate noisy data and labels
all_noisy_data = []
all_labels = []

for idx, data in enumerate(data_matrix):
    noisy_data = generate_noisy_data(data)
    all_noisy_data.append(noisy_data)

    # Generate labels
    labels = np.full((noisy_data.shape[0],), idx)
    all_labels.append(labels)

# Combine all noisy data and labels
all_noisy_data = np.vstack(all_noisy_data)
all_labels = np.hstack(all_labels)

# Save the generated data and labels to local storage
data_save_path = "./dataset/generated_noisy_data.npy"
labels_save_path = "./dataset/generated_labels.npy"

np.save(data_save_path, all_noisy_data)
np.save(labels_save_path, all_labels)


# Function to load the saved data
def load_noisy_data(path=data_save_path):
    return np.load(path)


def load_labels(path=labels_save_path):
    return np.load(path)

# Example usage:
loaded_data = load_noisy_data()
loaded_labels = load_labels()
print(loaded_labels)
print(loaded_data)
