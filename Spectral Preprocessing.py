import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from pywt import wavedec, waverec

# Let's assume `S` is an array containing spectral data from all experiments
# E = [S1, S2, ..., SL]

def wavelet_transform(S):
    """
    Apply a wavelet transform to decompose the data into wavelet space.
    Using 'db1' wavelet and one level of decomposition for this example.
    """
    wavelet_coeffs = wavedec(S, 'db1', level=1)
    return wavelet_coeffs

def threshold_wavelet_coeffs(wavelet_coeffs, threshold):
    """
    Perform thresholding on the wavelet coefficients to suppress noise.
    """
    coeffs = wavelet_coeffs
    coeffs[wavelet_coeffs < threshold] = 0
    return coeffs

def polynomial_baseline_model(S, degree):
    """
    Model the baseline drift using a polynomial function.
    """
    x = np.arange(len(S))
    coeffs = np.polyfit(x, S, degree)
    return np.poly1d(coeffs)

def subtract_baseline(S, model):
    """
    Subtract the baseline model from the noise-reduced spectral data
    to correct variations in the baseline.
    """
    return S - model(np.arange(len(S)))

def normalize_spectrum(S):
    """
    Normalize the spectrum by dividing each point by the maximum intensity
    in the corrected spectrum.
    """
    return S / np.max(S)

def align_spectrum(S, alignment_values):
    """
    Align the normalized spectrum to match peak frequencies using interpolation.
    """
    spline = UnivariateSpline(np.arange(len(S)), S, k=1)
    return spline(np.arange(len(S)) + alignment_values)

# Preprocess a single experiment's spectral data
def preprocess_spectrum(S, wavelet_threshold, baseline_degree, alignment_values):
    """
    Preprocess the spectral data from a single experiment following the steps:
    1. Wavelet transform
    2. Thresholding of wavelet coefficients
    3. Baseline correction
    4. Normalization
    5. Alignment
    """
    # 1. Wavelet transform
    wavelet_coeffs = wavelet_transform(S)
    # 2. Threshold the wavelet coefficients
    denoised_coeffs = threshold_wavelet_coeffs(wavelet_coeffs, wavelet_threshold)
    # Reconstruct the denoised spectrum from the thresholded coefficients
    denoised_S = waverec(denoised_coeffs, 'db1')
    # 3. Correct for baseline drift
    baseline_model = polynomial_baseline_model(denoised_S, baseline_degree)
    corrected_S = subtract_baseline(denoised_S, baseline_model)
    # 4. Normalize the spectrum
    normalized_S = normalize_spectrum(corrected_S)
    # 5. Align the spectrum
    aligned_S = align_spectrum(normalized_S, alignment_values)
    
    return aligned_S

# Example: Preprocess the spectral data from the first experiment
# Assume wavelet_threshold, baseline_degree, and alignment_values are defined
preprocessed_S = preprocess_spectrum(S[0], wavelet_threshold, baseline_degree, alignment_values)

# Plot the original and preprocessed spectral data
plt.figure(figsize=(10, 5))
plt.plot(S[0], label='Original Spectrum')
plt.plot(preprocessed_S, label='Preprocessed Spectrum')
plt.legend()
plt.show()
