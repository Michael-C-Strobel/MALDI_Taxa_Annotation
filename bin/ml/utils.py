import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys

def mirror_plot_1d(spectrum_a, spectrum_b, output_path, tolerance, **kwargs):
    fig = plt.figure(figsize=(10, 6))

    spectrum_a = np.array(spectrum_a)
    spectrum_b = np.array(spectrum_b)
    
    max_nonzero_idx = max(np.max(np.nonzero(spectrum_a)), np.max(np.nonzero(spectrum_b)))
    
    # Highlight matched peaks within a tolerance
    matched_indices = np.where(np.abs(spectrum_a - spectrum_b) < tolerance)[0]
    
    # Plot unmatched peaks
    unmatched_a = np.setdiff1d(np.arange(len(spectrum_a)), matched_indices)
    unmatched_b = np.setdiff1d(np.arange(len(spectrum_b)), matched_indices)
    
    plt.stem(unmatched_a, spectrum_a[unmatched_a], linefmt='b-', markerfmt=' ', basefmt=' ')  # spectrum_a unmatched
    plt.stem(unmatched_b, -spectrum_b[unmatched_b], linefmt='r-', markerfmt=' ', basefmt=' ')  # spectrum_b unmatched

    # Plot matched peaks in green
    plt.stem(matched_indices, spectrum_a[matched_indices], linefmt='g-', markerfmt=' ', basefmt=' ')
    plt.stem(matched_indices, -spectrum_b[matched_indices], linefmt='g-', markerfmt=' ', basefmt=' ')

    title = kwargs.get('title')
    if title:
        plt.title(title)

    x_label = kwargs.get('x_label')
    if x_label:
        plt.xlabel(x_label)

    y_label = kwargs.get('y_label')
    if y_label:
        plt.ylabel(y_label)

    top_label = kwargs.get('top_label')
    if top_label:
        plt.text(max_nonzero_idx-200, 0.8, top_label, fontsize=12, verticalalignment='center')
    
    bottom_label = kwargs.get('bottom_label')
    if bottom_label:
        plt.text(max_nonzero_idx-200, -0.8, bottom_label, fontsize=12, verticalalignment='center')
        
    plt.xlim(-50, max_nonzero_idx)
    plt.ylim(-1, 1)
    plt.savefig(output_path)
    plt.close(fig)
        
def mirror_plot_2d(spectrum_a, spectrum_b, output_path, tolerance, **kwargs):
    fig = plt.figure(figsize=(10, 6))

    spectrum_a = np.array(spectrum_a)
    spectrum_b = np.array(spectrum_b)

    # Remove all instances of 0 m/z peaks, these are padding
    spectrum_a = spectrum_a[spectrum_a[:, 0] != -1]
    spectrum_b = spectrum_b[spectrum_b[:, 0] != -1]
    
    max_mz = max(np.max(spectrum_a[:, 0]), np.max(spectrum_b[:, 0])) + 200
    min_mz = min(np.min(spectrum_a[:, 0]), np.min(spectrum_b[:, 0]))
    min_mz = max(0, min_mz-200)

    # Highlight matched peaks within a tolerance. Note: This allows peaks to be matched multiple times!
    matched_indices = np.abs(np.subtract.outer(spectrum_a[:, 0], spectrum_b[:, 0])) < tolerance
    matched_a = np.where(np.any(matched_indices, axis=1))[0]
    matched_b = np.where(np.any(matched_indices, axis=0))[0]

    # Plot unmatched peaks
    unmatched_a = np.setdiff1d(np.arange(len(spectrum_a)), matched_a)
    unmatched_b = np.setdiff1d(np.arange(len(spectrum_b)), matched_b)

    if len(unmatched_a) > 0:
        plt.stem(spectrum_a[unmatched_a, 0], spectrum_a[unmatched_a, 1], linefmt='b-', markerfmt=' ', basefmt=' ')  # spectrum_a unmatched
    if len(unmatched_b) > 0:
        plt.stem(spectrum_b[unmatched_b, 0], -spectrum_b[unmatched_b, 1], linefmt='r-', markerfmt=' ', basefmt=' ')  # spectrum_b unmatched

    # Plot matched peaks in green
    if len(matched_a) > 0:
        plt.stem(spectrum_a[matched_a, 0], spectrum_a[matched_a, 1], linefmt='g-', markerfmt=' ', basefmt=' ')
    if len(matched_b) > 0:
        plt.stem(spectrum_b[matched_b, 0], -spectrum_b[matched_b, 1], linefmt='g-', markerfmt=' ', basefmt=' ')

    title = kwargs.get('title')
    if title:
        plt.title(title)

    x_label = kwargs.get('x_label')
    if x_label:
        plt.xlabel(x_label)

    y_label = kwargs.get('y_label')
    if y_label:
        plt.ylabel(y_label)

    top_label = kwargs.get('top_label')
    if top_label:
        plt.text(max_mz-200, 0.8, top_label, fontsize=12, verticalalignment='center')

    bottom_label = kwargs.get('bottom_label')
    if bottom_label:
        plt.text(max_mz-200, -0.8, bottom_label, fontsize=12, verticalalignment='center')

    plt.xlim(min_mz, max_mz)
    plt.ylim(-1, 1)
    plt.savefig(output_path)
    plt.close(fig)

def mirror_plot(spectrum_a: np.array, spectrum_b: np.array, output_path: Path, tolerance: float = 0.01, **kwargs):
    
    # Squeeze both
    spectrum_a = np.squeeze(spectrum_a)
    spectrum_b = np.squeeze(spectrum_b)

    if len(spectrum_a.shape) != len(spectrum_b.shape):
        raise ValueError("Spectrum A and Spectrum B must have the same number of dimensions.")
    
    if len(spectrum_a.shape) == 1:
        mirror_plot_1d(spectrum_a, spectrum_b, output_path, tolerance, **kwargs)

    elif len(spectrum_a.shape) == 2:
        mirror_plot_2d(spectrum_a, spectrum_b, output_path, tolerance, **kwargs)

    else:
        raise ValueError("Spectrum A and Spectrum B must have 1 or 2 dimensions.")
    
def shannon_entropy(spectrum: np.array, padding_value:float=-1, bin_size:float=10.0)->float:
    """Calculate the Shannon entropy of a spectrum.
    
    Args:
        spectrum (np.array): A 2D array with m/z and intensity values.
        
    Returns:
        float: The Shannon entropy of the spectrum.
    """
    # if len(spectrum.shape) != 2:
    #     raise ValueError("Expected a 2D array with m/z and intensity values.")
    
    # if spectrum.shape[1] != 2:
    #     raise ValueError("Expected a 2D array with two columns (m/z and intensity).")
    
    spectrum = np.array(spectrum)

    if len(spectrum.shape) == 1 or min(spectrum[:, 1]) == 0:
        print("Spectrum is likely already binned. Entropy calculation will be incorrect.", file=sys.stderr)
        binned_spectrum = spectrum[:]
    else:
        spectrum = spectrum[spectrum[:, 0] != padding_value] # Remove padding

        # Bin spectra 
        bins = np.arange(min(spectrum[:, 0]), max(spectrum[:, 0]), bin_size)
        binned_spectrum = np.histogram(spectrum[:, 0], bins=bins, weights=spectrum[:, 1])[0]

    # Add a small value to avoid log(0)
    binned_spectrum += 1e-12
    
    # Normalize the spectrum
    binned_spectrum = binned_spectrum / np.sum(binned_spectrum)

    # Calculate the Shannon entropy
    entropy = -np.sum(binned_spectrum * np.log(binned_spectrum))

    normalized_entropy = entropy / np.log(len(binned_spectrum))
    
    return normalized_entropy.item()