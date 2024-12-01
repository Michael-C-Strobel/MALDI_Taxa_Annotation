import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def mirror_plot(spectrum_a: np.array, spectrum_b: np.array, output_path: Path, tolerance: float = 0.01, **kwargs):
    fig = plt.figure(figsize=(10, 6))

    spectrum_a = np.array(spectrum_a)
    spectrum_b = np.array(spectrum_b)
    
    max_nonzero_idx = max(np.max(np.nonzero(spectrum_a)), np.max(np.nonzero(spectrum_b)))
    max_non_zero_idx = min(max(len(spectrum_a), len(spectrum_b)), max_nonzero_idx+50)
    
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
        
    plt.xlim(-50, max_nonzero_idx)
    plt.ylim(-1, 1)
    plt.savefig(output_path)
    plt.close(fig)
        