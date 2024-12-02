import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import pytest

class PadSequence(object):
    """ Pad the input sequence to the target shape with the padding value.

    Args:
        shape (Tuple[int,]): The target shape of the output tensor.
        padding_value (int): The value for the padded elements.

    Returns:
        Tensor: The output tensor with the target shape.

    Example:
        >>> transformer = PadSequence((5,))
        >>> input_tensor = torch.tensor([1, 2, 3])
        >>> transformer(input_tensor)
        tensor([1, 2, 3, 0, 0]) 
    """
    def __init__(self, shape:Tuple[int,], padding_value=0):
        self.shape = shape
        self.padding_value = padding_value

    def __call__(self, vector):
        # Create an output tensor filled with the padding value, with the target shape
        output_vector = torch.full(self.shape, self.padding_value)
        
        # Create slices for each dimension based on the smaller of the output shape and input vector's shape
        slices = tuple(slice(0, min(v, o)) for v, o in zip(vector.shape, self.shape))
        
        # Assign the input vector into the corresponding slice of the output vector
        output_vector[slices] = vector
        
        return output_vector
    
class BinSpectrum(object):
    """ Bin the input spectrum into m/z bins of fixed width. Intensity values within each bin are summed.
    
    Args:
        bin_width (float): The width of each bin.
        min_mz (float): The minimum m/z value.
        max_mz (float): The maximum m/z value.
        
    Returns:
        Tensor: The binned spectrum.

    Example:
        
    """

    def __init__(self, bin_width:float, min_mz:float, max_mz:float):
        self.bin_width = bin_width
        self.min_mz = min_mz
        self.max_mz = max_mz

    def __call__(self, spectrum):
        # Create a list of bin edges from min_mz to max_mz with bin_width
        bins = np.arange(self.min_mz, self.max_mz+self.bin_width, self.bin_width)
        
        # Bin the spectrum
        binned_spectrum = np.histogram(spectrum[:, 0], bins=bins, weights=spectrum[:, 1])[0]
        return binned_spectrum
    
class SquareRootTransform(object):
    """ Performs a square root transformation on the intensities of the spectrum.
    
    Args:
        None
        
    Returns:
        np.ndarray: The transformed spectrum.
    """
    def __init__(self):
        pass

    def __call__(self, spectrum):
        if len(spectrum.shape) == 1:
            spectrum[1] = np.sqrt(spectrum[1])
        elif len(spectrum.shape) == 2:
            spectrum[:, 1] = np.sqrt(spectrum[:, 1])
        else:
            raise ValueError(f"Expected a 1D or 2D array with m/z and intensity values. Instead got {spectrum.shape}")
        return spectrum

class NormalizeIntensity(object):
    """Normalizes intensity values along a specified dimension using the Euclidean norm,
    while preserving other dimensions such as m/z.
    
    Args:
        dim (int): The dimension to normalize (default is -1, the last dimension).
        
    Returns:
        np.ndarray: The array with normalized intensities along the specified dimension.
    """
    
    def __init__(self,):
        pass

    def __call__(self, spectrum):
        # If one dimensional, apply the norm
        if len(spectrum.shape) == 1:
            # Eucliden norm of the intensity values
            norm = np.linalg.norm(spectrum)
            spectrum = spectrum / norm
            return spectrum

        # If two dimensional, apply the norm to the second dimension
        if len(spectrum.shape) == 2:
            # Eucliden norm of the intensity values
            norm = np.linalg.norm(spectrum[:, 1])
            spectrum[:, 1] = spectrum[:, 1] / norm
            return spectrum
        
        raise ValueError(f"Expected a 1D or 2D array with m/z and intensity values. Instead got {spectrum.shape}")
    
class SelectMassRange(object):
    """ Selects the mass range of a spectrum to the specified range. Both endpoints are inclusive.

    Args:
        min_mz (float): The minimum m/z value.
        max_mz (float): The maximum m/z value.

    Returns:
        np.ndarray: The reduced spectrum.
    """

    def __init__(self, min_mz:float, max_mz:float):
        self.min_mz = min_mz
        self.max_mz = max_mz

    def __call__(self, spectrum):
        if len(spectrum.shape) != 2:
            raise ValueError(f"Expected a 2D array with m/z and intensity values. Instead got {spectrum.shape}")
        
        # Filter the spectrum based on the m/z values
        mask = (spectrum[:, 0] >= self.min_mz) & (spectrum[:, 0] <= self.max_mz)
        spectrum = spectrum[mask]
        return spectrum
    
class ExcludeMassRange(object):
    """ Excludes the mass range of a spectrum to the specified range. Both endpoints are inclusive.
    
    Args:
        min_mz (float): The minimum m/z value.
        max_mz (float): The maximum m/z value.
        
    Returns:
        np.ndarray: The reduced spectrum.
    """
    def __init__(self, min_mz:float, max_mz:float):
        self.min_mz = min_mz
        self.max_mz = max_mz

    def __call__(self, spectrum):
        if len(spectrum.shape) != 2:
            raise ValueError(f"Expected a 2D array with m/z and intensity values. Instead got {spectrum.shape}")
        
        # Filter the spectrum based on the m/z values
        mask = (spectrum[:, 0] < self.min_mz) | (spectrum[:, 0] > self.max_mz)
        spectrum = spectrum[mask]
        return spectrum
    
class PadToLength(object):
    """Pads a sequence along a specified dimension to a fixed length.
    
    Args:
        length (int): The target length of the specified dimension.
        dim (int): The dimension to pad (default is _, the last dimension).
        padding_value (float): The value for the padded elements (default is np.nan).
        
    Returns:
        np.ndarray: The padded sequence.
    """
    
    def __init__(self, length: int, dim: int=0, padding_value=np.nan):
        self.length = length
        self.dim = dim
        self.padding_value = padding_value

    def __call__(self, sequence):
        sequence = np.asarray(sequence)  # Ensure input is an array
        dim = self.dim if self.dim >= 0 else sequence.ndim + self.dim
        current_length = sequence.shape[dim]

        if current_length >= self.length:
            slicing = [slice(None)] * sequence.ndim
            slicing[dim] = slice(0, self.length)
            return sequence[tuple(slicing)]

        pad_width = [(0, 0)] * sequence.ndim
        pad_width[dim] = (0, self.length - current_length)

        return np.pad(sequence, pad_width, constant_values=self.padding_value)


@pytest.fixture
def test_spectrum():
    yield np.array([[1, 20], [2, 30], [3, 40], [4, 50], [5, 60]])   # m/z, intensity

def test_bin_spectrum(test_spectrum):
    transformer = BinSpectrum(3, 1, 6)
    output = transformer(test_spectrum)

    assert output.shape == (2,)
    assert output[0] == 20 + 30 + 40, f"First bin should contain the sum of intensities 20, 30, and 40, but got {output[0]}"
    assert output[1] == 50 + 60, f"Second bin should contain the sum of intensities 50 and 60, but got {output[1]}"

def test_reduce_mass_range(test_spectrum):
    transformer = SelectMassRange(2, 4)
    output = transformer(test_spectrum)

    assert output.shape == (3, 2), f"Expected shape (3, 2), but got {output.shape}"
    assert np.all(output[:, 0] >= 2), "All m/z values should be greater than or equal to 2"
    assert np.all(output[:, 0] <= 4), "All m/z values should be less than or equal to 4"

def test_exclude_mass_range(test_spectrum):
    transformer = ExcludeMassRange(2, 4)
    output = transformer(test_spectrum)

    assert output.shape == (2, 2), f"Expected shape (2, 2), but got {output.shape}"
    assert np.all((output[:, 0] < 2) | (output[:, 0] > 4)), "All m/z values should be less than 2 or greater than 4"