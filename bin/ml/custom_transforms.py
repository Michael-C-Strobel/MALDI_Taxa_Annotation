import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


# Pytorch Transform that pads vectors to fixed length
class PadSequence(object):
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