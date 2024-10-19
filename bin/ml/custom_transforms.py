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
        padding = tuple((torch.Tensor(list(self.shape)).to(int) - torch.Tensor(list(vector.shape)).to(int)).tolist())
        return F.pad(vector, padding, mode='constant', value=self.padding_value)