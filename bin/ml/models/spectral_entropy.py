from torch import optim, nn
import torchmetrics
import torch.nn.functional as F
import torch
import lightning as L

def spectral_entropy(spec_a, spec_b):
    

class RawCosine(L.LightningModule):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        if x[0].shape != x[1].shape:
            raise ValueError("Input tensors must have the same shape")
        return spectral_entropy(x[0], x[1])
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        spectrum_a, spectrum_b, similarity, metadata = batch
        preds = self((spectrum_a, spectrum_b))
        if similarity is not None:
            loss = nn.functional.mse_loss(preds, similarity)
        else:
            loss = None

        return {'predictions': preds, 'similarity': similarity, 'loss': loss, 'metadata': metadata}