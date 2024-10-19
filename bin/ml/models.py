import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import lightning as L

class MLP(L.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layers):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def training_step(self, batch, batch_idx):
        spectrum_a, spectrum_b, similarity = batch
        embed_1 = self(spectrum_a)
        embed_2 = self(spectrum_b)
        pred_sim = F.cosine_similarity(embed_1, embed_2)

        loss = nn.functional.mse_loss(pred_sim, similarity)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters())