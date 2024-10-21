import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torchmetrics
import torch.nn.functional as F
import torch
import lightning as L

class MLP(L.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layers, **kwargs):
        super().__init__()

        # Training metrics
        self.train_metrics = torchmetrics.MetricCollection({
            'mse': torchmetrics.MeanSquaredError(),
            'mae': torchmetrics.MeanAbsoluteError()
            }, 
            prefix='train_'
        )
        self.val_metrics = torchmetrics.MetricCollection({
            'mse': torchmetrics.MeanSquaredError(),
            'mae': torchmetrics.MeanAbsoluteError()
            },
            prefix='val_'
        )

        # Optimizer
        self.lr = kwargs.get('lr', 1e-5)
        if 'lr' in kwargs:
            print(f"Using learning rate: {self.lr}")

        # Model
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
        batch_value = self.train_metrics(pred_sim, similarity)
        self.log_dict(batch_value, on_epoch=True)

        avg_pred_mag = torch.mean(torch.abs(pred_sim))
        avg_real_mag = torch.mean(torch.abs(similarity))
        self.log('train_avg_pred_magnitude', avg_pred_mag, on_step=True, on_epoch=True)
        self.log('train_avg_real_magnitude', avg_real_mag, on_step=True, on_epoch=True)


        return loss
    
    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        spectrum_a, spectrum_b, similarity = batch
        embed_1 = self(spectrum_a)
        embed_2 = self(spectrum_b)
        pred_sim = F.cosine_similarity(embed_1, embed_2)

        loss = nn.functional.mse_loss(pred_sim, similarity)
        batch_value = self.val_metrics(pred_sim, similarity)
        self.log_dict(batch_value, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        self.val_metrics.reset()
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)