from torch import optim, nn
import torch.nn.utils as utils
import torchmetrics
import torch.nn.functional as F
import torch
import lightning as L
import logging
from .maldi_transformer import SinusoidalPositionalEncoding
from .CLIP_MALDI import CLIP_MALDI



class CLIP_MALDI_Classifier(CLIP_MALDI):
    def __init__(self, hyperparameters, pretrained_embedder=None):
        super().__init__(hyperparameters, pretrained_embedder)
        self.output_dim = self.hparams.n_classes
        self.is_classifier = True

        self.projection = nn.Linear(128, self.output_dim)

        if self.output_dim == 2:
            self.task = 'binary'
        elif self.output_dim > 2:
            self.task = 'multiclass'
        
    def forward(self, x):
        x = self.embedder(x)
        x = self.projection(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x, _, _ = self.embedder(x)
        x = self.projection(x)
        loss = F.cross_entropy(x, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        acc = torchmetrics.functional.accuracy(x, y, self.task, num_classes=self.output_dim)
        self.log('train_acc', acc, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, _, _ = self.embedder(x)
        x = self.projection(x)
        loss = F.cross_entropy(x, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        preds = torch.argmax(x, dim=1)
        acc = torchmetrics.functional.accuracy(preds, y, self.task, num_classes=self.output_dim)
        self.log('val_acc', acc, on_step=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            x = self.embedder(x)
            x = self.projection(x)
            loss = F.cross_entropy(x, y)
            self.log('test_loss', loss, on_step=True, on_epoch=True)
            preds = torch.argmax(x, dim=1)
            acc = torchmetrics.functional.accuracy(preds, y, self.task, num_classes=self.output_dim)
            self.log('test_acc', acc, on_step=True, on_epoch=True)
            return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        with torch.no_grad():
            x = batch
            x = self.embedder(x)[0]
            x = self.projection(x)
            preds = torch.argmax(x, dim=1)
            return preds
        
    def embed_step(self, batch):
        with torch.no_grad():
            x = batch
            x, _, _ = self.embedder(x)
            return x