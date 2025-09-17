from torch import optim, nn
import torch.nn.utils as utils
import torchmetrics
import torch.nn.functional as F
import torch
import lightning as L
import logging
from .maldi_transformer import SinusoidalPositionalEncoding
from .CLIP_MALDI import CLIP_MALDI, clip_contrastive_loss



class Transformer_MulitLoss(CLIP_MALDI):
    def __init__(self, hyperparameters, pretrained_embedder=None):
        super().__init__(hyperparameters, pretrained_embedder)
        self.output_dim = self.hparams.n_classes
        self.is_classifier = True

        self.projection_embed = nn.Linear(128, self.output_dim)
        self.projection_cls = nn.Linear(128, self.output_dim)
        self.is_classifier = True
        self.is_embedder = True

        if self.output_dim == 2:
            self.task = 'binary'
        elif self.output_dim > 2:
            self.task = 'multiclass'
        
    def forward(self, anchors, queries):
        anchors_embeds, achor_raw_embeds, _ = self.embedder(anchors)
        q_embeds, q_raw_embeds, _ = self.embedder(queries)
    
        anchors_latent = self.projection_embed(anchors_embeds)
        q_latent = self.projection_embed(q_embeds)
    
        anchor_class_preds = self.projection_cls(anchors_embeds)
        query_class_preds = self.projection_cls(q_embeds)
    
        return (anchors_latent, q_latent), (anchor_class_preds, query_class_preds)
    
    def training_step(self, batch, batch_idx):
        spectra = batch[0]
        bare_classes = batch[1][0]['class']
        anchor_class = batch[1][0]['class_as_int']
        anchors = spectra[0]
        queries = spectra[1]
        (anchor_embeds, query_embeds), (anchor_class_preds, query_class_preds) = self.forward(anchors, queries)
        embedding_loss = clip_contrastive_loss(anchor_embeds, query_embeds, bare_classes, temperature=self.tau)
        
        classification_loss = F.cross_entropy(anchor_class_preds, anchor_class)

        loss = embedding_loss + 0.05 * classification_loss 

        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Only do embedding loss
        spectra = batch[0]
        anchor_class = batch[1][0]['class']
        anchors = spectra[0]
        queries = spectra[1]
        (anchor_embeds, query_embeds), (anchor_class_preds, query_class_preds) = self.forward(anchors, queries)
        embedding_loss = clip_contrastive_loss(anchor_embeds, query_embeds, anchor_class, temperature=self.tau)

        self.log('val_loss', embedding_loss, on_step=True, on_epoch=True)

        return embedding_loss
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            raise NotImplementedError
            x, y = batch
            x = self.embedder(x)
            x = self.projection_cls(x)
            loss = F.cross_entropy(x, y)
            self.log('test_loss', loss, on_step=True, on_epoch=True)
            preds = torch.argmax(x, dim=1)
            acc = torchmetrics.functional.accuracy(preds, y, self.task, num_classes=self.output_dim)
            self.log('test_acc', acc, on_step=True, on_epoch=True)
            return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        raise NotImplementedError("Predict step is not implemented for this model.")
        with torch.no_grad():
            x = batch
            x = self.embedder(x)[0]
            x = self.projection_cls(x)
            preds = torch.argmax(x, dim=1)
            return preds
        
    def embed_step(self, batch):
        with torch.no_grad():
            x = batch
            x = self.embedder(x)[0]
            x = self.projection_embed(x)
            return x
        
    def classifier_step(self, batch):
        with torch.no_grad():
            x = batch
            x = self.embedder(x)[0]
            x = self.projection_cls(x)
            return x