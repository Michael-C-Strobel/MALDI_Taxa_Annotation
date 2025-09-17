import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics
from torchmetrics.classification import BinaryAUROC, MulticlassAccuracy
from .self_supervised_metrics import cosine_metric, PercentageZeros, BCE_Metric
from .maldi_transformer import MaldiTransformer
from .CLIP_MALDI import CLIP_MALDI
from .autoencoder import MLP
import logging

# TODO: Investigate nadaraya_watson
# https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-pooling.html
class WeightedAggregation(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.weights = nn.Linear(dim, 1)
        self.output_layer = nn.Linear(dim, dim_out)

    def forward(self, x):
        inital_shape = x.shape
        weights = torch.softmax(self.weights(x), dim=1)  # Shape: (batch_size, sequence_length, 1)
        assert weights.shape == (inital_shape[0], inital_shape[1], 1)
        weighted_sum = torch.sum(weights * x, dim=1)  # Shape: (batch_size, dim)
        return self.output_layer(weighted_sum)  # Shape: (batch_size, dim_out)

class BinaryTransformerPredictionHead(LightningModule):
    def __init__(self, hyperparameters):
        super().__init__()

        task = 'binary'
        self.output_dim = 2
        self.is_binary_classifier = True
        self.is_embedder = True

        # Training metrics
        self.train_metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(num_classes=self.output_dim, task=task),
            'precision': torchmetrics.Precision(num_classes=self.output_dim, average='macro', task=task),
            'recall': torchmetrics.Recall(num_classes=self.output_dim, average='macro', task=task),
            # 'binary_accuracy': CustromBinaryMetric()
            }, 
            prefix='train_'
        )
        self.val_metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(num_classes=self.output_dim, task=task),
            'precision': torchmetrics.Precision(num_classes=self.output_dim, average='macro', task=task),
            'recall': torchmetrics.Recall(num_classes=self.output_dim, average='macro', task=task),
            },
            prefix='val_'
        )

        for key in hyperparameters.keys():
            self.hparams.update({key: hyperparameters[key]})
        self.output_bin_edges = torch.tensor(self.hparams['output_bin_edges'], device=self.device)
        self.tau = 1.0

        self.save_hyperparameters()

        # Optimizer
        self.lr = self.hparams.get('mlp_lr', 1e-5)
        self.weight_decay = self.hparams.get('mlp_weight_decay', 0.0)
        if 'mlp_lr' in self.hparams:
            print(f"Using learning rate: {self.lr}")
        if 'mlp_weight_decay' in self.hparams:
            print(f"Using weight decay: {self.weight_decay}")

        # Model
        self.hidden_dim = self.hparams['mlp_hidden_dim']
        self.latent_dim = 512 #self.hparams['latent_dim']

        self.hidden_layers = self.hparams['mlp_hidden_layers']
        self.dropout_rate = self.hparams.get('mlp_dropout', 0.2)  # Default dropout rate is 0.0
        
        if self.dropout_rate > 1.0 or self.dropout_rate < 0.0:
            raise ValueError("Dropout rate must be between 0.0 and 1.0")

        encoder = CLIP_MALDI.load_from_checkpoint(self.hparams['encoder_path'])
        if self.hparams.get('freeze_encoder', True):     # TODO: Optionally decrease the learning rate of the encoder
            for param in encoder.parameters():
                param.requires_grad = False

        # Remove the last layer of the encoder
        encoder.output_head = nn.Identity()
        self.encoder = encoder

        # Get encoder output size
        encoder_output_size = 128#self.encoder.hparams['dim']
        print("Got Encoder output size: ", encoder_output_size)

        # self.agg = WeightedAggregation(encoder_output_size, self.latent_dim) # I don't think this is needed

        self.prediction_head = MLP( input_dim=self.latent_dim, 
                                    hidden_dim=self.hidden_dim,
                                    output_dim=1,
                                    hidden_layers=self.hidden_layers,
                                    dropout=self.dropout_rate)

    def forward(self, x1, x2):
        (x1, x2), (_,_) = self.encoder(x1, x2)
        merged = torch.cat((x1, x2, torch.abs(x1 - x2), x1 * x2), dim=1)
        out = self.prediction_head(merged)

        return out


    def training_step(self, batch, batch_idx):
        # Unpack the batch
        spectra = batch[0]
        anchors = [x[0] for x in spectra]
        positives = [x[1] for x in spectra]
        negatives = [x[2] for x in spectra]

        # Forward pass
        pos_preds = self.forward(torch.stack(anchors), torch.stack(positives))
        neg_preds = self.forward(torch.stack(anchors), torch.stack(negatives))

        # Compute loss
        # Stack everything
        embeds = torch.cat([pos_preds, neg_preds], dim=0).squeeze()
        pos_labels = torch.ones(pos_preds.shape[0], device=embeds.device)
        neg_labels = torch.zeros(neg_preds.shape[0], device=embeds.device)
        labels = torch.cat((pos_labels, neg_labels), dim=0).squeeze()

        # BCE logit loss
        loss = F.binary_cross_entropy_with_logits(embeds, labels)

        # Compute metrics
        metric_results = self.train_metrics(embeds, labels)
        self.log_dict({f'train_loss': loss, **metric_results}, on_step=True, on_epoch=True)

        return loss
    
    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        # Unpack the batch
        spectra = batch[0]
        anchors = [x[0] for x in spectra]
        positives = [x[1] for x in spectra]
        negatives = [x[2] for x in spectra]

        # Forward pass
        pos_preds = self.forward(torch.stack(anchors), torch.stack(positives))
        neg_preds = self.forward(torch.stack(anchors), torch.stack(negatives))

        # Compute loss
        # Stack everything
        preds = torch.cat([pos_preds, neg_preds], dim=0).squeeze()
        pos_labels = torch.ones(pos_preds.shape[0], device=preds.device)
        neg_labels = torch.zeros(neg_preds.shape[0], device=preds.device)
        labels = torch.cat((pos_labels, neg_labels), dim=0).squeeze()

        # BCE logit loss
        loss = F.binary_cross_entropy_with_logits(preds, labels)

        # Compute metrics
        metric_results = self.val_metrics(preds, labels)
        self.log_dict({f'val_loss': loss, **metric_results}, on_step=True, on_epoch=True)

        return loss
    
    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Test step not implemented")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def transform_to_classification(self, similarities):
        """
        Converts similarity scores into classification labels based on bin edges.

        Args:
            similarities (torch.Tensor): A tensor of shape (batch_size, 1) containing similarity scores.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 1) containing classification labels.
        """
        # Digitize the similarities into bins
        classifications = torch.bucketize(similarities.squeeze(-1), self.output_bin_edges.to(similarities.device))

        # Add a dimension to match the shape (batch_size, 1)
        return classifications
    
    def embed_step(self, batch):
        with torch.no_grad():
            spectrum = batch
            spectrum = spectrum.to(self.device)
            embed, _, _ = self.encoder.embedder.forward(spectrum)
            return embed.squeeze()
                

    def binary_predict_step(self, batch,):
        with torch.no_grad():
            # Unpack the batch
            spectrum_a, spectrum_b, = batch
            spectrum_a = spectrum_a.to(self.device)
            spectrum_b = spectrum_b.to(self.device)

            # Forward pass
            pred = self.forward(spectrum_a, spectrum_a).squeeze()

            return pred