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
from .autoencoder import MLP

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

class TransformerPredictionHead(LightningModule):
    def __init__(self, hyperparameters):
        super().__init__()

        # Training metrics
        self.train_metrics = torchmetrics.MetricCollection({
            'mse': torchmetrics.MeanSquaredError(),
            'mae': torchmetrics.MeanAbsoluteError(),
            # 'cosine_similarity': cosine_metric(),
            # 'bce': BCE_Metric(),
            'perc_non_zeros': PercentageZeros()
            }, 
            prefix='train_'
        )
        self.val_metrics = torchmetrics.MetricCollection({
            'mse': torchmetrics.MeanSquaredError(),
            'mae': torchmetrics.MeanAbsoluteError(),
            # 'cosine_similarity': cosine_metric(),
            # 'bce': BCE_Metric(),
            'perc_non_zeros': PercentageZeros()
            },
            prefix='val_'
        )
        self.r2_score = torchmetrics.R2Score()

        for key in hyperparameters.keys():
            self.hparams.update({key: hyperparameters[key]})
        self.save_hyperparameters()

        # Optimizer
        self.lr = self.hparams.get('lr', 1e-5)
        self.weight_decay = self.hparams.get('weight_decay', 0.0)
        if 'lr' in self.hparams:
            print(f"Using learning rate: {self.lr}")
        if 'weight_decay' in self.hparams:
            print(f"Using weight decay: {self.weight_decay}")

        # Model
        self.hidden_dim = self.hparams['hidden_dim']
        self.latent_dim = self.hparams['latent_dim']

        self.hidden_layers = self.hparams['hidden_layers']
        self.dropout_rate = self.hparams.get('dropout', 0.0)  # Default dropout rate is 0.0
        
        if self.dropout_rate > 1.0 or self.dropout_rate < 0.0:
            raise ValueError("Dropout rate must be between 0.0 and 1.0")

        encoder = MaldiTransformer.load_from_checkpoint(self.hparams['encoder_path'])
        if self.hparams.get('freeze_encoder', True):     # TODO: Optionally decrease the learning rate of the encoder
            for param in encoder.parameters():
                param.requires_grad = False

        # Remove the last layer of the encoder
        encoder.output_head = nn.Identity()
        self.encoder = encoder

        # Get encoder output size
        encoder_output_size = self.encoder.hparams['dim']
        print("Got Encoder output size: ", encoder_output_size)

        self.agg = WeightedAggregation(encoder_output_size, self.latent_dim)

        self.state = 'pretrain'
        self.prediction_head = MLP( input_dim=self.latent_dim, 
                                    hidden_dim=self.hidden_dim,
                                    output_dim=self.latent_dim,
                                    hidden_layers=self.hidden_layers,
                                    dropout=self.dropout_rate)

    def forward(self, x):
        x = F.relu(self.encoder(x)[0]) # Returns (B x S x F) :: S = sequence length, F = feature length
        x = self.agg(x)             # Returns (B x F')
        x = self.prediction_head(x) # Returns (B x F'')
        return x


    def training_step(self, batch, batch_idx):
        spectrum_a, spectrum_b, similarity, metadata = batch
        embed_1 = self(spectrum_a)
        embed_2 = self(spectrum_b)
        pred_sim = F.cosine_similarity(embed_1, embed_2)

        loss = nn.functional.mse_loss(pred_sim, similarity)
        batch_value = self.train_metrics(pred_sim, similarity)
        self.log_dict(batch_value, on_epoch=True)

        return loss
    
    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        spectrum_a, spectrum_b, similarity, metadata = batch
        embed_1 = self(spectrum_a)
        embed_2 = self(spectrum_b)
        pred_sim = F.cosine_similarity(embed_1, embed_2)

        loss = nn.functional.mse_loss(pred_sim, similarity)
        batch_value = self.val_metrics(pred_sim, similarity)
        self.log_dict(batch_value, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        spectrum_a, spectrum_b, similarity, metadata = batch
        embed_1 = self(spectrum_a)
        embed_2 = self(spectrum_b)
        preds = F.cosine_similarity(embed_1, embed_2)
        loss = nn.functional.mse_loss(preds, similarity)
        return {'predictions': preds, 'similarity': similarity, 'loss': loss}
    

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        spectrum_a, spectrum_b, similarity, metadata = batch
        embed_1 = self(spectrum_a)
        embed_2 = self(spectrum_b)
        preds = F.cosine_similarity(embed_1, embed_2)
        if similarity is not None:
            loss = nn.functional.mse_loss(preds, similarity)
        else:
            loss = None

        return {'predictions': preds, 'similarity': similarity, 'loss': loss, 'metadata': metadata}

    def convert(self,):
        pass

    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
