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
        spectra = [x[0] for x in batch]
        metadata = [x[1] for x in batch]
        similarities = [x[2] for x in batch]

        # Unpack the batch
        anchors = [x[0] for x in spectra]
        positives = [x[1] for x in spectra]
        negatives = [x[2] for x in spectra]

        pos_similarities = torch.tensor([x[1] for x in similarities], device=self.device)
        neg_similarities = torch.tensor([x[2] for x in similarities], device=self.device)

        # Forward pass
        pos_embeds = self.forward(torch.stack(positives))
        neg_embeds = self.forward(torch.stack(negatives))
        anchor_embeds = self.forward(torch.stack(anchors))
        pos_preds = (F.cosine_similarity(anchor_embeds, pos_embeds) + 1)/2
        neg_preds = (F.cosine_similarity(anchor_embeds, neg_embeds) + 1)/2

        # Convert preds to a probability vector
        temp_preds = torch.ones((pos_preds.shape[0], 2), device=self.device)
        temp_preds[:, 0] = pos_preds
        temp_preds[:, 1] = 1 - pos_preds
        pos_preds = temp_preds
        temp_preds = torch.ones((neg_preds.shape[0], 2), device=self.device)
        temp_preds[:, 0] = neg_preds
        temp_preds[:, 1] = 1 - neg_preds
        neg_preds = temp_preds

        # Apply temperature scaling
        pos_preds /= self.tau
        neg_preds /= self.tau

        # Convert similarities to classification targets
        pos_targets = self.transform_to_classification(pos_similarities)
        neg_targets = torch.ones_like(neg_similarities, device=self.device, dtype=torch.long) * (self.output_dim - 1)

        # Compute loss with class weighting
        if self.output_dim != 2:
            class_weights = torch.ones(self.output_dim, device=self.device)
            class_weights[-1] = 0.1  # Underweight negative samples
            pos_loss = F.cross_entropy(pos_preds, pos_targets, weight=class_weights)
            neg_loss = F.cross_entropy(neg_preds, neg_targets, weight=class_weights)
            logging.warning(f"Cosine embedding is not used in training")
        else:
            pos_loss = F.cross_entropy(pos_preds, pos_targets)
            neg_loss = F.cross_entropy(neg_preds, neg_targets)
            margin = 0.1
            pos_cosine_embedding_loss = F.cosine_embedding_loss(anchor_embeds, pos_embeds, torch.ones(pos_embeds.shape[0], device=self.device), margin=margin)
            neg_cosine_embedding_loss = F.cosine_embedding_loss(anchor_embeds, neg_embeds, -torch.ones(neg_embeds.shape[0], device=self.device), margin=margin)

        # Final loss
        # loss = pos_loss + neg_loss #+ (0.05 * pos_cosine_embedding_loss) + (0.05 * neg_cosine_embedding_loss)
        loss = pos_cosine_embedding_loss + neg_cosine_embedding_loss

        # Logging
        preds = torch.cat((pos_preds, neg_preds), dim=0)
        targets = torch.cat((pos_targets, neg_targets), dim=0)

        batch_value = self.train_metrics(torch.argmax(preds, dim=1), targets)
        self.log_dict(batch_value, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss
    
    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        spectra = [x[0] for x in batch]
        metadata = [x[1] for x in batch]
        similarities = [x[2] for x in batch]

        # Unpack the batch
        anchors = [x[0] for x in spectra]
        positives = [x[1] for x in spectra]
        negatives = [x[2] for x in spectra]

        anchor_metadata = [x[0] for x in metadata]
        positive_metadata = [x[1] for x in metadata]
        negative_metadata = [x[2] for x in metadata]
        
        pos_similarities = torch.tensor([x[1] for x in similarities], device=self.device)
        neg_similarities = torch.tensor([x[2] for x in similarities])

        # Forward pass
        pos_embeds = self.forward(torch.stack(positives))
        neg_embeds = self.forward(torch.stack(negatives))
        anchor_embeds = self.forward(torch.stack(anchors))
        pos_preds = (F.cosine_similarity(anchor_embeds, pos_embeds) + 1)/2
        neg_preds = (F.cosine_similarity(anchor_embeds, neg_embeds) + 1)/2

        # Convert preds to prob vector
        temp_preds = torch.ones((pos_preds.shape[0], 2), device=self.device)
        temp_preds[:, 0] = pos_preds
        temp_preds[:, 1] = 1 - pos_preds
        pos_preds = temp_preds
        temp_preds = torch.ones((neg_preds.shape[0], 2), device=self.device)
        temp_preds[:, 0] = neg_preds
        temp_preds[:, 1] = 1 - neg_preds
        neg_preds = temp_preds

        # Calculate loss
        pos_targets = self.transform_to_classification(pos_similarities)
        neg_targets = torch.ones_like(neg_similarities, device=self.device, dtype=torch.long) * (self.output_dim - 1)

        if self.output_dim != 2:
            # Underweight the negative samples (since they are guarenteed 50%, other classes split the remaining 50%)
            class_weights = torch.ones(self.output_dim, device=self.device)
            class_weights[-1] = 0.1

            pos_loss = F.cross_entropy(pos_preds, pos_targets, weight=class_weights)
            neg_loss = F.cross_entropy(neg_preds, neg_targets, weight=class_weights)
        else:
            pos_loss = F.cross_entropy(pos_preds, pos_targets,)
            neg_loss = F.cross_entropy(neg_preds, neg_targets,)

        loss = pos_loss + neg_loss
        preds = torch.cat((pos_preds, neg_preds), dim=0)
        targets = torch.cat((pos_targets, neg_targets), dim=0)

        batch_value = self.val_metrics(torch.argmax(preds, dim=1), targets)
        self.log_dict(batch_value, on_epoch=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True)

        return loss
    
    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Test step not implemented")


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        spectrum_a, spectrum_b, similarity, metadata = batch

        embed_a = self.forward(spectrum_a)
        embed_b = self.forward(spectrum_b)

        preds = (F.cosine_similarity(embed_a, embed_b) + 1)/2
        if similarity is not None:
            loss = nn.functional.mse_loss(preds, similarity) # Who knows why we're doing this, but it's here
        else:
            loss = None

        return {'predictions': preds, 'similarity': similarity, 'loss': loss, 'metadata': metadata}

    def convert(self,):
        pass



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