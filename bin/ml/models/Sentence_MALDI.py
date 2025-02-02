from torch import optim, nn
import torch.nn.utils as utils
import torchmetrics
import torch.nn.functional as F
import torch
import lightning as L

class CustromBinaryMetric(torchmetrics.Metric):
    """This metric calculates the binary accuracy of a model's same genus/different genus classification.
    """
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        num_classes = preds.shape[1]
        target = (target == num_classes - 1).long()

        preds = torch.argmax(preds, dim=1)
        preds = (preds == num_classes - 1).long()
        correct = torch.sum(preds == target)
        total = len(target)
        self.correct += correct
        self.total += total

    def compute(self):
        return self.correct / self.total

class Classifier(L.LightningModule):
    # def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim):
    #     super().__init__()
    #     self.layers = nn.ModuleList()
        
    #     # Input layer
    #     self.layers.append(nn.Linear(input_dim, hidden_dim))
    #     self.layers.append(nn.ReLU())
    #     # self.layers.append(nn.Dropout(self.dropout_rate))  # Dropout after input layer

    #     # Hidden layers with dropout
    #     for _ in range(hidden_layers):
    #         self.layers.append(nn.Linear(hidden_dim, hidden_dim))
    #         self.layers.append(nn.ReLU())
    #         # self.layers.append(nn.Dropout(self.dropout_rate))  # Dropout after each hidden layer

    #     # Output layer for classification
    #     self.layers.append(nn.Linear(hidden_dim, output_dim))

    def __init__(self, input_dim, hidden_dim, _, output_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, output_dim))
        # self.layers.append(nn.Dropout(self.dropout_rate))  # Dropout after input layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, dropout_rate, activation=nn.GELU):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList()

        # Input layer with WeightNorm
        self.layers.append(utils.weight_norm(nn.Linear(input_dim, hidden_dim)))
        self.layers.append(nn.LayerNorm(hidden_dim))  # LayerNorm added
        self.layers.append(activation())
        self.layers.append(nn.Dropout(self.dropout_rate))

        # Hidden layers with WeightNorm + LayerNorm
        for _ in range(hidden_layers):
            self.layers.append(utils.weight_norm(nn.Linear(hidden_dim, hidden_dim)))
            self.layers.append(nn.LayerNorm(hidden_dim))  # LayerNorm added
            self.layers.append(activation())
            self.layers.append(nn.Dropout(self.dropout_rate))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Sentence_MALDI(L.LightningModule):
    """ This is an MLP (for now) implementation that loosely follows the SBERT setup. See:
    Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks, Figure 1 For More Details.
    """
    def __init__(self, hyperparameters):
        super().__init__()

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
        self.input_dim = self.hparams['input_dim']
        self.output_bin_edges = torch.tensor(self.hparams['output_bin_edges'], device=self.device)
        self.output_dim = len(self.hparams['output_bin_edges']) +1 # We will predict between each bin edge, + 1 for "different genus"
        self.hidden_dim = self.hparams['hidden_dim']
        self.hidden_layers = self.hparams['hidden_layers']
        self.dropout_rate = self.hparams.get('dropout', 0.0)  # Default dropout rate is 0.0
        
        if self.dropout_rate > 1.0 or self.dropout_rate < 0.0:
            raise ValueError("Dropout rate must be between 0.0 and 1.0")
        
        self.embedder = Embedder(self.input_dim, self.hidden_dim, self.hidden_layers, self.dropout_rate)
        self.classifier = Classifier(self.hidden_dim*3, self.hidden_dim, 3, self.output_dim)

        # Training metrics
        task = 'multiclass'
        if self.output_dim == 2:
            task = 'binary'

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

    def forward(self, anchors, positives, negatives):
        anchors = self.embedder(anchors)
        positives = self.embedder(positives)
        negatives = self.embedder(negatives)

        pos_inputs = torch.cat((anchors, positives, torch.abs(anchors - positives)), dim=1)
        neg_inputs = torch.cat((anchors, negatives, torch.abs(anchors - negatives)), dim=1)

        pos_preds = self.classifier(pos_inputs)
        neg_preds = self.classifier(neg_inputs)

        return pos_preds, neg_preds
        

    def training_step(self, batch, batch_idx):
        # Batch is a list of:
        # spectra: (anchor, postive, negative)
        # metadata: (anchor_metadata, positive_metadata, negative_metadata)
        # similarity: (None, pos_sim, neg_sim)

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
        pos_preds, neg_preds = self(torch.stack(anchors), torch.stack(positives), torch.stack(negatives))

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

        print('preds: ', torch.argmax(preds, dim=1))
        print('targets: ', targets)

        batch_value = self.train_metrics(torch.argmax(preds, dim=1), targets)
        self.log_dict(batch_value, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        if False:   # Handy for debugging
            avg_pred_mag = torch.mean(torch.abs(pred_sim))
            avg_real_mag = torch.mean(torch.abs(similarity))
            self.log('train_avg_pred_magnitude', avg_pred_mag, on_step=True, on_epoch=True)
            self.log('train_avg_real_magnitude', avg_real_mag, on_step=True, on_epoch=True)

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
        pos_preds, neg_preds = self(torch.stack(anchors), torch.stack(positives), torch.stack(negatives))

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
    
    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Test step not implemented")
        spectrum_a, spectrum_b, similarity, metadata = batch
        embed_1 = self(spectrum_a)
        embed_2 = self(spectrum_b)
        preds = F.cosine_similarity(embed_1, embed_2)
        loss = nn.functional.mse_loss(preds, similarity)
        return {'predictions': preds, 'similarity': similarity, 'loss': loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        spectrum_a, spectrum_b, similarity, metadata = batch

        embed_a = self.embedder(spectrum_a)
        embed_b = self.embedder(spectrum_b)

        # input = torch.cat((embed_a, embed_b, torch.abs(embed_a - embed_b)), dim=1)
        # preds = F.softmax(self.classifier(input))
        # # Reverse prediction classes
        # preds = torch.flip(preds, dims=[1])
        # # Return probability of same genus
        # preds = preds[:, -1]
        # loss = None

        preds = (F.cosine_similarity(embed_a, embed_b) + 1)/2
        if similarity is not None:
            loss = nn.functional.mse_loss(preds, similarity) # Who knows why we're doing this, but it's here
        else:
            loss = None

        return {'predictions': preds, 'similarity': similarity, 'loss': loss, 'metadata': metadata}

    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)