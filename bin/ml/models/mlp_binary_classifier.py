from torch import optim, nn
import torchmetrics
import torch.nn.functional as F
import torch
import lightning as L

class Embedder(L.LightningModule):
    def __init__(self,input_dim, hidden_dim, hidden_layers, ):
        super().__init__() 
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        # self.layers.append(nn.Dropout(self.dropout_rate))  # Dropout after input layer

        # Hidden layers with dropout
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            # self.layers.append(nn.Dropout(self.dropout_rate))  # Dropout after each hidden layer
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MLPBinaryClassifier(L.LightningModule):
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
        self.output_dim = 2
        self.hidden_dim = self.hparams['hidden_dim']
        self.hidden_layers = self.hparams['hidden_layers']
        self.dropout_rate = self.hparams.get('dropout', 0.0)  # Default dropout rate is 0.0
        
        if self.dropout_rate > 1.0 or self.dropout_rate < 0.0:
            raise ValueError("Dropout rate must be between 0.0 and 1.0")
        
        self.embedder = Embedder(self.input_dim, self.hidden_dim, self.hidden_layers)

                # Training metrics
        self.train_metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(num_classes=self.output_dim, task="binary"),
            'precision': torchmetrics.Precision(num_classes=self.output_dim, average='macro', task="binary"),
            'recall': torchmetrics.Recall(num_classes=self.output_dim, average='macro', task="binary"),
            }, 
            prefix='train_'
        )
        self.val_metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(num_classes=self.output_dim, task="binary"),
            'precision': torchmetrics.Precision(num_classes=self.output_dim, average='macro', task="binary"),
            'recall': torchmetrics.Recall(num_classes=self.output_dim, average='macro', task="binary"),
            },
            prefix='val_'
        )

    def forward(self, anchors, positives, negatives):
        anchors = self.embedder(anchors)
        positives = self.embedder(positives)
        negatives = self.embedder(negatives)

        return anchors, positives, negatives
        

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
        embedded_anchor, embedded_pos, embedded_neg = self.forward(torch.stack(anchors), torch.stack(positives), torch.stack(negatives))

        # Print the representations
        # print('embedded_anchor[0]: ', embedded_anchor[0])
        # print('embedded_pos[0]: ', embedded_pos[0])
        # print('embedded_neg[0]: ', embedded_neg[0])

        pos_targets = torch.ones(len(anchors), device=self.device, dtype=torch.long)
        neg_targets = -torch.ones(len(anchors), device=self.device, dtype=torch.long)
        
        # pos_loss = F.cosine_embedding_loss(embedded_anchor, embedded_pos, pos_targets, margin=0.7)
        # neg_loss = F.cosine_embedding_loss(embedded_anchor, embedded_neg, neg_targets, margin=0.7)

        # loss = pos_loss + neg_loss
        loss = F.triplet_margin_loss(embedded_anchor, embedded_pos, embedded_neg, margin=1.0)

        # Metrics: Convert cosine similarity to binary (0 or 1)
        pos_preds = F.cosine_similarity(embedded_anchor, embedded_pos)
        neg_preds = F.cosine_similarity(embedded_anchor, embedded_neg)
        print(pos_preds)
        print(neg_preds)

        # Convert predictions to binary values (0 or 1)
        pos_preds_binary = (pos_preds > 0).float()  # Similar -> 1, Dissimilar -> 0
        neg_preds_binary = (neg_preds < 0).float()  # Dissimilar -> 0, Similar -> 1
        preds = torch.cat((pos_preds_binary, neg_preds_binary), dim=0)
        targets = torch.cat((pos_targets, neg_targets), dim=0)
        targets = torch.where(targets == -1, torch.tensor(0, dtype=targets.dtype), targets)

        batch_value = self.train_metrics(preds, targets)
        self.log_dict(batch_value, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss
    
    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):    
        # Batch is a list of:
        # spectra: (anchor, positive, negative)
        # metadata: (anchor_metadata, positive_metadata, negative_metadata)
        # similarity: (None, pos_sim, neg_sim)

        spectra = [x[0] for x in batch]
        metadata = [x[1] for x in batch]
        similarities = [x[2] for x in batch]

        # Unpack the batch
        anchors = torch.stack([x[0] for x in spectra])
        positives = torch.stack([x[1] for x in spectra])
        negatives = torch.stack([x[2] for x in spectra])

        pos_similarities = torch.tensor([x[1] for x in similarities], device=self.device)
        neg_similarities = torch.tensor([x[2] for x in similarities], device=self.device)

        # Forward pass
        embedded_anchor = self.embedder(anchors)
        embedded_pos = self.embedder(positives)
        embedded_neg = self.embedder(negatives)

        # Cosine Embedding Loss
        pos_targets = torch.ones(len(anchors), device=self.device, dtype=torch.long)
        neg_targets = -torch.ones(len(anchors), device=self.device, dtype=torch.long)
        
        # pos_loss = F.cosine_embedding_loss(embedded_anchor, embedded_pos, pos_targets, margin=0.5)
        # neg_loss = F.cosine_embedding_loss(embedded_anchor, embedded_neg, neg_targets, margin=0.5)

        # loss = pos_loss + neg_loss
        loss = F.triplet_margin_loss(embedded_anchor, embedded_pos, embedded_neg, margin=1.0)


        # Metrics: Convert cosine similarity to binary (0 or 1)
        pos_preds = F.cosine_similarity(embedded_anchor, embedded_pos)
        neg_preds = F.cosine_similarity(embedded_anchor, embedded_neg)

        # Convert predictions to binary values (0 or 1)
        pos_preds_binary = (pos_preds > 0).float()  # Similar -> 1, Dissimilar -> 0
        neg_preds_binary = (neg_preds < 0).float()  # Dissimilar -> 0, Similar -> 1

        # Concatenate positive and negative predictions and targets for metric calculation
        preds = torch.cat((pos_preds_binary, neg_preds_binary), dim=0)
        targets = torch.cat((pos_targets, neg_targets), dim=0)
        preds = torch.where(preds == -1, torch.tensor(0, dtype=preds.dtype), preds)
        targets = torch.where(targets == -1, torch.tensor(0, dtype=targets.dtype), targets)

        batch_value = self.val_metrics(preds, targets)
        self.log_dict(batch_value, on_epoch=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True)

        return loss

    
    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Test step not implemented")


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        spectrum_a, spectrum_b, similarity, metadata = batch

        embed_a = self.embedder(spectrum_a)
        embed_b = self.embedder(spectrum_b)

        preds = F.sigmoid(F.cosine_similarity(embed_a, embed_b))
        loss = None

        return {'predictions': preds, 'similarity': similarity, 'loss': loss, 'metadata': metadata}

    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)