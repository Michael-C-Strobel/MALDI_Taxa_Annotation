from torch import optim, nn
import torchmetrics
import torch.nn.functional as F
import torch
import lightning as L

class MLPClassifier(L.LightningModule):
    def __init__(self, hyperparameters):
        super().__init__()

        # Training and validation metrics
        self.train_metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(num_classes=hyperparameters['output_dim'], task="multiclass"),
            'precision': torchmetrics.Precision(num_classes=hyperparameters['output_dim'], average='macro', task="multiclass"),
            'recall': torchmetrics.Recall(num_classes=hyperparameters['output_dim'], average='macro', task="multiclass"),
        }, prefix='train_')

        self.val_metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(num_classes=hyperparameters['output_dim'], task="multiclass"),
            'precision': torchmetrics.Precision(num_classes=hyperparameters['output_dim'], average='macro', task="multiclass"),
            'recall': torchmetrics.Recall(num_classes=hyperparameters['output_dim'], average='macro', task="multiclass"),
        }, prefix='val_')

        # Save hyperparameters
        for key in hyperparameters.keys():
            self.hparams.update({key: hyperparameters[key]})
        self.save_hyperparameters()

        # Optimizer parameters
        self.lr = self.hparams.get('lr', 1e-5)
        self.weight_decay = self.hparams.get('weight_decay', 0.0)

        # Model architecture
        self.input_dim = self.hparams['input_dim']
        self.output_dim = self.hparams['output_dim']
        self.hidden_dim = self.hparams['hidden_dim']
        self.hidden_layers = self.hparams['hidden_layers']
        self.dropout_rate = self.hparams.get('dropout', 0.0)

        if not (0.0 <= self.dropout_rate <= 1.0):
            raise ValueError("Dropout rate must be between 0.0 and 1.0")

        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(self.hidden_layers):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(self.dropout_rate))

        # Output layer for classification
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        # Space to store one-hot encoder
        self.one_hot_encoder = None

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def training_step(self, batch, batch_idx):
        spectrum, metadata = batch
        if self.one_hot_encoder is None:
            raise ValueError("One-hot encoder not initialized")
        else:
            targets = self.one_hot_encoder(metadata['class']).to(self.device)
        logits = self(spectrum)
        loss = F.cross_entropy(logits, targets)

        # Log metrics
        preds = torch.argmax(logits, dim=1)
        batch_metrics = self.train_metrics(preds, targets)
        self.log_dict(batch_metrics, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        spectrum, metadata = batch
        if self.one_hot_encoder is None:
            raise ValueError("One-hot encoder not initialized")
        else:
            targets = self.one_hot_encoder(metadata['class']).to(self.device)
        logits = self(spectrum)
        loss = F.cross_entropy(logits, targets)

        # Log metrics
        preds = torch.argmax(logits, dim=1)
        batch_metrics = self.val_metrics(preds, targets)
        self.log_dict(batch_metrics, on_epoch=True)
        self.log('val_loss', loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        spectrum, metadata = batch
        if self.one_hot_encoder is None:
            raise ValueError("One-hot encoder not initialized")
        else:
            targets = self.one_hot_encoder(metadata['class']).to(self.device)
        logits = self(spectrum)
        preds = torch.argmax(logits, dim=1)
        loss = F.cross_entropy(logits, targets)
        return {'predictions': preds, 'targets': targets, 'loss': loss}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)