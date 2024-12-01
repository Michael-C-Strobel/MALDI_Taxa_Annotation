from torch import optim, nn
import torchmetrics
import torch.nn.functional as F
import torch
import lightning as L

class MLP(L.LightningModule):
    def __init__(self, hyperparameters):
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
        self.input_dim = self.hparams['input_dim']
        self.output_dim = self.hparams['output_dim']
        self.hidden_dim = self.hparams['hidden_dim']
        self.hidden_layers = self.hparams['hidden_layers']
        self.dropout_rate = self.hparams.get('dropout', 0.0)  # Default dropout rate is 0.0
        
        if self.dropout_rate > 1.0 or self.dropout_rate < 0.0:
            raise ValueError("Dropout rate must be between 0.0 and 1.0")

        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        self.layers.append(nn.ReLU())
        # self.layers.append(nn.Dropout(self.dropout_rate))  # Dropout after input layer

        # Hidden layers with dropout
        for _ in range(self.hidden_layers):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.ReLU())
            # self.layers.append(nn.Dropout(self.dropout_rate))  # Dropout after each hidden layer

        # Output layer
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def training_step(self, batch, batch_idx):
        spectrum_a, spectrum_b, similarity, metadata = batch
        embed_1 = self(spectrum_a)
        embed_2 = self(spectrum_b)
        pred_sim = F.cosine_similarity(embed_1, embed_2)
        
        loss = nn.functional.mse_loss(pred_sim, similarity)
        batch_value = self.train_metrics(pred_sim, similarity)
        self.log_dict(batch_value, on_epoch=True)

        if False:   # Handy for debugging
            avg_pred_mag = torch.mean(torch.abs(pred_sim))
            avg_real_mag = torch.mean(torch.abs(similarity))
            self.log('train_avg_pred_magnitude', avg_pred_mag, on_step=True, on_epoch=True)
            self.log('train_avg_real_magnitude', avg_real_mag, on_step=True, on_epoch=True)

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
        return {'predictions': preds, 'similarity': similarity, 'loss': loss}

    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)