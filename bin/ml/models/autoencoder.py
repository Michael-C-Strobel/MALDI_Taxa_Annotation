import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torchmetrics
import torch.nn.functional as F
import torch
import lightning as L
from torchmetrics import MetricCollection

class MLP(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.input_size = kwargs['input_dim']
        self.output_size = kwargs['output_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.hidden_layers = kwargs['hidden_layers']
        self.dropout = kwargs.get('dropout', 0.0)

        input_size = self.input_size
        self.layers = nn.ModuleList()

        for _ in range(self.hidden_layers):
            self.layers.append(nn.Linear(input_size, self.hidden_dim))
            self.layers.append(nn.BatchNorm1d(self.hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(self.dropout))
            input_size = self.hidden_dim

        self.layers.append(nn.Linear(self.hidden_dim, self.output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class PercentageZeros(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("zeros", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        self.total += preds.numel()
        self.zeros += torch.sum(preds > 1e-5)

    def compute(self):
        return self.zeros / self.total
    
class BCE_Metric(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        # Add state variables to keep track of accumulated loss and sample count
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update the state with predictions and targets.
        
        Args:
            preds (torch.Tensor): Model predictions (logits or probabilities).
            targets (torch.Tensor): Ground truth labels (binary).
        """
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets.float(), reduction='sum')

        self.total_loss += bce_loss
        self.total_samples += targets.numel()

    def compute(self):
        """
        Compute the average BCE loss over all updates.
        """
        return self.total_loss / self.total_samples
    
class cosine_metric(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update the state with predictions and targets.
        
        Args:
            preds (torch.Tensor): Model predictions (logits or probabilities).
            targets (torch.Tensor): Ground truth labels (binary).
        """
        cosine_loss = F.cosine_similarity(preds, targets, dim=1)

        self.total_loss += cosine_loss
        self.total_samples += targets.numel()

    def compute(self):
        """
        Compute the average BCE loss over all updates.
        """
        return self.total_loss / self.total_samples


class Autoencoder(L.LightningModule):
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
        self.input_dim = self.hparams['input_dim']
        self.hidden_dim = self.hparams['hidden_dim']
        self.hidden_layers = self.hparams['hidden_layers']
        self.bottleneck_dim = self.hparams['bottleneck_dim']
        self.dropout_rate = self.hparams.get('dropout', 0.0)  # Default dropout rate is 0.0
        
        if self.dropout_rate > 1.0 or self.dropout_rate < 0.0:
            raise ValueError("Dropout rate must be between 0.0 and 1.0")

        encoder_params = self.hparams
        encoder_params['output_dim'] = self.bottleneck_dim

        decoder_params = self.hparams.copy()
        decoder_params['input_dim'] = self.bottleneck_dim
        decoder_params['output_dim'] = self.input_dim

        self.encoder = MLP(**encoder_params)
        print("Self.encoder")
        print(self.encoder)
        self.decoder = MLP(**decoder_params)
        print("Self.decoder")
        print(self.decoder)

        self.state = 'pretrain'
        self.prediction_head = None

    def forward(self, x):
        if self.state == 'pretrain':
            x = F.relu(self.encoder(x))
            x = self.decoder(x)
            return x
        else:
            x = F.relu(self.encoder(x))
            x = self.prediction_head(x)
            return x


    def training_step(self, batch, batch_idx):
        if self.state == 'pretrain':
            spectrum, _ = batch

            reconstruction = self(spectrum)

            const_weight = torch.tensor([13.81]).to(spectrum.device)

            loss = nn.functional.binary_cross_entropy_with_logits(reconstruction, spectrum, pos_weight=const_weight)

            batch_value = self.train_metrics(reconstruction, spectrum)
            self.log_dict(batch_value, on_epoch=True)
            return loss
        else:
            spectrum_a, spectrum_b, similarity, metadata = batch
            embed_1 = self(spectrum_a)
            embed_2 = self(spectrum_b)
            pred_sim = F.cosine_similarity(embed_1, embed_2)

            loss = nn.functional.mse_loss(pred_sim, similarity)
            batch_value = self.train_metrics(pred_sim, similarity)
            self.log_dict(batch_value, on_epoch=True)
            
            # # Print gradients for each new layer
            # print("Prediction Head gradients")
            # for param in self.prediction_head.parameters():
            #     print(param.grad)

            return loss
    
    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        if self.state == 'pretrain':
            spectrum, _ = batch
            reconstruction = self(spectrum)
            loss = nn.functional.mse_loss(reconstruction, spectrum)
            batch_value = self.val_metrics(reconstruction, spectrum)
            self.log_dict(batch_value, on_epoch=True)
            return loss
        else:
            spectrum_a, spectrum_b, similarity, metadata = batch
            embed_1 = self(spectrum_a)
            embed_2 = self(spectrum_b)
            pred_sim = F.cosine_similarity(embed_1, embed_2)

            loss = nn.functional.mse_loss(pred_sim, similarity)
            batch_value = self.val_metrics(pred_sim, similarity)
            self.log_dict(batch_value, on_epoch=True)
            return loss
    
    def test_step(self, batch, batch_idx):
        if self.state == 'pretrain':
            raise ValueError("Cannot test pretraining model")
        spectrum_a, spectrum_b, similarity, metadata = batch
        embed_1 = self(spectrum_a)
        embed_2 = self(spectrum_b)
        preds = F.cosine_similarity(embed_1, embed_2)
        loss = nn.functional.mse_loss(preds, similarity)
        return {'predictions': preds, 'similarity': similarity, 'loss': loss}
    

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if self.state == 'pretrain':
            raise ValueError("Cannot predict using pretraining model")
        spectrum_a, spectrum_b, similarity, metadata = batch
        embed_1 = self(spectrum_a)
        embed_2 = self(spectrum_b)
        preds = F.cosine_similarity(embed_1, embed_2)
        if similarity is not None:
            loss = nn.functional.mse_loss(preds, similarity)
        else:
            loss = None

        return {'predictions': preds, 'similarity': similarity, 'loss': loss, 'metadata': metadata}

    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def convert(self, freeze_encoder=False):
        """Convert the model from an autoencoder to a feature extractor"""

        # self.encoder.eval()
        # self.decoder.eval()
        self.state = 'feature_extractor'

        # Remove BCE From the training metrics (This doesn't work)
        # metrics_to_remove = ['bce', 'cosine_similarity']

        # self.train_metrics = torchmetrics.MetricCollection({k: v for k, v in self.train_metrics.items() if k not in metrics_to_remove}, 
        #                                                 prefix='train_')
        # self.val_metrics = torchmetrics.MetricCollection({k: v for k, v in self.val_metrics.items() if k not in metrics_to_remove},
        #                                                  prefix='val_')


        self.encoder = self.encoder

        self.decoder = None
        self.prediction_head = MLP(input_dim=self.bottleneck_dim, output_dim=self.bottleneck_dim, hidden_dim=self.bottleneck_dim, hidden_layers=2)
        # Make prediction head a no-op
        self.prediction_head = nn.Identity()
        # self.prediction_head.forward = lambda x: x

        # print(len(list(self.parameters())))

        # Freeze the encoder
        if freeze_encoder:
            for param in list(self.encoder.parameters())[:-1]:
                param.requires_grad = False


    def load_converted_from_checkpoint(self, checkpoint_path):
        self.convert()
        self.load_state_dict(torch.load(checkpoint_path, map_location=self.device)['state_dict'], strict=True)
        # self.prediction_head = nn.Identity()
        