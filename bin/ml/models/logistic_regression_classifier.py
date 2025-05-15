from torch import nn
import torch.nn.functional as F
import torch
import lightning as L
import torchmetrics

class MultinomialLogisticClassifier(L.LightningModule):
    def __init__(self, **hyperparameters):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = self.hparams['input_dim']
        self.output_dim = self.hparams['n_classes']
        self.lr = self.hparams.get('lr', 1e-3)
        self.weight_decay = self.hparams.get('weight_decay', 0.0)
        self.is_classifier = True

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        self.task = 'multiclass' if self.output_dim > 2 else 'binary'
        self.train_metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(num_classes=self.output_dim, task=self.task),
            'precision': torchmetrics.Precision(num_classes=self.output_dim, average='macro', task=self.task),
            'recall': torchmetrics.Recall(num_classes=self.output_dim, average='macro', task=self.task),
        }, prefix='train_')

        self.val_metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(num_classes=self.output_dim, task=self.task),
            'precision': torchmetrics.Precision(num_classes=self.output_dim, average='macro', task=self.task),
            'recall': torchmetrics.Recall(num_classes=self.output_dim, average='macro', task=self.task),
        }, prefix='val_')

        self.test_metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(num_classes=self.output_dim, task=self.task),
        }, prefix='test_')

        self.train_set_stats = None # Set after constructor

    # def on_save_checkpoint(self, checkpoint):
    #     checkpoint['train_set_stats'] = self.train_set_stats
    #     return checkpoint

    # def on_load_checkpoint(self, checkpoint):
    #     if 'train_set_stats' in checkpoint:
    #         self.train_set_stats = checkpoint['train_set_stats']

    def set_train_set_stats(self, train_set_stats):
        """
        Set the training set statistics for normalization.
        :param train_set_stats: A dictionary containing 'mean' and 'std' for normalization.
        """
        # Get on same device as model
        self.train_set_stats = {
            'mean': train_set_stats['mean'].float(),
            'std': train_set_stats['std'].float()+1e-10,
        }

    def forward(self, x):
        if self.train_set_stats is not None:
            x = (x - self.train_set_stats['mean']) / self.train_set_stats['std']


        return self.linear(x)

    def _step(self, batch, metrics, prefix):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        metric_results = metrics(logits, y)
        self.log_dict({f'{prefix}_loss': loss, **metric_results}, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, self.train_metrics, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, self.val_metrics, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, self.test_metrics, 'test')

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        with torch.no_grad():
            logits = self.forward(batch)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            return preds
        
    def embed_step(self, batch):
        return batch    # Embeddings are just the transformed inputs

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
