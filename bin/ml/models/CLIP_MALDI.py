from torch import optim, nn
import torch.nn.utils as utils
import torchmetrics
import torch.nn.functional as F
import torch
import lightning as L
import logging
from .maldi_transformer import SinusoidalPositionalEncoding


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

class SimpleSelfAttention(nn.Module):
    def __init__(
        self,
        depth,
        dim,
        n_heads=8,
        dropout=0.2,
        reduce="none",
        output_head_dim=64,
        padding_value=None
    ):
        super().__init__()

        self.embed = nn.Linear(1, dim)
        self.padding_value = padding_value

        # This is particularly unfaithful to the source code
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=depth,
        )

        self.reduce = reduce
        self.positional_encoding = SinusoidalPositionalEncoding(dim)

        self.output_head = nn.Linear(dim, output_head_dim)

    def forward(self, spectrum):
        if self.padding_value is not None:
            padding = (spectrum[:,:,0] == self.padding_value).bool()    # True indicates padding
            # print(f"Found a total of {torch.sum(padding)} padding values, {torch.sum(padding)/padding.numel() }%")
        else:
            padding = None
        z = spectrum[:,:,1]

        if torch.isnan(z).any():
            raise ValueError("Nan values in z")

        # z[padding] = 0

        # print(("Number of standard values", torch.sum(~padding)))
        # print(("Number of padding values", torch.sum(padding)))

        z = self.embed(z.unsqueeze(-1))   # Embed the intensity values
        # If all nan, raise an error
        if torch.isnan(z).all():
            raise ValueError("All intensity embeddings are NaN")
        if padding is not None:
            if torch.isnan(z)[~padding].any():
                raise ValueError("Nan values in z that aren't in padding")
        z = self.positional_encoding(z, pos=spectrum[:,:,0])    # Use m/z values as positions

        if torch.isnan(z)[~padding].any():
            raise ValueError("Nan values in z that aren't in padding")

        z = self.transformer(z, src_key_padding_mask=padding) # Somehow, we don't actually allow padding?

        if torch.isnan(z).all():
            raise ValueError("All values in z are NaN")
        if torch.isnan(z)[~padding].any():
            raise ValueError("Nan values in z that aren't in padding")


        # if torch.isnan(z).any():
        #     raise ValueError("Nan values in z")

        if self.reduce == "mean":
            return self.output_head(z.sum(1))
        elif self.reduce == "max":
            return self.output_head(z.max(1).values)
        elif self.reduce == "cls":
            return self.output_head(z[:, 0, :])
        elif self.reduce == "none":
            return z, padding

# def clip_contrastive_loss(anchor_embeds, pair_embeds, anchor_class, temperature=0.07):
#     """
#     Compute the contrastive loss between image and text embeddings.

#     Parameters:
#     - anchor_embeds (torch.Tensor): A tensor of shape (batch_size, embed_size) containing image embeddings.
#     - pair_embeds (torch.Tensor): A tensor of shape (batch_size, embed_size) containing text embeddings.
#     - temperature (float): A temperature scaling factor for the similarity computation.

#     Returns:
#     - loss (torch.Tensor): The computed contrastive loss value.
#     """
#     # Normalize embeddings to unit length
#     anchor_embeds = F.normalize(anchor_embeds, p=2, dim=-1)
#     pair_embeds = F.normalize(pair_embeds, p=2, dim=-1)
    
#     # Compute cosine similarity between all image-text pairs
#     similarity_matrix = torch.matmul(anchor_embeds, pair_embeds.T)  # (batch_size, batch_size)

#     # Apply temperature scaling
#     similarity_matrix /= temperature

#     # Clip logits to prevent numerical instability
#     similarity_matrix = torch.clamp(similarity_matrix, min=-100, max=100)
    
#     # Create labels: for each image, the corresponding text is the positive pair
#     labels = torch.arange(anchor_embeds.size(0), device=anchor_embeds.device)
    
#     # Compute cross-entropy loss using the similarity matrix
#     # We concatenate the positive pairs for image-text and text-image
#     loss_image_to_text = F.cross_entropy(similarity_matrix, labels)
#     loss_text_to_image = F.cross_entropy(similarity_matrix.T, labels)
    
#     # Final loss is the sum of both directions (image -> text and text -> image)
#     loss = (loss_image_to_text + loss_text_to_image) / 2.0
    
#     return loss

def clip_contrastive_loss(anchor_embeds, pair_embeds, anchor_class, temperature=0.07):
    """
    Compute a contrastive loss that accounts for multiple instances of the same class.
    
    Parameters:
    - anchor_embeds (torch.Tensor): (batch_size, embed_size) image embeddings.
    - pair_embeds (torch.Tensor): (batch_size, embed_size) text embeddings.
    - anchor_class (List[str]): List of class labels corresponding to each sample.
    - temperature (float): Scaling factor for similarity.

    Returns:
    - torch.Tensor: Contrastive loss.
    """

    # Normalize embeddings
    anchor_embeds = F.normalize(anchor_embeds, p=2, dim=-1)
    pair_embeds = F.normalize(pair_embeds, p=2, dim=-1)

    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(anchor_embeds, pair_embeds.T) / temperature
    
    # Convert class labels to numerical indices
    unique_classes = list(set(anchor_class))  # Get unique class labels
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}  # Create a mapping
    class_indices = torch.tensor([class_to_idx[cls] for cls in anchor_class], device=anchor_embeds.device)
    
    # Create a mask where positive pairs (same class) are 1, else 0
    pos_mask = class_indices.unsqueeze(1) == class_indices.unsqueeze(0)  # (batch_size, batch_size)

    # Compute log probabilities
    log_probs = F.log_softmax(similarity_matrix, dim=-1)
    
    # Average log likelihood over positive samples (avoid NaNs by adding eps)
    eps = 1e-8
    loss = -(pos_mask * log_probs).sum(dim=-1) / (pos_mask.sum(dim=-1) + eps)
    
    return loss.mean()


class CLIP_MALDI(L.LightningModule):
    """ 
    """
    def __init__(self, hyperparameters, pretrained_embedder=None):
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
        self.tau = self.hparams.get('tau', 1.0)  # Softmax temperature at train-time
        
        if self.dropout_rate > 1.0 or self.dropout_rate < 0.0:
            raise ValueError("Dropout rate must be between 0.0 and 1.0")
        
        if not pretrained_embedder:
            # self.embedder = Embedder(self.input_dim, self.hidden_dim, self.hidden_layers, self.dropout_rate)
            self.embedder = SimpleSelfAttention(self.input_dim, self.hidden_dim, self.hidden_dim, n_heads=10, dropout_rate=self.dropout_rate)
        else:
            self.embedder = pretrained_embedder

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

    def forward(self, anchors, positives):
        anchors = self.embedder(anchors)
        pos_embeds = self.embedder(positives)

        return anchors, pos_embeds
        

    def training_step(self, batch, batch_idx):
        # Batch is a list of:
        # spectra: (anchor, positive)
        # metadata: (anchor_metadata, positive_metadata)
        # similarity: (sim,)

        spectra = [x[0] for x in batch]
        metadata = [x[1] for x in batch]
        anchor_class = [x[0]['class'] for x in metadata]
        similarities = [x[2] for x in batch]

        # Unpack the batch
        anchors = [x[0] for x in spectra]
        positives = [x[1] for x in spectra]


        # Forward pass
        anchor_embeds, positive_embeds = self.forward(torch.stack(anchors), torch.stack(positives))

        loss = clip_contrastive_loss(anchor_embeds, positive_embeds, anchor_class, temperature=self.tau)

        # Logging

        # batch_value = self.train_metrics(torch.argmax(preds, dim=1), targets)
        # self.log_dict(batch_value, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss
    
    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        spectra = [x[0] for x in batch]
        metadata = [x[1] for x in batch]
        anchor_class = [x[0]['class'] for x in metadata]
        similarities = [x[2] for x in batch]

        # Unpack the batch
        anchors = [x[0] for x in spectra]
        positives = [x[1] for x in spectra]

        # Forward pass
        anchor_embeds, positive_embeds = self.forward(torch.stack(anchors), torch.stack(positives))

        loss = clip_contrastive_loss(anchor_embeds, positive_embeds, anchor_class, temperature=self.tau)

        # Logging
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
        # preds = F.softmax(self.classifier(input), dim=1)
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