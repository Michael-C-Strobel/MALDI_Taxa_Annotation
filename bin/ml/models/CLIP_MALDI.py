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
        padding_value=None,
        prepend_cls=False,
        no_attn_mask=False,
        concat_pos=False,
    ):
        super().__init__()

        self.embed = nn.Linear(1, dim)
        self.padding_value = padding_value
        self.prepend_cls = prepend_cls
        self.no_attn_mask = no_attn_mask
        self.concat_pos = concat_pos
        self.fixed_cls_encoding = True
        if not self.fixed_cls_encoding
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        _dim = dim
        if self.concat_pos:
            self.proj = nn.Linear(dim*2, dim)
            # _dim *= 2

        # This is particularly unfaithful to the source code
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=_dim,
                nhead=n_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=depth,
        )

        self.reduce = reduce
        self.positional_encoding = SinusoidalPositionalEncoding(dim, concat=concat_pos)

        self.output_head = nn.Linear(_dim, output_head_dim)

    def forward(self, spectrum):
        # Spectrum (B x L x 2) where last dimension is [mz, intensity]
        cls_appended = False

        if (self.reduce == 'cls' or self.prepend_cls is not None) and (self.fixed_cls_encoding):
            # Prepend a CLS token
            # assert cls_token_val.item() != self.padding_value, "CLS token value is the same as padding value"
            # cls_token_val = torch.tensor([-2], device=spectrum.device)
            # cls_tokens = torch.ones(spectrum.shape[0], 1, spectrum.shape[2], device=spectrum.device) * cls_token_val
            cls_tokens = self.cls_token.expand(spectrum.shape[0], -1, -1)  # B x 1 x D
            spectrum = torch.cat([cls_tokens, spectrum], dim=1)
            cls_appended = True

        if self.padding_value is not None:
            padding = (spectrum[:,:,0] == self.padding_value).bool()    # True indicates padding
            # print(f"Found a total of {torch.sum(padding)} padding values, {torch.sum(padding)/padding.numel() }%")
        else:
            padding = None

        z = spectrum[:,:,1] # B x L x [mz, intensity]

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
        if self.concat_pos:
            z = self.proj(z)    # Reduce dim by half

        if (self.prepend_cls or self.reduce == 'cls') and (not self.fixed_cls_encoding):
            cls_tokens = self.cls_token.expand(z.shape[0], -1, -1)
            z = torch.cat([cls_tokens, z], dim=1)  # Prepend CLS token

            if padding is not None:
                # Add a false padding value for the CLS token
                cls_padding = torch.zeros((z.shape[0], 1), dtype=torch.bool, device=z.device)
                padding = torch.cat([cls_padding, padding], dim=1)

        if padding is not None:
            if torch.isnan(z)[~padding].any():
                raise ValueError("Nan values in z that aren't in padding")

        z = self.transformer(z, src_key_padding_mask=padding) # Somehow, we don't actually allow padding?

        if torch.isnan(z).all():
            raise ValueError("All values in z are NaN")
        if padding is not None:
            if torch.isnan(z)[~padding].any():
                raise ValueError("Nan values in z that aren't in padding")

        if self.reduce == "sum":
            if cls_appended:
                z_org = z[:, 1:, :]
            else:
                z_org = z
            return self.output_head(z_org.sum(1)), z, padding
        elif self.reduce == "mean":
            if cls_appended:
                z_org = z[:, 1:, :]
            else:
                z_org = z
            return self.output_head(z_org.mean(1)), z, padding
        elif self.reduce == "max":
            if cls_appended:
                z_org = z[:, 1:, :]
            else:
                z_org = z
            return self.output_head(z_org.max(1).values), z, padding
        elif self.reduce == "cls":
            return self.output_head(z[:, 0, :]), z, padding
        elif self.reduce == "none":
            if cls_appended:
                z_org = z[:, 1:, :]
            else:
                z_org = z
            return z_org, z, padding
        else:
            raise ValueError("Invalid reduction method")

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

def reconstruction_loss(recon, target, padding_value=None):
    """
    Compute MSE loss on reconstruction, on 10 randomly sampled positive (peak intensity > 0.02) and negative
    (peak intensity < 0.02) peaks.

    Parameters:
    - recon (torch.Tensor): (batch_size, num_bins) reconstructed spectra.
    - target (torch.Tensor): (batch_size, num_bins) target spectra.

    Returns:
    - torch.Tensor: Reconstruction loss.
    """

    # Get Random 10 positive and negative peaks
    pos_mask = target > 0.05
    neg_mask = target < 0.02

    if padding_value is not None:
        non_padding_mask = target != padding_value
        pos_mask = pos_mask & non_padding_mask
        neg_mask = neg_mask & non_padding_mask

    pos_indices = torch.nonzero(pos_mask, as_tuple=True)
    neg_indices = torch.nonzero(neg_mask, as_tuple=True)

    pos_indices = torch.stack([pos_indices[0], pos_indices[1]], dim=1)
    neg_indices = torch.stack([neg_indices[0], neg_indices[1]], dim=1)

    # Sample 10 random indices
    pos_indices = pos_indices[torch.randperm(pos_indices.size(0))[:10]]
    neg_indices = neg_indices[torch.randperm(neg_indices.size(0))[:10]]

    # Compute MSE loss
    # pos_loss = nn.functional.mse_loss(recon[pos_indices[:, 0], pos_indices[:, 1]], target[pos_indices[:, 0], pos_indices[:, 1]])
    # neg_loss = nn.functional.mse_loss(recon[neg_indices[:, 0], neg_indices[:, 1]], target[neg_indices[:, 0], neg_indices[:, 1]])

    # Compute Binary Cross Entropy
    binarized_target = (target > 0.02).float()

    pos_loss = nn.functional.binary_cross_entropy(recon[pos_indices[:, 0], pos_indices[:, 1]], binarized_target[pos_indices[:, 0], pos_indices[:, 1]])
    neg_loss = nn.functional.binary_cross_entropy(recon[neg_indices[:, 0], neg_indices[:, 1]], binarized_target[neg_indices[:, 0], neg_indices[:, 1]])

    return (pos_loss + neg_loss) / 2
    

class BinSpectrum(torch.nn.Module):
    """Bin the input spectrum into m/z bins of fixed width and apply L2 normalization. 
    Intensity values within each bin are summed.
    
    Args:
        bin_width (float): The width of each bin.
        min_mz (float): The minimum m/z value.
        max_mz (float): The maximum m/z value.
        
    Returns:
        Tensor: The binned and L2-normalized spectrum.
    """
    
    def __init__(self, bin_width: float, min_mz: float, max_mz: float):
        super().__init__()
        self.bin_width = bin_width
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.num_bins = int((max_mz - min_mz) / bin_width)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrum (Tensor): A tensor of shape (batch_size, seq_len, 2) where the last dimension contains 
                                m/z values (index 0) and intensity values (index 1).
        
        Returns:
            Tensor: A 2D tensor of shape (batch_size, num_bins) where each row corresponds to the 
                    binned and L2-normalized intensity of a spectrum in the batch.
        """
        batch_size, seq_len, _ = spectrum.shape
        
        # Initialize the output tensor for binned spectra
        binned_spectra = torch.zeros(batch_size, self.num_bins, dtype=torch.float32, device=spectrum.device)

        # Loop over the batch
        for i in range(batch_size):
            # Extract m/z and intensity for the current spectrum
            mz_values = spectrum[i, :, 0]
            intensity_values = spectrum[i, :, 1]

            # Compute bin indices for each m/z value
            bin_indices = ((mz_values - self.min_mz) / self.bin_width).long()

            # Ensure indices are within the valid bin range
            bin_indices = torch.clamp(bin_indices, min=0, max=self.num_bins - 1)

            # Sum intensities into bins using scatter_add_
            binned_spectra[i].scatter_add_(0, bin_indices, intensity_values)

            # Apply L2 normalization (only if nonzero to avoid NaN)
            norm = torch.linalg.norm(binned_spectra[i], ord=2)
            if norm > 0:
                binned_spectra[i] /= norm

        return binned_spectra
        
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
        self.warmup_steps = self.hparams.get('warmup_steps', 0)

        # Model
        self.input_dim = self.hparams['input_dim']
        self.output_bin_edges = torch.tensor(self.hparams['output_bin_edges'], device=self.device)
        self.output_dim = len(self.hparams['output_bin_edges']) +1 # We will predict between each bin edge, + 1 for "different genus"
        self.hidden_dim = self.hparams['hidden_dim']
        self.hidden_layers = self.hparams['hidden_layers']
        self.dropout_rate = self.hparams.get('dropout', 0.0)  # Default dropout rate is 0.0
        self.tau = self.hparams.get('tau', 1.0)  # Softmax temperature at train-time
        self.padding_value = self.hparams.get('padding_value', None)
        self.ss_task = self.hparams.get('ss_task', None)
        self.rcon_head_dim = self.hparams.get('rcon_head_dim', None)
        self.concat_pos = self.hparams.get('concat_pos', False)
        if self.ss_task == 'recon':
            assert self.rcon_head_dim is None, "Reconstruction head dimension must be specified for reconstruction task"
        else:
            self.rcon_head_dim = None



        if self.dropout_rate > 1.0 or self.dropout_rate < 0.0:
            raise ValueError("Dropout rate must be between 0.0 and 1.0")
        
        if not pretrained_embedder:
            # self.embedder = Embedder(self.input_dim, self.hidden_dim, self.hidden_layers, self.dropout_rate)
            self.transformer_reduction = 'cls'
            self.embedder = SimpleSelfAttention(2,  # depth
                                                self.hidden_dim,
                                                n_heads=10,
                                                dropout=self.dropout_rate,
                                                output_head_dim=128,
                                                padding_value=self.padding_value,
                                                reduce=self.transformer_reduction,
                                                prepend_cls=True,
                                                concat_pos=self.concat_pos)
            if self.ss_task == 'recon':
                if self.rcon_head_dim != 1700: 
                    raise NotImplementedError("Reconstruction head dimension must be 1700")
                self.rcon_target_generator = BinSpectrum(10.0, 3_000.0, 20_000.0)
                # Linear Decoder Head
                self.rcon_head = nn.Linear(self.hidden_dim, self.rcon_head_dim)

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
        anchors_embeds, achor_raw_embeds, _ = self.embedder(anchors)
        pos_embeds, pos_raw_embeds, _ = self.embedder(positives)

        anchor_rcon = None
        pos_rcon = None

        if self.ss_task == 'recon':
            assert self.embedder.prepend_cls == True, "Prepend CLS token must be enabled for reconstruction task"
            anchor_rcon = self.rcon_head(achor_raw_embeds[:, 0, :]) # TODO
            pos_rcon = self.rcon_head(pos_raw_embeds[:, 0, :])      # For now, this is always using the CLS token, maybe using a different aggregation is better?

            anchor_rcon = F.sigmoid(anchor_rcon)
            pos_rcon = F.sigmoid(pos_rcon)

        if self.ss_task == 'MLM':
            raise NotImplementedError()

        return (anchors_embeds, pos_embeds), (anchor_rcon, pos_rcon)
        

    def training_step(self, batch, batch_idx):
        # Batch is a list of:
        # spectra: (anchor, positive)
        # metadata: (anchor_metadata, positive_metadata)
        # similarity: (sim,)

        # spectra = [x[0] for x in batch]
        # metadata = [x[1] for x in batch]
        # anchor_class = [x[0]['class'] for x in metadata]
        # similarities = [x[2] for x in batch]

        # # Unpack the batch
        # anchors = [x[0] for x in spectra]
        # positives = [x[1] for x in spectra]


        spectra = batch[0]
        # metadata = batch[1]
        anchor_class = batch[1][0]['class']
        similarities = batch[2]
        anchors = spectra[0]
        positives = spectra[1]
        

        # Forward pass
        # (anchor_embeds, positive_embeds), (anchor_rcon, pos_rcon) = self.forward(torch.stack(anchors), torch.stack(positives))
        (anchor_embeds, positive_embeds), (anchor_rcon, pos_rcon) = self.forward(anchors, positives)

        # Bin anchor and positive spectra to get rcon targets
        if self.rcon_head_dim is not None:
            anchor_rcon_target  = self.rcon_target_generator(torch.stack(anchors))
            pos_rcon_target     = self.rcon_target_generator(torch.stack(positives))

            clip_loss = clip_contrastive_loss(anchor_embeds, positive_embeds, anchor_class, temperature=self.tau)
            rcon_loss = (reconstruction_loss(anchor_rcon, anchor_rcon_target, self.padding_value) + reconstruction_loss(pos_rcon, pos_rcon_target, self.padding_value))/2

            loss = clip_loss  + (3 * rcon_loss)

            self.log('train_clip_loss', clip_loss, on_step=True, on_epoch=True)
            self.log('train_rcon_loss', rcon_loss, on_step=True, on_epoch=True)

        else:
            clip_loss = clip_contrastive_loss(anchor_embeds, positive_embeds, anchor_class, temperature=self.tau)
            rcon_loss = None
            loss = clip_loss

        # Logging

        # batch_value = self.train_metrics(torch.argmax(preds, dim=1), targets)
        # self.log_dict(batch_value, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss
    
    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        # spectra = [x[0] for x in batch]
        # metadata = [x[1] for x in batch]
        # anchor_class = [x[0]['class'] for x in metadata]
        # similarities = [x[2] for x in batch]

        # # Unpack the batch
        # anchors = [x[0] for x in spectra]
        # positives = [x[1] for x in spectra]


        spectra = batch[0]
        # metadata = batch[1]
        anchor_class = batch[1][0]['class']
        similarities = batch[2]
        anchors = spectra[0]
        positives = spectra[1]


        if self.ss_task == "MLM":
            # Randomly set peask to -1 based on intensity value
            anchors, anchor_masks, anchor_targets = self.mask_spectra(anchors)
            positives, pos_masks, pos_targets = self.mask_spectra(positives)

        # Forward pass
        # (anchor_embeds, positive_embeds), (anchor_rcon, pos_rcon) = self.forward(torch.stack(anchors), torch.stack(positives))
        (anchor_embeds, positive_embeds), (anchor_rcon, pos_rcon) = self.forward(anchors, positives)

        clip_loss = clip_contrastive_loss(anchor_embeds, positive_embeds, anchor_class, temperature=self.tau)
        self.log('val_loss', clip_loss, on_step=True, on_epoch=True)


        # Bin anchor and positive spectra to get rcon targets
        if self.ss_task == 'recon':
            anchor_rcon_target  = self.rcon_target_generator(torch.stack(anchors))
            pos_rcon_target     = self.rcon_target_generator(torch.stack(positives))

            rcon_loss = (reconstruction_loss(anchor_rcon, anchor_rcon_target, self.padding_value) + reconstruction_loss(pos_rcon, pos_rcon_target, self.padding_value))/2

            self.log('val_rcon_loss', rcon_loss, on_step=True, on_epoch=True)

        # Masked Language Loss
        if self.ss_task == "MLM":
            # Compute masked language model loss
            anchor_masked_loss  = nn.functional.cross_entropy(anchor_embeds[anchor_masks], anchor_targets[anchor_masks])
            pos_masked_loss     = nn.functional.cross_entropy(positive_embeds[pos_masks], pos_targets[pos_masks])

            masked_loss = (anchor_masked_loss + pos_masked_loss)
            self.log('val_masked_loss', masked_loss, on_step=True, on_epoch=True)
            

        return clip_loss
    
    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Test step not implemented")
        spectrum_a, spectrum_b, similarity, metadata = batch
        embed_1 = self(spectrum_a)
        embed_2 = self(spectrum_b)
        preds = F.cosine_similarity(embed_1, embed_2)
        loss = nn.functional.mse_loss(preds, similarity)
        return {'predictions': preds, 'similarity': similarity, 'loss': loss}

    def on_predict_start(self):
        torch.set_grad_enabled(True)
    #     self.embedder.train()

    #     for layer in self.modules():
    #         if isinstance(layer, torch.nn.BatchNorm1d) or isinstance(layer, torch.nn.BatchNorm2d):
    #             print("Freezing BatchNorm")
    #             layer.track_running_stats = False
    #             layer.weight.requires_grad = False
    #             layer.bias.requires_grad = False

    #     for layer in self.modules():
    #         if isinstance(layer, torch.nn.LayerNorm):
    #             print("Freezing LayerNorm")
    #             layer.weight.requires_grad = False
    #             layer.bias.requires_grad = False

    def TTT_helper(self, spectrum):
        if self.TTT_steps == 0:
            embed, raw_embed, _ = self.embedder(spectrum)
        else:
            temp_weights = self.state_dict()

            opt = optim.SGD(self.embedder.parameters(), lr=1e-4, momentum=0.0, weight_decay=0.0)

            for _ in range(self.TTT_steps):
                # Forward pass
                embed, raw_embed, _ = self.embedder(spectrum)
                rcon = F.sigmoid(self.rcon_head(raw_embed[:, 0, :]))

                with torch.no_grad():
                    target = self.rcon_target_generator(spectrum)

                reconstruction_loss_val = reconstruction_loss(rcon, target, self.padding_value)

                # Backpropagation
                opt.zero_grad()
                reconstruction_loss_val.backward()
                opt.step()

            # Reload model weights
            self.load_state_dict(temp_weights)

        return embed

    def TTT_step(self, batch):
        
        spectrum_a, spectrum_b, similarity, metadata = batch

        spectrum_a_embeds = [self.TTT_helper(spectrum.unsqueeze(0)) for spectrum in spectrum_a]
        spectrum_b_embeds = [self.TTT_helper(spectrum.unsqueeze(0)) for spectrum in spectrum_b]

        spectrum_a_embeds = torch.cat(spectrum_a_embeds, dim=0)
        spectrum_b_embeds = torch.cat(spectrum_b_embeds, dim=0)

        return spectrum_a_embeds, spectrum_b_embeds

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        self.TTT_steps = 0

        spectrum_a, spectrum_b, similarity, metadata = batch

        embed_a = None
        embed_b = None


        if self.TTT_steps == 0:
            embed_a, rcon_a, _ = self.embedder(spectrum_a)
            embed_b, rcon_b, _ = self.embedder(spectrum_b)

        else:
            embed_a, embed_b = self.TTT_step(batch)

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

        return {'predictions': preds.detach(), 'similarity': similarity.detach(), 'loss': loss.detach(), 'metadata': metadata}
    
    def embed_step(self, batch):
        inputs = batch[0]

        if len(inputs.shape) == 2:
            # Add a batch dimension ( S, B, F)
            inputs = inputs.unsqueeze(0)

        # Run the model in inference mode
        self.eval()
        with torch.no_grad():
            embeddings, _, _ = self.embedder.forward(inputs)

        return embeddings

    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return 1.0  # keep lr constant after warmup

        scheduler = {
            'scheduler': optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            'interval': 'step',  # update lr every step
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def mask_spectra(self, spectra, target_bins=(50, 2000, 10), masked_mz_val=-1.0):
        """
        Randomly masks peaks in the input spectra relative weight by their intensity values.

        Args:
            spectra (List[Tensor]): A list of input spectra.

        Returns:
            List[Tensor]: A list of masked spectra.
        """

        # Assert all spectrum shapes are the same
        assert all([spectrum.shape[1] == spectra[0].shape[1] for spectrum in spectra]), "Spectra shapes must be the same"

        assert self.ss_task == "MLM", "Masked Language Model task must be enabled"

        assert self.padding_value is not None, "Padding value must be specified for MLM task"
        assert self.padding_value != masked_mz_val, "Padding value and masked mz value must be different"
        
        masked_spectra = []
        masks = []
        targets = torch.ones_like((len(spectra), spectra[0][:, :, 0]))

        for i, spectrum in enumerate(spectra):
            # Randomly mask peaks based on intensity values
            mask = torch.rand_like(spectrum[:, :, 1]) < spectrum[:, :, 1]

            # Generate labels for the masked peaks
            targets[i][mask] = spectrum[i][mask]

            masked_spectrum = spectrum.clone()
            masked_spectrum[:, :, 0][mask] = masked_mz_val

            masked_spectra.append(masked_spectrum)
            masks.append(mask)

        return masked_spectra, masks, targets