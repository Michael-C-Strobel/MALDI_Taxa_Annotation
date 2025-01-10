import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics
from torchmetrics.classification import BinaryAUROC, MulticlassAccuracy
from .autoencoder import BCE_Metric

class SinusoidalPositionalEncoding(nn.Module):  # TODO: Somehow creating nan vals
    def __init__(self, dim):
        super().__init__()
        # Register div_term as a buffer to avoid being updated by the optimizer
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        self.register_buffer("div_term", div_term)

    def forward(self, x, pos):#, mask):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, dim) - Input features.
            pos: Tensor of shape (batch_size, seq_len) - Continuous positions.
        """
        pos = pos.unsqueeze(-1)  # (batch_size, seq_len, 1) for broadcasting
        pe = torch.zeros_like(x)
        
        # print('pe.shape', pe.shape)
        # print('torch.sin(pos * self.div_term).shape', torch.sin(pos * self.div_term).shape)
        # Compute the sin and cosine embeddings
        pe[..., 0::2] = torch.sin(pos * self.div_term)  # Sine on even indices
        pe[..., 1::2] = torch.cos(pos * self.div_term)  # Cosine on odd indices

        # if mask is not None:
        #     pe = pe.masked_fill(mask.unsqueeze(-1), 0)

            # if torch.isnan(pe[~mask]).any():
            #     print('x.shape', x.shape)
            #     print('pos.shape', pos.shape)
            #     print('pe.shape', pe.shape)
            #     print('mask.shape', mask.shape)
            #     raise ValueError("Nan values in pe")

        return x + pe
    
# Adapted From: https://github.com/gdewael/maldi-nn/blob/92464a1325b273efa639fbf31de5372a5d672a72/maldi_nn/nn.py#L48
class Transformer(nn.Module):
    def __init__(
        self,
        depth,
        dim,
        n_heads=8,
        dropout=0.2,
        reduce="none",
        output_head_dim=64,
    ):
        super().__init__()

        self.embed = nn.Linear(1, dim)

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
        padding = (spectrum[:,:,0] == 0).bool()    # True indicates padding
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
        if torch.isnan(z)[~padding].any():
            raise ValueError("Nan values in z that aren't in padding")
        z = self.positional_encoding(z, pos=spectrum[:,:,0])    # Use m/z values as positions

        if torch.isnan(z)[~padding].any():
            raise ValueError("Nan values in z that aren't in padding")

        modified_z = z.detach().clone()
        modified_z[padding] = 0

        if torch.isnan(modified_z).any():
            raise ValueError("Nan values in modified_z that aren't in padding")

        z = self.transformer(z, src_key_padding_mask=None) # Somehow, we don't actually allow padding?

        # Assert nothing is nan that isn't padded
        # print(z.shape)
        # print("torch.isnan(z)[~padding]", torch.isnan(z)[~padding].shape)
        # print(z)

        if torch.isnan(z).all():
            raise ValueError("All values in z are NaN")    # WHY IS IT ERRORING HERE?
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

# Adapted from: https://github.com/gdewael/maldi-nn/blob/92464a1325b273efa639fbf31de5372a5d672a72/maldi_nn/models.py#L283
class MaldiTransformer(LightningModule):
    def __init__(
        self,
        hyperparameters,
        padding_value:float=None
    ):
        super().__init__()

        self.train_metrics = torchmetrics.MetricCollection({
            # binary_cross_entropy_with_logits
            'bce': BCE_Metric(),
        },
        prefix='train_')
        self.val_metrics = torchmetrics.MetricCollection({
            'bce': BCE_Metric(),
            # 'val_mlmmicro_auc': BinaryAUROC(),
        },
        prefix='val_'
        )

        # super().__init__(
        #     lr=lr,
        #     weight_decay=weight_decay,
        #     lr_decay_factor=lr_decay_factor,
        #     warmup_steps=warmup_steps,
        # )

        for key in hyperparameters.keys():
            self.hparams.update({key: hyperparameters[key]})
        self.save_hyperparameters()

        self.padding_value = padding_value

        depth = self.hparams.depth
        dim = self.hparams.dim
        n_classes = self.hparams.n_classes
        n_heads = self.hparams.n_heads
        dropout = self.hparams.dropout
        p = self.hparams.p
        lmbda = self.hparams.lmbda
        proportional = self.hparams.proportional

        self.lr = self.hparams.get('lr', 5e-4)
        self.weight_decay = self.hparams.get('weight_decay', 0.0)
        if 'lr' in self.hparams:
            print(f"Using learning rate: {self.lr}")


        self.transformer = Transformer(
            depth,
            dim,
            n_heads=n_heads,
            dropout=dropout,
            reduce="none",
            output_head_dim=n_classes,
        )

        self.output_head = nn.Linear(dim, 1)

        # self.n_species = n_classes
        # self.clf = clf
        # self.clf_train_p = clf_train_p
        self.p = p
        self.lmbda = lmbda
        self.prop = proportional

    def forward(self, batch):
        z, padding = self.transformer(batch)
        mlm_logits = self.output_head(z).squeeze(-1)

        return mlm_logits, padding

    def train_indices_select(self, in_, train_indices):
        if train_indices.dtype == torch.long:
            return torch.gather(in_, 1, train_indices)
        elif train_indices.dtype == torch.bool:
            return in_[train_indices]
        else:
            raise ValueError("train_indices should be either .long or .bool")

    # def training_step(self, batch, batch_idx):
    #     batch, metadata = batch
    #     batch_intensity = batch[:,:,1].to(self.dtype)
    #     batch_mz = batch[:,:,0].to(self.dtype)

    #     shuffled = self.shuffler(batch)
    #     batch = shuffled['modified_batch']
    #     train_indices = shuffled['train_indices']
    #     intensity_true = shuffled['intensity_true']

    #     mlm_logits, padding_mask = self.forward(batch)

    #     mlm_logits_train = self.train_indices_select(mlm_logits, train_indices)
    #     trues_train = self.train_indices_select(
    #         intensity_true, train_indices
    #     )
    #     mlm_loss = F.binary_cross_entropy_with_logits(
    #         mlm_logits_train, trues_train.to(self.dtype)
    #     )

    #     self.log(
    #         "train_loss",
    #         mlm_loss,
    #         batch_size=len(batch[:,:,1]),
    #     )
    #     return mlm_loss
    def training_step(self, batch, batch_idx):
        batch, metadata = batch
        batch_intensity = batch[:,:,1].to(self.dtype)
        batch_mz = batch[:,:,0].to(self.dtype)

        shuffled = self.shuffler(batch)
        batch = shuffled['modified_batch']
        train_indices = shuffled['train_indices']
        intensity_true = shuffled['intensity_true']

        mlm_logits, padding_mask = self.forward(batch)
        
        
        # Apply the mask to logits and true values
        mlm_logits_train = self.train_indices_select(mlm_logits, train_indices)
        trues_train = self.train_indices_select(intensity_true, train_indices)

        if torch.isnan(mlm_logits_train).any():
            raise ValueError("Nan values in mlm_logits_train")
        
        if torch.isnan(trues_train).any():
            raise ValueError("Nan values in trues_train")
        # Use the valid_mask to exclude padding during loss calculation
        mlm_loss = F.binary_cross_entropy_with_logits(
            mlm_logits_train,  # Only valid entries
            trues_train.to(self.dtype)  # Only valid entries
        )

        # self.log(
        #     "train_loss",
        #     mlm_loss,
        #     batch_size=len(batch[:,:,1]),
        # )
        # print('mlm_loss', mlm_loss)

        batch_value = self.train_metrics(mlm_logits_train, trues_train)
        self.log_dict(batch_value, on_step=True, on_epoch=True)

        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", current_lr, prog_bar=True, on_step=True, on_epoch=True)
        self.log("true_percent_zeros", torch.sum(trues_train == 0) / len(trues_train), prog_bar=True, on_step=True, on_epoch=True)
        as_predictions = torch.sigmoid(mlm_logits_train) > 0.5

        self.log("pred_percent_zeros", torch.sum(as_predictions == 0) / len(as_predictions), prog_bar=False, on_step=True, on_epoch=True)

        # print('mlm_loss',mlm_loss)
        return mlm_loss


    def validation_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        shuffled = self.shuffler(batch)
        batch = shuffled['modified_batch']
        train_indices = shuffled['train_indices']
        intensity_true = shuffled['intensity_true']

        mlm_logits = self.forward(batch)

        mlm_logits_train = self.train_indices_select(mlm_logits, train_indices)
        trues_train = self.train_indices_select(
            intensity_true, train_indices
        )
        mlm_loss = F.binary_cross_entropy_with_logits(
            mlm_logits_train, trues_train.to(self.dtype)
        )

        # self.log(
        #     "val_mlmloss",
        #     mlm_loss,
        #     on_step=False,
        #     on_epoch=True,
        #     batch_size=len(mlm_logits_train),
        #     sync_dist=True,
        # )

        # self.auroc(mlm_logits_train, trues_train)
        # self.log(
        #     "val_mlmmicro_auc",
        #     self.auroc,
        #     on_step=False,
        #     on_epoch=True,
        #     batch_size=len(mlm_logits_train),
        #     sync_dist=True,
        # )

        batch_value = self.val_metrics(mlm_logits_train, trues_train)
        self.log_dict(batch_value, on_epoch=True)



        return mlm_loss

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError("Predict step not implemented")
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        shuffled = self.shuffler(batch)
        batch = shuffled['modified_batch']
        train_indices = shuffled['train_indices']
        intensity_true = shuffled['intensity_true']

        mlm_logits = self.forward(batch)

        logits_train = self.train_indices_select(mlm_logits, train_indices)
        trues_train = self.train_indices_select(
            intensity_true, train_indices
        )
        mzs_train = self.train_indices_select(batch["mz"], train_indices)
        intensities_train = self.train_indices_select(
            batch["intensity"], train_indices
        )

        return (
            np.array(batch["loc"])[
                torch.where(train_indices)[0].cpu().numpy()
            ],
            batch["species"][torch.where(train_indices)[0]],
            mzs_train,
            intensities_train,
            logits_train,
            trues_train,
        )

    # def shuffler(self, batch):
    #     batch = batch.detach().clone()
    #     mz = batch[:, :, 0]
    #     intensity = batch[:, :, 1]

    #     all_indices = torch.stack(torch.where(mz)).T
    #     # all_indices = torch.stack(torch.where(~torch.isnan(intensity))).T

    #     if self.prop:
    #         raise NotImplementedError("Proportional shuffling not implemented")
    #         intensities_norm = (
    #             (intensity / intensity.sum(1)[:, None]).reshape(-1).cpu().numpy()
    #         )
    #         shuff, pos = torch.chunk(
    #             torch.tensor(
    #                 np.random.choice(
    #                     len(all_indices),
    #                     int(len(all_indices) * self.p),
    #                     replace=False,
    #                     p=(intensities_norm / intensities_norm.sum()),
    #                 ),
    #                 device=all_indices.device,
    #             ),
    #             2,
    #         )

    #     else:
    #         desired_num = int(len(all_indices) * self.p)
    #         if desired_num % 2 != 0:
    #             desired_num -= 1

    #         shuff, pos = torch.chunk(
    #             torch.randperm(len(all_indices), device=all_indices.device)[
    #                 : desired_num
    #             ],
                
    #             2,
    #         )

    #     shuffled_shuff = shuff[torch.randperm(len(shuff), device=shuff.device)]

    #     # Boolean mask that merges where peaks will be disjoint?
    #     indexer = (all_indices[shuff] != all_indices[shuffled_shuff])[:, 0]

    #     shuff = shuff[indexer]
    #     shuffled_shuff = shuffled_shuff[indexer]
    #     pos = pos[indexer]

    #     train_indices = torch.zeros_like(mz.bool())
    #     train_indices[all_indices[pos][:, 0], all_indices[pos][:, 1]] = True
    #     train_indices[all_indices[shuff][:, 0], all_indices[shuff][:, 1]] = True

    #     intensity_true = torch.ones_like(mz).long()
    #     intensity_true[all_indices[shuff][:, 0], all_indices[shuff][:, 1]] = 0

    #     all_indices[shuff] = all_indices[shuffled_shuff]

    #     # Repad the batch
    #     # new_mz = torch.ones_like(mz) * torch.nan
    #     new_mz = torch.zeros_like(mz)
    #     # new_intensity = torch.ones_like(intensity) * torch.nan
    #     new_intensity = torch.zeros_like(intensity)# * torch.nan
    #     if (mz[all_indices[:, 0], all_indices[:, 1]]==0).any():
    #         raise ValueError("Nan values in new_mz")
    #     if (intensity[all_indices[:, 0], all_indices[:, 1]]==0).any():
    #         raise ValueError("Nan values in new_intensity")
        
    #     mz_to_copy = mz[all_indices[:, 0], all_indices[:, 1]].view(mz.shape[0], -1)
    #     new_mz[:mz_to_copy.shape[0], :mz_to_copy.shape[1]] = mz_to_copy
        
    #     intensity_to_copy = intensity[all_indices[:, 0], all_indices[:, 1]].view(intensity.shape[0], -1)
    #     new_intensity[:intensity_to_copy.shape[0], :intensity_to_copy.shape[1]] = intensity_to_copy

    #     batch[:, :, 0] = new_mz
    #     batch[:, :, 1] = new_intensity

    #     # Not idea what these are yet
    #     return {
    #             'modified_batch': batch,
    #             'train_indices': train_indices,
    #             'intensity_true': intensity_true,
    #             }
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        
        # Define the warmup + constant LR scheduler
        def lr_lambda(step):
            if step < 250:
                return step / 250  # Linearly scale from 0 to 1 over 250 steps
            return 1.0  # Keep LR constant after warmup

        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda),
            'interval': 'step',  # Update every step
            'frequency': 1       # Ensure it runs each step
        }

        return [optimizer], [scheduler]
    
    def _padded_shuffler(self, batch):
        """Shuffles peaks between spectra, and returns the shuffled spectra. Will not shuffle padding tokents.
        Notably this function is not fully vectorized and may inversely affect performance.
        
        Args:
            batch: Tensor of shape (batch_size, seq_len, 2) - Batch of spectra.

        Returns:
            modified_batch: Tensor of shape (batch_size, seq_len, 2) - Shuffled spectra.
            train_indices: Tensor of shape (batch_size, seq_len) - Boolean mask of shuffled peaks.
            intensity_true: Tensor of shape (batch_size, seq_len) - Boolean mask of shuffled peaks.
        """
        batch = batch.detach().clone()
        mz = batch[:, :, 0]
        intensity = batch[:, :, 1]
        batch_size = len(batch)
        assert batch_size > 1, "Batch size must be greater than 1 for padded shuffling"
        assert self.padding_value is not None, "Padding value must be set for padded shuffling"

        lengths_per_batch = torch.sum(mz != self.padding_value, dim=1)
        desired_nums = (lengths_per_batch * self.p).long()

        # For each spectrum, randomly sample desired_nums[i] from (0,lengths_per_batch[i]) 
        rand_perms = [torch.randperm(lengths_per_batch[i]) for i in range(batch_size)]
        indices_to_drop = [perm[:desired_nums[i]] for i,perm in enumerate(rand_perms)]

        # Randomly sample additional indices to train on 
        positive_indices = [perm[desired_nums[i]:desired_nums[i]*2] for i,perm in enumerate(rand_perms)]

        # For each spectrum, randomly sample desired_nums[i] for 0,lengths_per_batch[k]) for k!= i, where k is in the index of random source spectra
        # For each target spectrum, generate a list of source spectra of length desired_nums[i]
        # Weighted sample
        weights = torch.ones(batch_size, batch_size)
        weights[torch.eye(batch_size).bool()] = 0
        source_spectra = [torch.multinomial(weights[i], desired_nums[i], replacement=True) for i in range(batch_size)]
        # For each source spectra, sample desired_nums[i] from (0,lengths_per_batch[i])
        source_indices = [(source_spectra[i], torch.stack([torch.randperm(lengths_per_batch[k])[0] for k in source_spectra[i]])) for i in range(batch_size)]
        # source_indices is now a list of [(source_spectra_index, peak_num)] for each target spectrum
        
        # For each spectrum, replace the indices to drop with the source indices
        new_mz = mz.clone()
        new_intensity = intensity.clone()
        for i in range(batch_size):
            new_mz[i, indices_to_drop[i]] = mz[source_indices[i]]
            new_intensity[i, indices_to_drop[i]] = intensity[source_indices[i]]

        # Create the boolean mask that selects the indices we will train on (shuffled peaks, and positive/static peaks)
        train_indices = torch.zeros_like(mz).bool()
        for i in range(batch_size):
            train_indices[i, torch.concat([indices_to_drop[i], positive_indices[i]])] = True

        # Create the boolean target vector for shuffled/not shuffled predictionv
        intensity_true = torch.ones_like(mz).long()
        for i in range(batch_size):
            intensity_true[i, indices_to_drop[i]] = 0

        return {
            'modified_batch': torch.stack([new_mz, new_intensity], dim=-1),
            'train_indices': train_indices,
            'intensity_true': intensity_true,
        }

    def _standard_shuffler(self, batch)->dict:
        """
        Shuffles peaks between spectra, and returns the shuffled spectra. Can include padding tokens

        Args:
            batch: Tensor of shape (batch_size, seq_len, 2) - Batch of spectra.

        Returns:
            modified_batch: Tensor of shape (batch_size, seq_len, 2) - Shuffled spectra.
            train_indices: Tensor of shape (batch_size, seq_len) - Boolean mask of shuffled peaks.
            intensity_true: Tensor of shape (batch_size, seq_len) - Boolean mask of shuffled peaks.
        """
        batch = batch.detach().clone()
        mz = batch[:, :, 0]
        intensity = batch[:, :, 1]

        all_indices = torch.stack(torch.where(mz)).T
        
        desired_num = int(len(all_indices) * self.p)
        if desired_num % 2 != 0:
            desired_num -= 1

        shuff, pos = torch.chunk(
            torch.randperm(len(all_indices), device=all_indices.device)[
                : desired_num
            ],
            2,
        )
        shuffled_shuff = shuff[torch.randperm(len(shuff), device=shuff.device)] # Randomly shuffle the indices we will shuffle

        indexer = (all_indices[shuff] != all_indices[shuffled_shuff])[:, 0] # Boolean mask for only the indices not equal to each other


        shuff = shuff[indexer]
        shuffled_shuff = shuffled_shuff[indexer]
        pos = pos[indexer]

        train_indices = torch.zeros_like(mz.bool())
        train_indices[all_indices[pos][:, 0], all_indices[pos][:, 1]] = True
        train_indices[all_indices[shuff][:, 0], all_indices[shuff][:, 1]] = True

        intensity_true = torch.ones_like(mz).long()
        intensity_true[all_indices[shuff][:, 0], all_indices[shuff][:, 1]] = 0

        all_indices[shuff] = all_indices[shuffled_shuff]

        new_mz = mz[all_indices[:, 0], all_indices[:, 1]].view(mz.shape[0], -1)
        new_intensity = intensity[all_indices[:, 0], all_indices[:, 1]].view(intensity.shape[0], -1)
        return {
            'modified_batch': torch.stack([new_mz, new_intensity], dim=-1),
            'train_indices': train_indices,
            'intensity_true': intensity_true,
        }

    def shuffler(self, batch):
        if self.padding_value is not None:
            return self._padded_shuffler(batch)
        else:
            return self._standard_shuffler(batch)