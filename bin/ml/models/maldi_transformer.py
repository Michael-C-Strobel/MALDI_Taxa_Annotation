import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lightning import LightningModule
from torchmetrics.classification import BinaryAUROC, MulticlassAccuracy
# from maldi_nn.utils.drug import (
#     DrugOneHotEmbedding,
#     DrugMLP,
#     DrugGRU,
#     DrugCNN,
#     DrugTransformer,
#     DrugImageCNN,
# )
# import maldi_nn.nn as maldinn
# from maldi_nn.utils.metrics import *

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_pos=20_000):
        super().__init__()
        position = torch.arange(0, max_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        
        # Create the positional encodings
        pe = torch.zeros(max_pos, dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Sine on even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Cosine on odd indices
        
        self.register_buffer("pe", pe)

    def forward(self, x, pos=None):
        if pos is None:
            # Default to sequential positions
            pe = self.pe[: x.shape[1]]
        else:
            # Use custom positions; pos should have shape (batch_size, seq_len)
            pe = self.pe[pos]

        # Add the positional encodings to the input
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

        self.embed = nn.Embedding(1, dim)

        # self.transformer = TransformerEncoder(
        #     depth=depth,
        #     dim=dim,
        #     nh=n_heads,
        #     attentiontype="vanilla",
        #     attention_args={"dropout": dropout},
        #     plugintype="sinusoidal",
        #     plugin_args={"dim": dim, "divide": 10},
        #     only_apply_plugin_at_first=True,
        #     dropout=dropout,
        #     glu_ff=True,
        #     activation="gelu",
        # )

        # This is particularly unfaithful to the source code
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
            ),
            num_layers=depth,
        )

        self.reduce = reduce
        self.positional_encoding = SinusoidalPositionalEncoding(dim)

        self.output_head = nn.Linear(dim, output_head_dim)

    def forward(self, spectrum):
        z = self.embed(spectrum["intensity"])
        z = self.positional_encoding(z, pos=spectrum["mz"])
        z = self.transformer(z)

        if self.reduce == "mean":
            return self.output_head(z.sum(1))
        elif self.reduce == "max":
            return self.output_head(z.max(1).values)
        elif self.reduce == "cls":
            return self.output_head(z[:, 0, :])
        elif self.reduce == "none":
            return z

# Adapted from: https://github.com/gdewael/maldi-nn/blob/92464a1325b273efa639fbf31de5372a5d672a72/maldi_nn/models.py#L283
class MaldiTransformer(LightningModule):
    def __init__(
        self,
        depth,
        dim,
        n_classes=64,   # TODO: Rename
        n_heads=8,
        dropout=0.2,
        p=0.125,
        # clf=False,
        # clf_train_p=1 / 100,
        lr=0.0005,
        weight_decay=0,
        lr_decay_factor=1,
        warmup_steps=2500,
        lmbda=1,
        proportional=False,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            lr_decay_factor=lr_decay_factor,
            warmup_steps=warmup_steps,
        )
        self.save_hyperparameters()
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
        z = self.transformer(batch)
        mlm_logits = self.output_head(z[:, 1:]).squeeze(-1)

        return mlm_logits

    def training_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        batch = self.shuffler(batch)

        mlm_logits = self.forward(batch)

        mlm_logits_train = self.train_indices_select(mlm_logits, batch["train_indices"])
        trues_train = self.train_indices_select(
            batch["intensity_true"], batch["train_indices"]
        )
        mlm_loss = F.binary_cross_entropy_with_logits(
            mlm_logits_train, trues_train.to(self.dtype)
        )

        self.log(
            "train_loss",
            mlm_loss,
            batch_size=len(batch["intensity"]),
        )
        return mlm_loss

    def validation_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        batch = self.shuffler(batch)

        mlm_logits = self.forward(batch)

        mlm_logits_train = self.train_indices_select(mlm_logits, batch["train_indices"])
        trues_train = self.train_indices_select(
            batch["intensity_true"], batch["train_indices"]
        )
        mlm_loss = F.binary_cross_entropy_with_logits(
            mlm_logits_train, trues_train.to(self.dtype)
        )

        self.log(
            "val_mlmloss",
            mlm_loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(mlm_logits_train),
            sync_dist=True,
        )

        self.auroc(mlm_logits_train, trues_train)
        self.log(
            "val_mlmmicro_auc",
            self.auroc,
            on_step=False,
            on_epoch=True,
            batch_size=len(mlm_logits_train),
            sync_dist=True,
        )

    def predict_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        batch = self.shuffler(batch)

        mlm_logits = self.forward(batch)

        logits_train = self.train_indices_select(mlm_logits, batch["train_indices"])
        trues_train = self.train_indices_select(
            batch["intensity_true"], batch["train_indices"]
        )
        mzs_train = self.train_indices_select(batch["mz"], batch["train_indices"])
        intensities_train = self.train_indices_select(
            batch["intensity"], batch["train_indices"]
        )

        return (
            np.array(batch["loc"])[
                torch.where(batch["train_indices"])[0].cpu().numpy()
            ],
            batch["species"][torch.where(batch["train_indices"])[0]],
            mzs_train,
            intensities_train,
            logits_train,
            trues_train,
        )

    def shuffler(self, batch):
        mz = batch["mz"]
        intensity = batch["intensity"]

        all_indices = torch.stack(torch.where(mz)).T

        if self.prop:
            intensities_norm = (
                (intensity / intensity.sum(1)[:, None]).reshape(-1).cpu().numpy()
            )
            shuff, pos = torch.chunk(
                torch.tensor(
                    np.random.choice(
                        len(all_indices),
                        int(len(all_indices) * self.p),
                        replace=False,
                        p=(intensities_norm / intensities_norm.sum()),
                    ),
                    device=all_indices.device,
                ),
                2,
            )

        else:
            shuff, pos = torch.chunk(
                torch.randperm(len(all_indices), device=all_indices.device)[
                    : int(len(all_indices) * self.p)
                ],
                2,
            )
        shuffled_shuff = shuff[torch.randperm(len(shuff), device=shuff.device)]

        indexer = (all_indices[shuff] != all_indices[shuffled_shuff])[:, 0]

        shuff = shuff[indexer]
        shuffled_shuff = shuffled_shuff[indexer]
        pos = pos[indexer]

        train_indices = torch.zeros_like(mz.bool())
        train_indices[all_indices[pos][:, 0], all_indices[pos][:, 1]] = True
        train_indices[all_indices[shuff][:, 0], all_indices[shuff][:, 1]] = True

        intensity_true = torch.ones_like(mz).long()
        intensity_true[all_indices[shuff][:, 0], all_indices[shuff][:, 1]] = 0

        all_indices[shuff] = all_indices[shuffled_shuff]

        batch["mz"] = mz[all_indices[:, 0], all_indices[:, 1]].view_as(mz)
        batch["intensity"] = intensity[all_indices[:, 0], all_indices[:, 1]].view_as(
            intensity
        )
        batch["intensity_true"] = intensity_true
        batch["train_indices"] = train_indices
        return batch