from maldi_nn.models import MaldiTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import maldi_nn.nn as maldinn
from maldi_nn.utils.metrics import *


class MaldiTransformerWrapper(MaldiTransformer):
    def __init__(
        self,
        depth,
        dim,
        n_classes=64,
        n_heads=8,
        dropout=0.2,
        p=0.125,
        clf=False,
        clf_train_p=1 / 100,
        lr=0.0005,
        weight_decay=0,
        lr_decay_factor=1,
        warmup_steps=2500,
        lmbda=1,
        proportional=False,
    ):
        super().__init__(
            depth,
            dim,
            n_classes=n_classes,
            n_heads=n_heads,
            dropout=dropout,
            p=p,
            clf=clf,
            clf_train_p=clf_train_p,
            lr=lr,
            weight_decay=weight_decay,
            lr_decay_factor=lr_decay_factor,
            warmup_steps=warmup_steps,
            lmbda=lmbda,
            proportional=proportional,
        )

        # super().__init__(
        #     lr=lr,
        #     weight_decay=weight_decay,
        #     lr_decay_factor=lr_decay_factor,
        #     warmup_steps=warmup_steps,
        # )
        # self.save_hyperparameters()
        # self.transformer = maldinn.Transformer(
        #     depth,
        #     dim,
        #     n_heads=n_heads,
        #     dropout=dropout,
        #     cls=True,
        #     reduce="none",
        #     output_head_dim=n_classes,
        # )

        # self.output_head = nn.Linear(dim, 1)

        # self.n_species = n_classes
        # self.clf = clf
        # self.clf_train_p = clf_train_p
        # self.p = p
        # self.lmbda = lmbda
        # self.prop = proportional

        # self.auroc = BinaryAUROC()
        # self.accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro")
        # self.top5_accuracy = MulticlassAccuracy(
        #     num_classes=n_classes, top_k=5, average="micro"
        # )

    def forward(self, batch):
        z = self.transformer(batch)
        mlm_logits = self.output_head(z[:, 1:]).squeeze(-1)

        clf_logits = self.transformer.output_head(z[:, 0, :])
        return mlm_logits, clf_logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        mz = x[:,:,0]
        intensity = x[:,:,1]
        
        # Create a MaldiTransformer-Like Batch (slow in model but effective)
        _batch = dict()
        _batch["intensity"] = mz.to(self.dtype)
        _batch["mz"]        = intensity.to(self.dtype)
        _batch["loc"]       = None                  # Other metadata
        _batch["species"]   = y                     # Species labels (probably integers?)
        batch = _batch

        batch = self.shuffler(batch)                # Adds "train_indices" and "intensity_true" to batch

        mlm_logits, clf_logits = self(batch)

        mlm_logits_train = self.train_indices_select(mlm_logits, batch["train_indices"])
        trues_train = self.train_indices_select(
            batch["intensity_true"], batch["train_indices"]
        )
        mlm_loss = F.binary_cross_entropy_with_logits(
            mlm_logits_train, trues_train.to(self.dtype)
        )

        indexer = batch["species"] < self.n_species
        clf_loss = F.cross_entropy(clf_logits[indexer], batch["species"][indexer])

        self.log(
            "train_loss",
            mlm_loss + clf_loss * (1 if self.clf else 0),
            batch_size=len(batch["intensity"]),
            on_step=True,   # Enable per step logging (different from original)
            on_epoch=True,  # Enable per epoch logging
        )
        return mlm_loss + clf_loss * (
            self.lmbda
            if ((torch.rand(1) < self.clf_train_p).item() and self.clf)
            else 0
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mz = x[:,:,0]
        intensity = x[:,:,1]
        
        # Create a MaldiTransformer-Like Batch (slow in model but effective)
        _batch = dict()
        _batch["intensity"] = mz.to(self.dtype)
        _batch["mz"]        = intensity.to(self.dtype)
        _batch["loc"]       = None                  # Other metadata
        _batch["species"]   = y                     # Species labels (probably integers?)
        batch = _batch
        batch = self.shuffler(batch)

        mlm_logits, clf_logits = self(batch)

        mlm_logits_train = self.train_indices_select(mlm_logits, batch["train_indices"])
        trues_train = self.train_indices_select(
            batch["intensity_true"], batch["train_indices"]
        )
        mlm_loss = F.binary_cross_entropy_with_logits(
            mlm_logits_train, trues_train.to(self.dtype)
        )

        # Change from original: log val_mlmlss as "val_loss" for early stopping
        self.log(
            "val_loss",
            mlm_loss,
            on_step=True,   # Enable per step logging (different from original)
            on_epoch=True,
            batch_size=len(mlm_logits_train),
            sync_dist=True,
        )

        self.auroc(mlm_logits_train, trues_train)
        self.log(
            "val_mlmmicro_auc",
            self.auroc,
            on_step=True,   # Enable per step logging (different from original)
            on_epoch=True,
            batch_size=len(mlm_logits_train),
            sync_dist=True,
        )

        if self.clf:
            indexer = batch["species"] < self.n_species
            if indexer.sum() > 0:
                clf_loss = F.cross_entropy(
                    clf_logits[indexer], batch["species"][indexer]
                )

                self.log(
                    "val_clfloss",
                    clf_loss,
                    on_step=False,
                    on_epoch=True,
                    batch_size=indexer.sum(),
                    sync_dist=True,
                )
                self.accuracy(clf_logits[indexer], batch["species"][indexer])
                self.log(
                    "val_clfacc",
                    self.accuracy,
                    on_step=False,
                    on_epoch=True,
                    batch_size=indexer.sum(),
                    sync_dist=True,
                )
                self.top5_accuracy(clf_logits[indexer], batch["species"][indexer])
                self.log(
                    "val_clftop5_acc",
                    self.top5_accuracy,
                    on_step=False,
                    on_epoch=True,
                    batch_size=indexer.sum(),
                    sync_dist=True,
                )

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError("Predict step not implemented")
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        batch = self.shuffler(batch)

        mlm_logits, clf_logits = self(batch)

        logits_train = self.train_indices_select(mlm_logits, batch["train_indices"])
        trues_train = self.train_indices_select(
            batch["intensity_true"], batch["train_indices"]
        )
        mzs_train = self.train_indices_select(batch["mz"], batch["train_indices"])
        intensities_train = self.train_indices_select(
            batch["intensity"], batch["train_indices"]
        )

        if self.clf:
            return (
                np.array(batch["loc"])[
                    torch.where(batch["train_indices"])[0].cpu().numpy()
                ],
                batch["species"][torch.where(batch["train_indices"])[0]],
                mzs_train,
                intensities_train,
                logits_train,
                trues_train,
                clf_logits,
                batch["species"],
            )
        else:
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

    def modified_shuffler(self, batch):
        """Implementation of the same shuffler, that avoids nan values (padding tokens)"""

    def shuffler(self, batch):
        mz = batch["mz"]
        intensity = batch["intensity"]

        all_indices = torch.stack(torch.where(mz)).T

        # Only change made: require number of samples to be divisible by 2
        desired_num_samples = int(len(all_indices) * self.p)
        if desired_num_samples % 2 != 0:
            desired_num_samples -= 1

        if self.prop:
            intensities_norm = (
                (intensity / intensity.sum(1)[:, None]).reshape(-1).cpu().numpy()
            )
            shuff, pos = torch.chunk(
                torch.tensor(
                    np.random.choice(
                        len(all_indices),
                        desired_num_samples,
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
                    : desired_num_samples
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