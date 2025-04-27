from torch import optim, nn
import torch.nn.utils as utils
import torchmetrics
import torch.nn.functional as F
import torch
import lightning as L
import logging
from .maldi_transformer import SinusoidalPositionalEncoding
import numpy as np

from .CLIP_MALDI import CLIP_MALDI

class PrototyicalTransformer(CLIP_MALDI):
    def __init__(self, n_classes, n_support_samples, n_query_samples, hparams):
        self.n_classes = n_classes                  # Only used at training time to automatically split the data
        self.n_support_samples = n_support_samples  # Only used at training time to automatically split the data
        self.n_query_samples = n_query_samples      # Only used at training time to automatically split the data
        super().__init__(hparams)

    def compute_logits(self, query_embeddings, prototypes):
        # Use Euclidean distance here
        dists = torch.cdist(query_embeddings, prototypes)**2 # Original work squares distance  # shape (Nq_total, Nc)
        return -dists  # logits

    def compute_prototypes(self, embeddings, labels):
        """ Computes prototypes for each class based on the support embeddings """

        # Compute mean embedding per class
        labels = labels.squeeze()
        prototypes = []
        prototype_labels = torch.unique(labels)
        for label in prototype_labels:
            cls_mask = (labels == label)
            cls_embeddings = embeddings[cls_mask]

            # Compute mean embedding for this class
            cls_prototype = cls_embeddings.mean(dim=0)
            prototypes.append(cls_prototype)
        prototypes = torch.stack(prototypes)
        return prototypes, prototype_labels
    
    def compute_loss(self, logits, query_labels):
        return F.cross_entropy(logits, query_labels)
    
    def forward(self, x):
        embeds, raw_embeds, _ = self.embedder(x)

        anchor_rcon = None

        if self.ss_task == 'recon':
            assert self.embedder.prepend_cls == True, "Prepend CLS token must be enabled for reconstruction task"
            anchor_rcon = self.rcon_head(raw_embeds[:, 0, :]) # TODO

            anchor_rcon = F.sigmoid(anchor_rcon)
        if self.ss_task == 'MLM':
            raise NotImplementedError()

        return embeds, anchor_rcon
        

    def training_step(self, batch, batch_idx):
        inputs = batch[0]
        labels = batch[1].to(self.device)

        # Get the support and query samples
        # Nc = len(torch.unique(labels))
        Ns = self.n_support_samples
        Nq = self.n_query_samples
        samples_per_class = Ns + Nq

        support_inputs = []
        support_labels = []
        query_inputs = []
        query_labels = []

        for label in torch.unique(labels):
            cls_mask = (labels == label)
            cls_indices = torch.nonzero(cls_mask)
            if len(cls_indices.shape) > 2:
                cls_indices = cls_indices.squeeze()
            cls_indices = cls_indices[:samples_per_class]

            support_idx = cls_indices[:Ns]
            query_idx = cls_indices[Ns:Ns+Nq]

            support_inputs.append(inputs[support_idx])
            support_labels.append(labels[support_idx])

            query_inputs.append(inputs[query_idx])
            query_labels.append(labels[query_idx])

        support_inputs = torch.cat(support_inputs, dim=0).squeeze()
        support_labels = torch.cat(support_labels, dim=0)
        query_inputs = torch.cat(query_inputs, dim=0).squeeze()
        query_labels = torch.cat(query_labels, dim=0)

        support_embeddings, _ = self.forward(support_inputs)
        query_embeddings, _ = self.forward(query_inputs)

        prototypes, prototype_labels = self.compute_prototypes(support_embeddings, support_labels)
        logits = self.compute_logits(query_embeddings, prototypes)

        label_map = {label.item(): i for i, label in enumerate(prototype_labels)}
        query_labels_remapped = torch.tensor([label_map[l.item()] for l in query_labels], device=query_labels.device)

        loss = self.compute_loss(logits, query_labels_remapped)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == query_labels_remapped).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch[0]
        labels = batch[1].to(self.device)


        # Get the support and query samples
        # Nc = len(torch.unique(labels))
        Ns = self.n_support_samples
        Nq = self.n_query_samples
        samples_per_class = Ns + Nq

        support_inputs = []
        support_labels = []
        query_inputs = []
        query_labels = []

        # print("torch.unique(labels): ", torch.unique(labels))

        for label in torch.unique(labels):
            # print("Label: ", label)
            cls_mask = (labels == label)
            cls_indices = torch.nonzero(cls_mask)
            if len(cls_indices.shape) > 2:
                cls_indices = cls_indices.squeeze()
            cls_indices = cls_indices[:samples_per_class]

            # For each class get Ns and Nq samples
            support_idx = cls_indices[:Ns]
            query_idx = cls_indices[Ns:Ns+Nq]

            support_inputs.append(inputs[support_idx])
            support_labels.append(labels[support_idx])
            # print("Support labels: ", support_labels[-1])

            query_inputs.append(inputs[query_idx])
            query_labels.append(labels[query_idx])
            # print("Query labels: ", query_labels[-1])

        support_inputs = torch.cat(support_inputs, dim=0).squeeze()
        support_labels = torch.cat(support_labels, dim=0)
        query_inputs = torch.cat(query_inputs, dim=0).squeeze()
        query_labels = torch.cat(query_labels, dim=0)

        support_embeddings, _ = self.forward(support_inputs)
        query_embeddings, _ = self.forward(query_inputs)

        # print("Support labels: ", support_labels)


        prototypes, prototype_labels = self.compute_prototypes(support_embeddings, support_labels)
        logits = self.compute_logits(query_embeddings, prototypes)

        label_map = {label.item(): i for i, label in enumerate(prototype_labels)}
        query_labels_remapped = torch.tensor([label_map[l.item()] for l in query_labels], device=query_labels.device)

        # print("Predictions: ", logits.argmax(dim=1))
        # print("Query labels remapped: ", query_labels_remapped)

        loss = self.compute_loss(logits, query_labels_remapped)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == query_labels_remapped).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss
    
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