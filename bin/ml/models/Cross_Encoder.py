from torch import optim, nn
import torch.nn.utils as utils
import torchmetrics
import torch.nn.functional as F
import torch
import lightning as L
import logging
from .maldi_transformer import SinusoidalPositionalEncoding
from .CLIP_MALDI import CLIP_MALDI


from .CLIP_MALDI import CustromBinaryMetric, Embedder, SimpleSelfAttention, clip_contrastive_loss, reconstruction_loss
from collections import Counter
     
class Cross_Encoder(CLIP_MALDI):
    """ 
    """
    def __init__(self, hyperparameters, pretrained_embedder=None):
        super().__init__(hyperparameters, pretrained_embedder, transformer_reduction='none')
        self.is_binary_classifier = True

        self.max_in_batch_negatives = hyperparameters.get("max_in_batch_negatives", 4)

        self.sep_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        self.cross_att = SimpleSelfAttention(2,  # depth
                                            self.hidden_dim,
                                            n_heads=10,
                                            dropout=self.dropout_rate,
                                            output_head_dim=128,
                                            padding_value=self.padding_value,
                                            reduce='max',
                                            prepend_cls=True,
                                            concat_pos=self.concat_pos)
        self.postprocess = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, anchors, others):
        # TODO Concat to product a single tensor
        anchors_embeds, anchor_raw_embeds, _ = self.embedder(anchors)
        pos_embeds, pos_raw_embeds, _ = self.embedder(others)

        fw_actual_batch_size = anchors_embeds.shape[0]

        # print("anchor shape", anchors_embeds.shape)
        # print("pos shape", pos_embeds.shape)
        
        _sep_token = torch.repeat_interleave(self.sep_token, fw_actual_batch_size, dim=0)

        # print("_sep_token shape", _sep_token.shape)

        x = torch.cat([anchors_embeds, _sep_token, pos_embeds], dim=1)

        # print("Combined shape", x.shape)

        # Forward through the cross-attention module
        x, _, _ = self.cross_att(x)

        # assert x.ndim == 2 and x.shape[1] == self.cross_att.output_head, (
        #     f"cross_att output shape mismatch: expected (B, hidden_dim={self.cross_att.output_head_dim}), "
        #     f"got {tuple(x.shape)} — check reduce='max' in SimpleSelfAttention."
        # )

        # print("After attention shape", x.shape)
        x = self.postprocess(x)
        # print("After postprocess shape", x.shape)

        return x
    
    def mz_only_cosine(
        self,
        anchors,
        others,
    ):
        """
        Given a batch of spectra, compute the cosine similarity matrix based on the discretized precursor m/z values.
        
        Args:
            anchors: Tensor of shape (B1, N1, 2) where each entry is (mz, intensity)
            others: Tensor of shape (B2, N2, 2) where each entry is (mz, intensity)

        Returns:
            cosine_sim: Tensor of shape (B1, B2) with cosine similarity scores        
        """
        # Cast everything to an int 
        resolution = 10 # 10 Da bins
        anchors_mz = torch.round(anchors[:, :, 0] / resolution).long()
        others_mz = torch.round(others[:, :, 0] / resolution).long()

        max_mz = max(anchors_mz.max(), others_mz.max())

        # Create binary vectors for each spectrum | TODO: Consider using sets?
        anchors_bin = torch.zeros((anchors.shape[0], int(max_mz) + 1), device=anchors.device)
        others_bin = torch.zeros((others.shape[0], int(max_mz) + 1), device=others.device)
        anchors_bin.scatter_(1, anchors_mz, 1)
        others_bin.scatter_(1, others_mz, 1)

        # l2 Norm
        anchors_bin = F.normalize(anchors_bin, p=2, dim=1)
        others_bin = F.normalize(others_bin, p=2, dim=1)

        # Compute cosine similarity
        cosine_sim = torch.matmul(anchors_bin, others_bin.T)
        return cosine_sim


    def construct_inbatch_pairs(
        self,
        anchors,
        positives,
        anchor_class,
        max_negatives=None,
        hard_negatives=True
    ):
        """
        Construct positive + in-batch negative pairs for cross-encoder training.
        
        Args:
            anchors: Tensor of shape (B, F) anchor embeddings/spectra
            positives: Tensor of shape (B, F) positive embeddings/spectra
            anchor_class: list or tensor of length B indicating class for each anchor
            max_negatives: int, maximum number of in-batch negatives per anchor
            hard_negatives: bool, if True, sample negatives proportional to cosine similarity
        
        Returns:
            anchors_flat: Tensor of shape (B*(1+N_neg), F)
            candidates_flat: Tensor of shape (B*(1+N_neg), F)
            N_neg: number of negatives used per anchor
        """
        B = anchors.shape[0]
        N_neg = min(max_negatives or self.max_in_batch_negatives, B - 1)

        all_anchor_pairs = []
        all_candidate_pairs = []

        for i in range(B):
            # Positive pair
            all_anchor_pairs.append(anchors[i])
            all_candidate_pairs.append(positives[i])

            # Candidate negatives: different class and not self
            neg_candidates = [j for j in range(B) if j != i and anchor_class[j] != anchor_class[i]]

            if len(neg_candidates) == 0:
                raise ValueError(
                    "No negative candidates found. Ensure that your batch contains multiple classes."
                )

            # Repeat if too few negatives
            if len(neg_candidates) < N_neg:
                neg_candidates = neg_candidates * (N_neg // len(neg_candidates)) + neg_candidates[:N_neg % len(neg_candidates)]

            # Hard negative sampling
            if hard_negatives:
                # Compute cosine similarity between anchor[i] and candidate negatives
                anchor_vec = anchors[i].unsqueeze(0)  # shape (1, F)
                neg_vectors = torch.stack([positives[j] for j in neg_candidates])  # shape (len(neg_candidates), F)

                sim = self.mz_only_cosine(anchor_vec, neg_vectors)**2  # shape (len(neg_candidates),)

                # Convert similarity to probabilities (higher similarity = higher prob)
                probs = sim / sim.sum()
                requires_replacement = len(probs) < N_neg

                sampled_idxs = torch.multinomial(probs, N_neg, replacement=requires_replacement).squeeze()
                if False:
                    print("sim.shape", sim.shape, flush=True)
                    print("sampled_idxs.shape", sampled_idxs.shape, flush=True)

                    # Debug sanity check: Show the cosine score of the sampled negatives vs all candidates
                    # Take root
                    with torch.no_grad():
                        sampled_sims = torch.sqrt(sim)
                        sampled_sims = sampled_sims/sampled_sims.sum()
                        sampled_sims = sampled_sims.squeeze()[sampled_idxs]
                        all_sims = sim
                        print("Sampled sims:", sampled_sims, flush=True)
                        print("All sims:", all_sims, flush=True)

                        print(f"Anchor {i}: Sampled negative sims: mean {sampled_sims.mean()}, std {sampled_sims.std()}", flush=True)
                        print(f"Anchor {i}: All candidate sims: mean {all_sims.mean()}, std {all_sims.std()}", flush=True)
            else:
                # Uniform random sampling
                sampled_idxs = torch.randperm(len(neg_candidates))[:N_neg].tolist()

            # Append sampled negatives
            for j in sampled_idxs:
                all_anchor_pairs.append(anchors[i])
                all_candidate_pairs.append(positives[neg_candidates[j]])

        anchors_flat = torch.stack(all_anchor_pairs)
        candidates_flat = torch.stack(all_candidate_pairs)
        return anchors_flat, candidates_flat, N_neg

    def training_step(self, batch, batch_idx):
        # Batch is a list of:
        # spectra: (anchor, positive)
        # metadata: (anchor_metadata, positive_metadata)
        # similarity: (sim,)

        spectra = batch[0]
        anchor_class = batch[1][0]['class'] # N.b., we may have multiple positives in our batch for small datasets, this is not handled
        # Count number of identical classes
        anchor_class_counts = Counter(anchor_class)
        # Check how many duplicates there are
        # duplicates_exist = any(count > 1 for count in anchor_class_counts.values())
        # if duplicates_exist:
        #     logging.warning(f"Duplicate classes found in batch: {anchor_class_counts}")
        anchors = spectra[0]
        positives = spectra[1]
        B = anchors.shape[0]
        
        anchors_flat, candidates_flat, N_neg = self.construct_inbatch_pairs(
            anchors, positives, anchor_class
        )

        scores_flat = self.forward(anchors_flat, candidates_flat)
        scores = scores_flat.view(B, 1 + N_neg)
        labels = torch.zeros(B, dtype=torch.long, device=scores.device)
        ranking_loss = torch.nn.functional.cross_entropy(scores, labels)
        preds = torch.argmax(scores, dim=1)
        acc = (preds == labels).float().mean()

        loss = ranking_loss
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True)

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
        B = anchors.shape[0]
    
        anchors_flat, candidates_flat, N_neg = self.construct_inbatch_pairs(
            anchors, positives, anchor_class
        )

        scores_flat = self.forward(anchors_flat, candidates_flat)
        scores = scores_flat.view(B, 1 + N_neg)
        labels = torch.zeros(B, dtype=torch.long, device=scores.device)
        ranking_loss = torch.nn.functional.cross_entropy(scores, labels)
        preds = torch.argmax(scores, dim=1)
        acc = (preds == labels).float().mean()

        loss = ranking_loss
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Test step not implemented")
        spectrum_a, spectrum_b, similarity, metadata = batch
        embed_1 = self(spectrum_a)
        embed_2 = self(spectrum_b)
        preds = F.cosine_similarity(embed_1, embed_2)
        loss = nn.functional.mse_loss(preds, similarity)
        return {'predictions': preds, 'similarity': similarity, 'loss': loss}

    def on_predict_start(self):
        raise NotImplementedError("Predict step not implemented")
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

    def binary_predict_step(self, batch, dataloader_idx=None):

        spectrum_a, spectrum_b = batch

        embed_a = None
        embed_b = None

        preds = self.forward(spectrum_a, spectrum_b).squeeze()

        return F.sigmoid(preds)
    
    def embed_step(self, batch):
        raise NotImplementedError("Embed step not implemented")
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
    