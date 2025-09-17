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
        # print("After attention shape", x.shape)
        x = self.postprocess(x)
        # print("After postprocess shape", x.shape)

        return x

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
        
        # ----------------------------
        # Construct in-batch negatives
        # ----------------------------
        N_neg = min(self.max_in_batch_negatives, B-1)  # configurable
        all_anchor_pairs = []
        all_candidate_pairs = []

        for i in range(B):
            all_anchor_pairs.append(anchors[i])
            all_candidate_pairs.append(positives[i])

            # Only sample negatives that are a different class
            neg_candidates = [j for j in range(B) if j != i and anchor_class[j] != anchor_class[i]]
            # If neg_candidates < N_neg, repeat it
            if len(neg_candidates) == 0:
                raise ValueError("No negative candidates found. Ensure that your batch contains multiple classes.")
            if len(neg_candidates) < N_neg:
                neg_candidates = neg_candidates * (N_neg // len(neg_candidates)) + neg_candidates[:N_neg % len(neg_candidates)]
            if len(neg_candidates) > 0:
                sampled = torch.randperm(len(neg_candidates))[:N_neg]
                for idx in sampled:
                    j = neg_candidates[idx]
                    all_anchor_pairs.append(anchors[i])
                    all_candidate_pairs.append(positives[j])

        anchors_flat = torch.stack(all_anchor_pairs)
        candidates_flat = torch.stack(all_candidate_pairs)

        # Forward pass
        # (anchor_embeds, positive_embeds), (anchor_rcon, pos_rcon) = self.forward(torch.stack(anchors), torch.stack(positives))
        scores_flat = self.forward(anchors_flat, candidates_flat)

        scores = scores_flat.view(B, 1 + N_neg)
        # print('scores.shape', scores.shape)
        labels = torch.zeros(B, dtype=torch.long, device=scores.device)
        ranking_loss = torch.nn.functional.cross_entropy(scores, labels)
        preds = torch.argmax(scores, dim=1) 
        acc = (preds == labels).float().mean()

        loss = ranking_loss

        # ----------------------------
        # Logging
        # ----------------------------
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
    
        # ----------------------------
        # Construct in-batch negatives
        # ----------------------------
        N_neg = min(self.max_in_batch_negatives, B-1)  # configurable
        all_anchor_pairs = []
        all_candidate_pairs = []

        for i in range(B):
            all_anchor_pairs.append(anchors[i])
            all_candidate_pairs.append(positives[i])

            # Only sample negatives that are a different class
            neg_candidates = [j for j in range(B) if j != i and anchor_class[j] != anchor_class[i]]
            # If neg_candidates < N_neg, repeat it
            if len(neg_candidates) == 0:
                raise ValueError("No negative candidates found. Ensure that your batch contains multiple classes.")
            if len(neg_candidates) < N_neg:
                neg_candidates = neg_candidates * (N_neg // len(neg_candidates)) + neg_candidates[:N_neg % len(neg_candidates)]
            if len(neg_candidates) > 0:
                sampled = torch.randperm(len(neg_candidates))[:N_neg]
                for idx in sampled:
                    j = neg_candidates[idx]
                    all_anchor_pairs.append(anchors[i])
                    all_candidate_pairs.append(positives[j])

        anchors_flat = torch.stack(all_anchor_pairs)
        candidates_flat = torch.stack(all_candidate_pairs)

        # Forward pass
        # (anchor_embeds, positive_embeds), (anchor_rcon, pos_rcon) = self.forward(torch.stack(anchors), torch.stack(positives))
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

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        raise NotImplementedError("Predict step not implemented")
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