import torch
import torchmetrics
import torch.nn.functional as F
from torch import Tensor

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