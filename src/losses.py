"""
Loss functions for keypoint detection training.
"""

import torch


class FocalHeatmapLoss(torch.nn.Module):
    """Focal loss for heatmap regression - focuses training on hard examples."""
    def __init__(self, alpha=2, beta=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred, target):
        """
        pred: predicted heatmaps (B, K, H, W) - output of sigmoid
        target: ground truth heatmaps (B, K, H, W)
        """
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        
        # Positive locations (peaks) - use threshold > 0.5 for better peak detection
        pos_mask = (target > 0.5).float()
        # Negative locations (background)
        neg_mask = (target <= 0.5).float()
        
        # Focal loss for positive and negative samples
        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred) * pos_mask
        neg_loss = -((1 - target) ** self.beta) * (pred ** self.alpha) * torch.log(1 - pred) * neg_mask
        
        # FIXED: Normalize by total number of pixels, not just positive pixels
        # This prevents massive loss values
        num_pos = pos_mask.sum().clamp(min=1)
        num_total = pred.numel()  # Total number of pixels
        
        pos_loss = pos_loss.sum() / num_pos  # Keep high weight on positive samples
        neg_loss = neg_loss.sum() / num_total  # Normalize negative loss properly
        
        return pos_loss + neg_loss


class UnimodalHeatmapLoss(torch.nn.Module):
    """
    Weighted MSE + vertical focus for keypoint heatmap regression.
    Keeps only the domain-relevant regularizer: tight vertical localization for ruler/pole calibration.
    """
    def __init__(self, pos_weight=8.0, vertical_focus_weight=0.1):
        super().__init__()
        self.pos_weight = pos_weight
        self.vertical_focus_weight = vertical_focus_weight

    def _compute_vertical_focus_loss(self, heatmap):
        """
        Encourage vertical concentration to reduce vertical shifts.
        For ruler keypoints, we want tight vertical localization.
        Returns normalized loss (0-1 range approximately).
        """
        B, K, H, W = heatmap.shape
        
        # Project heatmap to vertical axis (sum over width)
        vertical_projection = heatmap.sum(dim=3)  # (B, K, H)
        
        # Normalize to probability
        vertical_projection = vertical_projection + 1e-8
        vert_prob = vertical_projection / vertical_projection.sum(dim=2, keepdim=True)
        
        # Compute expected vertical position and variance
        y_coords = torch.arange(H, device=heatmap.device, dtype=torch.float32).view(1, 1, H)
        mean_y = (vert_prob * y_coords).sum(dim=2)  # (B, K)
        
        # Variance measures spread - we want low variance (tight vertical concentration)
        y_coords_expanded = y_coords.expand(B, K, H)
        mean_y_expanded = mean_y.unsqueeze(2).expand(B, K, H)
        variance_y = ((y_coords_expanded - mean_y_expanded) ** 2 * vert_prob).sum(dim=2)  # (B, K)
        
        # Normalize variance by maximum possible variance (H^2/12 for uniform distribution)
        # This brings the loss to a 0-1 scale
        max_variance = (H ** 2) / 12.0
        normalized_variance = variance_y / (max_variance + 1e-8)
        
        return normalized_variance.mean()

    def forward(self, pred, target):
        """
        pred: predicted heatmaps (B, K, H, W) - output of sigmoid
        target: ground truth heatmaps (B, K, H, W)
        """
        # Base MSE loss
        mse = (pred - target) ** 2
        
        # Weight positive regions more heavily
        pos_mask = (target > 0.5).float()
        weights = torch.ones_like(target)
        weights = weights + pos_mask * (self.pos_weight - 1)
        
        weighted_mse = (mse * weights).mean()
        vertical_focus_loss = self._compute_vertical_focus_loss(pred)
        base_scale = weighted_mse.detach() + 1e-8
        return weighted_mse + self.vertical_focus_weight * base_scale * vertical_focus_loss

