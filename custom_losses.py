import torch
import torch.nn as nn
from monai.losses import (DiceCELoss, GeneralizedDiceLoss, GeneralizedWassersteinDiceLoss, GeneralizedDiceFocalLoss,
                          DiceFocalLoss)


class CelDlLoss(DiceCELoss):
    """
    Combined Categorical Cross-Entropy and Dice Loss.

    This loss function combines CEL for probabilistic accuracy with Generalized Dice Loss
    for handling class imbalance and overlap in multiclass segmentation tasks.

    Args:
        weight_cel (float): Weighting factor for balancing CEL and Dice losses (default: 0.5).
        weight (Tensor): Class weights for CEL and DL.
        to_onehot_y (bool): Must be set to True if y_true shape is (B, L) (default: True)
        softmax (bool): Whether to apply softmax to y_pred. (default: True)
        reduction (str): Reduction method for the losses ('mean', 'sum', or 'none').
    """

    def __init__(self, weight_cel=0.5, weight=None, to_onehot_y=True, softmax=True, reduction='mean'):
        self.to_onehot_y = to_onehot_y
        super().__init__(lambda_ce=weight_cel, lambda_dice=1 - weight_cel,
                         weight=weight, to_onehot_y=to_onehot_y, softmax=softmax, reduction=reduction)

    def forward(self, y_pred, y_true):
        """
        Forward pass.

        Args:
            y_pred (torch.Tensor): Predicted logits with shape (B, C, L), where B is batch size,
                                   C is number of classes, L is sequence length.
            y_true (torch.Tensor): Ground-truth labels with shape (B, L) (class indices) or one-hot (B, C, L).

        Returns:
            torch.Tensor: Combined loss value.
        """
        if self.to_onehot_y:
            y_true = y_true.view(y_true.shape[0], 1, y_true.shape[1])
        return super().forward(y_pred, y_true)


class CelGdlLoss(nn.Module):
    """
    Combined Categorical Cross-Entropy and Generalized Dice Loss.

    This loss function combines CEL for probabilistic accuracy with Generalized Dice Loss
    for handling class imbalance and overlap in multiclass segmentation tasks.

    Args:
        weight_cel (float): Weighting factor for balancing CEL and Dice losses (default: 0.5).
        weight (Tensor): Class weights for CEL.
        w_type (str): Weighting type for Generalized Dice ('square' for 1/sum(g^2), 'simple' for 1/sum(g), 'uniform'
        for equal weights).
        to_onehot_y (bool): Must be set to True if y_true shape is (B, L) (default: True)
        softmax (bool): Whether to apply softmax to y_pred. (default: True)
        reduction (str): Reduction method for the losses ('mean', 'sum', or 'none').
    """

    def __init__(self, weight_cel=0.5, weight=None, w_type='square', to_onehot_y=True, softmax=True, reduction='mean'):
        super(CelGdlLoss, self).__init__()
        self.weight_cel = weight_cel
        self.weight = weight
        self.to_onehot_y = to_onehot_y
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.dice = GeneralizedDiceLoss(
            include_background=True,
            to_onehot_y=to_onehot_y,  # Automatically convert y_true to one-hot if needed
            softmax=softmax,  # Apply softmax to y_pred (assumes y_pred are logits)
            w_type=w_type,  # 'square' for generalized weighting based on class frequencies
            reduction=reduction,
            smooth_nr=1e-5,  # Smoothing for numerator
            smooth_dr=1e-5  # Smoothing for denominator
        )

        # For loss scaling:
        self.ema_decay = 0.999
        # Initialize EMA as floats (not tensors) to avoid CUDA issues
        self.register_buffer('ema_ce', torch.tensor(1.0))  # Stored on same device as model
        self.register_buffer('ema_dice', torch.tensor(1.0))

    def forward(self, y_pred, y_true):
        """
        Forward pass.

        Args:
            y_pred (torch.Tensor): Predicted logits with shape (B, C, L), where B is batch size,
                                   C is number of classes, L is sequence length.
            y_true (torch.Tensor): Ground-truth labels with shape (B, L) (class indices) or one-hot (B, C, L).

        Returns:
            torch.Tensor: Combined loss value.
        """
        loss_ce = self.ce(y_pred, y_true)
        if self.to_onehot_y:
            y_true = y_true.view(y_true.shape[0], 1, y_true.shape[1])
        loss_dice = self.dice(y_pred, y_true)

        # Update EMAs (use .item() to detach and convert to scalar)
        with torch.no_grad():  # Ensure no gradients for EMA updates
            self.ema_ce.mul_(self.ema_decay).add_((1 - self.ema_decay) * loss_ce.item())
            self.ema_dice.mul_(self.ema_decay).add_((1 - self.ema_decay) * loss_dice.item())

        # Normalize losses using EMAs (ensure min_ema to avoid division by zero)
        norm_ce = loss_ce / self.ema_ce.clamp(min=1e-5)
        norm_dice = loss_dice / self.ema_dice.clamp(min=1e-5)

        return self.weight_cel * norm_ce + (1 - self.weight_cel) * norm_dice


class CelGwdlLoss(nn.Module):
    """
    Combined Categorical Cross-Entropy and Generalized Wasserstein Dice Loss.

    This loss function combines CEL for probabilistic accuracy with Generalized Dice Loss
    for handling class imbalance and overlap in multiclass segmentation tasks.

    Args:
        dist_matrix (2D Tensor): Classes distance matrix.
        weight_cel (float): Weighting factor for balancing CEL and Dice losses (default: 0.5).
        weight (Tensor): Class weights for CEL.
        weighting_mode (str): Weighting mode for GWDL, ('default' for original paper implementation, 'GDL' for GDL like implementation.
        reduction (str): Reduction method for the losses ('mean', 'sum', or 'none').
    """

    def __init__(self, dist_matrix, weight_cel=0.5, weight=None, weighting_mode="default", reduction='mean'):
        super(CelGwdlLoss, self).__init__()
        self.weight_cel = weight_cel
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.dice = GeneralizedWassersteinDiceLoss(
            dist_matrix=dist_matrix,
            weighting_mode=weighting_mode,  # 'square' for generalized weighting based on class frequencies
            reduction=reduction,
            smooth_nr=1e-5,  # Smoothing for numerator
            smooth_dr=1e-5  # Smoothing for denominator
        )

        # For loss scaling:
        self.ema_decay = 0.999
        # Initialize EMA as floats (not tensors) to avoid CUDA issues
        self.register_buffer('ema_ce', torch.tensor(1.0))  # Stored on same device as model
        self.register_buffer('ema_dice', torch.tensor(1.0))

    def forward(self, y_pred, y_true):
        """
        Forward pass.

        Args:
            y_pred (torch.Tensor): Predicted logits with shape (B, C, L), where B is batch size,
                                   C is number of classes, L is sequence length.
            y_true (torch.Tensor): Ground-truth labels with shape (B, L) (class indices)

        Returns:
            torch.Tensor: Combined loss value.
        """
        loss_ce = self.ce(y_pred, y_true)
        loss_dice = self.dice(y_pred, y_true)

        # Update EMAs (use .item() to detach and convert to scalar)
        with torch.no_grad():  # Ensure no gradients for EMA updates
            self.ema_ce.mul_(self.ema_decay).add_((1 - self.ema_decay) * loss_ce.item())
            self.ema_dice.mul_(self.ema_decay).add_((1 - self.ema_decay) * loss_dice.item())

        # Normalize losses using EMAs (ensure min_ema to avoid division by zero)
        norm_ce = loss_ce / self.ema_ce.clamp(min=1e-5)
        norm_dice = loss_dice / self.ema_dice.clamp(min=1e-5)

        return self.weight_cel * norm_ce + (1 - self.weight_cel) * norm_dice


class FlDlLoss(DiceFocalLoss):
    """
    Combined Focal Loss and Dice Loss.

    This loss function combines Fl for probabilistic accuracy with Generalized Dice Loss
    for handling class imbalance and overlap in multiclass segmentation tasks.

    Args:
        weight_fl (float): Weighting factor for balancing FL and Dice losses (default: 0.5).
        weight (Tensor): Class weights for FL and DL.
        gamma (float): Focal gamma parameter.
        w_type (str): Weighting type for Generalized Dice ('square' for 1/sum(g^2), 'simple' for 1/sum(g), 'uniform' for equal weights).
        to_onehot_y (bool): Must be set to True if y_true shape is (B, L) (default: True)
        softmax (bool): Whether to apply softmax to y_pred. (default: True)
        reduction (str): Reduction method for the losses ('mean', 'sum', or 'none').
    """

    def __init__(self, weight_fl=0.5, weight=None, gamma=2.0, to_onehot_y=True, softmax=True,
                 reduction='mean'):
        self.to_onehot_y = to_onehot_y
        super().__init__(lambda_focal=weight_fl, lambda_dice=1 - weight_fl,
                         weight=weight, gamma=gamma, to_onehot_y=to_onehot_y,
                         softmax=softmax, reduction=reduction)

    def forward(self, y_pred, y_true):
        """
        Forward pass.

        Args:
            y_pred (torch.Tensor): Predicted logits with shape (B, C, L), where B is batch size,
                                   C is number of classes, L is sequence length.
            y_true (torch.Tensor): Ground-truth labels with shape (B, L) (class indices) or one-hot (B, C, L).

        Returns:
            torch.Tensor: Combined loss value.
        """
        if self.to_onehot_y:
            y_true = y_true.view(y_true.shape[0], 1, y_true.shape[1])
        return super().forward(y_pred, y_true)


class FlGdlLoss(GeneralizedDiceFocalLoss):
    """
    Combined Focal Loss and Generalized Dice Loss.

    This loss function combines Fl for probabilistic accuracy with Generalized Dice Loss
    for handling class imbalance and overlap in multiclass segmentation tasks.

    Args:
        weight_fl (float): Weighting factor for balancing FL and GDL (default: 0.5).
        weight (Tensor): Class weights for FL.
        gamma (float): Focal gamma parameter.
        w_type (str): Weighting type for Generalized Dice ('square' for 1/sum(g^2), 'simple' for 1/sum(g), 'uniform' for equal weights).
        to_onehot_y (bool): Must be set to True if y_true shape is (B, L) (default: True)
        softmax (bool): Whether to apply softmax to y_pred. (default: True)
        reduction (str): Reduction method for the losses ('mean', 'sum', or 'none').
    """

    def __init__(self, weight_fl=0.5, weight=None, gamma=2.0, w_type='square', to_onehot_y=True, softmax=True,
                 reduction='mean'):
        self.to_onehot_y = to_onehot_y
        super().__init__(lambda_focal=weight_fl, lambda_gdl=1 - weight_fl,
                         weight=weight, w_type=w_type, gamma=gamma, to_onehot_y=to_onehot_y,
                         softmax=softmax, reduction=reduction)

    def forward(self, y_pred, y_true):
        """
        Forward pass.

        Args:
            y_pred (torch.Tensor): Predicted logits with shape (B, C, L), where B is batch size,
                                   C is number of classes, L is sequence length.
            y_true (torch.Tensor): Ground-truth labels with shape (B, L) (class indices) or one-hot (B, C, L).

        Returns:
            torch.Tensor: Combined loss value.
        """
        if self.to_onehot_y:
            y_true = y_true.view(y_true.shape[0], 1, y_true.shape[1])
        return super().forward(y_pred, y_true)
