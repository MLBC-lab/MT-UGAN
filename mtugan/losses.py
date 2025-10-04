import torch
import torch.nn as nn
import torch.nn.functional as F

from mtugan.utils import gradient_penalty


# Loss Functions
def dice_loss(y_true, y_pred, epsilon=1e-8):
    numerator = 2 * torch.sum(y_true * y_pred, dim=(1, 2, 3))
    denominator = torch.sum(y_true + y_pred, dim=(1, 2, 3)) + epsilon
    return 1 - (numerator / denominator).mean()

def focal_loss(y_pred, y_true, gamma=2.0, alpha=None, reduction='mean'):
    log_pt = F.log_softmax(y_pred, dim=1)
    pt = torch.exp(log_pt)
    y_true_onehot = F.one_hot(y_true, num_classes=3).float()
    if alpha is not None:
        focal_weight = alpha[y_true] * (1 - pt[range(len(y_true)), y_true]) ** gamma
    else:
        focal_weight = (1 - pt[range(len(y_true)), y_true]) ** gamma
    loss = -focal_weight * log_pt[range(len(y_true)), y_true]
    return loss.mean() if reduction == 'mean' else loss.sum()

def generator_loss(y_true_seg, y_pred_seg, fake_output, y_true_cls, y_pred_cls, class_weights, 
                  lambda_dice=2.0, lambda_adv=0.1, lambda_cls=0.3):  # Adjusted to prioritize segmentation
    seg_loss = dice_loss(y_true_seg, y_pred_seg)
    adv_loss = nn.BCELoss()(fake_output, torch.ones_like(fake_output))
    cls_loss = focal_loss(y_pred_cls, y_true_cls, alpha=class_weights)
    return lambda_dice * seg_loss + lambda_adv * adv_loss + lambda_cls * cls_loss

def discriminator_loss(discriminator, real_output, fake_output, ground_truth, lambda_dice=0.5, lambda_bce=0.5, lambda_gp=1.0, device='cuda'):
    dice_real = dice_loss(ground_truth, real_output)
    bce_real = nn.BCELoss()(real_output, torch.ones_like(real_output))
    dice_fake = dice_loss(ground_truth, fake_output)
    bce_fake = nn.BCELoss()(fake_output, torch.zeros_like(fake_output))
    gp = gradient_penalty(discriminator, ground_truth, fake_output.detach(), device)
    return lambda_dice * (dice_real + dice_fake) / 2 + lambda_bce * (bce_real + bce_fake) / 2 + lambda_gp * gp

