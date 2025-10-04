import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from medpy.metric.binary import assd
from sklearn.metrics import precision_recall_fscore_support

def iou_score(y_true, y_pred, epsilon=1e-8):
    intersection = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    union = torch.sum(y_true, dim=(1, 2, 3)) + torch.sum(y_pred, dim=(1, 2, 3)) - intersection
    return ((intersection + epsilon) / (union + epsilon)).mean()

# Refine Mask
def refine_mask(masks, device, threshold=0.3, min_area=300):
    masks = (masks > threshold).float()
    masks = (masks.detach().cpu().numpy() * 255).astype(np.uint8).squeeze(1)
    refined_masks = []
    for mask in masks:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered_mask[labels == i] = 255
        refined_masks.append(filtered_mask[np.newaxis, :])
    return torch.from_numpy(np.stack(refined_masks) / 255.0).float().to(device)

def gradient_penalty(discriminator, real, fake, device):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)
    disc_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(disc_interpolates),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# Evaluation and Visualization
def evaluate(model, loader, device):
    model.eval()
    dice_scores, iou_scores, hd_scores, prec_scores, rec_scores, assd_scores = [], [], [], [], [], []
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, masks, labels in loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            seg_pred, cls_pred = model(images)
            seg_pred = refine_mask(seg_pred, device)
            dice = 2 * torch.sum(seg_pred * masks, dim=(1, 2, 3)) / (seg_pred.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-8)
            iou = iou_score(masks, seg_pred)
            hd = hausdorff_distance(masks, seg_pred)
            prec, rec = precision_recall(masks, seg_pred)
            assd_val = avg_symmetric_surface_distance(masks, seg_pred)
            dice_scores.extend(dice.cpu().numpy())
            iou_scores.append(iou.item())
            hd_scores.append(hd)
            prec_scores.append(prec)
            rec_scores.append(rec)
            assd_scores.append(assd_val)
            _, preds = torch.max(cls_pred, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    class_prec, class_rec, class_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2])
    
    return {
        'dice': np.mean(dice_scores), 'iou': np.mean(iou_scores), 'hd': np.mean(hd_scores),
        'prec': np.mean(prec_scores), 'rec': np.mean(rec_scores), 'assd': np.mean(assd_scores),
        'acc': acc, 'class_prec': class_prec.tolist(), 'class_rec': class_rec.tolist(), 'class_f1': class_f1.tolist()
    }

def hausdorff_distance(y_true, y_pred):
    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
    hd_scores = []
    for t, p in zip(y_true, y_pred):
        t_points = np.argwhere(t.squeeze())
        p_points = np.argwhere(p.squeeze())
        if len(t_points) == 0 or len(p_points) == 0:
            hd_scores.append(0.0)
        else:
            hd = max(directed_hausdorff(t_points, p_points)[0], directed_hausdorff(p_points, t_points)[0])
            hd_scores.append(hd)
    return np.mean(hd_scores)

def precision_recall(y_true, y_pred, epsilon=1e-8):
    y_true, y_pred = y_true.flatten(1), y_pred.flatten(1)
    tp = torch.sum(y_true * y_pred, dim=1)
    fp = torch.sum(y_pred * (1 - y_true), dim=1)
    fn = torch.sum(y_true * (1 - y_pred), dim=1)
    precision = (tp / (tp + fp + epsilon)).mean().item()
    recall = (tp / (tp + fn + epsilon)).mean().item()
    return precision, recall

def avg_symmetric_surface_distance(y_true, y_pred):
    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
    assd_scores = []
    for t, p in zip(y_true, y_pred):
        if t.sum() == 0 or p.sum() == 0:
            assd_scores.append(0.0)
        else:
            assd_scores.append(assd(t.squeeze(), p.squeeze()))
    return np.mean(assd_scores)

def visualize_predictions(model, loader, device, num_samples=3, plot_dir="plots"):
    model.eval()
    images, masks, labels = next(iter(loader))
    images, masks, labels = images[:num_samples].to(device), masks[:num_samples].to(device), labels[:num_samples].to(device)
    with torch.no_grad():
        seg_pred, cls_pred = model(images)
        seg_pred = refine_mask(seg_pred, device)
    # mean, std = torch.tensor(full_dataset.mean).view(1, 3, 1, 1).to(device), torch.tensor(full_dataset.std).view(1, 3, 1, 1).to(device)
    # images_denorm = (images * std + mean).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    images_denorm = images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()  # Remove mean/std operations

    class_names = ['DG', 'CA1', 'CA3']

    plt.figure(figsize=(5 * num_samples, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images_denorm[i])
        plt.contour(masks[i].cpu().squeeze(), colors='green', linewidths=1)
        plt.contour(seg_pred[i].cpu().squeeze(), colors='red', linewidths=1)
        pred_label = class_names[cls_pred[i].argmax().item()]
        true_label = class_names[labels[i].item()]
        plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=12)
        plt.axis('off')
    plt.legend(['Ground Truth', 'Predicted'], loc='upper right', fontsize=10)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "multitask_predictions_neun.png"), dpi=500, bbox_inches='tight')
    plt.close()

def plot_training_curves(gen_losses, disc_losses, val_metrics, plot_dir):
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(range(1, len(gen_losses) + 1), gen_losses, label='Generator Loss', color='blue')
    ax1.plot(range(1, len(disc_losses) + 1), disc_losses, label='Discriminator Loss', color='red')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Losses', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(val_metrics['dice'], label='Dice', color='blue')
    ax2.plot(val_metrics['iou'], label='IoU', color='green')
    ax2.plot(val_metrics['acc'], label='Accuracy', color='orange')
    ax2.plot(val_metrics['prec'], label='Precision', color='purple')
    ax2.plot(val_metrics['rec'], label='Recall', color='red')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Validation Metrics', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_curves_neun.png"), dpi=500, bbox_inches='tight')
    plt.close()
