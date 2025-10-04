import os
import argparse
import logging

import torch
from torch.utils.data import DataLoader


from sklearn.model_selection import train_test_split

from mtugan.config import CONFIG
from mtugan.dataset import MultiTaskDataset, compute_class_weights
from mtugan.gan import MultiTaskGenerator, Discriminator
from mtugan.train import *

# creating directory for storing sample output and logs
os.makedirs(CONFIG['plot_dir'], exist_ok=True)

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{CONFIG['plot_dir']}/mtugan_training.log"),
            logging.StreamHandler()
        ]
    )

# Main Execution
if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description='Train MT-UGAN for hippocampal subregion segmentation and classification.')
    parser.add_argument('--image_dir', default=CONFIG['image_dir'], help='Directory with input images.')
    parser.add_argument('--mask_dir', default=CONFIG['mask_dir'], help='Directory with mask images.')
    # parser.add_argument('--class_dir_dict', default=CONFIG['class_dir_dict'], help='Directory paths for each class.')
    args = parser.parse_args()
    
    # update config with command-line arguments
    CONFIG['image_dir'] = args.image_dir
    CONFIG['mask_dir'] = args.mask_dir
    # CONFIG['class_dir_dict'] = args.class_dir_dict


    # Dataset and Loaders with split-specific augmentation
    full_dataset = MultiTaskDataset(CONFIG['image_dir'], CONFIG['mask_dir'], CONFIG['class_dir_dict'], split='train')  # Train split with augmentation
    labels = full_dataset.labels
    indices = list(range(len(full_dataset)))

    train_val_idx, test_idx = train_test_split(indices, test_size=CONFIG['test_split'], stratify=labels, random_state=CONFIG['random_seed'])
    train_idx, val_idx = train_test_split(train_val_idx, test_size= CONFIG['val_split'], stratify=[labels[i] for i in train_val_idx], random_state=CONFIG['random_seed'])

    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)  # Augmentation applied (split='train')
    val_dataset = torch.utils.data.Subset(MultiTaskDataset(CONFIG['image_dir'], CONFIG['mask_dir'], CONFIG['class_dir_dict'], split='val'), val_idx)  # No augmentation
    test_dataset = torch.utils.data.Subset(MultiTaskDataset(CONFIG['image_dir'], CONFIG['mask_dir'], CONFIG['class_dir_dict']), test_idx)  # No augmentation

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size_train'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size_val'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size_test'], shuffle=False)

    # Compute and print class distribution for each split
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    test_labels = [labels[i] for i in test_idx]

    logging.info(f"Total samples: {len(labels)}")
    logging.info(f"Train data: {len(train_idx)}")
    logging.info(f"  DG: {train_labels.count(0)}, CA1: {train_labels.count(1)}, CA3: {train_labels.count(2)}")
    logging.info(f"Validation data: {len(val_idx)}")
    logging.info(f"  DG: {val_labels.count(0)}, CA1: {val_labels.count(1)}, CA3: {val_labels.count(2)}")
    logging.info(f"Test data: {len(test_idx)}")
    logging.info(f"  DG: {test_labels.count(0)}, CA1: {test_labels.count(1)}, CA3: {test_labels.count(2)}")

    class_weights = compute_class_weights(labels).to(CONFIG['device'])
    logging.info(f"Class weights (DG, CA1, CA3): {class_weights.tolist()}")

    # Initialize Models
    generator = MultiTaskGenerator(in_channels=3, out_channels=1, num_classes=len(set(labels))).to(CONFIG['device'])
    discriminator = Discriminator().to(CONFIG['device'])

    # Clear GPU memory
    torch.cuda.empty_cache()

    # Train and Evaluate
    generator, test_metrics = train(generator, discriminator, train_loader, val_loader, test_loader, 
                                    epochs=CONFIG['epochs'], patience= CONFIG['patience'], device=CONFIG['device'], class_weights=class_weights, plot_dir=CONFIG['plot_dir'])
    
    # Final Performance Summary
    logging.info("\nPerformance Summary:")
    logging.info("| Split | Dice (%) | IoU (%) | HD (px) | Prec (%) | Rec (%) | ASSD (px) | Acc (%) |")
    logging.info("|-------|----------|---------|---------|----------|---------|-----------|---------|")
    logging.info(f"| Test  | {test_metrics['dice']*100:.2f} | {test_metrics['iou']*100:.2f} | {test_metrics['hd']:.2f} | "
          f"{test_metrics['prec']*100:.2f} | {test_metrics['rec']*100:.2f} | {test_metrics['assd']:.2f} | {test_metrics['acc']*100:.2f} |")
    logging.info(f"Class Metrics (DG, CA1, CA3):")
    logging.info(f"Precision: {test_metrics['class_prec']}")
    logging.info(f"Recall: {test_metrics['class_rec']}")
    logging.info(f"F1-score: {test_metrics['class_f1']}")

