import os
import torch


# Configuration dictionary
CONFIG = {
    # Dataset paths
    'image_dir': '',  # Directory for input images; os.path.join(DATA_DIR, 'images')
    'mask_dir': '',    # Directory for segmentation masks; os.path.join(DATA_DIR, 'masks')
    'class_dir_dict':  {
            'DG': './c1_c2_classification_masks/r_DG',
            'CA1': './c1_c2_classification_masks/r_CA1',
            'CA3': './c1_c2_classification_masks/r_CA3'
            },

    'plot_dir': 'PLOT_DIR',                           # Directory for logs and plots
    'model_dir': 'MODEL_DIR',                         # Directory for model checkpoints
    

    # Dataset splitting
    'test_split': 0.15,      # Proportion of dataset for testing
    'val_split': 0.0588235,  # Proportion of training data for validation
    'random_seed': 42,      # Random seed for reproducibility

    # DataLoader settings
    'batch_size_train': 8,  # Batch size for training
    'batch_size_val': 8,    # Batch size for validation
    'batch_size_test': 8,   # Batch size for testing

    # Model configuration
    'image_size': [256, 256],  # Input image dimensions


    # Training configuration
    'epochs': 100,             # Number of training epochs
    'patience': 10,
    'learning_rate_g': 5e-4, # Learning rate for generator
    'learning_rate_d': 1e-6, # Learning rate for discriminator     
}

# Device configuration
CONFIG['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'