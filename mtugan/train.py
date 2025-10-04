import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
from mtugan.losses import *
from mtugan.utils import *
from main import setup_logging


# Training Step
def train_step(generator, discriminator, images, masks, labels, gen_optimizer, disc_optimizer, class_weights, device, train_disc=True):
    images, masks, labels = images.to(device), masks.to(device), labels.to(device)
    
    gen_optimizer.zero_grad()
    disc_optimizer.zero_grad()
    
    seg_pred, cls_pred = generator(images)
    real_output = discriminator(masks)
    fake_output = discriminator(seg_pred)
    
    if train_disc:
        loss = discriminator_loss(discriminator,real_output, fake_output.detach(), masks, device=device)
        loss.backward()
        disc_optimizer.step()
    else:
        loss = generator_loss(masks, seg_pred, fake_output, labels, cls_pred, class_weights)
        loss.backward()
        gen_optimizer.step()
    
    return loss.item()


# Training Function
def train(generator, discriminator, train_loader, val_loader, test_loader, epochs, patience, device, class_weights, plot_dir="plots"):
    setup_logging()
    gen_optimizer = optim.Adam(generator.parameters(), lr=5e-4)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-6)
    
    best_val_dice = 0.0
    epochs_no_improve = 0

    gen_losses, disc_losses, val_metrics = [], [], {'dice': [], 'iou': [], 'hd': [], 'prec': [], 'rec': [], 'assd': [], 'acc': [], 'class_prec': [], 'class_rec': [], 'class_f1': []}

    # Warm-up phase
    print("Starting warm-up phase (segmentation only)...")
    for epoch in range(10):
        generator.train()
        epoch_gen_loss = 0.0
        for images, masks, labels in tqdm(train_loader, desc=f"Warm-up Epoch {epoch+1}/10"):
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            gen_optimizer.zero_grad()
            seg_pred, _ = generator(images)
            loss = dice_loss(masks, seg_pred)
            loss.backward()
            gen_optimizer.step()
            epoch_gen_loss += loss.item()
        gen_losses.append(epoch_gen_loss / len(train_loader))
        print(f"Warm-up - Seg Loss: {gen_losses[-1]:.4f}")

    # Full GAN training
    print("Starting full multitask GAN training...")
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (images, masks, labels) in enumerate(pbar):
            num_batches += 1
            train_disc = (i % 3 == 0)
            loss = train_step(generator, discriminator, images, masks, labels, gen_optimizer, disc_optimizer, class_weights, device, train_disc)
            
            if train_disc:
                epoch_disc_loss += loss
            else:
                epoch_gen_loss += loss
            pbar.set_postfix({"Gen Loss" if not train_disc else "Disc Loss": loss})

        gen_losses.append(epoch_gen_loss / (num_batches * 2 / 3))
        disc_losses.append(epoch_disc_loss / (num_batches / 3))

        # Validation
        metrics = evaluate(generator, val_loader, device)
        for key, value in metrics.items():
            val_metrics[key].append(value)
        print(f"Val - Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}, HD: {metrics['hd']:.2f}, "
              f"Prec: {metrics['prec']:.4f}, Rec: {metrics['rec']:.4f}, ASSD: {metrics['assd']:.2f}, Acc: {metrics['acc']:.4f}")
        print(f"Class Prec: {metrics['class_prec']}, Rec: {metrics['class_rec']}, F1: {metrics['class_f1']}")

        if metrics['dice'] > best_val_dice:
            best_val_dice = metrics['dice']
            torch.save(generator.state_dict(), os.path.join(plot_dir, 'best_multitask_generator_neun.pth'))
            logging.info(f"New best model saved at {epoch} epoch, with Val Dice: {best_val_dice:.4f} and Acc: {metrics['acc']:.4f} \
                         Class Prec: {metrics['class_prec']}, Rec: {metrics['class_rec']}, F1: {metrics['class_f1']}")
            epochs_no_improve = 0
            if epoch>50:
                visualize_predictions(generator, val_loader, device, plot_dir=plot_dir)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered at {epochs} epoch")
                break

        if epoch == 5:
            visualize_predictions(generator, val_loader, device, plot_dir=plot_dir)

    generator.load_state_dict(torch.load(os.path.join(plot_dir, 'best_multitask_generator_neun.pth')))
    test_metrics = evaluate(generator, test_loader, device)
    print(f"Test - Dice: {test_metrics['dice']:.4f}, IoU: {test_metrics['iou']:.4f}, HD: {test_metrics['hd']:.2f}, "
          f"Prec: {test_metrics['prec']:.4f}, Rec: {test_metrics['rec']:.4f}, ASSD: {test_metrics['assd']:.2f}, Acc: {test_metrics['acc']:.4f}")
    print(f"Class Prec: {test_metrics['class_prec']}, Rec: {test_metrics['class_rec']}, F1: {test_metrics['class_f1']}")

    plot_training_curves(gen_losses, disc_losses, val_metrics, plot_dir)
    visualize_predictions(generator, test_loader, device, plot_dir=plot_dir)
    
    return generator, test_metrics