"""
Training script for wav2vec model.
No torchaudio dependency - uses scipy and soundfile instead.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
from pathlib import Path
from tqdm import tqdm

from model import Wav2Vec, Wav2VecLarge, ContrastiveLoss, InfoNCELoss
from data import create_dataloader
from utils import (
    setup_logger, save_checkpoint, load_checkpoint, 
    find_latest_checkpoint, plot_loss_curve, log_gpu_memory
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train wav2vec model")
    
    # Data parameters
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--segment-length", type=int, default=32000, help="Audio segment length in samples (~2s)")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="wav2vec", choices=["wav2vec", "wav2vec_large"], 
                        help="Model architecture")
    parser.add_argument("--channels", type=int, default=512, help="Number of channels in model")
    parser.add_argument("--num-steps", type=int, default=12, help="Number of steps for prediction")
    parser.add_argument("--num-negatives", type=int, default=10, help="Number of negative samples")
    parser.add_argument("--loss-type", type=str, default="contrastive", choices=["contrastive", "infonce"],
                        help="Type of contrastive loss to use")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps for learning rate scheduler")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Gradient clipping value")
    parser.add_argument("--accumulate-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer")
    
    # System parameters
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for saving logs")
    parser.add_argument("--vis-dir", type=str, default="visualizations", help="Directory for saving visualizations")
    parser.add_argument("--checkpoint-interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--use-transforms", action="store_true", help="Use data augmentation transforms")
    
    return parser.parse_args()

def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    """Create a cosine learning rate schedule with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, 
                epoch, args, logger, step, losses):
    """Train for one epoch."""
    model.train()
    
    # Set up progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    # For accumulating gradients
    optimizer.zero_grad()
    
    # For tracking loss
    epoch_loss = 0.0
    samples_processed = 0
    
    # For mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    for i, batch in enumerate(pbar):
        # Move batch to device
        batch = batch.to(device, non_blocking=True)
        
        # Forward pass with mixed precision if enabled
        if args.fp16:
            with torch.cuda.amp.autocast():
                z, c = model(batch)
                predictions = model.get_predictions(c)
                loss = criterion(z, c, predictions)
        else:
            z, c = model(batch)
            predictions = model.get_predictions(c)
            loss = criterion(z, c, predictions)
        
        # Check for NaN loss
        if torch.isnan(loss):
            logger.warning(f"NaN loss detected at step {step}, skipping batch")
            continue
        
        # Scale loss by accumulation steps
        loss = loss / args.accumulate_steps
        
        # Backward pass with mixed precision if enabled
        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Step optimizer after accumulation
        if (i + 1) % args.accumulate_steps == 0:
            if args.fp16:
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                # Update weights
                scaler.step(optimizer)
                scaler.update()
            else:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                # Update weights
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Zero gradients
            optimizer.zero_grad()
        
        # Update progress bar
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            'loss': loss.item() * args.accumulate_steps,
            'lr': f'{current_lr:.6f}'
        })
        
        # Track loss
        epoch_loss += loss.item() * args.accumulate_steps * batch.size(0)
        samples_processed += batch.size(0)
        
        # Track global step and loss
        step += 1
        losses.append(loss.item() * args.accumulate_steps)
        
        # Log every 100 steps
        if step % 100 == 0:
            logger.info(f"Step {step}: Loss = {loss.item() * args.accumulate_steps:.4f}, LR = {current_lr:.6f}")
            if step % 500 == 0:  # Log GPU memory less frequently
                log_gpu_memory()
    
    # Calculate average loss for the epoch
    avg_epoch_loss = epoch_loss / samples_processed if samples_processed > 0 else float('inf')
    
    return avg_epoch_loss, step, losses

def main():
    args = parse_args()
    
    # Set up directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    
    # Set up logger
    logger = setup_logger(args.log_dir)
    logger.info(f"Arguments: {args}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create data loader
    try:
        dataloader = create_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            sample_rate=args.sample_rate,
            segment_length=args.segment_length,
            num_workers=args.num_workers,
            use_transforms=args.use_transforms
        )
        logger.info(f"Created dataloader with {len(dataloader)} batches")
    except Exception as e:
        logger.error(f"Failed to create dataloader: {e}")
        return
    
    # Create model
    if args.model == "wav2vec":
        model = Wav2Vec(
            channels=args.channels,
            num_steps=args.num_steps
        )
    else:  # wav2vec_large
        model = Wav2VecLarge(
            channels=args.channels,
            num_steps=args.num_steps
        )
    
    model = model.to(device)
    logger.info(f"Model: {args.model}")
    
    # Create loss function
    if args.loss_type == "contrastive":
        criterion = ContrastiveLoss(num_negatives=args.num_negatives)
    else:  # infonce
        criterion = InfoNCELoss(num_negatives=args.num_negatives)
    
    logger.info(f"Using {args.loss_type} loss with {args.num_negatives} negatives")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Create learning rate scheduler
    total_steps = len(dataloader) * args.epochs // args.accumulate_steps
    scheduler = cosine_schedule_with_warmup(
        optimizer, 
        warmup_steps=args.warmup_steps,
        total_steps=total_steps
    )
    
    # Initialize training variables
    start_epoch = 0
    step = 0
    best_loss = float('inf')
    losses = []
    
    # Resume training if requested
    if args.resume:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        if checkpoint_path:
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            try:
                model, optimizer, scheduler, start_epoch, step, _, best_loss = load_checkpoint(
                    checkpoint_path, model, optimizer, scheduler
                )
                start_epoch += 1  # Start from next epoch
                logger.info(f"Resumed from epoch {start_epoch-1}, step {step}, best loss {best_loss:.4f}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.info("Starting training from scratch")
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch
        try:
            avg_loss, step, losses = train_epoch(
                model, dataloader, criterion, optimizer, scheduler,
                device, epoch, args, logger, step, losses
            )
        except Exception as e:
            logger.error(f"Error during training epoch {epoch}: {e}")
            continue
        
        # Log epoch results
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                logger.info(f"New best loss: {best_loss:.4f}")
            
            try:
                checkpoint_path = save_checkpoint(
                    model, optimizer, scheduler, epoch, step, avg_loss, best_loss,
                    args.checkpoint_dir, is_best
                )
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
        
        # Plot loss curve
        if (epoch + 1) % 5 == 0 and losses:
            try:
                plot_loss_curve(losses, os.path.join(args.vis_dir, f"loss_epoch_{epoch}.png"))
                # Also save losses as numpy array
                np.save(os.path.join(args.vis_dir, "losses.npy"), np.array(losses))
            except Exception as e:
                logger.error(f"Failed to plot loss curve: {e}")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
