"""
Visualization tools for wav2vec model.
No torchaudio dependency - uses scipy and soundfile instead.
"""
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from model import Wav2Vec, Wav2VecLarge
from data import create_dataloader
from utils import load_checkpoint, load_audio, normalize_audio, plot_spectrogram, plot_embedding_similarity, compute_spectrogram

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize wav2vec model")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default="wav2vec", choices=["wav2vec", "wav2vec_large"], 
                        help="Model architecture")
    parser.add_argument("--channels", type=int, default=512, help="Number of channels in model")
    parser.add_argument("--num-steps", type=int, default=12, help="Number of steps for prediction")
    
    # Data parameters
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--segment-length", type=int, default=32000, help="Audio segment length in samples (~2s)")
    
    # Visualization parameters
    parser.add_argument("--output-dir", type=str, default="visualizations", help="Directory for saving visualizations")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of audio samples to visualize")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--max-batches", type=int, default=20, help="Maximum number of batches to process for embeddings")
    
    return parser.parse_args()

def visualize_spectrograms(model, dataloader, device, output_dir, num_samples=5):
    """Visualize spectrograms of input audio and latent representations."""
    os.makedirs(os.path.join(output_dir, "spectrograms"), exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            z, c = model(batch)
            
            # Process each item in batch
            for j in range(min(batch.size(0), 3)):  # Limit to 3 samples per batch
                # Get audio sample
                audio = batch[j, 0].cpu()
                
                # Get latent representations
                z_sample = z[j].cpu()
                c_sample = c[j].cpu()
                
                # Plot and save input spectrogram
                try:
                    plot_spectrogram(
                        audio, 
                        16000, 
                        os.path.join(output_dir, f"spectrograms/sample_{i*3+j}_input.png")
                    )
                except Exception as e:
                    print(f"Error plotting input spectrogram: {e}")
                
                # Plot and save latent representation spectrograms
                for k, (latent, name) in enumerate([(z_sample, "encoder"), (c_sample, "context")]):
                    try:
                        plt.figure(figsize=(10, 4))
                        plt.imshow(latent.numpy(), aspect='auto', origin='lower')
                        plt.colorbar()
                        plt.title(f"{name.capitalize()} Representation")
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f"spectrograms/sample_{i*3+j}_{name}.png"))
                        plt.close()
                    except Exception as e:
                        print(f"Error plotting {name} representation: {e}")

def visualize_embeddings(model, dataloader, device, output_dir, max_batches=20):
    """Visualize embeddings using t-SNE and PCA."""
    os.makedirs(os.path.join(output_dir, "embeddings"), exist_ok=True)
    
    model.eval()
    
    # Collect embeddings
    encoder_embeddings = []
    context_embeddings = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Collecting embeddings")):
            if i >= max_batches:
                break
                
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            z, c = model(batch)
            
            # Store embeddings (average over time dimension)
            encoder_embeddings.append(z.mean(dim=2).cpu())
            context_embeddings.append(c.mean(dim=2).cpu())
    
    if not encoder_embeddings:
        print("No embeddings collected, skipping embedding visualization")
        return
    
    # Concatenate embeddings
    encoder_embeddings = torch.cat(encoder_embeddings, dim=0).numpy()
    context_embeddings = torch.cat(context_embeddings, dim=0).numpy()
    
    print(f"Collected {encoder_embeddings.shape[0]} embeddings")
    
    # Visualize with t-SNE and PCA
    for name, embeddings in [("encoder", encoder_embeddings), ("context", context_embeddings)]:
        try:
            # Limit number of samples for t-SNE (it's slow)
            max_samples = min(1000, embeddings.shape[0])
            embeddings_subset = embeddings[:max_samples]
            
            # t-SNE
            print(f"Computing t-SNE for {name} embeddings...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max_samples//4))
            embeddings_tsne = tsne.fit_transform(embeddings_subset)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], alpha=0.6, s=20)
            plt.title(f"t-SNE of {name.capitalize()} Embeddings")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"embeddings/tsne_{name}.png"), dpi=150)
            plt.close()
            
            # PCA
            print(f"Computing PCA for {name} embeddings...")
            pca = PCA(n_components=2)
            embeddings_pca = pca.fit_transform(embeddings_subset)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], alpha=0.6, s=20)
            plt.title(f"PCA of {name.capitalize()} Embeddings")
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"embeddings/pca_{name}.png"), dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"Error visualizing {name} embeddings: {e}")

def visualize_similarity(model, dataloader, device, output_dir):
    """Visualize similarity matrices of embeddings."""
    os.makedirs(os.path.join(output_dir, "similarity"), exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 2:  # Just do a few batches
                break
            
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            z, c = model(batch)
            
            # Plot similarity matrices
            for name, embeddings in [("encoder", z), ("context", c)]:
                try:
                    # Reshape: [batch_size, channels, time] -> [batch_size * time, channels]
                    batch_size, channels, time_steps = embeddings.shape
                    
                    # Limit the number of time steps to avoid huge matrices
                    max_time_steps = min(100, time_steps)
                    embeddings_subset = embeddings[:, :, :max_time_steps]
                    
                    embeddings_flat = embeddings_subset.transpose(1, 2).reshape(-1, channels)
                    
                    # Limit to first 200 samples to keep matrix manageable
                    max_samples = min(200, embeddings_flat.shape[0])
                    embeddings_flat = embeddings_flat[:max_samples]
                    
                    # Plot similarity matrix
                    plot_embedding_similarity(
                        embeddings_flat,
                        os.path.join(output_dir, f"similarity/batch_{i}_{name}.png")
                    )
                except Exception as e:
                    print(f"Error plotting similarity for {name}: {e}")

def visualize_nearest_neighbors(model, dataloader, device, output_dir, k=5):
    """Visualize nearest neighbors in latent space."""
    os.makedirs(os.path.join(output_dir, "nearest_neighbors"), exist_ok=True)
    
    model.eval()
    
    # Collect embeddings and corresponding audio segments
    embeddings = []
    audio_segments = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Collecting embeddings for NN")):
            if i >= 10:  # Limit to avoid memory issues
                break
                
            # Move batch to device
            batch = batch.to(device)
            
            # Store audio segments
            audio_segments.append(batch.cpu())
            
            # Forward pass
            z, c = model(batch)
            
            # Store context embeddings (average over time dimension)
            embeddings.append(c.mean(dim=2).cpu())
    
    if not embeddings:
        print("No embeddings collected for nearest neighbors")
        return
    
    # Concatenate embeddings and audio segments
    embeddings = torch.cat(embeddings, dim=0)
    audio_segments = torch.cat(audio_segments, dim=0)
    
    print(f"Computing nearest neighbors for {embeddings.shape[0]} samples")
    
    # Normalize embeddings for cosine similarity
    embeddings_norm = embeddings / (torch.norm(embeddings, dim=1, keepdim=True) + 1e-8)
    
    # Compute similarity matrix
    similarity = torch.mm(embeddings_norm, embeddings_norm.t())
    
    # For each embedding, find k nearest neighbors
    _, indices = torch.topk(similarity, k=k+1, dim=1)  # +1 because the closest is itself
    
    # Visualize nearest neighbors for a few samples
    num_viz_samples = min(5, embeddings.size(0))
    for i in range(num_viz_samples):
        try:
            plt.figure(figsize=(15, 8))
            
            # Plot query audio
            plt.subplot(k+1, 1, 1)
            plt.plot(audio_segments[i, 0].numpy())
            plt.title("Query Audio")
            plt.axis('off')
            
            # Plot nearest neighbors
            for j in range(1, k+1):
                nn_idx = indices[i, j].item()
                plt.subplot(k+1, 1, j+1)
                plt.plot(audio_segments[nn_idx, 0].numpy())
                plt.title(f"Nearest Neighbor {j} (similarity: {similarity[i, nn_idx]:.3f})")
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"nearest_neighbors/sample_{i}.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error plotting nearest neighbors for sample {i}: {e}")

def plot_training_curves(output_dir):
    """Plot training loss curves if available."""
    try:
        # Look for saved losses
        loss_files = [
            os.path.join(output_dir, "losses.npy"),
            os.path.join(os.path.dirname(output_dir), "visualizations", "losses.npy"),
            os.path.join("visualizations", "losses.npy")
        ]
        
        losses = None
        for loss_file in loss_files:
            if os.path.exists(loss_file):
                losses = np.load(loss_file)
                print(f"Loaded losses from {loss_file}")
                break
        
        if losses is not None:
            plt.figure(figsize=(12, 5))
            
            # Plot full loss curve
            plt.subplot(1, 2, 1)
            plt.plot(losses)
            plt.title('Training Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.grid(True)
            
            # Plot smoothed loss curve
            plt.subplot(1, 2, 2)
            window_size = max(1, len(losses) // 100)
            if len(losses) > window_size:
                smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                plt.plot(smoothed_losses)
                plt.title(f'Smoothed Training Loss (window={window_size})')
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=150)
            plt.close()
            print("Saved training loss plot")
        else:
            print("No loss data found")
            
    except Exception as e:
        print(f"Error plotting training curves: {e}")

def main():
    args = parse_args()
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    # Load checkpoint
    try:
        model, _, _, _, _, _, _ = load_checkpoint(args.checkpoint, model)
        model = model.to(device)
        model.eval()
        print(f"Loaded model from {args.checkpoint}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Create data loader
    try:
        dataloader = create_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            sample_rate=args.sample_rate,
            segment_length=args.segment_length,
            num_workers=args.num_workers,
            shuffle=True
        )
        print(f"Created dataloader with {len(dataloader)} batches")
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        return
    
    # Visualize spectrograms
    print("Visualizing spectrograms...")
    try:
        visualize_spectrograms(model, dataloader, device, args.output_dir, args.num_samples)
        print("Spectrograms visualization completed")
    except Exception as e:
        print(f"Error visualizing spectrograms: {e}")
    
    # Visualize embeddings
    print("Visualizing embeddings...")
    try:
        visualize_embeddings(model, dataloader, device, args.output_dir, args.max_batches)
        print("Embeddings visualization completed")
    except Exception as e:
        print(f"Error visualizing embeddings: {e}")
    
    # Visualize similarity matrices
    print("Visualizing similarity matrices...")
    try:
        visualize_similarity(model, dataloader, device, args.output_dir)
        print("Similarity matrices visualization completed")
    except Exception as e:
        print(f"Error visualizing similarity matrices: {e}")
    
    # Visualize nearest neighbors
    print("Visualizing nearest neighbors...")
    try:
        visualize_nearest_neighbors(model, dataloader, device, args.output_dir)
        print("Nearest neighbors visualization completed")
    except Exception as e:
        print(f"Error visualizing nearest neighbors: {e}")
    
    # Plot training curves
    print("Plotting training curves...")
    try:
        plot_training_curves(args.output_dir)
        print("Training curves plotting completed")
    except Exception as e:
        print(f"Error plotting training curves: {e}")
    
    print(f"All visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
