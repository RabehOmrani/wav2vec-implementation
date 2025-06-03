"""
Utility functions for wav2vec model.
Includes logging, checkpointing, audio I/O, etc.
No torchaudio dependency - uses scipy and soundfile instead.
"""
import os
import json
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf
from scipy import signal
import time
from datetime import datetime

def setup_logger(log_dir):
    """Set up logger for training and evaluation."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger()

def load_audio(file_path, target_sr=16000):
    """Load audio file and resample if necessary using scipy."""
    # Load audio using soundfile
    waveform, sample_rate = sf.read(file_path, dtype='float32')
    
    # Convert to mono if stereo
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    
    # Resample if needed using scipy
    if sample_rate != target_sr:
        # Calculate resampling ratio
        ratio = target_sr / sample_rate
        # Use scipy's resample function
        num_samples = int(len(waveform) * ratio)
        waveform = signal.resample(waveform, num_samples)
    
    # Convert to torch tensor
    waveform = torch.from_numpy(waveform.astype(np.float32))
    
    return waveform, target_sr

def normalize_audio(waveform):
    """Normalize audio to have zero mean and unit variance."""
    waveform = waveform - waveform.mean()
    waveform = waveform / (waveform.std() + 1e-8)
    return waveform

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, best_loss, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'best_loss': best_loss
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model if applicable
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(checkpoint, best_path)
    
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', float('inf'))
    best_loss = checkpoint.get('best_loss', float('inf'))
    
    return model, optimizer, scheduler, epoch, step, loss, best_loss

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the checkpoint directory."""
    checkpoints = list(Path(checkpoint_dir).glob("epoch_*.pt"))
    if not checkpoints:
        return None
    
    # Extract epoch numbers and find the latest
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
    return str(latest_checkpoint)

def plot_loss_curve(losses, save_path):
    """Plot and save loss curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def compute_spectrogram(waveform, sample_rate, n_fft=1024, hop_length=512):
    """Generate spectrogram from waveform using scipy."""
    # Convert to numpy if it's a tensor
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    
    # Compute STFT using scipy
    f, t, Zxx = signal.stft(waveform, fs=sample_rate, nperseg=n_fft, noverlap=n_fft-hop_length)
    
    # Convert to magnitude spectrogram
    magnitude = np.abs(Zxx)
    
    # Convert to dB
    magnitude_db = 20 * np.log10(magnitude + 1e-8)
    
    return magnitude_db, f, t

def plot_spectrogram(waveform, sample_rate, save_path):
    """Generate and save spectrogram from waveform."""
    plt.figure(figsize=(10, 4))
    
    # Generate spectrogram
    spectrogram_db, f, t = compute_spectrogram(waveform, sample_rate)
    
    plt.imshow(spectrogram_db, aspect='auto', origin='lower', 
               extent=[t[0], t[-1], f[0], f[-1]])
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_embedding_similarity(embeddings, save_path):
    """Plot similarity matrix of embeddings."""
    # Convert to numpy if tensor
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embeddings = embeddings / (norms + 1e-8)
    
    # Compute similarity matrix
    similarity = np.dot(norm_embeddings, norm_embeddings.T)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity, cmap='viridis')
    plt.colorbar()
    plt.title('Embedding Similarity Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def log_gpu_memory():
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
            logging.info(f"GPU {i}: Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")

def apply_window(waveform, window_type='hann'):
    """Apply windowing function to reduce spectral leakage."""
    if isinstance(waveform, torch.Tensor):
        waveform_np = waveform.numpy()
    else:
        waveform_np = waveform
    
    if window_type == 'hann':
        window = signal.windows.hann(len(waveform_np))
    elif window_type == 'hamming':
        window = signal.windows.hamming(len(waveform_np))
    else:
        window = np.ones(len(waveform_np))
    
    windowed = waveform_np * window
    
    if isinstance(waveform, torch.Tensor):
        return torch.from_numpy(windowed.astype(np.float32))
    else:
        return windowed

def preemphasis(waveform, coeff=0.97):
    """Apply pre-emphasis filter to the waveform."""
    if isinstance(waveform, torch.Tensor):
        waveform_np = waveform.numpy()
    else:
        waveform_np = waveform
    
    # Apply pre-emphasis: y[n] = x[n] - coeff * x[n-1]
    emphasized = np.append(waveform_np[0], waveform_np[1:] - coeff * waveform_np[:-1])
    
    if isinstance(waveform, torch.Tensor):
        return torch.from_numpy(emphasized.astype(np.float32))
    else:
        return emphasized
