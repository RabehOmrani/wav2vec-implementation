"""
Custom PyTorch Dataset and DataLoader for wav2vec model.
No torchaudio dependency - uses scipy and soundfile instead.
"""
import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from utils import load_audio, normalize_audio

class AudioDataset(Dataset):
    """Dataset for loading and preprocessing audio files."""
    
    def __init__(self, data_dir, sample_rate=16000, segment_length=32000, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory containing audio files
            sample_rate (int): Target sample rate
            segment_length (int): Length of audio segments in samples (~2s at 16kHz)
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.transform = transform
        
        # Find all audio files recursively (support multiple formats)
        audio_extensions = ['*.wav', '*.flac', '*.mp3', '*.m4a', '*.ogg']
        self.audio_files = []
        
        for ext in audio_extensions:
            self.audio_files.extend(list(self.data_dir.glob(f'**/{ext}')))
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {data_dir}")
            
        print(f"Found {len(self.audio_files)} audio files in {data_dir}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """
        Get a random segment from an audio file.
        
        Returns:
            torch.Tensor: Audio segment of shape [1, segment_length]
        """
        audio_path = self.audio_files[idx]
        
        try:
            # Load and preprocess audio
            waveform, sr = load_audio(audio_path, self.sample_rate)
            
            # Normalize audio
            waveform = normalize_audio(waveform)
            
            # If audio is shorter than segment_length, pad with zeros
            if waveform.shape[0] < self.segment_length:
                padding = self.segment_length - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            # If audio is longer than segment_length, take a random segment
            elif waveform.shape[0] > self.segment_length:
                start = random.randint(0, waveform.shape[0] - self.segment_length)
                waveform = waveform[start:start + self.segment_length]
            
            # Apply additional transforms if specified
            if self.transform:
                waveform = self.transform(waveform)
            
            # Add channel dimension [segment_length] -> [1, segment_length]
            waveform = waveform.unsqueeze(0)
            
            return waveform
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return a zero tensor if loading fails
            return torch.zeros(1, self.segment_length)


class AudioTransforms:
    """Collection of audio transforms that can be applied to waveforms."""
    
    @staticmethod
    def add_noise(waveform, noise_factor=0.01):
        """Add Gaussian noise to the waveform."""
        noise = torch.randn_like(waveform) * noise_factor
        return waveform + noise
    
    @staticmethod
    def time_shift(waveform, shift_limit=0.1):
        """Randomly shift the waveform in time."""
        shift_amount = int(random.uniform(-shift_limit, shift_limit) * waveform.shape[-1])
        if shift_amount > 0:
            # Shift right
            shifted = torch.cat([torch.zeros_like(waveform[..., :shift_amount]), 
                               waveform[..., :-shift_amount]], dim=-1)
        elif shift_amount < 0:
            # Shift left
            shifted = torch.cat([waveform[..., -shift_amount:], 
                               torch.zeros_like(waveform[..., :shift_amount])], dim=-1)
        else:
            shifted = waveform
        return shifted
    
    @staticmethod
    def amplitude_scale(waveform, scale_range=(0.8, 1.2)):
        """Randomly scale the amplitude of the waveform."""
        scale = random.uniform(*scale_range)
        return waveform * scale
    
    @staticmethod
    def compose(*transforms):
        """Compose multiple transforms."""
        def composed_transform(waveform):
            for transform in transforms:
                waveform = transform(waveform)
            return waveform
        return composed_transform


def create_dataloader(data_dir, batch_size=8, sample_rate=16000, segment_length=32000, 
                      num_workers=4, shuffle=True, use_transforms=False):
    """
    Create a DataLoader for audio data.
    
    Args:
        data_dir (str): Directory containing audio files
        batch_size (int): Batch size
        sample_rate (int): Target sample rate
        segment_length (int): Length of audio segments in samples
        num_workers (int): Number of worker processes
        shuffle (bool): Whether to shuffle the dataset
        use_transforms (bool): Whether to apply data augmentation transforms
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    # Set up transforms if requested
    transform = None
    if use_transforms:
        transform = AudioTransforms.compose(
            AudioTransforms.add_noise,
            AudioTransforms.time_shift,
            AudioTransforms.amplitude_scale
        )
    
    dataset = AudioDataset(
        data_dir=data_dir,
        sample_rate=sample_rate,
        segment_length=segment_length,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader

def collate_fn(batch):
    """Custom collate function for handling variable length audio."""
    # Filter out None values (failed loads)
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return torch.zeros(1, 1, 32000)  # Return dummy tensor
    
    # Stack the batch
    return torch.stack(batch, dim=0)
