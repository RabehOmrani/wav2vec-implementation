"""
Implementation of the wav2vec model architecture.
GPU-accelerated PyTorch implementation without torchaudio dependency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class GroupNormConvLayer(nn.Module):
    """Convolutional layer with group normalization and ReLU activation."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 causal=False, dilation=1):
        super().__init__()
        
        # For causal convolution, add padding only on the left side
        if causal:
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size, 
                stride=stride, padding=0, dilation=dilation
            )
        else:
            self.padding = padding
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size, 
                stride=stride, padding=padding, dilation=dilation
            )
        
        self.norm = nn.GroupNorm(1, out_channels)  # Group=1 is equivalent to Layer Norm
        self.relu = nn.ReLU()
        self.causal = causal
    
    def forward(self, x):
        if self.causal:
            x = F.pad(x, (self.padding, 0))
        
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    """Encoder network for wav2vec."""
    
    def __init__(self, channels=512):
        super().__init__()
        
        # 5-layer convolutional network with kernel sizes (10, 8, 4, 4, 4) and strides (5, 4, 2, 2, 2)
        self.conv_layers = nn.ModuleList([
            GroupNormConvLayer(1, channels, kernel_size=10, stride=5, padding=0),
            GroupNormConvLayer(channels, channels, kernel_size=8, stride=4, padding=0),
            GroupNormConvLayer(channels, channels, kernel_size=4, stride=2, padding=0),
            GroupNormConvLayer(channels, channels, kernel_size=4, stride=2, padding=0),
            GroupNormConvLayer(channels, channels, kernel_size=4, stride=2, padding=0)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, sequence_length]
            
        Returns:
            torch.Tensor: Encoded features of shape [batch_size, channels, encoded_length]
        """
        for layer in self.conv_layers:
            x = layer(x)
        return x


class ContextNetwork(nn.Module):
    """Context network for wav2vec."""
    
    def __init__(self, channels=512, kernel_size=3, layers=9):
        super().__init__()
        
        # 9-layer causal convolutional network with kernel size 3 and stride 1
        self.conv_layers = nn.ModuleList([
            GroupNormConvLayer(channels, channels, kernel_size=kernel_size, stride=1, causal=True)
            for _ in range(layers)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass through the context network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, sequence_length]
            
        Returns:
            torch.Tensor: Contextualized features of shape [batch_size, channels, sequence_length]
        """
        for layer in self.conv_layers:
            x = layer(x)
        return x


class Wav2Vec(nn.Module):
    """
    Complete wav2vec model with encoder and context networks.
    
    As described in the paper:
    "wav2vec: Unsupervised Pre-training for Speech Recognition" (arXiv:1904.05862v4)
    """
    
    def __init__(self, channels=512, context_layers=9, num_steps=12):
        super().__init__()
        
        self.encoder = Encoder(channels=channels)
        self.context_network = ContextNetwork(channels=channels, layers=context_layers)
        
        # Projection heads for k-step predictions
        self.proj_heads = nn.ModuleList([
            nn.Linear(channels, channels) for _ in range(num_steps)
        ])
        
        self.num_steps = num_steps
        self.channels = channels
        
        # Initialize projection heads
        for head in self.proj_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
    
    def forward(self, x):
        """
        Forward pass through the wav2vec model.
        
        Args:
            x (torch.Tensor): Raw audio input of shape [batch_size, 1, sequence_length]
            
        Returns:
            tuple: 
                - z (torch.Tensor): Encoded features [batch_size, channels, encoded_length]
                - c (torch.Tensor): Contextualized features [batch_size, channels, encoded_length]
        """
        # Encode raw audio
        z = self.encoder(x)
        
        # Get context vectors
        c = self.context_network(z)
        
        return z, c
    
    def get_predictions(self, c):
        """
        Apply projection heads to context vectors for k-step predictions.
        
        Args:
            c (torch.Tensor): Context vectors [batch_size, channels, sequence_length]
            
        Returns:
            list: List of k prediction tensors, each of shape [batch_size, channels, sequence_length]
        """
        # Transpose for linear layer: [batch_size, channels, seq_len] -> [batch_size, seq_len, channels]
        c_t = c.transpose(1, 2)
        
        # Apply projection heads
        predictions = []
        for k in range(self.num_steps):
            pred_k = self.proj_heads[k](c_t)
            # Transpose back: [batch_size, seq_len, channels] -> [batch_size, channels, seq_len]
            pred_k = pred_k.transpose(1, 2)
            predictions.append(pred_k)
        
        return predictions


class Wav2VecLarge(Wav2Vec):
    """
    Larger variant of wav2vec with increased capacity.
    
    Features:
    - Two additional linear transformations in the encoder
    - Larger context network with 12 layers and increasing kernel sizes
    - Skip connections in the context network
    """
    
    def __init__(self, channels=512, num_steps=12):
        # Initialize with base parameters but we'll override the context network
        super().__init__(channels=channels, context_layers=0, num_steps=num_steps)
        
        # Add linear transformations to encoder
        self.encoder_linear1 = nn.Linear(channels, channels)
        self.encoder_linear2 = nn.Linear(channels, channels)
        
        # Create context network with increasing kernel sizes (2, 3, ..., 13)
        self.context_layers = nn.ModuleList([
            GroupNormConvLayer(channels, channels, kernel_size=k, stride=1, causal=True)
            for k in range(2, 14)  # 12 layers with kernel sizes 2 to 13
        ])
        
        # Initialize additional layers
        nn.init.xavier_uniform_(self.encoder_linear1.weight)
        nn.init.zeros_(self.encoder_linear1.bias)
        nn.init.xavier_uniform_(self.encoder_linear2.weight)
        nn.init.zeros_(self.encoder_linear2.bias)
    
    def forward(self, x):
        """
        Forward pass through the wav2vec large model.
        
        Args:
            x (torch.Tensor): Raw audio input of shape [batch_size, 1, sequence_length]
            
        Returns:
            tuple: 
                - z (torch.Tensor): Encoded features [batch_size, channels, encoded_length]
                - c (torch.Tensor): Contextualized features [batch_size, channels, encoded_length]
        """
        # Encode raw audio
        z = self.encoder(x)
        
        # Apply additional linear transformations
        z_t = z.transpose(1, 2)
        z_t = self.encoder_linear1(z_t)
        z_t = F.relu(z_t)
        z_t = self.encoder_linear2(z_t)
        z = z_t.transpose(1, 2)
        
        # Apply context network with skip connections
        c = z
        for i, layer in enumerate(self.context_layers):
            c_new = layer(c)
            # Add skip connection every 3 layers
            if i % 3 == 2 and i > 0:
                c_new = c_new + c
            c = c_new
        
        return z, c


class ContrastiveLoss(nn.Module):
    """
    Contrastive predictive coding loss for wav2vec.
    
    L_k = -log σ(z_{i+k}^T W_k c_i) + λ ∑_{neg} log σ(-z̃^T W_k c_i)
    """
    
    def __init__(self, num_negatives=10, temperature=1.0):
        super().__init__()
        self.num_negatives = num_negatives
        self.temperature = temperature
    
    def forward(self, z, c, predictions):
        """
        Compute contrastive loss.
        
        Args:
            z (torch.Tensor): Encoded features [batch_size, channels, sequence_length]
            c (torch.Tensor): Contextualized features [batch_size, channels, sequence_length]
            predictions (list): List of k prediction tensors from projection heads
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        batch_size, channels, seq_len = z.shape
        device = z.device
        
        # Initialize total loss
        total_loss = 0.0
        valid_steps = 0
        
        # For each step k
        for k, pred_k in enumerate(predictions, 1):
            if seq_len <= k:
                continue  # Skip if sequence is too short for this step
            
            # Get positive examples: z_{i+k}
            pos_z = z[:, :, k:seq_len]  # [batch_size, channels, seq_len-k]
            
            # Get context predictions: h_k(c_i)
            context_pred = pred_k[:, :, :seq_len-k]  # [batch_size, channels, seq_len-k]
            
            # Ensure dimensions match
            if pos_z.shape != context_pred.shape:
                min_len = min(pos_z.shape[2], context_pred.shape[2])
                pos_z = pos_z[:, :, :min_len]
                context_pred = context_pred[:, :, :min_len]
            
            if pos_z.shape[2] == 0:  # Skip if no valid positions
                continue
            
            # Compute positive similarities
            # [batch_size, seq_len-k]
            pos_sim = torch.sum(pos_z * context_pred, dim=1) / self.temperature
            
            # Sample negative examples from the batch
            # Reshape z for easier sampling: [batch_size, channels, seq_len] -> [batch_size*seq_len, channels]
            z_flat = z.transpose(1, 2).contiguous().view(-1, channels)  # [batch_size*seq_len, channels]
            
            # Sample random indices for negatives
            num_total_samples = z_flat.shape[0]
            num_pos_samples = pos_z.shape[0] * pos_z.shape[2]  # batch_size * (seq_len-k)
            
            # Create negative samples
            neg_indices = torch.randint(
                0, num_total_samples, 
                (num_pos_samples, self.num_negatives), 
                device=device
            )
            neg_z = z_flat[neg_indices]  # [num_pos_samples, num_negatives, channels]
            
            # Reshape context predictions for negative similarity computation
            context_pred_flat = context_pred.transpose(1, 2).contiguous().view(-1, channels)  # [num_pos_samples, channels]
            
            # Compute negative similarities
            # [num_pos_samples, num_negatives]
            neg_sim = torch.sum(
                neg_z * context_pred_flat.unsqueeze(1), dim=2
            ) / self.temperature
            
            # Compute contrastive loss for this step
            # Positive term: -log(sigmoid(positive_similarity))
            pos_loss = -F.logsigmoid(pos_sim.view(-1)).mean()
            
            # Negative term: -log(sigmoid(-negative_similarity))
            neg_loss = -F.logsigmoid(-neg_sim).mean()
            
            # Combine positive and negative losses
            step_loss = pos_loss + neg_loss
            
            # Add to total loss
            total_loss += step_loss
            valid_steps += 1
        
        # Average over valid steps
        if valid_steps > 0:
            return total_loss / valid_steps
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class InfoNCELoss(nn.Module):
    """
    Alternative InfoNCE loss implementation for wav2vec.
    This is a more standard implementation of the contrastive loss.
    """
    
    def __init__(self, num_negatives=10, temperature=0.1):
        super().__init__()
        self.num_negatives = num_negatives
        self.temperature = temperature
    
    def forward(self, z, c, predictions):
        """
        Compute InfoNCE loss.
        
        Args:
            z (torch.Tensor): Encoded features [batch_size, channels, sequence_length]
            c (torch.Tensor): Contextualized features [batch_size, channels, sequence_length]
            predictions (list): List of k prediction tensors from projection heads
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        batch_size, channels, seq_len = z.shape
        device = z.device
        
        total_loss = 0.0
        valid_steps = 0
        
        for k, pred_k in enumerate(predictions, 1):
            if seq_len <= k:
                continue
            
            # Get positive examples and predictions
            pos_z = z[:, :, k:seq_len]  # [batch_size, channels, seq_len-k]
            context_pred = pred_k[:, :, :seq_len-k]  # [batch_size, channels, seq_len-k]
            
            # Ensure dimensions match
            if pos_z.shape != context_pred.shape:
                min_len = min(pos_z.shape[2], context_pred.shape[2])
                pos_z = pos_z[:, :, :min_len]
                context_pred = context_pred[:, :, :min_len]
            
            if pos_z.shape[2] == 0:  # Skip if no valid positions
                continue
            
            # Reshape for easier computation
            pos_z_flat = pos_z.transpose(1, 2).contiguous().view(-1, channels)  # [batch_size*(seq_len-k), channels]
            context_pred_flat = context_pred.transpose(1, 2).contiguous().view(-1, channels)  # [batch_size*(seq_len-k), channels]
            
            # Compute positive similarities
            pos_sim = torch.sum(pos_z_flat * context_pred_flat, dim=1) / self.temperature  # [batch_size*(seq_len-k)]
            
            # Sample negatives from the entire batch
            z_all = z.transpose(1, 2).contiguous().view(-1, channels)  # [batch_size*seq_len, channels]
            
            # For each positive, sample negatives
            num_pos = pos_z_flat.shape[0]
            neg_indices = torch.randint(0, z_all.shape[0], (num_pos, self.num_negatives), device=device)
            neg_z = z_all[neg_indices]  # [num_pos, num_negatives, channels]
            
            # Compute negative similarities
            neg_sim = torch.sum(
                neg_z * context_pred_flat.unsqueeze(1), dim=2
            ) / self.temperature  # [num_pos, num_negatives]
            
            # Combine positive and negative similarities
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [num_pos, 1+num_negatives]
            
            # Labels: positive is always at index 0
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
            
            # Compute cross-entropy loss
            step_loss = F.cross_entropy(logits, labels)
            
            total_loss += step_loss
            valid_steps += 1
        
        if valid_steps > 0:
            return total_loss / valid_steps
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
