# wav2vec: Unsupervised Pre-training for Speech Recognition

This repository contains a PyTorch implementation of the wav2vec model as described in the paper:
[wav2vec: Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/1904.05862v4) (Schneider et al., 2019).

**Note: This implementation does not use torchaudio and relies on scipy and soundfile for audio processing, making it more lightweight while maintaining full GPU acceleration.**

## Overview

wav2vec is an unsupervised pre-training approach for speech recognition that learns representations of raw audio. The model is trained on large amounts of unlabeled audio data and the resulting representations can then be used to improve acoustic model training, especially in low-resource settings.

The model consists of two main components:
1. **Encoder Network**: A 5-layer convolutional network that converts raw audio into latent representations
2. **Context Network**: A 9-layer causal convolutional network that aggregates the encoder's output to produce context-aware representations

The model is trained using a contrastive predictive coding objective, where it learns to distinguish future audio samples from random negative samples.

## File Structure

```
wav2vec_scratch/
├── data/                  # [provided by user] contains folders of audio files
├── utils.py               # utility functions (logging, checkpointing, audio I/O, etc.)
├── data.py                # custom PyTorch Dataset + DataLoader using soundfile
├── model.py               # full wav2vec model (encoder + context network)
├── main.py                # training script with CLI args
├── vis.py                 # generates loss plots, embeddings, spectrograms, etc.
├── requirements.txt       # all required packages (torch, soundfile, scipy, etc.)
├── README.md              # how to install, run, train, and visualize
└── checkpoints/           # stores .pt models
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/wav2vec_scratch.git
cd wav2vec_scratch
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Supported Audio Formats

This implementation supports multiple audio formats through soundfile:
- WAV (.wav)
- FLAC (.flac) 
- MP3 (.mp3)
- M4A (.m4a)
- OGG (.ogg)

## Training

To train the wav2vec model, you need a directory containing audio files. The script will recursively search for all supported audio files in the specified directory.

Basic training command:
```bash
python main.py --data-dir data/ --epochs 100 --batch-size 8 --lr 1e-3
```

### Training Options

**Data Parameters:**
- `--data-dir`: Directory containing audio files (required)
- `--sample-rate`: Target sample rate (default: 16000)
- `--segment-length`: Audio segment length in samples (default: 32000, ~2s)

**Model Parameters:**
- `--model`: Model architecture (`wav2vec` or `wav2vec_large`)
- `--channels`: Number of channels in model (default: 512)
- `--num-steps`: Number of steps for prediction (default: 12)
- `--num-negatives`: Number of negative samples (default: 10)
- `--loss-type`: Type of contrastive loss (`contrastive` or `infonce`)

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-3)
- `--fp16`: Use mixed precision training
- `--use-transforms`: Enable data augmentation
- `--resume`: Resume training from latest checkpoint

**Example with all options:**
```bash
python main.py \\
    --data-dir data/ \\
    --model wav2vec_large \\
    --epochs 200 \\
    --batch-size 16 \\
    --lr 5e-4 \\
    --fp16 \\
    --use-transforms \\
    --num-negatives 10 \\
    --loss-type infonce
```

For a full list of options, run:
```bash
python main.py --help
```

## Visualization

After training, you can visualize the model's representations using the `vis.py` script:

```bash
python vis.py --checkpoint checkpoints/best_model.pt --data-dir data/ --num-samples 10
```

This will generate:
- **Loss curves**: Training progress over time
- **Spectrograms**: Input audio and latent representations
- **Embedding visualizations**: t-SNE and PCA plots of learned representations
- **Similarity heatmaps**: Cosine similarity matrices of embeddings
- **Nearest neighbors**: Similar audio samples in latent space

### Visualization Options

- `--checkpoint`: Path to model checkpoint (required)
- `--data-dir`: Directory containing audio files (required)
- `--output-dir`: Directory for saving visualizations (default: visualizations)
- `--num-samples`: Number of audio samples to visualize (default: 10)
- `--max-batches`: Maximum batches for embedding collection (default: 20)

## Model Architecture

### Encoder Network
- 5-layer convolutional network
- Kernel sizes: (10, 8, 4, 4, 4)
- Strides: (5, 4, 2, 2, 2)
- Group normalization and ReLU after each layer
- Xavier weight initialization

### Context Network
- 9-layer causal convolutional network
- Kernel size: 3, stride: 1
- Group normalization and ReLU after each layer
- Causal padding for autoregressive modeling

### wav2vec Large
- Additional linear transformations in the encoder
- 12-layer context network with increasing kernel sizes (2-13)
- Skip connections every 3 layers in the context network
- Increased model capacity for larger datasets

## Loss Functions

### Contrastive Loss
Standard contrastive predictive coding loss as described in the paper:
$$
L_k = -log σ(z_{i+k}^T W_k c_i) + λ ∑_{neg} log σ(-z̃^T W_k c_i)
$$

### InfoNCE Loss
Alternative InfoNCE implementation for more stable training:
- Uses cross-entropy formulation
- More numerically stable
- Often converges faster

## Data Augmentation

When `--use-transforms` is enabled, the following augmentations are applied:
- **Gaussian noise**: Adds random noise to improve robustness
- **Time shifting**: Randomly shifts audio in time
- **Amplitude scaling**: Randomly scales audio amplitude

## GPU Acceleration

The implementation is fully GPU-accelerated:
- Automatic mixed precision (AMP) support with `--fp16`
- Efficient memory management
- CUDA memory monitoring and logging
- Gradient accumulation for large effective batch sizes

## Key Features

✅ **No torchaudio dependency** - Uses scipy and soundfile for audio processing  
✅ **Multiple audio formats** - Supports WAV, FLAC, MP3, M4A, OGG  
✅ **GPU acceleration** - Full CUDA support with mixed precision  
✅ **Robust training** - Gradient clipping, learning rate scheduling, checkpointing  
✅ **Data augmentation** - Built-in audio transforms  
✅ **Comprehensive visualization** - Spectrograms, embeddings, similarity analysis  
✅ **Two loss functions** - Contrastive and InfoNCE implementations  
✅ **Model variants** - Standard and large architectures  

## Troubleshooting

**Out of Memory Errors:**
- Reduce `--batch-size`
- Use `--fp16` for mixed precision
- Increase `--accumulate-steps` to maintain effective batch size

**Audio Loading Errors:**
- Ensure audio files are not corrupted
- Check that soundfile supports your audio format
- Verify file permissions

**Training Instability:**
- Try `--loss-type infonce` for more stable training
- Reduce learning rate
- Enable gradient clipping (default: 5.0)

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{schneider2019wav2vec,
  title={wav2vec: Unsupervised Pre-training for Speech Recognition},
  author={Schneider, Steffen and Baevski, Alexei and Collobert, Ronan and Auli, Michael},
  journal={arXiv preprint arXiv:1904.05862},
  year={2019}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
