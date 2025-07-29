"""
Utility functions for TTS training and data processing.
"""

import torch
import numpy as np


def pad_1D(inputs, PAD=0):
    """
    Pad a list of 1D tensors to the same length.

    Args:
        inputs: List of 1D tensors
        PAD: Padding value

    Returns:
        Padded tensor with shape [batch_size, max_length]
    """

    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return torch.from_numpy(padded)


def pad_2D(inputs, maxlen=None, PAD=0):
    """
    Pad a list of 2D tensors (e.g., mel-spectrograms) to the same shape.

    Args:
        inputs: List of 2D tensors with shape [features, time]
        maxlen: Maximum length to pad to (if None, use max length in batch)
        PAD: Padding value

    Returns:
        Padded tensor with shape [batch_size, features, max_time]
    """

    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, ((0, 0), (0, length - x.shape[1])), mode="constant", constant_values=PAD
        )
        return x_padded

    if maxlen:
        output_len = maxlen
    else:
        output_len = max(x.shape[1] for x in inputs)

    padded = np.stack([pad_data(x, output_len, PAD) for x in inputs])

    return torch.from_numpy(padded)


def pad_3D(inputs, B, T, L):
    """
    Pad a list of 3D tensors to the same shape.

    Args:
        inputs: List of 3D tensors
        B: Batch size
        T: Time dimension
        L: Feature dimension

    Returns:
        Padded tensor
    """
    inputs_padded = np.zeros((B, T, L), dtype=np.float32)
    for i, batch in enumerate(inputs):
        inputs_padded[i, : batch.shape[0], :] = batch
    return inputs_padded


def get_mask_from_lengths(lengths, max_len=None):
    """
    Create a mask tensor from sequence lengths.

    Args:
        lengths: Tensor of sequence lengths
        max_len: Maximum length (if None, use max length in batch)

    Returns:
        Boolean mask tensor
    """
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


def to_gpu(x):
    """
    Move tensor to GPU if available.

    Args:
        x: Input tensor

    Returns:
        Tensor on GPU or original tensor if GPU not available
    """
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def sequence_mask(length, max_length=None):
    """
    Create a sequence mask.

    Args:
        length: Tensor of sequence lengths
        max_length: Maximum length

    Returns:
        Boolean mask tensor
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def compute_same_padding(kernel_size, dilation):
    """
    Compute padding for 'same' convolution.

    Args:
        kernel_size: Size of convolution kernel
        dilation: Dilation rate

    Returns:
        Padding value
    """
    return (kernel_size - 1) * dilation // 2


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)

    Returns:
        Loaded checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load model state
    model.load_state_dict(checkpoint["model"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def save_checkpoint(checkpoint_path, model, optimizer=None, **kwargs):
    """
    Save model checkpoint.

    Args:
        checkpoint_path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save (optional)
        **kwargs: Additional data to save
    """
    checkpoint = {"model": model.state_dict(), **kwargs}

    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()

    torch.save(checkpoint, checkpoint_path)


def plot_attention(attention, save_path=None):
    """
    Plot attention weights.

    Args:
        attention: Attention weights tensor
        save_path: Path to save plot (optional)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(attention, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xlabel("Decoder timestep")
    ax.set_ylabel("Encoder timestep")
    ax.set_title("Attention weights")
    plt.colorbar(im, ax=ax)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_spectrogram(spectrogram, save_path=None, title="Mel-Spectrogram"):
    """
    Plot mel-spectrogram.

    Args:
        spectrogram: Spectrogram tensor
        save_path: Path to save plot (optional)
        title: Plot title
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel channels")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
