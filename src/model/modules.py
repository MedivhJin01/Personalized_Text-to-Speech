import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.text.symbols import symbols
from model.blocks import ConvNorm


class TextEncoder(nn.Module):
    """Simple text encoder using convolutional layers followed by a bidirectional LSTM."""

    def __init__(self, embedding_dim=256, n_convolutions=3, kernel_size=5, hidden_dim=256):
        super().__init__()

        self.embedding = nn.Embedding(len(symbols), embedding_dim)

        convs = []
        for _ in range(n_convolutions):
            convs.append(
                nn.Sequential(
                    ConvNorm(
                        embedding_dim,
                        embedding_dim,
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) // 2,
                    ),
                    nn.BatchNorm1d(embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                )
            )
        self.convolutions = nn.ModuleList(convs)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, text, text_lengths):
        """Encode a batch of padded token sequences.

        Args:
            text (LongTensor): [B, T] token ids
            text_lengths (LongTensor): [B] valid lengths before padding

        Returns:
            Tensor: [B, T, hidden_dim] encoded outputs
        """
        x = self.embedding(text).transpose(1, 2)
        for conv in self.convolutions:
            x = conv(x)
        x = x.transpose(1, 2)

        lengths = text_lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs

    def inference(self, text):
        """Inference without padding info."""
        x = self.embedding(text).transpose(1, 2)
        for conv in self.convolutions:
            x = conv(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs