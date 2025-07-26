"""C V A E Tacotron2 Wrapper (updated)

Adds a Conditional‑VAE branch to a pretrained NVIDIA Tacotron2.  
**Text input must already be tokenised** to int IDs (LongTensor) using the same
Grapheme/phoneme cleaner that the pre‑trained model expects – typically
`text_to_sequence(<raw str>, ['english_cleaners'])` from tacotron2 utils.

Key fixes vs previous draft
---------------------------
* **Correct encoder call** – first run `self.tts.embedding(text)` to turn
  `LongTensor` into float embeddings *before* the convolution stack.
* Clarified docstrings and comments so it’s obvious that `text` must be
  token IDs, not raw strings.
* Minor renames for clarity.
"""

from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

N_MELS = 80

# -----------------------------------------------------------------------------
# Reference Encoder ------------------------------------------------------------
# -----------------------------------------------------------------------------

class ReferenceEncoder(nn.Module):
    """3×Conv2d → GRU → μ, logσ² as in Skerry‑Ryan 2018."""

    def __init__(self, z_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            )
            for in_ch, out_ch in zip([1, 32, 64], [32, 64, 128])
        ])
        self.gru = nn.GRU(128 * (N_MELS // 8), 256, batch_first=True)
        self.mu = nn.Linear(256, z_dim)
        self.logvar = nn.Linear(256, z_dim)

    def forward(self, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # mel: [B, 80, T]
        x = mel.unsqueeze(1)               # [B,1,80,T]
        x = self.conv(x)                   # [B,128,F',T']
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        _, h = self.gru(x)                 # h: [1,B,256]
        h = h.squeeze(0)
        return self.mu(h), self.logvar(h)

# -----------------------------------------------------------------------------
# CVAE‑Tacotron2 ----------------------------------------------------------------
# -----------------------------------------------------------------------------

class CVAETacotron2(nn.Module):
    """Freeze pretrained Tacotron2; learn ReferenceEncoder + speaker projection."""

    def __init__(self, ckpt_path: str, *, z_dim: int = 64,
                 spk_dim_raw: int = 256, spk_dim_proj: int = 128):
        super().__init__()
        tacotron2 = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2',
            pretrained=False)
        tacotron2.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        tacotron2 = tacotron2.eval()
        for p in tacotron2.parameters():
            p.requires_grad = False
        self.tts = tacotron2

        self.ref_enc   = ReferenceEncoder(z_dim)
        self.spk_proj  = nn.Linear(spk_dim_raw, spk_dim_proj)
        self.cond_proj = nn.Linear(z_dim + spk_dim_proj, 512)  # match encoder dim

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, text_ids, text_lens, mel_gt, speaker_embed):
        """
        Parameters
        ----------
        text_ids : LongTensor [B, L]
            **Already tokenised** indices.
        text_lens : LongTensor [B]
        mel_gt : FloatTensor [B, 80, T]
        speaker_embed : FloatTensor [B, 256]
            Resemblyzer d‑vector (frozen).
        """
        # latent z
        mu, logvar = self.ref_enc(mel_gt)
        z = self.reparameterize(mu, logvar)            # [B, z_dim]
        # speaker condition
        e_spk = self.spk_proj(speaker_embed)            # [B,128]
        cond  = self.cond_proj(torch.cat([e_spk, z], -1))

        # encoder: embed → conv → bLSTM (pretrained & frozen)
        embedded = self.tts.embedding(text_ids).transpose(1, 2)  # [B,512?,L]
        enc_out  = self.tts.encoder(embedded, text_lens)          # [B,L,512]
        enc_out  = enc_out + cond.unsqueeze(1)                   # broadcast add

        # decoder (teacher forcing)
        mel_out, mel_post, gate_out, _ = self.tts.decoder(
            enc_out, text_lens, mel_gt)

        return mel_post, mel_out, gate_out, mu, logvar

# -----------------------------------------------------------------------------
# Loss helper
# -----------------------------------------------------------------------------

def cvae_taco_loss(mel_post, mel_gt, gate_out, gate_tgt, mu, logvar, *, beta=1e-4):
    l1  = F.l1_loss(mel_post, mel_gt)
    gate = F.binary_cross_entropy_with_logits(gate_out, gate_tgt)
    kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return l1 + gate + beta * kl, {'l1': l1.item(), 'gate': gate.item(), 'kl': kl.item()}

# -----------------------------------------------------------------------------
# sanity check
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    B, L, T = 4, 50, 400
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CVAETacotron2('tacotron2_pretrained.pt').to(dev)

    txt   = torch.randint(1, 150, (B, L)).long().to(dev)
    tlens = torch.randint(L//2, L, (B,)).long().to(dev)
    mel   = torch.randn(B, 80, T).to(dev)
    spk   = torch.randn(B, 256).to(dev)
    gate  = torch.zeros(B, T//model.tts.decoder.n_frames_per_step).to(dev)

    mel_post, mel_out, gate_out, mu, logvar = model(txt, tlens, mel, spk)
    loss, logs = cvae_taco_loss(mel_post, mel, gate_out, gate, mu, logvar)
    print('sanity OK', loss.item(), logs)
