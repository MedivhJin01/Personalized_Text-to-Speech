
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
                 spk_dim_raw: int = 256, spk_dim_proj: int = 128, spk_emb_lookup_path: str = "src/speaker_embedding/speaker_emb_lookup.npy"):
        super().__init__()
        tacotron2 = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2',
            pretrained=False)
        tacotron2.load_state_dict(torch.load(ckpt_path))
        tacotron2 = tacotron2.eval()
        for p in tacotron2.parameters():
            p.requires_grad = False
        self.tts = tacotron2

        # speaker embedding lookup table
        speaker_emb_dict = np.load(spk_emb_lookup_path, allow_pickle=True).item()
        speaker_look_up = torch.from_numpy(np.stack(list(speaker_emb_dict.values())))
        self.register_buffer("spk_emb", speaker_look_up)  # [N_spk, 256]

        self.ref_enc   = ReferenceEncoder(z_dim)
        self.spk_proj  = nn.Linear(spk_dim_raw, spk_dim_proj)
        self.cond_proj = nn.Linear(z_dim + spk_dim_proj, 512)  # match encoder dim

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, text_ids, text_lens, mel_gt, speaker_embed):
        # ---------- sort for pack_padded_sequence ----------
        text_lens, sort_idx = text_lens.sort(descending=True)
        text_ids      = text_ids[sort_idx]
        mel_gt        = mel_gt[sort_idx]
        speaker_embed = speaker_embed[sort_idx]

        # ---------- VAE + speaker condition ----------
        mu, logvar = self.ref_enc(mel_gt)
        z      = self.reparameterize(mu, logvar)
        e_spk  = self.spk_proj(speaker_embed)          # [B,128]
        cond   = self.cond_proj(torch.cat([e_spk, z], -1))  # [B,512]

        # ---------- Embed text ----------
        embedded = self.tts.embedding(text_ids)        # [B, T, 512]  (Float)
        embedded = embedded.transpose(1, 2)            # [B, 512, T]

        # ---------- Encoder  ----------
        enc_out = self.tts.encoder(embedded, text_lens)    # [B, T, 512]

        # ---------- add condition ----------
        enc_out = enc_out + cond.unsqueeze(1)          # broadcast add

        # ---------- Decoder (teacher forcing) ----------
        # order: decoder(memory, decoder_inputs, memory_lengths)
        mel_out, gate_out, align = self.tts.decoder(
            enc_out, mel_gt, text_lens)

        mel_post = mel_out + self.tts.postnet(mel_out)

        # ---------- to right order ----------
        inv_idx  = sort_idx.argsort()
        return (mel_post[inv_idx], mel_out[inv_idx],
                gate_out[inv_idx], mu[inv_idx], logvar[inv_idx])



# -----------------------------------------------------------------------------
# Loss helper
# -----------------------------------------------------------------------------

def cvae_taco_loss(mel_post, mel_gt, gate_out, gate_tgt, mu, logvar, *, beta=1e-4):
    l1  = F.l1_loss(mel_post, mel_gt)
    gate = F.binary_cross_entropy_with_logits(gate_out, gate_tgt)
    kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return l1 + gate + beta * kl, {'l1': l1.item(), 'gate': gate.item(), 'kl': kl.item()}

