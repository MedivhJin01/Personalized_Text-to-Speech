#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference for CVAE-Tacotron2 (speaker-personalized TTS)
"""
import argparse, os, sys, numpy as np, torch
from scipy.io.wavfile import write

sys.path.append("src")                     
from model.cvae_tacotron_wrapper import CVAETacotron2       

# ------------------------- utils ---------------------------------
def device_select():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def load_speaker_lookup(path: str) -> dict[str, np.ndarray]:
    return np.load(path, allow_pickle=True).item()   # {spk_id: [256]}

# ------------------------- main ----------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_ckpt", required=True, help="fine-tuned CVAE ckpt")
    p.add_argument("--tacotron_ckpt", required=True, help="pretrained Tacotron2 ckpt")
    p.add_argument("--spk_lookup", default="src/speaker_embedding/speaker_emb_lookup.npy")
    p.add_argument("--text", required=True)
    p.add_argument("--speaker_id", default=None, help="e.g. p225")
    p.add_argument("--speaker_emb_path", default=None, help="npy file with 256-d vector")
    p.add_argument("--out_wav", default="out.wav")
    p.add_argument("--sigma", type=float, default=1.0, help="std scale for z")
    args = p.parse_args()

    dev = device_select()
    print("🔧 device:", dev)

    # ---------- build model ----------
    model = CVAETacotron2(
        ckpt_path=args.tacotron_ckpt,
        spk_emb_lookup_path=args.spk_lookup).to(dev)
    model.load_state_dict(torch.load(args.model_ckpt, map_location=dev))
    model.eval()

    # ---------- speaker embedding ----------
    if args.speaker_emb_path:
        spk_emb = torch.from_numpy(np.load(args.speaker_emb_path)).to(dev).unsqueeze(0)
    elif args.speaker_id:
        spk_lookup = load_speaker_lookup(args.spk_lookup)
        if args.speaker_id not in spk_lookup:
            raise KeyError(f"speaker_id {args.speaker_id} not in lookup")
        spk_emb = torch.from_numpy(spk_lookup[args.speaker_id]).to(dev).unsqueeze(0)
    else:
        print("Warning: No speaker embedding provided, using zero speaker embedding.")
        spk_emb = torch.zeros(1, 256, device=dev)

    # ---------- call infer ----------
    out = model.infer(args.text, spk_dvec=spk_emb.squeeze(0),
                      sigma=args.sigma, device=dev)

    mel = out["mel"]                          # [80,T]
    print("✅ mel generated:", mel.shape)

    # ---------- vocoder ----------
    print("→ Vocoder (WaveGlow)…")
    waveglow = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub",
                              "nvidia_waveglow", model_math="fp16").to(dev).eval()
    mel_in = mel.unsqueeze(0).to(dev)
    with torch.no_grad():
        audio = waveglow.infer(mel_in)[0].cpu().numpy()

    rate = 22050
    # Normalize audio and convert to int16 before saving
    audio = audio / np.max(np.abs(audio))
    write(args.out_wav, rate, (audio * 32767).astype(np.int16))
    print("🎵 saved to", args.out_wav)


if __name__ == "__main__":
    main()
    
    