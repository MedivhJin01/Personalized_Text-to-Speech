#!/usr/bin/env python3
"""
Inference script for the trained VAE with Tacotron2 for personalized TTS.
This script loads a trained model and synthesizes speech with personalized voice.
"""

import torch
import numpy as np
import argparse
import os
from scipy.io.wavfile import write

# Add src to path
import sys

sys.path.append("src")

from model.vae import VAE


def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_speaker_embeddings():
    """Load available speaker embeddings."""
    speaker_emb_path = "src/speaker_embedding/speaker_embeddings.npy"
    selected_speakers_path = "src/speaker_embedding/selected_speakers.txt"

    # Load speaker embeddings
    speaker_embeddings = np.load(speaker_emb_path, allow_pickle=True).item()

    # Load selected speakers
    with open(selected_speakers_path, "r") as f:
        selected_speakers = f.read().replace("'", "").replace("\n", "").split(",")
        selected_speakers = [s.strip() for s in selected_speakers if s.strip()]

    return speaker_embeddings, selected_speakers


def list_available_speakers():
    """List all available speakers."""
    speaker_embeddings, selected_speakers = load_speaker_embeddings()

    print("Available speakers:")
    for i, speaker_id in enumerate(selected_speakers):
        print(f"  {i+1:2d}. {speaker_id}")

    return selected_speakers


def get_speaker_embedding(speaker_id, device):
    """Get speaker embedding for a specific speaker ID."""
    speaker_embeddings, selected_speakers = load_speaker_embeddings()

    if speaker_id in speaker_embeddings:
        embedding = torch.tensor(
            speaker_embeddings[speaker_id], dtype=torch.float32, device=device
        )
        return embedding.unsqueeze(0)  # Add batch dimension
    else:
        raise ValueError(
            f"Speaker ID '{speaker_id}' not found. Available speakers: {selected_speakers}"
        )


def load_trained_model(model_path, device):
    """Load the trained VAE model."""
    checkpoint = torch.load(model_path, map_location=device)

    # Initialize VAE with same parameters as training
    vae = VAE(
        n_mels=80,
        spk_emb_dim=256,
        latent_dim=64,
        hidden_dim=256,
        use_tacotron2=True,
        device=device,
    ).to(device)

    # Load trained weights
    vae.load_state_dict(checkpoint["vae"])
    vae.eval()

    print(
        f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}"
    )
    return vae


def synthesize_speech(
    vae, text, speaker_embedding=None, latent_z=None, output_path="output.wav"
):
    """
    Synthesize speech using the trained VAE with Tacotron2.

    Args:
        vae: Trained VAE model
        text: Text to synthesize (string or list of strings)
        speaker_embedding: Speaker embedding tensor [1, 256] or None
        latent_z: Latent representation tensor [1, 64] or None
        output_path: Path to save the output audio
    """
    device = next(vae.parameters()).device

    # Convert text to list if it's a string
    if isinstance(text, str):
        text = [text]

    # Create speaker embedding if not provided
    if speaker_embedding is None and latent_z is None:
        print("No speaker embedding or latent provided, using random embedding...")
        speaker_embedding = torch.randn(1, 256).to(device)
    import torch
    with torch.no_grad():
        try:
            # Synthesize mel spectrogram
            mel_output = vae.synthesize(text, spk_emb=speaker_embedding, z=latent_z)

            print(f"Generated mel spectrogram shape: {mel_output.shape}")

            # Convert to audio using WaveGlow (if available)
            try:
                import torch.hub

                waveglow = torch.hub.load(
                    "NVIDIA/DeepLearningExamples:torchhub",
                    "nvidia_waveglow",
                    model_math="fp16",
                )
                waveglow = waveglow.remove_weightnorm(waveglow)
                waveglow = waveglow.to(device)
                waveglow.eval()

                print("Converting mel spectrogram to audio using WaveGlow...")
                audio = waveglow.infer(mel_output)
                audio_numpy = audio[0].data.cpu().numpy()
                rate = 22050

                # Save audio
                write(output_path, rate, audio_numpy)
                print(f"Audio saved to: {output_path}")

                return audio_numpy, rate

            except Exception as e:
                print(f"WaveGlow not available: {e}")
                print("Saving mel spectrogram instead...")

                # Save mel spectrogram
                mel_np = mel_output[0].cpu().numpy()
                np.save(output_path.replace(".wav", "_mel.npy"), mel_np)
                print(
                    f"Mel spectrogram saved to: {output_path.replace('.wav', '_mel.npy')}"
                )

                return mel_np, None

        except Exception as e:
            print(f"Error during synthesis: {e}")
            return None, None


def main():
    parser = argparse.ArgumentParser(description="Inference for VAE with Tacotron2")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/vae_tacotron2_best.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello world, this is a test of the personalized text-to-speech system.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="synthesized_audio.wav",
        help="Path to save output audio",
    )
    parser.add_argument(
        "--speaker_id",
        type=str,
        default=None,
        help="Speaker ID to use (e.g., 'p272', 'p298'). Use --list_speakers to see available speakers.",
    )
    parser.add_argument(
        "--speaker_index",
        type=int,
        default=None,
        help="Speaker index (1-based). Use --list_speakers to see available speakers.",
    )
    parser.add_argument(
        "--list_speakers",
        action="store_true",
        help="List all available speakers and exit",
    )
    parser.add_argument(
        "--speaker_embedding_path",
        type=str,
        default=None,
        help="Path to speaker embedding numpy file (optional, overrides speaker_id/speaker_index)",
    )
    parser.add_argument(
        "--latent_z_path",
        type=str,
        default=None,
        help="Path to latent representation numpy file (optional)",
    )

    args = parser.parse_args()

    # List speakers if requested
    if args.list_speakers:
        list_available_speakers()
        return

    # Set device
    device = get_device()
    print(f"Using device: {device}")

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        print("Please train the model first or provide correct path.")
        return

    # Load trained model
    print(f"Loading model from {args.model_path}...")
    vae = load_trained_model(args.model_path, device)

    # Load speaker embedding or latent
    speaker_embedding = None
    latent_z = None

    # Priority: speaker_embedding_path > speaker_id/speaker_index > latent_z_path > random
    if args.speaker_embedding_path and os.path.exists(args.speaker_embedding_path):
        print(f"Loading speaker embedding from {args.speaker_embedding_path}")
        spk_emb = np.load(args.speaker_embedding_path)
        speaker_embedding = torch.tensor(spk_emb, dtype=torch.float32, device=device)
        if speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)  # Add batch dimension

    elif args.speaker_id:
        print(f"Using speaker ID: {args.speaker_id}")
        speaker_embedding = get_speaker_embedding(args.speaker_id, device)

    elif args.speaker_index is not None:
        selected_speakers = list_available_speakers()
        if 1 <= args.speaker_index <= len(selected_speakers):
            speaker_id = selected_speakers[args.speaker_index - 1]
            print(f"Using speaker index {args.speaker_index}: {speaker_id}")
            speaker_embedding = get_speaker_embedding(speaker_id, device)
        else:
            print(
                f"Invalid speaker index {args.speaker_index}. Available range: 1-{len(selected_speakers)}"
            )
            return

    elif args.latent_z_path and os.path.exists(args.latent_z_path):
        print(f"Loading latent representation from {args.latent_z_path}")
        z = np.load(args.latent_z_path)
        latent_z = torch.tensor(z, dtype=torch.float32, device=device)
        if latent_z.dim() == 1:
            latent_z = latent_z.unsqueeze(0)  # Add batch dimension

    else:
        print("No speaker specified, using random speaker embedding...")
        speaker_embedding = torch.randn(1, 256).to(device)

    # Synthesize speech
    print(f"Synthesizing text: '{args.text}'")
    audio, rate = synthesize_speech(
        vae,
        args.text,
        speaker_embedding=speaker_embedding,
        latent_z=latent_z,
        output_path=args.output_path,
    )

    if audio is not None:
        print("Synthesis completed successfully!")
    else:
        print("Synthesis failed!")


if __name__ == "__main__":
    main()
