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
import matplotlib.pyplot as plt

# Add src to path
import sys

sys.path.append("src")

from model.vae import VAE


def visualize_alignments(alignments, text, output_path):
    """
    Visualize attention alignments.

    Args:
        alignments: Attention weights [T_decoder, T_encoder]
        text: Input text
        output_path: Path to save the visualization
    """
    try:
        plt.figure(figsize=(12, 8))
        plt.imshow(alignments.T, aspect="auto", origin="lower")
        plt.colorbar()
        plt.title(f'Attention Alignments\nText: "{text}"')
        plt.xlabel("Decoder Steps")
        plt.ylabel("Encoder Steps")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Alignment visualization saved to: {output_path}")
    except Exception as e:
        print(f"Failed to create alignment visualization: {e}")


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

    # Get model configuration from checkpoint
    use_tacotron2 = checkpoint.get("use_tacotron2", True)
    fine_tune_decoder = checkpoint.get("fine_tune_decoder", True)

    print(
        f"Loading model with use_tacotron2={use_tacotron2}, fine_tune_decoder={fine_tune_decoder}"
    )

    # Initialize VAE with same parameters as training
    vae = VAE(
        n_mels=80,
        spk_emb_dim=256,
        latent_dim=64,
        hidden_dim=256,
        use_tacotron2=use_tacotron2,
        fine_tune_decoder=fine_tune_decoder,
        device=device,
    ).to(device)

    # Load trained weights
    vae.load_state_dict(checkpoint["vae"])
    vae.eval()

    print(
        f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}"
    )

    # Print model information
    print("\n=== Model Information ===")
    total_params, trainable_params = vae.count_parameters()

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
    with torch.no_grad():
        try:
            # Synthesize mel spectrogram using the new decoder method
            if latent_z is not None:
                print("Using provided latent representation for synthesis...")
                mel_output, alignments = vae.synthesize_with_decoder(text, latent_z)
            else:
                print("Using speaker embedding for synthesis...")
                # Convert speaker embedding to latent space if needed
                if speaker_embedding.size(-1) != vae.latent_dim:
                    print(
                        f"Projecting speaker embedding from {speaker_embedding.size(-1)} to {vae.latent_dim} dimensions..."
                    )
                    # Use the speaker projection layer to convert to latent space
                    latent_z = vae.speaker_projection(speaker_embedding)
                else:
                    latent_z = speaker_embedding

                mel_output, alignments = vae.synthesize_with_decoder(text, latent_z)

            print(f"Generated mel spectrogram shape: {mel_output.shape}")
            print(f"Generated alignments shape: {alignments.shape}")

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

                # Save alignments for visualization
                alignments_path = output_path.replace(".wav", "_alignments.npy")
                np.save(alignments_path, alignments[0].cpu().numpy())
                print(f"Alignments saved to: {alignments_path}")

                return audio_numpy, rate

            except Exception as e:
                print(f"WaveGlow not available: {e}")
                print("Saving mel spectrogram instead...")

                # Save mel spectrogram
                mel_np = mel_output[0].cpu().numpy()
                mel_path = output_path.replace(".wav", "_mel.npy")
                np.save(mel_path, mel_np)
                print(f"Mel spectrogram saved to: {mel_path}")

                # Save alignments
                alignments_path = output_path.replace(".wav", "_alignments.npy")
                np.save(alignments_path, alignments[0].cpu().numpy())
                print(f"Alignments saved to: {alignments_path}")

                return mel_np, None

        except Exception as e:
            print(f"Error during synthesis: {e}")
            import traceback

            traceback.print_exc()
            return None, None


def extract_latent_from_audio(vae, mel_spectrogram, speaker_embedding, device):
    """
    Extract latent representation from a mel spectrogram using the VAE encoder.

    Args:
        vae: Trained VAE model
        mel_spectrogram: Mel spectrogram tensor [1, n_mels, T]
        speaker_embedding: Speaker embedding tensor [1, 256]
        device: Device to use

    Returns:
        latent_z: Latent representation tensor [1, latent_dim]
    """
    vae.eval()
    with torch.no_grad():
        # Encode the mel spectrogram and speaker embedding
        mu, logvar = vae.encode(mel_spectrogram, speaker_embedding)
        # Sample from the latent space
        latent_z = vae.reparameterize(mu, logvar)
        return latent_z


def main():
    parser = argparse.ArgumentParser(description="Inference for VAE with Tacotron2")
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/model/check_points/vae_tacotron2_best.pt",
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
    parser.add_argument(
        "--extract_latent",
        type=str,
        default=None,
        help="Extract latent from mel spectrogram file and save it",
    )
    parser.add_argument(
        "--max_decoder_steps",
        type=int,
        default=1000,
        help="Maximum number of decoder steps for synthesis",
    )
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Save intermediate outputs (mel spectrograms, alignments)",
    )
    parser.add_argument(
        "--visualize_alignments",
        action="store_true",
        help="Create and save alignment visualization",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
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

    with torch.no_grad():
        # Handle latent extraction if requested
        if args.extract_latent and os.path.exists(args.extract_latent):
            if speaker_embedding is None:
                print("Error: Speaker embedding required for latent extraction")
                return

            print(f"Extracting latent from {args.extract_latent}")
            mel_spectrogram = np.load(args.extract_latent)
            mel_tensor = torch.tensor(
                mel_spectrogram, dtype=torch.float32, device=device
            )
            if mel_tensor.dim() == 2:
                mel_tensor = mel_tensor.unsqueeze(0)  # Add batch dimension

            latent_z = extract_latent_from_audio(
                vae, mel_tensor, speaker_embedding, device
            )

            # Save extracted latent
            latent_path = args.extract_latent.replace(".npy", "_extracted_latent.npy")
            np.save(latent_path, latent_z[0].cpu().numpy())
            print(f"Extracted latent saved to: {latent_path}")

            # Use extracted latent for synthesis
            speaker_embedding = None

        # Synthesize speech
        print(f"Synthesizing text: '{args.text}'")

        # Use custom synthesis function with max_decoder_steps
        if latent_z is not None:
            print("Using provided latent representation for synthesis...")
            mel_output, alignments = vae.synthesize_with_decoder(
                [args.text], latent_z, max_decoder_steps=args.max_decoder_steps
            )
        else:
            print("Using speaker embedding for synthesis...")
            # Convert speaker embedding to latent space if needed
            if speaker_embedding.size(-1) != vae.latent_dim:
                print(
                    f"Projecting speaker embedding from {speaker_embedding.size(-1)} to {vae.latent_dim} dimensions..."
                )
                latent_z = vae.speaker_projection(speaker_embedding)
            else:
                latent_z = speaker_embedding

            mel_output, alignments = vae.synthesize_with_decoder(
                [args.text], latent_z, max_decoder_steps=args.max_decoder_steps
            )

        print(f"Generated mel spectrogram shape: {mel_output.shape}")
        print(f"Generated alignments shape: {alignments.shape}")

        # Save intermediate outputs if requested
        if args.save_intermediate:
            mel_path = args.output_path.replace(".wav", "_mel.npy")
            alignments_path = args.output_path.replace(".wav", "_alignments.npy")
            np.save(mel_path, mel_output[0].cpu().numpy())
            np.save(alignments_path, alignments[0].cpu().numpy())
            print(f"Intermediate outputs saved: {mel_path}, {alignments_path}")

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
            write(args.output_path, rate, audio_numpy)
            print(f"Audio saved to: {args.output_path}")

            # Save alignments for visualization
            alignments_path = args.output_path.replace(".wav", "_alignments.npy")
            np.save(alignments_path, alignments[0].cpu().numpy())
            print(f"Alignments saved to: {alignments_path}")

            # Create alignment visualization if requested
            if args.visualize_alignments:
                alignments_viz_path = args.output_path.replace(
                    ".wav", "_alignments.png"
                )
                visualize_alignments(
                    alignments[0].cpu().numpy(), args.text, alignments_viz_path
                )

            print("Synthesis completed successfully!")

        except Exception as e:
            print(f"WaveGlow not available: {e}")
            print("Saving mel spectrogram instead...")

            # Save mel spectrogram
            mel_np = mel_output[0].cpu().numpy()
            mel_path = args.output_path.replace(".wav", "_mel.npy")
            np.save(mel_path, mel_np)
            print(f"Mel spectrogram saved to: {mel_path}")

            # Save alignments
            alignments_path = args.output_path.replace(".wav", "_alignments.npy")
            np.save(alignments_path, alignments[0].cpu().numpy())
            print(f"Alignments saved to: {alignments_path}")

            # Create alignment visualization if requested
            if args.visualize_alignments:
                alignments_viz_path = args.output_path.replace(
                    ".wav", "_alignments.png"
                )
                visualize_alignments(
                    alignments[0].cpu().numpy(), args.text, alignments_viz_path
                )

            print("Synthesis completed (mel spectrogram only)!")

        # Print summary if verbose
        if args.verbose:
            print(f"\n=== Synthesis Summary ===")
            print(f"Input text: {args.text}")
            print(f"Mel spectrogram shape: {mel_output.shape}")
            print(f"Alignments shape: {alignments.shape}")
            if latent_z is not None:
                print(f"Latent representation shape: {latent_z.shape}")
            if speaker_embedding is not None:
                print(f"Speaker embedding shape: {speaker_embedding.shape}")
            print(f"Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()
