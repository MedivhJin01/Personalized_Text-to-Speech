#!/usr/bin/env python3
"""
Inference script for Capacitron-enhanced Tacotron2 with speaker conditioning.
Generate speech from text using trained multi-speaker model.
"""

import os
import argparse
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

try:
    from TTS.tts.configs.tacotron2_config import Tacotron2Config
    from TTS.tts.models.tacotron2 import Tacotron2
    from TTS.utils.audio import AudioProcessor
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.config import load_config

    print("Using installed TTS library")
except ImportError:
    print("TTS library not found. Please install with: pip install TTS")
    sys.exit(1)

# Local imports
from utils.dataset import VCTKDataset


class CapacitronInference:
    """
    Inference engine for Capacitron-enhanced Tacotron2
    """

    def __init__(
        self, model_path, config_path, vocoder_path=None, vocoder_config_path=None
    ):
        """
        Initialize the inference engine

        Args:
            model_path: Path to trained Tacotron2 model checkpoint
            config_path: Path to model configuration file
            vocoder_path: Path to vocoder model (optional, uses Griffin-Lim if not provided)
            vocoder_config_path: Path to vocoder configuration
        """

        # Load configuration
        self.config = load_config(config_path)

        # Initialize audio processor
        self.ap = AudioProcessor.init_from_config(self.config)

        # Initialize tokenizer
        self.tokenizer, self.config = TTSTokenizer.init_from_config(self.config)

        # Load model
        self.model = Tacotron2(self.config, self.ap, self.tokenizer)
        self.model.load_checkpoint(self.config, model_path, eval=True)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load vocoder if provided
        self.vocoder = None
        if vocoder_path:
            # TODO: Add vocoder loading (HiFiGAN, etc.)
            print("External vocoder not implemented yet, using Griffin-Lim")

        print(f"Model loaded on {self.device}")
        print(f"Number of speakers: {self.config.num_speakers}")

    def load_speaker_info(self, selected_speakers_path):
        """
        Load speaker information for inference
        """
        with open(selected_speakers_path, "r") as f:
            speakers = [line.strip() for line in f if line.strip()]

        self.speakers = speakers
        self.speaker_to_id = {speaker: idx for idx, speaker in enumerate(speakers)}

        print(f"Available speakers: {speakers}")
        return speakers

    def text_to_mel(self, text, speaker_id=0, reference_mel=None):
        """
        Convert text to mel-spectrogram using the trained model

        Args:
            text: Input text string
            speaker_id: ID of target speaker (0-based index)
            reference_mel: Reference mel-spectrogram for Capacitron conditioning (optional)

        Returns:
            mel_outputs: Generated mel-spectrogram
            alignments: Attention alignments
        """

        # Tokenize text
        text_inputs = self.tokenizer.text_to_ids(text)
        text_inputs = torch.LongTensor(text_inputs).unsqueeze(0).to(self.device)
        text_lengths = torch.LongTensor([len(text_inputs[0])]).to(self.device)

        # Speaker embedding
        speaker_ids = torch.LongTensor([speaker_id]).to(self.device)

        # Prepare auxiliary inputs
        aux_input = {"speaker_ids": speaker_ids}

        # Add reference mel for Capacitron if provided
        if reference_mel is not None:
            aux_input["reference_mel"] = reference_mel

        with torch.no_grad():
            # Generate mel-spectrogram
            outputs = self.model.inference(text_inputs, aux_input)

            mel_outputs = outputs["model_outputs"]
            mel_outputs_postnet = (
                outputs["model_outputs_postnet"]
                if "model_outputs_postnet" in outputs
                else mel_outputs
            )
            alignments = outputs["alignments"] if "alignments" in outputs else None

        return mel_outputs_postnet, alignments

    def mel_to_wav(self, mel):
        """
        Convert mel-spectrogram to waveform

        Args:
            mel: Mel-spectrogram tensor

        Returns:
            wav: Generated waveform
        """

        if self.vocoder:
            # Use neural vocoder
            with torch.no_grad():
                wav = self.vocoder(mel)
        else:
            # Use Griffin-Lim
            mel_np = mel.squeeze().cpu().numpy()
            wav = self.ap.griffin_lim(mel_np)

        return wav

    def tts(self, text, speaker_name=None, speaker_id=None, reference_audio=None):
        """
        Text-to-speech synthesis

        Args:
            text: Input text
            speaker_name: Name of target speaker
            speaker_id: ID of target speaker (alternative to speaker_name)
            reference_audio: Path to reference audio for Capacitron conditioning

        Returns:
            wav: Generated waveform
            sample_rate: Audio sample rate
        """

        # Determine speaker ID
        if speaker_name:
            if speaker_name not in self.speaker_to_id:
                raise ValueError(
                    f"Speaker '{speaker_name}' not found. Available speakers: {list(self.speaker_to_id.keys())}"
                )
            spk_id = self.speaker_to_id[speaker_name]
        elif speaker_id is not None:
            spk_id = speaker_id
        else:
            spk_id = 0  # Default to first speaker

        print(
            f"Generating speech for speaker: {self.speakers[spk_id] if hasattr(self, 'speakers') else spk_id}"
        )

        # Load reference mel if provided
        reference_mel = None
        if reference_audio:
            ref_wav, _ = sf.read(reference_audio)
            reference_mel = self.ap.melspectrogram(ref_wav)
            reference_mel = (
                torch.FloatTensor(reference_mel).unsqueeze(0).to(self.device)
            )

        # Generate mel-spectrogram
        mel, alignments = self.text_to_mel(text, spk_id, reference_mel)

        # Convert to waveform
        wav = self.mel_to_wav(mel)

        return wav, self.config.audio.sample_rate

    def save_attention_plot(self, alignments, output_path):
        """
        Save attention alignment plot
        """
        import matplotlib.pyplot as plt

        if alignments is None:
            print("No alignments to plot")
            return

        plt.figure(figsize=(12, 6))
        plt.imshow(
            alignments[0].cpu().numpy().T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )
        plt.xlabel("Decoder timestep")
        plt.ylabel("Encoder timestep")
        plt.title("Attention Alignment")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Attention plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Capacitron-Tacotron2 Inference")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to model configuration"
    )
    parser.add_argument(
        "--selected_speakers_path",
        type=str,
        default="src/speaker_embedding/selected_speakers.txt",
        help="Path to selected speakers file",
    )
    parser.add_argument(
        "--speaker_name", type=str, default=None, help="Name of target speaker"
    )
    parser.add_argument(
        "--speaker_id", type=int, default=None, help="ID of target speaker"
    )
    parser.add_argument(
        "--reference_audio",
        type=str,
        default=None,
        help="Reference audio for Capacitron conditioning",
    )
    parser.add_argument(
        "--output_path", type=str, default="output.wav", help="Output audio file path"
    )
    parser.add_argument(
        "--save_alignment", action="store_true", help="Save attention alignment plot"
    )
    parser.add_argument(
        "--vocoder_path", type=str, default=None, help="Path to vocoder model"
    )
    parser.add_argument(
        "--vocoder_config_path",
        type=str,
        default=None,
        help="Path to vocoder configuration",
    )

    args = parser.parse_args()

    # Initialize inference engine
    print("Loading Capacitron-Tacotron2 model...")
    tts_engine = CapacitronInference(
        model_path=args.model_path,
        config_path=args.config_path,
        vocoder_path=args.vocoder_path,
        vocoder_config_path=args.vocoder_config_path,
    )

    # Load speaker information
    speakers = tts_engine.load_speaker_info(args.selected_speakers_path)

    # Generate speech
    print(f"Synthesizing: '{args.text}'")
    wav, sample_rate = tts_engine.tts(
        text=args.text,
        speaker_name=args.speaker_name,
        speaker_id=args.speaker_id,
        reference_audio=args.reference_audio,
    )

    # Save audio
    sf.write(args.output_path, wav, sample_rate)
    print(f"Audio saved to: {args.output_path}")

    # Save attention plot if requested
    if args.save_alignment:
        _, alignments = tts_engine.text_to_mel(
            args.text,
            args.speaker_id or tts_engine.speaker_to_id.get(args.speaker_name, 0),
        )
        alignment_path = args.output_path.replace(".wav", "_alignment.png")
        tts_engine.save_attention_plot(alignments, alignment_path)


if __name__ == "__main__":
    main()
