#!/usr/bin/env python3
"""
Training script for Capacitron-enhanced Tacotron2 with multi-speaker support.
Based on Coqui TTS library approach, adapted for VCTK dataset with selected speakers.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# Add TTS to path if installed locally
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.speakers import SpeakerManager
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CapacitronVAEConfig

# Local imports
from utils.dataset import VCTKDataset
from utils.tools import pad_1D, pad_2D


class VCTKTTSDataset:
    """
    Adapter class to make VCTKDataset compatible with TTS training framework
    """

    def __init__(
        self,
        root_path,
        selected_speakers_path="src/speaker_embedding/selected_speakers.txt",
        cache_mel=True,
        sample_rate=22050,
    ):
        self.vctk_dataset = VCTKDataset(
            root=root_path,
            selecting_speaker=True,
            cache_mel=cache_mel,
            selected_speakers_path=selected_speakers_path,
        )
        self.sample_rate = sample_rate

    def get_samples(self):
        """Convert VCTKDataset format to TTS expected format"""
        samples = []
        for i in range(len(self.vctk_dataset)):
            wav_path, txt_path, spk_id = self.vctk_dataset.items[i]

            # Read text
            text = Path(txt_path).read_text().strip()

            # Get speaker name from mapping
            speaker_name = self.vctk_dataset.id2spk[spk_id]

            # Create unique name for audio file (used for phoneme caching)
            audio_unique_name = os.path.splitext(os.path.basename(wav_path))[0]

            # TTS expected format: dictionary with required keys
            sample = {
                "text": text,
                "audio_file": str(wav_path),
                "audio_unique_name": audio_unique_name,
                "speaker_name": speaker_name,
                "root_path": self.vctk_dataset.root,
                "language": "en",  # Add language field as well
            }
            samples.append(sample)

        return samples

    def get_speaker_manager_data(self):
        """Get speaker information for SpeakerManager"""
        speakers = list(self.vctk_dataset.spk2id.keys())
        speaker_ids = {spk: idx for idx, spk in enumerate(speakers)}
        return speakers, speaker_ids


def setup_model_config(output_path, dataset_path, selected_speakers_path):
    """
    Set up Tacotron2 configuration with Capacitron for multi-speaker training
    """

    # Audio configuration
    audio_config = BaseAudioConfig(
        sample_rate=22050,
        do_trim_silence=True,
        trim_db=60.0,
        signal_norm=False,
        mel_fmin=0.0,
        mel_fmax=11025,
        spec_gain=1.0,
        log_func="np.log",
        ref_level_db=20,
        preemphasis=0.0,
    )

    # Capacitron VAE configuration for speaker conditioning
    capacitron_config = CapacitronVAEConfig(
        capacitron_VAE_loss_alpha=1.0, capacitron_capacity=50
    )

    # Dataset configuration
    dataset_config = BaseDatasetConfig(
        formatter="vctk",
        meta_file_train="",  # We'll handle this differently
        meta_file_val="",
        path=dataset_path,
        language="en",
    )

    # Main model configuration
    config = Tacotron2Config(
        # Basic settings
        model="tacotron2",
        run_name="capacitron_vctk_multispeaker",
        epochs=1000,
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=2,
        # Learning rate and optimization
        lr=1e-3,
        optimizer="CapacitronOptimizer",
        optimizer_params={
            "RAdam": {"betas": [0.9, 0.998], "weight_decay": 1e-6},
            "SGD": {"lr": 1e-5, "momentum": 0.9},
        },
        lr_scheduler="ExponentialLR",
        lr_scheduler_params={"gamma": 0.999875},
        # Audio and text processing
        audio=audio_config,
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        text_cleaner="phoneme_cleaners",
        # Model architecture
        encoder_in_features=512,
        decoder_in_features=512,
        decoder_output_dim=80,
        r=1,  # reduction factor
        # Attention settings
        attention_type="original",
        location_attn=True,
        attention_norm="sigmoid",
        # Multi-speaker settings
        use_speaker_embedding=True,
        speaker_embedding_dim=128,
        num_speakers=10,  # Will be updated based on selected speakers
        # Capacitron VAE settings
        use_capacitron_vae=True,
        capacitron_vae=capacitron_config,
        # Training settings
        mixed_precision=False,  # Disabled to avoid BFloat16 logging issues
        grad_clip=5.0,
        loss_masking=True,
        # Logging and checkpointing
        output_path=output_path,
        print_step=25,
        plot_step=100,
        save_step=1000,
        save_n_checkpoints=5,
        # Dataset
        datasets=[dataset_config],
        # Evaluation
        run_eval=True,
        eval_split_size=0.1,
        test_delay_epochs=10,
    )

    return config


def collate_fn(batch):
    """
    Custom collate function for multi-speaker training with Capacitron
    """
    texts, mels, speaker_ids = zip(*batch)

    # Convert to tensors
    text_lengths = torch.LongTensor([t.size(0) for t in texts])
    mel_lengths = torch.LongTensor([m.shape[1] for m in mels])

    # Pad sequences
    text_padded = pad_1D(texts, pad_value=0)
    mel_padded = pad_2D(mels, pad_value=0.0)

    # Create gate (stop token) targets
    max_mel_len = mel_padded.shape[2]
    gate_padded = torch.zeros(len(batch), max_mel_len)
    for i, length in enumerate(mel_lengths):
        gate_padded[i, length - 1 :] = 1.0

    # Speaker IDs
    speaker_ids = torch.cat(speaker_ids, dim=0)

    return {
        "text": text_padded,
        "text_lengths": text_lengths,
        "mel": mel_padded,
        "mel_lengths": mel_lengths,
        "gate": gate_padded,
        "speaker_ids": speaker_ids,
    }


def setup_trainer(config, output_path, model, train_samples, eval_samples):
    """
    Set up the trainer with proper configuration
    """
    trainer_args = TrainerArgs(
        restore_path=None,
        skip_train_epoch=False,
        start_with_eval=False,
        grad_accum_steps=1,
    )

    trainer = Trainer(
        trainer_args,
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train Capacitron-enhanced Tacotron2")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to VCTK dataset root directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output",
        help="Output directory for models and logs",
    )
    parser.add_argument(
        "--selected_speakers_path",
        type=str,
        default="src/speaker_embedding/selected_speakers.txt",
        help="Path to selected speakers file",
    )
    parser.add_argument(
        "--restore_path",
        type=str,
        default=None,
        help="Path to checkpoint to restore from",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device to use")

    args = parser.parse_args()

    # Set up device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Set up dataset
    print("Loading VCTK dataset...")
    vctk_adapter = VCTKTTSDataset(
        root_path=args.dataset_path,
        selected_speakers_path=args.selected_speakers_path,
        cache_mel=True,
    )

    # Get samples in TTS format
    all_samples = vctk_adapter.get_samples()
    speakers, speaker_ids = vctk_adapter.get_speaker_manager_data()

    print(f"Loaded {len(all_samples)} samples from {len(speakers)} speakers")
    print(f"Selected speakers: {speakers}")

    # Split into train/eval
    num_eval = int(len(all_samples) * 0.1)
    eval_samples = all_samples[:num_eval]
    train_samples = all_samples[num_eval:]

    print(f"Training samples: {len(train_samples)}")
    print(f"Evaluation samples: {len(eval_samples)}")

    # Set up configuration
    config = setup_model_config(
        args.output_path, args.dataset_path, args.selected_speakers_path
    )
    # num_speakers will be set properly after speaker_manager initialization

    # Initialize audio processor
    ap = AudioProcessor.init_from_config(config)

    # Initialize tokenizer
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # Initialize speaker manager
    print("Setting up speaker manager...")
    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(
        train_samples + eval_samples, parse_key="speaker_name"
    )
    config.num_speakers = speaker_manager.num_speakers

    # Initialize model
    print("Initializing Capacitron-Tacotron2 model...")
    model = Tacotron2(config, ap, tokenizer, speaker_manager=speaker_manager)

    if args.restore_path:
        print(f"Restoring from checkpoint: {args.restore_path}")
        model.load_checkpoint(config, args.restore_path, eval=False)

    # Move model to device
    model = model.to(device)

    # Set up trainer
    trainer = setup_trainer(
        config, args.output_path, model, train_samples, eval_samples
    )

    # Start training
    print("Starting training...")
    print("=" * 50)
    print("Training Configuration:")
    print(f"  Model: Capacitron-enhanced Tacotron2")
    print(f"  Speakers: {len(speakers)}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Output path: {config.output_path}")
    print("=" * 50)

    trainer.fit()


if __name__ == "__main__":
    main()
