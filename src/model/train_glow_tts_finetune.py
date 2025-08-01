#!/usr/bin/env python3
"""
Multi-speaker TTS Fine-tuning Script using VCTK Dataset

This script fine-tunes a pre-trained TTS model on the VCTK dataset for multi-speaker synthesis.
Based on Coqui TTS documentation and examples.
"""

import os
import argparse
from pathlib import Path

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


def setup_paths():
    """Setup project paths"""
    script_path = Path(__file__).parent.absolute()
    project_root = script_path.parent
    output_path = project_root / "outputs"
    dataset_path = project_root / ".." / "dataset" / "VCTK"

    # Create directories if they don't exist
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_path.mkdir(parents=True, exist_ok=True)

    return str(output_path), str(dataset_path)


def download_vctk_if_needed(dataset_path):
    """Download VCTK dataset if not already present"""
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print(f"VCTK dataset not found at {dataset_path}. Downloading...")
        try:
            from TTS.utils.downloaders import download_vctk

            download_vctk(dataset_path)
            print("VCTK dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading VCTK dataset: {e}")
            print(
                "Please download VCTK manually and place it in the data/VCTK directory"
            )
            return False
    else:
        print(f"VCTK dataset found at {dataset_path}")
    return True


def create_config(
    output_path,
    dataset_path,
    restore_path=None,
    learning_rate=0.00001,
):
    """Create training configuration"""

    # Define dataset config
    dataset_config = BaseDatasetConfig(
        formatter="vctk", meta_file_train="", path=dataset_path
    )

    # Define audio config
    # Note: For faster training, resample the dataset externally using TTS/bin/resample.py
    # and set resample=False
    audio_config = BaseAudioConfig(
        sample_rate=22050, resample=True, do_trim_silence=True, trim_db=23.0
    )

    # Define model config for fine-tuning
    config = GlowTTSConfig(
        # Training parameters - reduced for fine-tuning
        batch_size=32,  # Reduced from 32 to 8 for more stable fine-tuning
        eval_batch_size=4,  # Reduced from 16 to 4
        num_loader_workers=4,  # Reduced for stability
        num_eval_loader_workers=4,
        precompute_num_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=100,  # Reduced from 100 to 50 to prevent overfitting
        # Learning rate for fine-tuning (much smaller than from scratch)
        lr=learning_rate,
        # Text processing
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        # Logging
        print_step=10,  # Increased frequency for better monitoring
        print_eval=True,  # Changed to True to monitor evaluation
        mixed_precision=True,
        # Output
        output_path=output_path,
        run_name="vctk_multispeaker_finetune",
        # Dataset
        datasets=[dataset_config],
        # Text and audio length constraints
        min_text_len=0,
        max_text_len=500,
        min_audio_len=0,
        max_audio_len=500000,
        # Audio config
        audio=audio_config,
        lr_scheduler="StepwiseGradualLR",
        lr_scheduler_params={
            "gradual_learning_rates": [
                [0, 1e-3],
                [1000, 1e-4],
                [5000, 5e-5],
                [10000, 1e-5],

            ]
        },
        scheduler_after_epoch=False,
        use_speaker_embedding=True,
    )
    return config


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TTS model on VCTK dataset")
    parser.add_argument(
        "--restore_path",
        type=str,
        required=True,
        help="Path to pre-trained model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate for fine-tuning (default: 0.00001)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default:100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)",
    )
    parser.add_argument(
        "--num_speakers",
        type=int,
        default=2,
        help="Number of speakers to use for training (default: 2)",
    )
    parser.add_argument(
        "--speakers",
        type=str,
        nargs="+",
        help="Specific speaker IDs to use (e.g., --speakers p225 p226)",
    )

    args = parser.parse_args()

    # Setup paths
    output_path, dataset_path = setup_paths()

    print(f"Output path: {output_path}")
    print(f"Dataset path: {dataset_path}")
    print(f"Restore path: {args.restore_path}")

    # Download VCTK dataset if needed
    if not download_vctk_if_needed(dataset_path):
        return

    # Verify restore path exists
    if not os.path.exists(args.restore_path):
        print(f"Error: Restore path {args.restore_path} does not exist!")
        print("Please download a pre-trained model first using:")
        print("tts --model_name tts_models/en/ljspeech/glow-tts --text 'Hello world'")
        return

    # Create configuration
    config = create_config(
        output_path,
        dataset_path,
        args.restore_path,
        args.learning_rate,
    )

    # Override config with command line arguments
    config.epochs = args.epochs
    config.batch_size = args.batch_size

    print("Initializing audio processor...")
    # Initialize the audio processor
    ap = AudioProcessor.init_from_config(config)

    print("Initializing tokenizer...")
    # Initialize the tokenizer
    tokenizer, config = TTSTokenizer.init_from_config(config)

    print("Loading data samples...")
    # Load data samples
    train_samples, eval_samples = load_tts_samples(
        config.datasets[0],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    print(
        f"Loaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples"
    )

    print("Setting up speaker manager...")
    # Initialize speaker manager for multi-speaker training
    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(
        train_samples + eval_samples, parse_key="speaker_name"
    )

    print(
        f"Available speakers in dataset: {speaker_manager.speaker_names[:10]}{'...' if len(speaker_manager.speaker_names) > 10 else ''}"
    )

    # Select speakers for fine-tuning based on arguments
    all_speakers = speaker_manager.speaker_names

    if args.speakers:
        # Use specific speakers provided by user
        selected_speakers = [spk for spk in args.speakers if spk in all_speakers]
        if len(selected_speakers) != len(args.speakers):
            missing = set(args.speakers) - set(selected_speakers)
            print(f"Warning: Speakers not found in dataset: {missing}")
        if not selected_speakers:
            print("Error: None of the specified speakers were found in the dataset!")
            print(f"Available speakers: {all_speakers[:10]}...")
            return
    else:
        # Use first N speakers from dataset
        selected_speakers = all_speakers[: args.num_speakers]

    print(
        f"Found {len(all_speakers)} speakers in dataset, selecting {len(selected_speakers)} for fine-tuning: {selected_speakers}"
    )

    # Filter samples to only include selected speakers
    print("Filtering samples for selected speakers...")
    train_samples_filtered = [
        sample
        for sample in train_samples
        if sample["speaker_name"] in selected_speakers
    ]
        _filtered = [
        sample for sample in eval_samples if sample["speaker_name"] in selected_speakers
    ]

    print(
        f"Filtered training samples: {len(train_samples)} -> {len(train_samples_filtered)}"
    )
    print(
        f"Filtered evaluation samples: {len(eval_samples)} -> {len(eval_samples_filtered)}"
    )

    # Re-initialize speaker manager with filtered data
    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(
        train_samples_filtered + eval_samples_filtered, parse_key="speaker_name"
    )
    config.num_speakers = speaker_manager.num_speakers

    # Update samples to use filtered versions
    train_samples = train_samples_filtered
    eval_samples = eval_samples_filtered

    print(
        f"Using {speaker_manager.num_speakers} speakers for training: {speaker_manager.speaker_names}"
    )

    print("Initializing model...")
    # Initialize model
    model = GlowTTS(config, ap, tokenizer, speaker_manager=speaker_manager)

    print("Initializing trainer...")
    # Initialize the trainer
    trainer = Trainer(
        TrainerArgs(restore_path=args.restore_path),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("Starting fine-tuning...")
    print(f"Training will run for {config.epochs} epochs")
    print(f"Results will be saved to: {output_path}")

    # Start training
    trainer.fit()

    print("Fine-tuning completed!")
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
