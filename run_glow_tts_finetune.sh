#!/bin/bash

# Glow-TTS Fine-tuning Script
# This script runs the Glow-TTS fine-tuning on VCTK dataset

echo "Starting Glow-TTS Fine-tuning..."
echo "=================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if required directories exist
if [ ! -d "wav48_silence_trimmed" ]; then
    echo "Error: wav48_silence_trimmed directory not found!"
    echo "Please ensure the VCTK dataset is properly downloaded and processed."
    exit 1
fi

if [ ! -f "src/speaker_embedding/speaker_emb_lookup.npy" ]; then
    echo "Error: Speaker embeddings not found!"
    echo "Please run the speaker embedding creation script first."
    exit 1
fi

# Set PYTHONPATH to include the current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the training script
echo "Running Glow-TTS fine-tuning..."
python src/model/train_glow_tts_finetune.py

echo "Training completed!" 