#!/bin/bash

# Fine-tune Capacitron T2 model on VCTK dataset
# Make sure you're in the virtual environment

echo "Starting Capacitron T2 fine-tuning on VCTK dataset..."

# Activate virtual environment if needed
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, using GPU for training"
    export CUDA_VISIBLE_DEVICES=0
else
    echo "No CUDA detected, using CPU (training will be slow)"
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create output directory
mkdir -p src/model/runs

# Run the fine-tuning
echo "Running fine-tuning script..."
cd src/model
python train_capacitron_t2_finetune.py

echo "Fine-tuning completed!" 