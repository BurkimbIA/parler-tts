#!/bin/bash

# xTTS Training Automation Script
# This script automates the complete xTTS training pipeline

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Configuration variables
REPO_URL="https://github.com/BurkimbIA/toolskit-tts.git"
PROJECT_DIR="toolskit-tts"
XTTS_DIR="$PROJECT_DIR/xtts"
CHECKPOINTS_DIR="checkpoints"
DATASET_DIR="dataset"
LANGUAGE="mos"
EXTENDED_VOCAB_SIZE=2000
NUM_EPOCHS=1500
BATCH_SIZE=2
GRAD_ACCUM=1
MAX_TEXT_LENGTH=400
MAX_AUDIO_LENGTH=330750
WEIGHT_DECAY=1e-2
LEARNING_RATE=5e-6
CUDA_DEVICE=0

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command_exists git; then
        error "Git is not installed. Please install git first."
    fi
    
    if ! command_exists python; then
        error "Python is not installed. Please install Python first."
    fi
    
    if ! command_exists pip; then
        error "Pip is not installed. Please install pip first."
    fi
    
    # Check if CUDA is available
    if ! command_exists nvidia-smi; then
        warning "NVIDIA GPU not detected. Training will be very slow on CPU."
    else
        log "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name --format=csv,noheader
    fi
}

# Step 1: Clone repository and install dependencies
step1_setup() {
    log "=== STEP 1: Setting up environment ==="
    
    # Remove existing directory if it exists
    if [ -d "$PROJECT_DIR" ]; then
        warning "Directory $PROJECT_DIR already exists. Removing it..."
        rm -rf "$PROJECT_DIR"
    fi
    
    # Clone repository
    log "Cloning repository..."
    git clone "$REPO_URL" || error "Failed to clone repository"
    
    # Navigate to xtts directory
    cd "$XTTS_DIR" || error "Failed to navigate to $XTTS_DIR"
    
    # Install requirements
    log "Installing requirements..."
    pip install -r requirements.txt || error "Failed to install requirements"
    
    # Upgrade TensorFlow and related packages
    log "Upgrading TensorFlow and related packages..."
    pip install --upgrade tensorflow tensorboard numpy || error "Failed to upgrade packages"
    
    # Run launch script if it exists
    if [ -f "launch_python_310.sh" ]; then
        log "Running launch script..."
        chmod +x launch_python_310.sh
        ./launch_python_310.sh || warning "Launch script failed, continuing anyway..."
    else
        warning "launch_python_310.sh not found, skipping..."
    fi
    
    log "Step 1 completed successfully!"
}

# Step 2: Generate data
step2_generate_data() {
    log "=== STEP 2: Generating data ==="
    
    if [ ! -f "generate_data.py" ]; then
        error "generate_data.py not found in current directory"
    fi
    
    log "Running data generation..."
    python generate_data.py || error "Data generation failed"
    
    log "Step 2 completed successfully!"
}

# Step 3: Download checkpoint
step3_download_checkpoint() {
    log "=== STEP 3: Downloading checkpoint ==="
    
    # Create checkpoints directory
    mkdir -p "$CHECKPOINTS_DIR"
    
    if [ ! -f "download_checkpoint.py" ]; then
        error "download_checkpoint.py not found in current directory"
    fi
    
    log "Downloading checkpoint..."
    python download_checkpoint.py --output_path "$CHECKPOINTS_DIR/" || error "Checkpoint download failed"
    
    log "Step 3 completed successfully!"
}

# Step 4: Extend vocabulary configuration
step4_extend_vocab() {
    log "=== STEP 4: Extending vocabulary configuration ==="
    
    # Check if required files exist
    if [ ! -f "xTTS/extend_vocab_config.py" ]; then
        error "xTTS/extend_vocab_config.py not found"
    fi
    
    if [ ! -f "$DATASET_DIR/metadata.csv" ]; then
        error "$DATASET_DIR/metadata.csv not found"
    fi
    
    # Create xTTS checkpoints directory
    mkdir -p "xTTS/checkpoints"
    
    log "Extending vocabulary configuration..."
    python xTTS/extend_vocab_config.py \
        --output_path="xTTS/checkpoints/" \
        --metadata_path "$DATASET_DIR/metadata.csv" \
        --language "$LANGUAGE" \
        --extended_vocab_size "$EXTENDED_VOCAB_SIZE" || error "Vocabulary extension failed"
    
    log "Step 4 completed successfully!"
}

# Step 5: Train the model
step5_train() {
    log "=== STEP 5: Training xTTS model ==="
    
    # Check if training script exists
    if [ ! -f "xTTS/train_gpt_xtts.py" ]; then
        error "xTTS/train_gpt_xtts.py not found"
    fi
    
    # Check if metadata files exist
    if [ ! -f "$DATASET_DIR/metadata_train.csv" ] || [ ! -f "$DATASET_DIR/metadata_eval.csv" ]; then
        error "Training metadata files not found in $DATASET_DIR/"
    fi
    
    log "Starting training with the following parameters:"
    log "  - Language: $LANGUAGE"
    log "  - Epochs: $NUM_EPOCHS"
    log "  - Batch size: $BATCH_SIZE"
    log "  - Learning rate: $LEARNING_RATE"
    log "  - CUDA device: $CUDA_DEVICE"
    
    # Set CUDA device and start training
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
    
    python xTTS/train_gpt_xtts.py \
        --output_path "xTTS/checkpoints/" \
        --metadatas "./$DATASET_DIR/metadata_train.csv,./$DATASET_DIR/metadata_eval.csv,$LANGUAGE" \
        --num_epochs "$NUM_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --grad_acumm "$GRAD_ACCUM" \
        --max_text_length "$MAX_TEXT_LENGTH" \
        --max_audio_length "$MAX_AUDIO_LENGTH" \
        --weight_decay "$WEIGHT_DECAY" \
        --lr "$LEARNING_RATE" || error "Training failed"
    
    log "Step 5 completed successfully!"
}

# Main execution function
main() {
    log "Starting xTTS training automation script..."
    
    # Store original directory
    ORIGINAL_DIR=$(pwd)
    
    # Create a trap to return to original directory on exit
    trap 'cd "$ORIGINAL_DIR"' EXIT
    
    # Check prerequisites
    check_prerequisites
    
    # Execute all steps
    step1_setup
    step2_generate_data
    step3_download_checkpoint
    step4_extend_vocab
    step5_train
    
    log "=== ALL STEPS COMPLETED SUCCESSFULLY! ==="
    log "Training has finished. Check the xTTS/checkpoints/ directory for your trained model."
}

# Script usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  --epochs N              Number of training epochs (default: $NUM_EPOCHS)"
    echo "  --batch-size N          Batch size (default: $BATCH_SIZE)"
    echo "  --lr RATE               Learning rate (default: $LEARNING_RATE)"
    echo "  --language LANG         Language code (default: $LANGUAGE)"
    echo "  --cuda-device N         CUDA device number (default: $CUDA_DEVICE)"
    echo "  --vocab-size N          Extended vocabulary size (default: $EXTENDED_VOCAB_SIZE)"
    echo ""
    echo "Example:"
    echo "  $0 --epochs 1000 --batch-size 4 --lr 1e-5"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        --cuda-device)
            CUDA_DEVICE="$2"
            shift 2
            ;;
        --vocab-size)
            EXTENDED_VOCAB_SIZE="$2"
            shift 2
            ;;
        *)
            error "Unknown option: $1. Use --help for usage information."
            ;;
    esac
done

# Run main function
main "$@"
