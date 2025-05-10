#!/bin/bash
# Enhanced Script to run Bayesian Optimization for LLaMA 2 inference parameters

# Enable error reporting and command echo
set -e
set -o pipefail

# Print banner
echo "===================================="
echo "LLaMA 2 Bayesian Optimization Runner"
echo "===================================="

# Create results directory
mkdir -p optimization_results

# Verify GPU is accessible
echo "Checking GPU availability:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    if [ $? -ne 0 ]; then
        echo "WARNING: nvidia-smi command failed. Are NVIDIA drivers installed properly?"
        echo "Continuing anyway, but optimization may fail if GPU is required."
    fi
else
    echo "WARNING: nvidia-smi not found. No NVIDIA GPU detected or drivers not installed."
    echo "Continuing anyway, but optimization may fail if GPU is required."
fi

# Define paths with variables for better flexibility
PROJ_DIR=$(pwd)
MODEL_DIR="${PROJ_DIR}/llama/llama-2-7b"
TOKENIZER_PATH="${PROJ_DIR}/llama/tokenizer.model"

# Process command line arguments with defaults
TARGET="${1:-throughput}"  # Default to optimizing for throughput if not specified
N_ITER="${2:-20}"          # Default to 20 iterations if not specified
RUN_GRID="${3:-false}"     # Default to not running grid search

# Parse target option
case "$TARGET" in
    throughput|latency|balanced|efficiency)
        echo "Optimization target: $TARGET"
        ;;
    *)
        echo "Invalid optimization target: $TARGET"
        echo "Valid options are: throughput, latency, balanced, efficiency"
        echo "Defaulting to throughput"
        TARGET="throughput"
        ;;
esac

# Print optimization configuration
echo "Optimization Configuration:"
echo "  - Target: $TARGET"
echo "  - Iterations: $N_ITER"
echo "  - Model: LLaMA 2 7B"
echo "  - Model path: $MODEL_DIR"
echo "  - Grid search: $RUN_GRID"

# Check if Weights & Biases is installed, install if needed
if ! pip list | grep -q "wandb"; then
    echo "Installing Weights & Biases for experiment tracking..."
    pip install wandb
    
    # Ask user to login to W&B
    echo "Please login to Weights & Biases:"
    wandb login
fi

# Install other required dependencies
echo "Installing required dependencies..."
pip install numpy==1.23.5 scikit-learn==1.2.2 bayesian-optimization==1.4.3 matplotlib tqdm

# Set environment variables for torch distributed
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# Set environment variables for memory efficiency
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Add current directory to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create a timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="optimization_${TARGET}_${TIMESTAMP}.log"

echo "Starting optimization (logging to $LOG_FILE)..."

# Construct the run command
RUN_CMD="python llama_bayesian_opt.py \
  --model_path ${MODEL_DIR} \
  --tokenizer_path ${TOKENIZER_PATH} \
  --model_size 7B \
  --max_gen_len 30 \
  --n_iter ${N_ITER} \
  --initial_points 5 \
  --optimization_target ${TARGET} \
  --output_dir ./optimization_results \
  --max_batch_size 1 \
  --max_seq_len 512"

# Add grid search if enabled
if [ "$RUN_GRID" = "true" ]; then
    RUN_CMD="${RUN_CMD} --run_grid_search"
fi

# Add quantization option (disabled by default)
if [ "${4:-false}" = "true" ]; then
    RUN_CMD="${RUN_CMD} --use_quantization"
    echo "Enabling quantization"
fi

# Run the optimization script
echo "Running command: $RUN_CMD"
$RUN_CMD | tee $LOG_FILE

# Check if optimization completed successfully
if [ $? -eq 0 ]; then
    echo "Bayesian optimization completed successfully!"
    
    # Display the best parameters
    echo "Best parameters:"
    grep -A5 "Best parameters" $LOG_FILE | tail -n 5
    
    # Create a summary file with timestamp
    SUMMARY_FILE="optimization_results/summary_${TARGET}_${TIMESTAMP}.txt"
    echo "Creating summary in $SUMMARY_FILE"
    
    echo "LLaMA 2 Optimization Summary" > $SUMMARY_FILE
    echo "=========================" >> $SUMMARY_FILE
    echo "Date: $(date)" >> $SUMMARY_FILE
    echo "Target: $TARGET" >> $SUMMARY_FILE
    echo "Iterations: $N_ITER" >> $SUMMARY_FILE
    echo "Model: LLaMA 2 7B" >> $SUMMARY_FILE
    echo "------------------------" >> $SUMMARY_FILE
    grep -A5 "Best parameters" $LOG_FILE | tail -n 5 >> $SUMMARY_FILE
    echo "------------------------" >> $SUMMARY_FILE
    echo "Results saved in optimization_results/" >> $SUMMARY_FILE
    
    # Print completion message
    echo "Optimization complete! Results saved in optimization_results/"
    echo "Summary saved to $SUMMARY_FILE"
else
    echo "Bayesian optimization failed with error code $?"
fi