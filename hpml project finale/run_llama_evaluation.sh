#!/bin/bash
# Script to run LLaMA 2 evaluation with optimized parameters on standard benchmarks

# Enable error reporting and command echo
set -e
set -o pipefail

# Print banner
echo "===================================="
echo "LLaMA 2 Optimized Parameters Evaluation"
echo "===================================="

# Create results directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="./evaluation_results_${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}

# Define paths
MODEL_PATH="./llama/llama-2-7b"
TOKENIZER_PATH="./llama"
MODEL_SIZE="7B"

# Install required packages
pip install datasets tqdm matplotlib pandas

# Ensure the evaluation script is available
if [ ! -f "run_llama_evaluation.py" ]; then
    echo "Error: run_llama_evaluation.py not found!"
    echo "Please make sure the evaluation script is in the current directory."
    exit 1
fi

# Run evaluation with optimized parameters
echo "Running evaluation with optimized parameters (temp=0.05, top_p=1.0)..."
python run_llama_evaluation.py \
    --model_path ${MODEL_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --model_size ${MODEL_SIZE} \
    --max_gen_len 128 \
    --num_runs 2 \
    --temperature 0.05 \
    --top_p 1.0 \
    --max_batch_size 1 \
    --max_seq_len 512 \
    --output_dir ${RESULTS_DIR}/optimized_params

# Run comparison with baseline parameters
echo "Running comparison between optimized and baseline parameters..."
python run_llama_evaluation.py \
    --model_path ${MODEL_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --model_size ${MODEL_SIZE} \
    --max_gen_len 128 \
    --num_runs 2 \
    --max_batch_size 1 \
    --max_seq_len 512 \
    --run_comparison \
    --output_dir ${RESULTS_DIR}/comparison
# Create a summary of results
echo "Creating summary report..."
echo "LLaMA 2 Optimized Parameters Evaluation Summary" > ${RESULTS_DIR}/summary.txt
echo "Date: $(date)" >> ${RESULTS_DIR}/summary.txt
echo "Model: LLaMA 2 ${MODEL_SIZE}" >> ${RESULTS_DIR}/summary.txt
echo "" >> ${RESULTS_DIR}/summary.txt
echo "Optimized Parameters:" >> ${RESULTS_DIR}/summary.txt
echo "  - Temperature: 0.05" >> ${RESULTS_DIR}/summary.txt
echo "  - Top-p: 1.0" >> ${RESULTS_DIR}/summary.txt
echo "" >> ${RESULTS_DIR}/summary.txt
echo "Baseline Parameters:" >> ${RESULTS_DIR}/summary.txt
echo "  - Temperature: 0.8" >> ${RESULTS_DIR}/summary.txt
echo "  - Top-p: 0.95" >> ${RESULTS_DIR}/summary.txt
echo "" >> ${RESULTS_DIR}/summary.txt
echo "Results directory: ${RESULTS_DIR}" >> ${RESULTS_DIR}/summary.txt

# Copy this script to the results directory for reproducibility
cp $0 ${RESULTS_DIR}/

# Print completion info
echo ""
echo "===================================="
echo "Evaluation Complete!"
echo "Results saved to: ${RESULTS_DIR}"
echo "See summary at: ${RESULTS_DIR}/summary.txt"
echo "===================================="