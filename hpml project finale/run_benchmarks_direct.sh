#!/bin/bash

# Print header
echo "=============================================="
echo "LLaMA 2 Benchmark Suite - DIRECT EXECUTION"
echo "=============================================="

# Create a directory for all results with timestamp
MAIN_DIR="llama2_results_direct_$(date +"%Y%m%d_%H%M%S")"
mkdir -p $MAIN_DIR
cd $MAIN_DIR

echo "Results will be saved in: $PWD"

# Create a log file
LOG_FILE="benchmark_log.txt"
echo "Starting benchmark suite at $(date)" > $LOG_FILE

# Function to run a benchmark and log its output
run_benchmark() {
    local name="$1"
    local cmd="$2"
    local output_dir="$3"
    
    echo "=============================================="
    echo "Running: $name"
    echo "Output directory: $output_dir"
    echo "Started at: $(date)"
    echo "=============================================="
    
    # Create the output directory
    mkdir -p "$output_dir"
    
    # Log the command
    echo "=============================================="
    echo "Running $name at $(date)" >> $LOG_FILE
    echo "Command: $cmd" >> $LOG_FILE
    
    # Run the command and capture output
    start_time=$(date +%s)
    
    # Execute the command
    eval "$cmd" | tee -a "${output_dir}/output.log"
    result=$?
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    # Log the result
    echo "Finished $name at $(date)" >> $LOG_FILE
    echo "Duration: $((duration / 60)) minutes $((duration % 60)) seconds" >> $LOG_FILE
    echo "Exit code: $result" >> $LOG_FILE
    echo "=============================================" >> $LOG_FILE
    
    echo "=============================================="
    echo "Finished: $name"
    echo "Duration: $((duration / 60)) minutes $((duration % 60)) seconds"
    echo "Exit code: $result"
    echo "=============================================="
    
    return $result
}

# Configure paths
MODEL_DIR="/scratch/am14661/FinalProject/llama2_benchmark/llama/llama-2-7b"
TOKENIZER_PATH="/scratch/am14661/FinalProject/llama2_benchmark/llama"

# 1. Run standard benchmark
STANDARD_CMD="python /scratch/am14661/FinalProject/llama2_benchmark/benchmark_llama.py \
    --model_path $MODEL_DIR \
    --tokenizer_path $TOKENIZER_PATH \
    --model_size 7B \
    --temperature 0.8 \
    --top_p 0.95 \
    --max_gen_len 64 \
    --num_runs 3 \
    --output_dir $PWD/standard_benchmark"

run_benchmark "Standard Benchmark" "$STANDARD_CMD" "$PWD/standard_benchmark"
standard_result=$?

# Take a break to allow GPU memory to clear
echo "Waiting 30 seconds to clear GPU memory..."
sleep 30
nvidia-smi

# 2. Run Bayesian optimization (with fewer iterations for quicker results)
BAYES_CMD="python /scratch/am14661/FinalProject/llama2_benchmark/llama_bayesian_opt.py \
    --model_path $MODEL_DIR \
    --tokenizer_path $TOKENIZER_PATH \
    --model_size 7B \
    --max_gen_len 20 \
    --n_iter 5 \
    --initial_points 2 \
    --optimization_target throughput \
    --output_dir $PWD/bayesian_optimization"

run_benchmark "Bayesian Optimization" "$BAYES_CMD" "$PWD/bayesian_optimization"
bayes_result=$?

# Take a break to allow GPU memory to clear
echo "Waiting 30 seconds to clear GPU memory..."
sleep 30
nvidia-smi

# 3. Run dataset benchmarks (one at a time)
# Helpfulness dataset
HELP_CMD="python /scratch/am14661/FinalProject/llama2_benchmark/benchmark_dataset.py \
    --model_path $MODEL_DIR \
    --tokenizer_path $TOKENIZER_PATH \
    --model_size 7B \
    --temperature 0.8 \
    --top_p 0.95 \
    --max_gen_len 20 \
    --dataset helpfulness \
    --max_samples 3 \
    --output_dir $PWD/dataset_helpfulness"

run_benchmark "Helpfulness Dataset Benchmark" "$HELP_CMD" "$PWD/dataset_helpfulness"
help_result=$?

# Take a break to allow GPU memory to clear
echo "Waiting 30 seconds to clear GPU memory..."
sleep 30
nvidia-smi

# Technical dataset
TECH_CMD="python /scratch/am14661/FinalProject/llama2_benchmark/benchmark_dataset.py \
    --model_path $MODEL_DIR \
    --tokenizer_path $TOKENIZER_PATH \
    --model_size 7B \
    --temperature 0.8 \
    --top_p 0.95 \
    --max_gen_len 20 \
    --dataset technical \
    --max_samples 3 \
    --output_dir $PWD/dataset_technical"

run_benchmark "Technical Dataset Benchmark" "$TECH_CMD" "$PWD/dataset_technical"
tech_result=$?

# Take a break to allow GPU memory to clear
echo "Waiting 30 seconds to clear GPU memory..."
sleep 30
nvidia-smi

# Creative dataset
CREATIVE_CMD="python /scratch/am14661/FinalProject/llama2_benchmark/benchmark_dataset.py \
    --model_path $MODEL_DIR \
    --tokenizer_path $TOKENIZER_PATH \
    --model_size 7B \
    --temperature 0.8 \
    --top_p 0.95 \
    --max_gen_len 20 \
    --dataset creative \
    --max_samples 3 \
    --output_dir $PWD/dataset_creative"

run_benchmark "Creative Dataset Benchmark" "$CREATIVE_CMD" "$PWD/dataset_creative"
creative_result=$?

# Create a summary file
echo "Creating benchmark summary..."
{
    echo "LLaMA 2 Benchmark Suite - Summary"
    echo "================================="
    echo "Date: $(date)"
    echo "Results directory: $PWD"
    echo ""
    echo "Benchmark Results:"
    echo "  Standard benchmark: $([ $standard_result -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
    echo "  Bayesian optimization: $([ $bayes_result -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
    echo "  Helpfulness dataset: $([ $help_result -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
    echo "  Technical dataset: $([ $tech_result -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
    echo "  Creative dataset: $([ $creative_result -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
    echo ""
    echo "See individual output directories for detailed results."
} > summary.txt

echo "=============================================="
echo "All benchmarks completed!"
echo "Results saved in: $PWD"
echo "Summary available in: $PWD/summary.txt"
echo "=============================================="