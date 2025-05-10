#!/bin/bash
#SBATCH --job-name=llama2_benchmark
#SBATCH --output=llama2_benchmark_%j.out
#SBATCH --error=llama2_benchmark_%j.err
#SBATCH --time=04:00:00  # Increased time for more comprehensive benchmarks
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --account=hpc

# Print job info
echo "=============================================="
echo "LLaMA 2 Performance Benchmarking Suite"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Starting at: $(date)"
echo "=============================================="

# Load necessary modules (adjust according to your HPC environment)
module purge
module load cuda/11.8.0
module load anaconda3/2023.3

# Activate virtual environment
source /scratch/am14661/FinalProject/llama2_benchmark/llama_env/bin/activate

# Install additional dependencies if needed
pip install --quiet wandb tqdm torch_tb_profiler

# Set up Weights & Biases (uncomment and configure if using W&B)
export WANDB_API_KEY=e2ce14bc345044e8c43020ccc0a10748e707433b
wandb login

# Create results directory with timestamp for better organization
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="./benchmark_results_${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}

# Define model paths
MODEL_PATH="./llama/llama-2-7b"
TOKENIZER_PATH="./llama"
MODEL_SIZE="7B"

# Set common parameters
TEMPERATURE=0.8
TOP_P=0.95
MAX_GEN_LEN=128
NUM_RUNS=3

# Function to log section headers
log_section() {
    echo ""
    echo "=============================================="
    echo "$1"
    echo "=============================================="
}

# Track start time for each benchmark
start_benchmark() {
    echo "Starting at: $(date)"
    START_TIME=$(date +%s)
}

# Track end time and duration for each benchmark
end_benchmark() {
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "Completed at: $(date)"
    echo "Duration: $((DURATION / 60)) minutes and $((DURATION % 60)) seconds"
}

# 1. Run standard benchmark
log_section "1. Running Standard Performance Benchmark"
start_benchmark
python benchmark_llama.py \
    --model_path ${MODEL_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --model_size ${MODEL_SIZE} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_gen_len ${MAX_GEN_LEN} \
    --num_runs ${NUM_RUNS} \
    --output_dir ${RESULTS_DIR}/standard_benchmark \
    --use_wandb
end_benchmark

# 2. Run memory analysis
log_section "2. Running Memory Usage Analysis"
start_benchmark
python benchmark_llama.py \
    --model_path ${MODEL_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --model_size ${MODEL_SIZE} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_gen_len ${MAX_GEN_LEN} \
    --output_dir ${RESULTS_DIR}/memory_analysis \
    --analyze_memory \
    --use_wandb
end_benchmark

# 3. Run batch size sweep
log_section "3. Running Batch Size Sweep"
start_benchmark
python benchmark_llama.py \
    --model_path ${MODEL_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --model_size ${MODEL_SIZE} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_gen_len ${MAX_GEN_LEN} \
    --output_dir ${RESULTS_DIR}/batch_sweep \
    --batch_sweep \
    --max_batch_sweep 8 \
    --use_wandb
end_benchmark

# 4. Run optimization comparison
log_section "4. Running Optimization Techniques Comparison"
start_benchmark
python benchmark_llama.py \
    --model_path ${MODEL_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --model_size ${MODEL_SIZE} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_gen_len ${MAX_GEN_LEN} \
    --num_runs 2 \
    --output_dir ${RESULTS_DIR}/optimization_comparison \
    --run_comparison \
    --use_wandb
end_benchmark

# 5. Run detailed profiling
log_section "5. Running Detailed Performance Profiling"
start_benchmark
python benchmark_llama.py \
    --model_path ${MODEL_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --model_size ${MODEL_SIZE} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_gen_len ${MAX_GEN_LEN} \
    --num_runs 1 \
    --output_dir ${RESULTS_DIR}/profiling \
    --use_profiler \
    --use_wandb
end_benchmark

# 6. Run quantization-only benchmark
log_section "6. Running Int8 Quantization Benchmark"
start_benchmark
python benchmark_llama.py \
    --model_path ${MODEL_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --model_size ${MODEL_SIZE} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_gen_len ${MAX_GEN_LEN} \
    --num_runs ${NUM_RUNS} \
    --output_dir ${RESULTS_DIR}/quantization \
    --use_quantization \
    --use_wandb
end_benchmark

# 7. Test different repetition penalties
log_section "7. Testing Impact of Repetition Penalty"
for penalty in 1.0 1.1 1.2 1.3; do
    echo "Testing repetition penalty: $penalty"
    start_benchmark
    python benchmark_llama.py \
        --model_path ${MODEL_PATH} \
        --tokenizer_path ${TOKENIZER_PATH} \
        --model_size ${MODEL_SIZE} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --max_gen_len ${MAX_GEN_LEN} \
        --num_runs 2 \
        --output_dir ${RESULTS_DIR}/repetition_penalty_${penalty} \
        --repetition_penalty ${penalty} \
        --use_wandb
    end_benchmark
done

# Create a summary of all benchmarks
log_section "Creating Summary Report"
echo "Benchmark Summary" > ${RESULTS_DIR}/summary.txt
echo "Date: $(date)" >> ${RESULTS_DIR}/summary.txt
echo "Model: LLaMA 2 ${MODEL_SIZE}" >> ${RESULTS_DIR}/summary.txt
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> ${RESULTS_DIR}/summary.txt
echo "" >> ${RESULTS_DIR}/summary.txt

echo "All benchmarks completed successfully" >> ${RESULTS_DIR}/summary.txt
echo "Results saved to: ${RESULTS_DIR}" >> ${RESULTS_DIR}/summary.txt

# Copy this script to the results directory for reproducibility
cp $0 ${RESULTS_DIR}/

# Print completion info
log_section "Benchmark Suite Completed"
echo "All benchmarks completed at: $(date)"
echo "Results saved to: ${RESULTS_DIR}"
echo "=============================================="