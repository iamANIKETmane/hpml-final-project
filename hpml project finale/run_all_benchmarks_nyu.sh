#!/bin/bash

# Print header
echo "=============================================="
echo "LLaMA 2 Benchmark Suite - NYU CUSTOM"
echo "=============================================="

# Create a directory for all results with timestamp
MAIN_DIR="llama2_results_$(date +"%Y%m%d_%H%M%S")"
mkdir -p $MAIN_DIR
cd $MAIN_DIR

# Set correct parameters for your NYU environment
PARTITION="g2-standard-12"  # This is the correct partition name
ACCOUNT="ece_gy_7123-2025sp"  # Using your course account

echo "Using partition: $PARTITION"
echo "Using account: $ACCOUNT"
echo "Results will be saved in: $PWD"

# Create a log file to track job submissions
LOG_FILE="job_submissions.log"
echo "Starting benchmark suite at $(date)" > $LOG_FILE

# Function to submit a job and log its ID
submit_job() {
    local script="$1"
    local args="${@:2}"  # All arguments after the script name
    
    echo "Submitting job: $script $args"
    JOB_ID=$(sbatch --parsable --partition=$PARTITION --account=$ACCOUNT ../$script $args 2>/dev/null)
    
    JOB_STATUS=$?
    
    if [ $JOB_STATUS -eq 0 ]; then
        echo "  Submitted job ID: $JOB_ID"
        echo "$(date) - Submitted $script $args - Job ID: $JOB_ID" >> $LOG_FILE
        echo $JOB_ID
    else
        echo "  Error submitting job: $JOB_ID"
        echo "$(date) - Error submitting $script $args" >> $LOG_FILE
        echo "0"
    fi
}

# 1. Run standard benchmark
echo "Submitting standard benchmark job..."
BENCH_JOB_ID=$(submit_job run_benchmark.sh)
echo "Standard benchmark job ID: $BENCH_JOB_ID"
sleep 2

# 2. Run Bayesian optimization
echo "Submitting Bayesian optimization job..."

# First modify the bayesian opt script to work with SLURM
cat > bayesian_opt_slurm.sh << 'EOL'
#!/bin/bash
#SBATCH --job-name=llama2_bayes
#SBATCH --output=llama2_bayes_%j.out
#SBATCH --error=llama2_bayes_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Starting at: $(date)"

# Load necessary modules
module purge

# Activate virtual environment
source /scratch/am14661/FinalProject/llama2_benchmark/eval_env/bin/activate

# Create results directory
mkdir -p optimization_results

# Define paths with variables for better flexibility
PROJ_DIR=$(pwd)
MODEL_DIR="${PROJ_DIR}/llama/llama-2-7b"
TOKENIZER_PATH="${PROJ_DIR}/llama/tokenizer.model"

# Parse command line args
target="${1:-throughput}"  # Default to optimizing for throughput if not specified
n_iter="${2:-10}"         # Default to 10 iterations if not specified

echo "Running Bayesian optimization for target: $target with $n_iter iterations"

# Install dependencies
pip install --quiet numpy==1.23.5 scikit-learn==1.2.2 bayesian-optimization==1.4.3 tqdm matplotlib

# Add current directory to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the optimization script
python llama_bayesian_opt.py \
  --model_path ${MODEL_DIR} \
  --tokenizer_path ${TOKENIZER_PATH} \
  --model_size 7B \
  --max_gen_len 30 \
  --n_iter ${n_iter} \
  --initial_points 3 \
  --optimization_target ${target} \
  --output_dir ./optimization_results \
  --max_batch_size 1 \
  --max_seq_len 512

# Check if optimization completed successfully
if [ $? -eq 0 ]; then
  echo "Bayesian optimization completed successfully!"
  
  # Display the best parameters
  echo "Best parameters:"
  cat ./optimization_results/optimization_results_${target}.json | grep -A3 "best_params"
else
  echo "Bayesian optimization failed with error code $?"
fi

echo "Completed at: $(date)"
EOL

chmod +x bayesian_opt_slurm.sh
cp bayesian_opt_slurm.sh ../bayesian_opt_slurm.sh

BAYES_JOB_ID=$(submit_job bayesian_opt_slurm.sh "throughput" 15)
echo "Bayesian optimization job ID: $BAYES_JOB_ID"
sleep 2

# 3. Create a modified dataset benchmark script
cat > dataset_benchmark_slurm.sh << 'EOL'
#!/bin/bash
#SBATCH --job-name=llama2_data
#SBATCH --output=llama2_data_%j.out
#SBATCH --error=llama2_data_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Starting at: $(date)"

# Load necessary modules
module purge

# Activate virtual environment
source /scratch/am14661/FinalProject/llama2_benchmark/eval_env/bin/activate

# Install tqdm if needed
pip list | grep tqdm > /dev/null || pip install --quiet tqdm

# Parse arguments
dataset="${1:-helpfulness}"
max_samples="${2:-5}"

echo "Running benchmark on dataset: $dataset with $max_samples samples"

# Create results directory
RESULT_DIR="dataset_${dataset}"
mkdir -p $RESULT_DIR

# Set environment variables for torch distributed
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# Add current directory to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the benchmark script
python benchmark_dataset.py \
  --model_path ./llama/llama-2-7b \
  --tokenizer_path ./llama \
  --model_size 7B \
  --temperature 0.8 \
  --top_p 0.95 \
  --max_gen_len 30 \
  --dataset ${dataset} \
  --max_samples ${max_samples} \
  --output_dir ${RESULT_DIR} \
  --local_rank 0 \
  --max_batch_size 1 \
  --max_seq_len 512

# Check if benchmark completed successfully
if [ $? -eq 0 ]; then
  echo "Dataset benchmark completed successfully!"
  echo "Results saved to ${RESULT_DIR}"
else
  echo "Dataset benchmark failed with error code $?"
fi

echo "Completed at: $(date)"
EOL

chmod +x dataset_benchmark_slurm.sh
cp dataset_benchmark_slurm.sh ../dataset_benchmark_slurm.sh

# Submit dataset benchmark jobs
echo "Submitting dataset benchmark jobs..."
HELP_JOB_ID=$(submit_job dataset_benchmark_slurm.sh "helpfulness" 5)
echo "Helpfulness dataset job ID: $HELP_JOB_ID"
sleep 2

TECH_JOB_ID=$(submit_job dataset_benchmark_slurm.sh "technical" 5)
echo "Technical dataset job ID: $TECH_JOB_ID"
sleep 2

CREATIVE_JOB_ID=$(submit_job dataset_benchmark_slurm.sh "creative" 5)
echo "Creative dataset job ID: $CREATIVE_JOB_ID"
sleep 2

# Create a summary file
echo "Creating job summary..."
echo "LLaMA 2 Benchmark Suite - Summary" > summary.txt
echo "=================================" >> summary.txt
echo "Date: $(date)" >> summary.txt
echo "Results directory: $PWD" >> summary.txt
echo "Using partition: $PARTITION" >> summary.txt
echo "Using account: $ACCOUNT" >> summary.txt
echo "" >> summary.txt
echo "Job IDs:" >> summary.txt
echo "  Standard benchmark: $BENCH_JOB_ID" >> summary.txt
echo "  Bayesian optimization: $BAYES_JOB_ID" >> summary.txt
echo "  Helpfulness dataset: $HELP_JOB_ID" >> summary.txt
echo "  Technical dataset: $TECH_JOB_ID" >> summary.txt
echo "  Creative dataset: $CREATIVE_JOB_ID" >> summary.txt
echo "" >> summary.txt
echo "Monitor jobs with: squeue -u $USER" >> summary.txt

echo "All jobs submitted. Check status with: squeue -u $USER"
echo "Results will be saved in: $PWD"
echo "Job IDs are recorded in: $PWD/summary.txt"
echo "=============================================="

# Add fallback direct run instructions
cat > direct_run_instructions.txt << 'EOL'
If job submission fails, you can run the benchmarks directly with these commands:

# Standard benchmark
cd /scratch/am14661/FinalProject/llama2_benchmark
source eval_env/bin/activate
python benchmark_llama.py \
  --model_path ./llama/llama-2-7b \
  --tokenizer_path ./llama \
  --model_size 7B \
  --temperature 0.8 \
  --top_p 0.95 \
  --max_gen_len 30 \
  --num_runs 1 \
  --output_dir ./benchmark_results_direct

# Check that the scripts run correctly before submitting to SLURM again
EOL

echo "If job submission fails, see direct_run_instructions.txt for manual commands"