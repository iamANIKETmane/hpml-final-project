#!/bin/bash
# Environment Information Gathering Script for LLaMA 2 Benchmarking

# Create output directory
OUTPUT_DIR="environment_info_$(date +"%Y%m%d_%H%M%S")"
mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR

echo "Gathering environment information..."
echo "Results will be saved in: $PWD"

# Function to run commands and save output
run_and_save() {
    local cmd="$1"
    local file="$2"
    
    echo "Running: $cmd"
    echo "$ $cmd" > "$file"
    eval "$cmd" >> "$file" 2>&1
    echo "  - Saved to $file"
}

# System information
echo "Getting system information..."
run_and_save "hostname" "hostname.txt"
run_and_save "whoami" "username.txt"
run_and_save "pwd" "current_dir.txt"
run_and_save "echo \$HOME" "home_dir.txt"
run_and_save "df -h" "disk_usage.txt"

# SLURM information
echo "Getting SLURM information..."
run_and_save "sinfo -o '%P %a'" "slurm_partitions.txt"
run_and_save "sinfo -o '%P %G'" "slurm_gpus.txt"
run_and_save "sacctmgr show associations -p | grep \$USER" "slurm_accounts.txt"
run_and_save "squeue -u \$USER" "slurm_queue.txt"

# Module information
echo "Getting module information..."
run_and_save "module avail" "modules_available.txt"
run_and_save "module list" "modules_loaded.txt"

# Directory structure
echo "Checking project directory structure..."
run_and_save "ls -la .." "parent_dir_listing.txt"
run_and_save "find .. -maxdepth 2 -type d | sort" "nearby_directories.txt"

# Look for important directories
echo "Looking for key directories..."
run_and_save "find /scratch -name llama -type d 2>/dev/null || echo 'Not found'" "llama_dir_location.txt"
run_and_save "find /scratch -name llama-2-7b -type d 2>/dev/null || echo 'Not found'" "model_dir_location.txt"
run_and_save "find /scratch -name eval_env -type d 2>/dev/null || echo 'Not found'" "env_dir_location.txt"

# Python environment
echo "Checking Python environment..."
run_and_save "which python" "python_path.txt"
run_and_save "python --version" "python_version.txt"
run_and_save "pip list" "pip_packages.txt"

# GPU information
echo "Checking GPU information..."
run_and_save "nvidia-smi" "nvidia_smi.txt"
run_and_save "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv" "gpu_memory.txt"

# Script existence check
echo "Checking for script files..."
for script in benchmark_llama.py benchmark_dataset.py llama_bayesian_opt.py run_benchmark.sh run_dataset_benchmark.sh run_bayesian_opt.sh; do
    if [ -f "../$script" ]; then
        echo "$script: Found ($(stat -c %s "../$script") bytes)" >> "script_files.txt"
    else
        echo "$script: Not found" >> "script_files.txt"
    fi
done

# Create a summary file
echo "Creating summary..."
{
    echo "====== Environment Information Summary ======"
    echo "Date: $(date)"
    echo "Hostname: $(cat hostname.txt)"
    echo "Username: $(cat username.txt)"
    echo "Current directory: $(cat current_dir.txt)"
    
    echo -e "\nAvailable SLURM partitions:"
    grep -v PARTITION slurm_partitions.txt
    
    echo -e "\nAvailable Python packages (first 10):"
    head -10 pip_packages.txt
    
    echo -e "\nGPU Information:"
    head -5 nvidia_smi.txt
    
    echo -e "\nScript Files:"
    cat script_files.txt
    
    echo -e "\nFor complete information, check all files in the $OUTPUT_DIR directory."
} > summary.txt

# Create a tar file with all the information
cd ..
tar -czf "${OUTPUT_DIR}.tar.gz" "$OUTPUT_DIR"

echo "Environment information gathering complete!"
echo "Summary available in: $OUTPUT_DIR/summary.txt"
echo "All information archived in: ${OUTPUT_DIR}.tar.gz"
echo "Please share the .tar.gz file for a fully tailored script."