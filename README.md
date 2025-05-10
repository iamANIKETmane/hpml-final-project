# LLaMA 2 Bayesian Optimization & Benchmarking

## Project Description
A reproducible pipeline to auto‑tune LLaMA 2 (7B) generation hyper‑parameters (temperature, top‑p) via Bayesian Optimization, and benchmark inference latency, throughput, and memory on the Nvidia A100 GPU.

---

## Milestones & Status

| Milestone                                      | Status      |
|------------------------------------------------|-------------|
| • Set up baseline LLaMA 2 inference benchmark  | ✅ Completed |
| • Implement Bayesian Optimization loop         | ✅ Completed |
| • Integrate FlashAttention‑2 & INT‑8 quant      | ✅ Completed |
| • Run benchmarks on NYU Greene cluster         | ✅ Completed |
| • Evaluate on MMLU, GSM8K, TruthfulQA, etc.    | ✅ Completed |
| • Generate figures & write report              | ✅ Completed |

---

## Repository structure:
- **benchmark_llama.py**  
  Runs inference sweeps to measure latency, throughput and memory.
- **llama_bayesian_opt.py**  
  Implements Bayesian Optimization loop for tuning temperature and top‑p.
- **run_llama_evaluation.py**  
  Evaluates optimized vs. baseline settings on MMLU, GSM8K, TruthfulQA, etc.
- **Shell wrappers (`.sh`)**  
  - `run_benchmark.sh`: SLURM wrapper for the standard benchmark  
  - `run_benchmarks_direct.sh`: direct/local execution fallback  
  - `run_bayesian_opt.sh`: local BO launcher  
  - `run_all_benchmarks_nyu.sh`: batch submission on NYU Greene  
  - `gather_environment_info.sh`: captures HPC environment details  
  - `run_llama_evaluation.sh`: shell frontend for `run_llama_evaluation.py`


---

## Example commands to execute the code

```bash
# 1. Change to project directory
cd "hpml project finale"

# 2. Run baseline benchmark (SLURM)
bash run_benchmark.sh

# 3. Run baseline benchmark locally (no SLURM)
bash run_benchmarks_direct.sh

# 4. Launch Bayesian Optimization (e.g. optimize throughput over 20 iterations)
bash run_bayesian_opt.sh throughput 20

# 5. Evaluate optimized vs. baseline on public tasks
bash run_llama_evaluation.sh

# 6. (Optional) Snapshot your environment for reproducibility
bash gather_environment_info.sh

# 7. (Alternative) Direct Python invocation with custom parameters
python benchmark_llama.py \
  --model_path ./llama/llama-2-7b \
  --tokenizer_path ./llama \
  --temperature 0.8 \
  --top_p 0.95 \
  --max_gen_len 64 \
  --num_runs 3



