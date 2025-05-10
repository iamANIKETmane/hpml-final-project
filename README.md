# LLaMA 2 Bayesian Optimization & Benchmarking

## Project Description
This project provides a complete pipeline to tune the temperature and top‑p hyperparameters of Meta’s LLaMA 2 7B model using Bayesian optimization on NVIDIA A100 GPUs. It includes scripts for running baseline inference benchmarks that collect latency, throughput, and memory usage data, and integrates FlashAttention‑2 kernels and INT‑8 quantization to improve performance. After the optimization step, the same tools evaluate the tuned settings on standard NLP benchmarks such as MMLU, GSM8K, and TruthfulQA to verify that output quality remains consistent.

---

## Milestones & Status
| Milestone                                   | Status      |
|---------------------------------------------|-------------|
| Set up baseline LLaMA 2 inference benchmark | ✅ Completed |
| Implement Bayesian Optimization loop        | ✅ Completed |
| Integrate FlashAttention‑2 & INT‑8 quant    | ✅ Completed |
| Run benchmarks on NVIDIA A100 GPU           | ✅ Completed |
| Evaluate on MMLU, GSM8K, TruthfulQA, etc.   | ✅ Completed |
| Generate figures & write report             | ✅ Completed |

---

## Repository structure
- **benchmark_llama.py**  
  Runs inference sweeps to measure latency, throughput, and memory.  
- **llama_bayesian_opt.py**  
  Implements Bayesian Optimization loop for tuning temperature and top‑p.  
- **run_llama_evaluation.py**  
  Evaluates optimized vs. baseline settings on MMLU, GSM8K, TruthfulQA, etc.  
- **Shell wrappers (`.sh`)**  
  - `run_benchmark.sh`: SLURM wrapper for the standard benchmark  
  - `run_benchmarks_direct.sh`: Direct/local execution fallback  
  - `run_bayesian_opt.sh`: Local BO launcher  
  - `run_all_benchmarks_nyu.sh`: Batch submission on NYU Greene  
  - `gather_environment_info.sh`: Captures HPC environment details  
  - `run_llama_evaluation.sh`: Shell frontend for `run_llama_evaluation.py`  

---
## Results & Observations
1. **Inference Benchmark**

| Batch Size | Avg. Latency (ms) | Throughput (tokens/sec) | Peak Memory (GB) |
|------------|-------------------|-------------------------|------------------|
| 1          | 210               | 9.5                     | 22.3             |
| 4          | 450               | 16.0                    | 26.1             |
| 8          | 800               | 20.0                    | 29.5             |

**Observations:**
- Latency increases roughly linearly with batch size.
- Throughput improves up to batch size 8 before plateauing.
- Peak GPU memory usage rises moderately as batch size grows.

2. **Bayesian Optimization Results**

| Configuration                  | Latency (ms) | Throughput (tokens/sec) | MMLU Accuracy (%) |
|--------------------------------|--------------|-------------------------|-------------------|
| Baseline (T=0.8, P=0.95)       | 210          | 9.5                     | 46.0              |
| BO-Optimized (T=0.6, P=0.90)   | 160          | 12.0                    | 45.8              |

**Observations:**
- BO-optimized settings reduce latency by ~24% and boost throughput by ~26%.
- MMLU accuracy drops by only 0.2%, indicating output quality is preserved.

3. **NLP Task Evaluation**

| Task       | Baseline Acc. (%) | Optimized Acc. (%) |
|------------|-------------------|--------------------|
| MMLU       | 46.0              | 45.8               |
| GSM8K      | 54.2              | 54.0               |
| TruthfulQA | 41.7              | 41.5               |

**Observations:**
- Accuracy degradation on downstream tasks is < 0.3% across the board.
- Demonstrates that performance gains come at negligible cost to model quality.


## Example commands to execute the code
```bash
cd "hpml project finale"
bash run_benchmark.sh
bash run_benchmarks_direct.sh
bash run_bayesian_opt.sh throughput 20
bash run_llama_evaluation.sh
bash gather_environment_info.sh
python benchmark_llama.py \
  --model_path ./llama/llama-2-7b \
  --tokenizer_path ./llama \
  --temperature 0.8 \
  --top_p 0.95 \
  --max_gen_len 64 \
  --num_runs 3










