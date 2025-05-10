"""
LLaMA 2 Standard Benchmark Evaluation Script
Validates the parameters from Bayesian optimization against standard benchmarks
"""

import os
import json
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import gc
import sys
from tqdm import tqdm

# Add standard benchmark datasets
from datasets import load_dataset

# Import your current benchmark utility functions
sys.path.append(os.getcwd())
from benchmark_llama import load_model, benchmark_inference

# Define standard benchmark prompts for different datasets
MMLU_PROMPTS = [
    "The process during which ATP synthase produces ATP using the proton motive force is called: (A) active transport (B) chemiosmosis (C) facilitated diffusion (D) substrate-level phosphorylation",
    "The difference between axons and dendrites is that: (A) dendrites are found only in the central nervous system (B) dendrites conduct action potentials and axons conduct graded potentials (C) dendrites are covered with a myelin sheath and axons are not (D) dendrites typically conduct signals toward the cell body and axons typically conduct signals away from the cell body",
    "The standard atomic weight of an element represents: (A) the mass of the most common isotope of the element (B) the average of the masses of the two most common isotopes (C) the weighted average of the masses of all naturally occurring isotopes (D) the mass of the isotope that has the number of neutrons equal to the number of protons",
    "A substance that is dissolved in a solution is known as a(n): (A) solute (B) solvent (C) suspension (D) colloid",
    "A chemical reaction in which the oxidation numbers of atoms are changed is called a(n): (A) combination reaction (B) synthesis reaction (C) decomposition reaction (D) oxidation-reduction reaction"
]

TRUTHFULQA_PROMPTS = [
    "Are there any animals that never sleep?",
    "What happens if you drink blood?",
    "Do eggplants cause insanity?",
    "What is the furthest humans have traveled in space?",
    "Do vaccines contain cells from aborted fetuses?"
]

COMMON_SENSE_PROMPTS = [
    "Can you cut a piece of paper with scissors that are too dull?",
    "If I drop a feather and a bowling ball at the same time from the same height in a vacuum, which will hit the ground first?",
    "Would a wooden boat float on a lake of mercury?",
    "If I have 10 apples and I give 3 to John and 2 to Sara, how many apples do I have left?",
    "Can you use a metal fork to get toast out of a toaster that's plugged in?"
]

GSM8K_PROMPTS = [
    "Janet's ducks lay 16 eggs per day. She eats 3 eggs for breakfast each morning and bakes muffins for her friends every day with 4 eggs in each batch of muffins. She bakes 2 batches of muffins daily. How many extra eggs will Janet have after 3 days?",
    "Elsa has 5 apples. She buys 7 more apples. Her sister gives her 2 apples. She bakes an apple pie using 8 apples. How many apples does she have now?",
    "A restaurant has 10 tables. Each table has 4 chairs. There are 15 customers in the restaurant. How many empty chairs are there?",
    "There are 15 trees in the grove. Each tree produces 12 apples. If Sam picks 35 apples and Lisa picks 40 apples, how many apples are left on the trees?",
    "John has 5 boxes of pencils. Each box has 10 pencils. He gives 3 pencils to each of his 8 friends. How many pencils does he have left?"
]

# Combine all benchmark prompts
BENCHMARK_DATASETS = {
    "MMLU": MMLU_PROMPTS,
    "TruthfulQA": TRUTHFULQA_PROMPTS,
    "CommonSense": COMMON_SENSE_PROMPTS,
    "GSM8K": GSM8K_PROMPTS
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLaMA 2 with optimized parameters on standard benchmarks")
    parser.add_argument("--model_size", type=str, default="7B", choices=["7B", "13B"],
                        help="Size of LLaMA 2 model to benchmark")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model weights directory")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to the tokenizer model")
    parser.add_argument("--max_gen_len", type=int, default=128,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--num_runs", type=int, default=2,
                        help="Number of runs to average over")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Directory to save results")
    
    # Add parameter options - default to optimized values
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Temperature for sampling (optimized value)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top p for nucleus sampling (optimized value)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="Repetition penalty")
    parser.add_argument("--max_batch_size", type=int, default=1,
                    help="Maximum batch size")
    parser.add_argument("--max_seq_len", type=int, default=512,
                    help="Maximum sequence length")
    parser.add_argument("--use_quantization", action="store_true", 
                    help="Enable dynamic int8 quantization")
    parser.add_argument("--use_flash_attention", action="store_true",
                    help="Enable flash attention if available")
    # Add comparison mode to test optimized vs. baseline parameters
    parser.add_argument("--run_comparison", action="store_true",
                        help="Compare optimized parameters with baseline")
    
    
    return parser.parse_args()

def run_benchmark_on_dataset(model, tokenizer, dataset_name, prompts, args):
    """Run benchmarks on a specific dataset with the provided parameters."""
    print(f"\n=== Running benchmark on {dataset_name} dataset ===")
    
    all_results = []
    
    for i, prompt in enumerate(prompts):
        print(f"Evaluating prompt {i+1}/{len(prompts)}")
        
        prompt_results = []
        for run in range(args.num_runs):
            try:
                # Use benchmark_inference from your existing code
                result = benchmark_inference(model, tokenizer, prompt, args)
                result["dataset"] = dataset_name
                result["prompt_idx"] = i
                result["prompt"] = prompt[:100] + "..." if len(prompt) > 100 else prompt
                result["run"] = run
                prompt_results.append(result)
                
                # Clear memory between runs
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"Error in run {run+1}: {e}")
                continue
        
        if prompt_results:
            # Calculate statistics for this prompt
            latencies = [r["latency"] for r in prompt_results]
            tokens_per_second = [r["tokens_per_second"] for r in prompt_results]
            
            print(f"  Prompt {i+1} avg latency: {np.mean(latencies):.3f}s")
            print(f"  Prompt {i+1} avg throughput: {np.mean(tokens_per_second):.2f} tokens/sec")
            print(f"  Sample output: {prompt_results[0]['output_text'][:100]}...")
            
            all_results.extend(prompt_results)
        else:
            print(f"  No successful runs for prompt {i+1}")
    
    # Calculate dataset average metrics
    if all_results:
        avg_latency = np.mean([r["latency"] for r in all_results])
        avg_throughput = np.mean([r["tokens_per_second"] for r in all_results])
        
        print(f"\n{dataset_name} Dataset Summary:")
        print(f"  Average latency: {avg_latency:.3f}s")
        print(f"  Average throughput: {avg_throughput:.2f} tokens/sec")
    
    return all_results

def compare_parameter_sets(model, tokenizer, args):
    """Compare the optimized parameters with baseline parameters."""
    datasets_to_test = ["MMLU", "GSM8K"]  # Use a subset for comparison
    
    parameter_sets = [
        {"name": "Optimized", "temperature": 0.05, "top_p": 1.0},
        {"name": "Baseline", "temperature": 0.8, "top_p": 0.95}
    ]
    
    comparison_results = {}
    
    for params in parameter_sets:
        print(f"\n=== Evaluating with {params['name']} parameters ===")
        print(f"Temperature: {params['temperature']}, Top-p: {params['top_p']}")
        
        # Update args with current parameters
        args.temperature = params['temperature']
        args.top_p = params['top_p']
        
        dataset_results = []
        
        for dataset_name in datasets_to_test:
            dataset_prompts = BENCHMARK_DATASETS[dataset_name]
            results = run_benchmark_on_dataset(model, tokenizer, dataset_name, dataset_prompts, args)
            dataset_results.extend(results)
        
        # Calculate overall metrics
        avg_latency = np.mean([r["latency"] for r in dataset_results])
        avg_throughput = np.mean([r["tokens_per_second"] for r in dataset_results])
        
        comparison_results[params["name"]] = {
            "avg_latency": avg_latency,
            "avg_throughput": avg_throughput,
            "temperature": params["temperature"],
            "top_p": params["top_p"]
        }
    
    # Create comparison visualization
    create_parameter_comparison_plot(comparison_results, args)
    
    return comparison_results

def evaluate_all_datasets(model, tokenizer, args):
    """Run benchmarks on all standard datasets."""
    all_dataset_results = {}
    
    for dataset_name, prompts in BENCHMARK_DATASETS.items():
        results = run_benchmark_on_dataset(model, tokenizer, dataset_name, prompts, args)
        all_dataset_results[dataset_name] = results
    
    return all_dataset_results

def create_parameter_comparison_plot(comparison_results, args):
    """Create plots comparing optimized vs baseline parameters."""
    labels = list(comparison_results.keys())
    throughputs = [comparison_results[label]["avg_throughput"] for label in labels]
    latencies = [comparison_results[label]["avg_latency"] for label in labels]
    
    plt.figure(figsize=(15, 6))
    
    # Throughput comparison
    plt.subplot(1, 2, 1)
    bars = plt.bar(labels, throughputs, color=['green', 'blue'])
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{throughputs[i]:.2f}', ha='center')
    plt.ylabel("Tokens per Second")
    plt.title("Throughput Comparison")
    plt.grid(axis='y', alpha=0.3)
    
    # Latency comparison
    plt.subplot(1, 2, 2)
    bars = plt.bar(labels, latencies, color=['green', 'blue'])
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{latencies[i]:.3f}s', ha='center')
    plt.ylabel("Latency (s)")
    plt.title("Latency Comparison")
    plt.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f"LLaMA 2 {args.model_size} Parameter Comparison", fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(args.output_dir, exist_ok=True)
    plot_file = os.path.join(args.output_dir, f"parameter_comparison_{args.model_size}.png")
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Parameter comparison plot saved to {plot_file}")

def create_dataset_comparison_plot(all_results, args):
    """Create visualization comparing performance across datasets."""
    # Calculate metrics per dataset
    dataset_metrics = {}
    
    for dataset_name, results in all_results.items():
        if results:
            avg_latency = np.mean([r["latency"] for r in results])
            avg_throughput = np.mean([r["tokens_per_second"] for r in results])
            avg_tokens = np.mean([r["output_length"] for r in results])
            
            dataset_metrics[dataset_name] = {
                "latency": avg_latency,
                "throughput": avg_throughput,
                "tokens": avg_tokens
            }
    
    if not dataset_metrics:
        print("No data available for dataset comparison plot")
        return
    
    # Extract data for plotting
    datasets = list(dataset_metrics.keys())
    latencies = [dataset_metrics[d]["latency"] for d in datasets]
    throughputs = [dataset_metrics[d]["throughput"] for d in datasets]
    tokens = [dataset_metrics[d]["tokens"] for d in datasets]
    
    # Create plot
    plt.figure(figsize=(18, 6))
    
    # Throughput by dataset
    plt.subplot(1, 3, 1)
    bars = plt.bar(datasets, throughputs)
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{throughputs[i]:.2f}', ha='center')
    plt.ylabel("Tokens per Second")
    plt.title("Throughput by Dataset")
    plt.grid(axis='y', alpha=0.3)
    
    # Latency by dataset
    plt.subplot(1, 3, 2)
    bars = plt.bar(datasets, latencies)
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{latencies[i]:.3f}s', ha='center')
    plt.ylabel("Latency (s)")
    plt.title("Latency by Dataset")
    plt.grid(axis='y', alpha=0.3)
    
    # Average output length by dataset
    plt.subplot(1, 3, 3)
    bars = plt.bar(datasets, tokens)
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{tokens[i]:.1f}', ha='center')
    plt.ylabel("Output Length (tokens)")
    plt.title("Average Output Length by Dataset")
    plt.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f"LLaMA 2 {args.model_size} Performance Across Datasets (temp={args.temperature}, top_p={args.top_p})", 
                 fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(args.output_dir, exist_ok=True)
    plot_file = os.path.join(args.output_dir, f"dataset_comparison_{args.model_size}.png")
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Dataset comparison plot saved to {plot_file}")

def main():
    args = parse_args()
    print(f"Evaluating LLaMA 2 {args.model_size} with optimized parameters on standard benchmarks")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model
    model, tokenizer = load_model(args.model_path, args.tokenizer_path, args.model_size, args)
    
    if args.run_comparison:
        # Compare optimized parameters with baseline
        comparison_results = compare_parameter_sets(model, tokenizer, args)
        
        # Save comparison results
        comparison_file = os.path.join(args.output_dir, f"parameter_comparison_{args.model_size}.json")
        with open(comparison_file, "w") as f:
            json.dump(comparison_results, f, indent=2)
        print(f"Parameter comparison results saved to {comparison_file}")
    else:
        # Evaluate all datasets with optimized parameters
        all_results = evaluate_all_datasets(model, tokenizer, args)
        
        # Create visualization
        create_dataset_comparison_plot(all_results, args)
        
        # Save all results
        all_results_merged = []
        for dataset_results in all_results.values():
            all_results_merged.extend(dataset_results)
        
        if all_results_merged:
            results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'output_text'} 
                                      for r in all_results_merged])
            results_file = os.path.join(args.output_dir, f"benchmark_results_{args.model_size}.csv")
            results_df.to_csv(results_file, index=False)
            print(f"Benchmark results saved to {results_file}")
            
            # Save a sample of output text separately
            sample_outputs = {
                f"{r['dataset']}_prompt_{r['prompt_idx']}": {
                    "prompt": r["prompt"],
                    "output": r["output_text"]
                }
                for r in all_results_merged[:10]  # Just save some samples
            }
            outputs_file = os.path.join(args.output_dir, f"sample_outputs_{args.model_size}.json")
            with open(outputs_file, "w") as f:
                json.dump(sample_outputs, f, indent=2)
            print(f"Sample outputs saved to {outputs_file}")
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()