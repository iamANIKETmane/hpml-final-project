"""
LLaMA 2 Inference Benchmarking Script (Enhanced Version)
"""

import os
import time
import json
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import psutil
import gc
import sys
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

# Initialize PyTorch distributed environment
if not torch.distributed.is_initialized():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    # Use the number of GPUs specified (default to 1)
    world_size = 1  # This will be updated by command line args
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    
    try:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=1,
            rank=0
        )
        print("PyTorch distributed initialized")
    except Exception as e:
        print(f"Warning: Failed to initialize distributed environment: {e}")
        print("Continuing without distributed support...")

# Add the current directory to sys.path to ensure LLaMA modules can be imported
sys.path.append(os.getcwd())

# FAIRSCALE MODEL-PARALLEL INITIALIZATION
try:
    from fairscale.nn.model_parallel.initialize import initialize_model_parallel
    initialize_model_parallel(1, 1)
    print("FairScale model parallel initialized")
except ImportError:
    print("Warning: FairScale not available. Continuing without model parallelism.")
except Exception as e:
    print(f"Warning: Failed to initialize model parallelism: {e}")
    print("Continuing without model parallelism...")

# Import llama model
# Import llama model - try all possible locations
try:
    # Add all possible paths
    sys.path.append(os.path.join(os.getcwd(), "llama"))
    sys.path.append(os.path.join(os.getcwd(), "llama-repo"))
    
    # Try different import patterns
    try:
        from llama.model import ModelArgs, Transformer
        from llama.tokenizer import Tokenizer
        print("Successfully imported from llama package")
    except ImportError:
        try:
            from llama.llama.model import ModelArgs, Transformer
            from llama.llama.tokenizer import Tokenizer
            print("Successfully imported from llama.llama package")
        except ImportError:
            # Direct import
            import sys
            sys.path.append("./llama")
            from model import ModelArgs, Transformer
            from tokenizer import Tokenizer
            print("Successfully imported directly")
except Exception as e:
    print(f"Fatal error importing LLaMA modules: {e}")
    sys.exit(1)
    
# Define test prompts
TEST_PROMPTS = [
    "What is quantum computing?",
    "Explain the theory of relativity in simple terms.",
    "How does machine learning work?",
    "What are the key features of Python programming language?",
    "Describe the process of photosynthesis."
]

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LLaMA 2 inference")
    parser.add_argument("--model_size", type=str, default="7B", choices=["7B", "13B"],
                        help="Size of LLaMA 2 model to benchmark")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model weights directory")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to the tokenizer model")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top p for nucleus sampling")
    parser.add_argument("--max_gen_len", type=int, default=128,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of runs to average over")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Directory to save results")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for distributed training")
    parser.add_argument("--max_batch_size", type=int, default=1,
                        help="Maximum batch size")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--use_profiler", action="store_true",
                        help="Use PyTorch profiler for detailed performance analysis")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for experiment tracking")
    parser.add_argument("--use_quantization", action="store_true",
                        help="Enable dynamic int8 quantization")
    parser.add_argument("--use_flash_attention", action="store_true",
                        help="Enable flash attention if available")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use for inference")
    parser.add_argument("--run_comparison", action="store_true",
                        help="Run comparison of different optimization techniques")
    parser.add_argument("--analyze_memory", action="store_true",
                        help="Run memory usage analysis")
    parser.add_argument("--batch_sweep", action="store_true",
                        help="Run benchmarks with different batch sizes")
    parser.add_argument("--max_batch_sweep", type=int, default=8,
                        help="Maximum batch size for batch size sweep")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="Repetition penalty (1.0 = no penalty)")
    return parser.parse_args()

def load_model(model_path, tokenizer_path, model_size, args):
    """Load LLaMA 2 model using Meta's official implementation with memory optimizations."""
    print(f"Loading LLaMA 2 {model_size} model...")

    # Clear CUDA cache before loading model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Model checkpoint locations
    ckpt_dir = Path(model_path)
    if os.path.isfile(tokenizer_path):
        tokenizer_path = tokenizer_path
    else:
        tokenizer_path = Path(tokenizer_path) / "tokenizer.model"
    
    print(f"Tokenizer path: {tokenizer_path}")
    print(f"Model path: {ckpt_dir}")

    # Load tokenizer
    try:
        tokenizer = Tokenizer(model_path=str(tokenizer_path))
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise

    # Load model params from params.json if available
    params_path = ckpt_dir / "params.json"
    if params_path.exists():
        print(f"Loading model parameters from {params_path}")
        with open(params_path, "r") as f:
            params_json = json.load(f)
        
        # Make a copy to avoid modifying the original
        params_dict = dict(params_json)
        
        # Override vocab_size if negative
        if "vocab_size" not in params_dict or params_dict["vocab_size"] <= 0:
            print(f"WARNING: Invalid vocab_size ({params_dict.get('vocab_size')}) in params.json. Setting to 32000.")
            params_dict["vocab_size"] = 32000
        
        # Use batch size and sequence length from args
        params_dict["max_batch_size"] = args.max_batch_size
        params_dict["max_seq_len"] = args.max_seq_len
        
        print(f"Modified parameters: {params_dict}")
        params = ModelArgs(**params_dict)
    else:
        print(f"Using default parameters for {model_size} model")
        if model_size == "7B":
            params = ModelArgs(
                dim=4096,
                n_layers=32,
                n_heads=32,
                norm_eps=1e-6,
                vocab_size=32000,
                multiple_of=256,
                max_batch_size=args.max_batch_size,
                max_seq_len=args.max_seq_len
            )
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
    
    print(f"Final model parameters: {params}")

    # Check for consolidated checkpoint
    checkpoint_path = ckpt_dir / "consolidated.00.pth"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        checkpoint_files = list(ckpt_dir.glob("*.pth"))
        if checkpoint_files:
            checkpoint_path = checkpoint_files[0]
            print(f"Using alternative checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

    # Create and load model with optimizations
    try:
        print("Creating transformer model...")
        
        # Enable flash attention if requested and available
        if hasattr(args, 'use_flash_attention') and args.use_flash_attention:
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
                print("Enabled flash attention")
            else:
                print("Flash attention not available in this PyTorch version")
        
        # Create model
        model = Transformer(params)
        print("Model created successfully")
        
        # Apply quantization if requested
        if hasattr(args, 'use_quantization') and args.use_quantization:
            try:
                print("Applying int8 quantization...")
                from torch.quantization import quantize_dynamic
                model = quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                print("Model quantized to int8")
            except Exception as e:
                print(f"Quantization failed: {e}. Continuing with original model.")
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        
        # Move to GPU with half precision
        print("Moving model to GPU with half precision...")
        model = model.half().cuda()
        model.eval()
        
        # Distribute across multiple GPUs if requested
        if hasattr(args, 'num_gpus') and args.num_gpus > 1:
            try:
                print(f"Distributing model across {args.num_gpus} GPUs...")
                model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))
                print("Model distributed successfully")
            except Exception as e:
                print(f"Failed to distribute model: {e}. Using single GPU.")
        
        # Clean up memory
        del checkpoint
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"Error creating/loading model: {e}")
        raise

    return model, tokenizer

def calculate_extended_metrics(result, device_info=None):
    """Calculate additional performance metrics beyond basic latency/throughput."""
    # Get device info if not provided
    if device_info is None:
        device_info = {
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "sms": torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else 1,
            "mem_total": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 1  # GB
        }
    
    # Calculate extended metrics
    metrics = {
        "per_token_latency": result["latency"] / max(result["output_length"], 1),  # seconds per token
        "memory_efficiency": result["output_length"] / max(result["memory_used"], 1),  # tokens per MB
        "compute_efficiency": result["tokens_per_second"] / device_info["sms"],  # tokens per second per SM
        "memory_utilization": result["memory_used"] / (device_info["mem_total"] * 1024) if device_info["mem_total"] > 0 else 0,  # percentage of GPU memory
        "first_token_latency": result.get("first_token_time", 0),  # first token generation time
        "processing_overhead": (result["latency"] - (result["output_length"] * result.get("per_token_latency", 0))) / result["latency"] if result["latency"] > 0 else 0  # overhead percentage
    }
    
    return metrics

def benchmark_inference(model, tokenizer, prompt, args):
    """Benchmark inference with profiling and memory optimizations."""
    print(f"Benchmarking prompt: {prompt}")
    
    # Tokenize the prompt
    try:
        tokens = tokenizer.encode(prompt, bos=True, eos=False)
        tokens = torch.tensor(tokens).cuda().unsqueeze(0)
        input_length = tokens.shape[1]
        print(f"Input tokens: {input_length}")
    except Exception as e:
        print(f"Error tokenizing: {e}")
        raise

    # Clear CUDA cache before inference
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    # Setup timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    first_token_start = torch.cuda.Event(enable_timing=True)
    first_token_end = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    first_token_start.record()

    # Generation parameters
    temperature = args.temperature
    top_p = args.top_p
    repetition_penalty = args.repetition_penalty if hasattr(args, 'repetition_penalty') else 1.0
    max_gen_len = args.max_gen_len

    # Use profiler if requested
    try:
        if hasattr(args, 'use_profiler') and args.use_profiler:
            # Create profiler output directory
            os.makedirs(os.path.join(args.output_dir, "profiler"), exist_ok=True)
            
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    os.path.join(args.output_dir, "profiler")),
                record_shapes=True,
                profile_memory=True
            ) as prof:
                # Generate output
                with torch.no_grad():
                    generated_tokens = []
                    
                    for i in range(max_gen_len):
                        # Process through model
                        if i == 0:
                            logits = model(tokens, 0)
                            # Mark end of first token generation
                            first_token_end.record()
                        else:
                            logits = model(tokens[:, -1:], i)
                        
                        # Apply temperature scaling
                        logits = logits[:, -1, :] / max(temperature, 1e-5)
                        
                        # Apply repetition penalty if enabled
                        if repetition_penalty > 1.0:
                            for token_id in set(tokens[0].tolist()):
                                logits[0, token_id] /= repetition_penalty
                        
                        # Apply top-p sampling
                        probs = torch.softmax(logits, dim=-1)
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        probs[indices_to_remove] = 0
                        
                        # Sample from filtered distribution
                        next_token = torch.multinomial(probs, num_samples=1)
                        
                        # Stop if EOS token
                        if next_token.item() == tokenizer.eos_id:
                            break
                        
                        # Save generated token
                        generated_tokens.append(next_token.item())
                        
                        # Update input for next iteration
                        tokens = torch.cat([tokens, next_token], dim=1)
                        
                        # Free memory periodically
                        if i % 5 == 0:
                            torch.cuda.empty_cache()
                        
                        # Step profiler
                        prof.step()
                
                # Print profiling summary
                print("\nProfiling Summary (top 10 operations by CUDA time):")
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        else:
            # Standard generation without profiling
            with torch.no_grad():
                generated_tokens = []
                
                for i in range(max_gen_len):
                    # Process through model
                    if i == 0:
                        logits = model(tokens, 0)
                        # Mark end of first token generation
                        first_token_end.record()
                    else:
                        logits = model(tokens[:, -1:], i)
                    
                    # Apply temperature scaling
                    logits = logits[:, -1, :] / max(temperature, 1e-5)
                    
                    # Apply repetition penalty if enabled
                    if repetition_penalty > 1.0:
                        for token_id in set(tokens[0].tolist()):
                            logits[0, token_id] /= repetition_penalty
                    
                    # Apply top-p sampling
                    probs = torch.softmax(logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    probs[indices_to_remove] = 0
                    
                    # Sample from filtered distribution
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Stop if EOS token
                    if next_token.item() == tokenizer.eos_id:
                        break
                    
                    # Save generated token
                    generated_tokens.append(next_token.item())
                    
                    # Update input for next iteration
                    tokens = torch.cat([tokens, next_token], dim=1)
                    
                    # Free memory periodically
                    if i % 5 == 0:
                        torch.cuda.empty_cache()
                    
                    # Print progress
                    if i % 10 == 0:
                        print(f"Generated {i} tokens...")
    except Exception as e:
        print(f"Error during generation: {e}")
        raise

    # End timing
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
    first_token_time = first_token_start.elapsed_time(first_token_end) / 1000  # Convert to seconds

    # Calculate metrics
    mem_peak = torch.cuda.max_memory_allocated()
    mem_used = mem_peak - mem_before
    output_length = len(generated_tokens)
    tokens_per_second = output_length / elapsed_time if elapsed_time > 0 else 0

    # Decode generated tokens
    try:
        output_text = tokenizer.decode([t for t in generated_tokens])
    except Exception as e:
        print(f"Error decoding tokens: {e}")
        output_text = f"[Error decoding: {e}]"

    print(f"Generated {output_length} tokens in {elapsed_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
    print(f"First token generated in {first_token_time:.4f}s")
    print(f"Memory used: {mem_used / (1024 * 1024):.2f} MB")
    print(f"Output: {output_text[:100]}{'...' if len(output_text) > 100 else ''}")

    # Create basic result dictionary
    result = {
        "latency": elapsed_time,
        "first_token_time": first_token_time,
        "memory_used": mem_used / (1024 * 1024),  # Convert to MB
        "output_length": output_length,
        "tokens_per_second": tokens_per_second,
        "input_length": input_length,
        "total_tokens": input_length + output_length,
        "output_text": output_text[:500] + "..." if len(output_text) > 500 else output_text,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty
    }
    
    # Add extended metrics
    extended_metrics = calculate_extended_metrics(result)
    result.update(extended_metrics)
    
    return result

def run_benchmarks(args):
    """Run benchmarks with memory optimizations."""
    # Initialize Weights & Biases if requested
    if hasattr(args, 'use_wandb') and args.use_wandb:
        try:
            import wandb
            use_wandb = True
            wandb.init(
                project="llama2-optimization",
                name=f"benchmark_{args.model_size}",
                config=vars(args)
            )
            print("Weights & Biases initialized for experiment tracking")
        except ImportError:
            use_wandb = False
            print("Weights & Biases not installed. Run 'pip install wandb' to enable tracking.")
    else:
        use_wandb = False
        
    # Update world size if using multiple GPUs
    if hasattr(args, 'num_gpus'):
        os.environ["WORLD_SIZE"] = str(args.num_gpus)
        print(f"Set world size to {args.num_gpus}")
    
    # Load the model
    model, tokenizer = load_model(args.model_path, args.tokenizer_path, args.model_size, args)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Store results
    all_results = []

    # Run benchmarks for each prompt
    for prompt_idx, prompt in enumerate(TEST_PROMPTS):
        print(f"Benchmarking prompt {prompt_idx+1}/{len(TEST_PROMPTS)}...")

        prompt_results = []
        for run in range(args.num_runs):
            print(f"  Run {run+1}/{args.num_runs}")
            try:
                result = benchmark_inference(model, tokenizer, prompt, args)
                result["prompt_idx"] = prompt_idx
                result["prompt"] = prompt[:100] + "..." if len(prompt) > 100 else prompt
                result["run"] = run
                prompt_results.append(result)
                
                # Clear memory between runs
                torch.cuda.empty_cache()
                gc.collect()
                
                # Log to W&B if enabled
                if use_wandb:
                    wandb.log({
                        "run": run,
                        "prompt_idx": prompt_idx,
                        "latency": result["latency"],
                        "tokens_per_second": result["tokens_per_second"],
                        "memory_used": result["memory_used"],
                        "output_length": result["output_length"],
                        "per_token_latency": result["per_token_latency"],
                        "first_token_latency": result["first_token_latency"],
                        "memory_efficiency": result["memory_efficiency"]
                    })
            except Exception as e:
                print(f"Error in run {run+1}: {e}")
                continue

        if prompt_results:
            # Calculate statistics
            latencies = [r["latency"] for r in prompt_results]
            memories = [r["memory_used"] for r in prompt_results]
            tokens_per_second = [r["tokens_per_second"] for r in prompt_results]
            first_token_latencies = [r["first_token_latency"] for r in prompt_results]

            print(f"  Average latency: {np.mean(latencies):.3f}s ± {np.std(latencies):.3f}")
            print(f"  Average memory: {np.mean(memories):.2f}MB ± {np.std(memories):.2f}")
            print(f"  Average tokens/sec: {np.mean(tokens_per_second):.2f} ± {np.std(tokens_per_second):.2f}")
            print(f"  Average first token latency: {np.mean(first_token_latencies):.4f}s ± {np.std(first_token_latencies):.4f}")

            all_results.extend(prompt_results)
        else:
            print(f"  No successful runs for prompt {prompt_idx+1}")

    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_file = os.path.join(args.output_dir, f"benchmark_{args.model_size}.csv")
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")

        # Save configuration
        config = vars(args)
        config_file = os.path.join(args.output_dir, f"config_{args.model_size}.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {config_file}")

        # Create summary plot
        try:
            fig = create_summary_plot(results_df, args)
            # Log to W&B if enabled
            if use_wandb:
                wandb.log({"summary_plot": wandb.Image(fig)})
        except Exception as e:
            print(f"Error creating summary plot: {e}")
        
        # Create detailed analysis plots
        try:
            fig = create_detailed_analysis_plots(results_df, args)
            # Log to W&B if enabled
            if use_wandb:
                wandb.log({"detailed_analysis": wandb.Image(fig)})
        except Exception as e:
            print(f"Error creating detailed analysis plots: {e}")
        
        # Log overall metrics to W&B
        if use_wandb:
            wandb.log({
                "avg_latency": np.mean([r["latency"] for r in all_results]),
                "avg_memory": np.mean([r["memory_used"] for r in all_results]),
                "avg_tokens_per_second": np.mean([r["tokens_per_second"] for r in all_results]),
                "std_latency": np.std([r["latency"] for r in all_results]),
                "std_tokens_per_second": np.std([r["tokens_per_second"] for r in all_results]),
                "avg_first_token_latency": np.mean([r["first_token_latency"] for r in all_results]),
                "avg_per_token_latency": np.mean([r["per_token_latency"] for r in all_results]),
                "avg_memory_efficiency": np.mean([r["memory_efficiency"] for r in all_results])
            })

        return results_df
    else:
        print("No results collected.")
        return None

def create_summary_plot(results_df, args):
    """Create summary plot of benchmark results."""
    plt.figure(figsize=(15, 10))
    
    # Calculate average metrics for each prompt
    summary = results_df.groupby("prompt_idx").agg({
        "latency": ["mean", "std"],
        "tokens_per_second": ["mean", "std"],
        "memory_used": ["mean", "std"],
        "first_token_latency": ["mean", "std"],
        "output_length": ["mean", "std"],
        "prompt": "first"
    })

    # Plot latency
    plt.subplot(2, 3, 1)
    bars = plt.bar(range(len(summary)), summary[("latency", "mean")])
    plt.errorbar(range(len(summary)), summary[("latency", "mean")], 
                yerr=summary[("latency", "std")], fmt='none', ecolor='black', capsize=5)
    plt.xlabel("Prompt")
    plt.ylabel("Latency (s)")
    plt.title(f"LLaMA 2 {args.model_size} Inference Latency")
    plt.grid(axis="y", alpha=0.3)
    
    # Add numeric labels above bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{summary[("latency", "mean")].iloc[i]:.2f}s',
                ha='center', va='bottom', rotation=0, fontsize=8)

    # Plot tokens per second
    plt.subplot(2, 3, 2)
    bars = plt.bar(range(len(summary)), summary[("tokens_per_second", "mean")])
    plt.errorbar(range(len(summary)), summary[("tokens_per_second", "mean")], 
                yerr=summary[("tokens_per_second", "std")], fmt='none', ecolor='black', capsize=5)
    plt.xlabel("Prompt")
    plt.ylabel("Tokens per Second")
    plt.title(f"LLaMA 2 {args.model_size} Throughput")
    plt.grid(axis="y", alpha=0.3)
    
    # Add numeric labels above bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{summary[("tokens_per_second", "mean")].iloc[i]:.2f}',
                ha='center', va='bottom', rotation=0, fontsize=8)
    
    # Plot memory usage
    plt.subplot(2, 3, 3)
    bars = plt.bar(range(len(summary)), summary[("memory_used", "mean")])
    plt.errorbar(range(len(summary)), summary[("memory_used", "mean")], 
                yerr=summary[("memory_used", "std")], fmt='none', ecolor='black', capsize=5)
    plt.xlabel("Prompt")
    plt.ylabel("Memory (MB)")
    plt.title(f"LLaMA 2 {args.model_size} Memory Usage")
    plt.grid(axis="y", alpha=0.3)
    
    # Add numeric labels above bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{summary[("memory_used", "mean")].iloc[i]:.1f}MB',
                ha='center', va='bottom', rotation=0, fontsize=8)
    
    # Plot first token latency
    plt.subplot(2, 3, 4)
    bars = plt.bar(range(len(summary)), summary[("first_token_latency", "mean")])
    plt.errorbar(range(len(summary)), summary[("first_token_latency", "mean")], 
                yerr=summary[("first_token_latency", "std")], fmt='none', ecolor='black', capsize=5)
    plt.xlabel("Prompt")
    plt.ylabel("First Token Latency (s)")
    plt.title(f"LLaMA 2 {args.model_size} First Token Latency")
    plt.grid(axis="y", alpha=0.3)
    
    # Add numeric labels above bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{summary[("first_token_latency", "mean")].iloc[i]:.3f}s',
                ha='center', va='bottom', rotation=0, fontsize=8)
    
    # Plot output lengths
    plt.subplot(2, 3, 5)
    bars = plt.bar(range(len(summary)), summary[("output_length", "mean")])
    plt.errorbar(range(len(summary)), summary[("output_length", "mean")], 
                yerr=summary[("output_length", "std")], fmt='none', ecolor='black', capsize=5)
    plt.xlabel("Prompt")
    plt.ylabel("Output Length (tokens)")
    plt.title(f"LLaMA 2 {args.model_size} Output Length")
    plt.grid(axis="y", alpha=0.3)
    
    # Add numeric labels above bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{summary[("output_length", "mean")].iloc[i]:.1f}',
                ha='center', va='bottom', rotation=0, fontsize=8)
    
    # Add a table of prompt descriptions
    ax = plt.subplot(2, 3, 6)
    ax.axis('off')
    prompt_table = []
    for i in range(len(summary)):
        prompt_text = summary[("prompt", "first")].iloc[i]
        if len(prompt_text) > 40:
            prompt_text = prompt_text[:37] + "..."
        prompt_table.append([f"Prompt {i}", prompt_text])
    
    table = ax.table(cellText=prompt_table, colLabels=["ID", "Prompt Text"], 
                    loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    plt.suptitle(f"LLaMA 2 {args.model_size} Benchmark Summary", fontsize=16, y=1.02)
    
    plot_file = os.path.join(args.output_dir, f"benchmark_{args.model_size}.png")
    plt.savefig(plot_file, bbox_inches='tight')
    print(f"Saved summary plot to {plot_file}")
    plt.close()
    
    return plt.gcf()  # Return the figure for W&B logging

def create_detailed_analysis_plots(results_df, args):
    """Create more detailed analysis plots of benchmark results."""
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Latency vs Output Length with regression line
    ax = axs[0, 0]
    ax.scatter(results_df["output_length"], results_df["latency"], alpha=0.7)
    
    # Add regression line
    if len(results_df) > 1:  # Need at least 2 points for regression
        z = np.polyfit(results_df["output_length"], results_df["latency"], 1)
        p = np.poly1d(z)
        x_range = np.linspace(results_df["output_length"].min(), results_df["output_length"].max(), 100)
        ax.plot(x_range, p(x_range), "r--", linewidth=2)
        ax.text(0.05, 0.95, f"y = {z[0]:.4f}x + {z[1]:.4f}", transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    ax.set_xlabel("Output Length (tokens)")
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Latency vs Output Length")
    ax.grid(True, alpha=0.3)
    
    # 2. Tokens per Second vs Output Length
    ax = axs[0, 1]
    ax.scatter(results_df["output_length"], results_df["tokens_per_second"], alpha=0.7)
    ax.set_xlabel("Output Length (tokens)")
    ax.set_ylabel("Tokens per Second")
    ax.set_title("Throughput vs Output Length")
    ax.grid(True, alpha=0.3)
    
    # 3. Memory Usage vs Output Length
    ax = axs[0, 2]
    ax.scatter(results_df["output_length"], results_df["memory_used"], alpha=0.7)
    
    # Add regression line for memory usage
    if len(results_df) > 1:
        z = np.polyfit(results_df["output_length"], results_df["memory_used"], 1)
        p = np.poly1d(z)
        x_range = np.linspace(results_df["output_length"].min(), results_df["output_length"].max(), 100)
        ax.plot(x_range, p(x_range), "r--", linewidth=2)
        ax.text(0.05, 0.95, f"y = {z[0]:.4f}x + {z[1]:.4f}", transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    ax.set_xlabel("Output Length (tokens)")
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_title("Memory Usage vs Output Length")
    ax.grid(True, alpha=0.3)
    
    # 4. Input Length vs Output Length
    ax = axs[1, 0]
    ax.scatter(results_df["input_length"], results_df["output_length"], alpha=0.7)
    ax.set_xlabel("Input Length (tokens)")
    ax.set_ylabel("Output Length (tokens)")
    ax.set_title("Input vs Output Length")
    ax.grid(True, alpha=0.3)
    
    # 5. Per-token Latency Distribution
    ax = axs[1, 1]
    per_token_latency = results_df["per_token_latency"]
    ax.hist(per_token_latency, bins=10, alpha=0.7)
    ax.axvline(per_token_latency.mean(), color='r', linestyle='--', linewidth=2)
    ax.text(0.95, 0.95, f"Mean: {per_token_latency.mean():.4f}s/token", transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    ax.set_xlabel("Latency per Output Token (s/token)")
    ax.set_ylabel("Frequency")
    ax.set_title("Per-token Latency Distribution")
    ax.grid(True, alpha=0.3)
    
    # 6. First Token vs Total Latency
    ax = axs[1, 2]
    ax.scatter(results_df["first_token_latency"], results_df["latency"], alpha=0.7)
    ax.set_xlabel("First Token Latency (s)")
    ax.set_ylabel("Total Latency (s)")
    ax.set_title("First Token vs Total Latency")
    ax.grid(True, alpha=0.3)
    
    # Add regression line
    if len(results_df) > 1:
        z = np.polyfit(results_df["first_token_latency"], results_df["latency"], 1)
        p = np.poly1d(z)
        x_range = np.linspace(results_df["first_token_latency"].min(), results_df["first_token_latency"].max(), 100)
        ax.plot(x_range, p(x_range), "r--", linewidth=2)
    
    plt.tight_layout()
    plt.suptitle(f"LLaMA 2 {args.model_size} Detailed Performance Analysis", fontsize=16, y=1.02)
    
    plot_file = os.path.join(args.output_dir, f"detailed_analysis_{args.model_size}.png")
    plt.savefig(plot_file, bbox_inches='tight')
    print(f"Saved detailed analysis to {plot_file}")
    plt.close()
    
    return fig  # Return the figure for W&B logging

def run_comparison_benchmarks(args):
    """Run benchmarks with different optimization settings for comparison."""
    results = {}
    
    # Run baseline (no optimizations)
    print("\n=== Running baseline (no optimizations) ===")
    baseline_args = argparse.Namespace(**vars(args))
    baseline_args.use_quantization = False
    baseline_args.use_flash_attention = False
    baseline_args.num_gpus = 1
    baseline_args.num_runs = 2  # Use fewer runs for faster comparison
    print(f"Baseline args: {vars(baseline_args)}")
    results['baseline'] = run_benchmarks(baseline_args)
    
    # Run with flash attention if available
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        print("\n=== Running with flash attention ===")
        flash_args = argparse.Namespace(**vars(args))
        flash_args.use_flash_attention = True
        flash_args.use_quantization = False
        flash_args.num_gpus = 1
        flash_args.num_runs = 2  # Use fewer runs for faster comparison
        print(f"Flash attention args: {vars(flash_args)}")
        results['flash_attention'] = run_benchmarks(flash_args)
    
    # Run with quantization
    print("\n=== Running with quantization ===")
    quant_args = argparse.Namespace(**vars(args))
    quant_args.use_quantization = True
    quant_args.use_flash_attention = False
    quant_args.num_gpus = 1
    quant_args.num_runs = 2  # Use fewer runs for faster comparison
    print(f"Quantization args: {vars(quant_args)}")
    results['quantization'] = run_benchmarks(quant_args)
    
    # Try running with both optimizations if flash attention is available
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        print("\n=== Running with flash attention + quantization ===")
        combined_args = argparse.Namespace(**vars(args))
        combined_args.use_flash_attention = True
        combined_args.use_quantization = True
        combined_args.num_gpus = 1
        combined_args.num_runs = 2  # Use fewer runs for faster comparison
        print(f"Combined optimizations args: {vars(combined_args)}")
        results['combined'] = run_benchmarks(combined_args)
    
    # Try distributed inference if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"\n=== Running with {torch.cuda.device_count()} GPUs ===")
        multi_gpu_args = argparse.Namespace(**vars(args))
        multi_gpu_args.num_gpus = torch.cuda.device_count()
        multi_gpu_args.use_flash_attention = False
        multi_gpu_args.use_quantization = False
        multi_gpu_args.num_runs = 2  # Use fewer runs for faster comparison
        print(f"Multi-GPU args: {vars(multi_gpu_args)}")
        results['multi_gpu'] = run_benchmarks(multi_gpu_args)
    
    # Create comparison plots and save results
    create_comparison_plots(results, args)
    
    return results

def create_comparison_plots(results_dict, args):
    """Create plots comparing different optimization techniques."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Make sure we have valid results
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if not valid_results:
        print("No valid results for comparison plotting")
        return
    
    # Extract metrics for comparison
    labels = list(valid_results.keys())
    
    # Calculate averages for each optimization technique
    avg_metrics = {}
    for technique, df in valid_results.items():
        if df is not None and not df.empty:
            avg_metrics[technique] = {
                "latency": df["latency"].mean(),
                "tokens_per_second": df["tokens_per_second"].mean(),
                "memory_used": df["memory_used"].mean(),
                "first_token_latency": df["first_token_latency"].mean() if "first_token_latency" in df.columns else 0
            }
    
    # Extract metrics into lists for plotting
    latencies = [avg_metrics[k]["latency"] for k in labels]
    throughputs = [avg_metrics[k]["tokens_per_second"] for k in labels]
    memories = [avg_metrics[k]["memory_used"] for k in labels]
    first_token_latencies = [avg_metrics[k]["first_token_latency"] for k in labels]
    
    # Create plots
    plt.figure(figsize=(18, 12))
    
    # Latency comparison
    plt.subplot(2, 2, 1)
    bars = plt.bar(labels, latencies)
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{latencies[i]:.2f}s', ha='center', va='bottom', rotation=0)
    plt.ylabel("Latency (s)")
    plt.title("Average Latency Comparison")
    plt.grid(axis='y', alpha=0.3)
    
    # Throughput comparison
    plt.subplot(2, 2, 2)
    bars = plt.bar(labels, throughputs)
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{throughputs[i]:.2f}', ha='center', va='bottom', rotation=0)
    plt.ylabel("Tokens per Second")
    plt.title("Average Throughput Comparison")
    plt.grid(axis='y', alpha=0.3)
    
    # Memory usage comparison
    plt.subplot(2, 2, 3)
    bars = plt.bar(labels, memories)
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{memories[i]:.1f}MB', ha='center', va='bottom', rotation=0)
    plt.ylabel("Memory Usage (MB)")
    plt.title("Average Memory Usage Comparison")
    plt.grid(axis='y', alpha=0.3)
    
    # First token latency comparison
    plt.subplot(2, 2, 4)
    bars = plt.bar(labels, first_token_latencies)
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{first_token_latencies[i]:.3f}s', ha='center', va='bottom', rotation=0)
    plt.ylabel("First Token Latency (s)")
    plt.title("Average First Token Latency Comparison")
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f"LLaMA 2 {args.model_size} Optimization Techniques Comparison", fontsize=16, y=1.02)
    
    comparison_file = os.path.join(args.output_dir, f"optimization_comparison_{args.model_size}.png")
    plt.savefig(comparison_file, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {comparison_file}")
    
    # Save comparison data
    comparison_df = pd.DataFrame({
        'optimization': labels,
        'avg_latency': latencies,
        'avg_throughput': throughputs,
        'avg_memory': memories,
        'avg_first_token_latency': first_token_latencies
    })
    comparison_csv = os.path.join(args.output_dir, f"optimization_comparison_{args.model_size}.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"Comparison data saved to {comparison_csv}")
    
    # Log to W&B if enabled
    if hasattr(args, 'use_wandb') and args.use_wandb:
        try:
            import wandb
            wandb.log({"optimization_comparison": wandb.Image(comparison_file)})
            for technique, metrics in avg_metrics.items():
                wandb.log({f"{technique}_latency": metrics["latency"],
                           f"{technique}_throughput": metrics["tokens_per_second"],
                           f"{technique}_memory": metrics["memory_used"],
                           f"{technique}_first_token_latency": metrics["first_token_latency"]})
        except ImportError:
            pass

def batch_size_sweep(args, max_batch=None):
    """Run benchmarks with different batch sizes to find optimal throughput."""
    results = []
    
    # Use provided max_batch or default to args.max_batch_sweep
    if max_batch is None:
        if hasattr(args, 'max_batch_sweep'):
            max_batch = args.max_batch_sweep
        else:
            max_batch = 8
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32]
    batch_sizes = [b for b in batch_sizes if b <= max_batch]
    
    for batch_size in batch_sizes:
        print(f"\n=== Testing batch size: {batch_size} ===")
        
        # Create a copy of args with updated batch size
        batch_args = argparse.Namespace(**vars(args))
        batch_args.max_batch_size = batch_size
        # Use fewer runs and prompts for faster sweeping
        batch_args.num_runs = 2
        
        # Run benchmark with this batch size
        try:
            # For batch sizes > 1, we need to modify the benchmarking approach
            # Here we create a simple benchmark that replicates the prompt
            model, tokenizer = load_model(batch_args.model_path, batch_args.tokenizer_path, batch_args.model_size, batch_args)
            
            # Run a single benchmark with the batched prompt
            prompt = TEST_PROMPTS[0]  # Use first prompt for all batch testing
            print(f"Using prompt: {prompt}")
            
            # Clear CUDA cache before inference
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            
            # Tokenize the prompt
            tokens = tokenizer.encode(prompt, bos=True, eos=False)
            tokens = torch.tensor(tokens).cuda()
            
            # Batch the tokens
            batched_tokens = tokens.unsqueeze(0).repeat(batch_size, 1)
            input_length = tokens.shape[0]
            
            # Setup timing
            start = time.time()
            torch.cuda.synchronize()
            
            # Forward pass through model
            with torch.no_grad():
                logits = model(batched_tokens, 0)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            # Calculate memory usage
            mem_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            
            # Calculate tokens per second for the batch
            tokens_per_second = (batch_size * input_length) / elapsed
            
            # Store results
            batch_result = {
                'batch_size': batch_size,
                'latency': elapsed,
                'tokens_per_second': tokens_per_second,
                'memory_used': mem_peak,
                'total_tokens': batch_size * input_length
            }
            
            results.append(batch_result)
            print(f"Batch size {batch_size}: {batch_result['tokens_per_second']:.2f} tokens/sec, {batch_result['memory_used']:.2f} MB")
            
            # Clean up
            del model, tokenizer, tokens, batched_tokens, logits
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error testing batch size {batch_size}: {e}")
            # If we hit an OOM error, don't try larger batch sizes
            if "CUDA out of memory" in str(e):
                print(f"Stopping batch size sweep due to OOM at batch size {batch_size}")
                break
    
    # Create and save batch size analysis
    if results:
        plot_batch_size_results(results, args)
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        batch_file = os.path.join(args.output_dir, f"batch_size_results_{args.model_size}.csv")
        results_df.to_csv(batch_file, index=False)
        print(f"Batch size results saved to {batch_file}")
        
        # Log to W&B if enabled
        if hasattr(args, 'use_wandb') and args.use_wandb:
            try:
                import wandb
                batch_plot_file = os.path.join(args.output_dir, f"batch_size_analysis_{args.model_size}.png")
                wandb.log({"batch_size_analysis": wandb.Image(batch_plot_file)})
                for result in results:
                    wandb.log({f"batch_{result['batch_size']}_latency": result["latency"],
                              f"batch_{result['batch_size']}_throughput": result["tokens_per_second"],
                              f"batch_{result['batch_size']}_memory": result["memory_used"]})
            except ImportError:
                pass
    
    return results

def plot_batch_size_results(results, args):
    """Plot the impact of batch size on performance metrics."""
    # Extract data
    batch_sizes = [r['batch_size'] for r in results]
    latencies = [r['latency'] for r in results]
    throughputs = [r['tokens_per_second'] for r in results]
    memories = [r['memory_used'] for r in results]
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Throughput vs Batch Size
    plt.subplot(2, 2, 1)
    plt.plot(batch_sizes, throughputs, 'o-', linewidth=2)
    for i, batch_size in enumerate(batch_sizes):
        plt.text(batch_size, throughputs[i] + 2, f'{throughputs[i]:.1f}', ha='center')
    plt.xlabel('Batch Size')
    plt.ylabel('Tokens per Second')
    plt.title('Throughput vs Batch Size')
    plt.grid(True)
    
    # Latency vs Batch Size
    plt.subplot(2, 2, 2)
    plt.plot(batch_sizes, latencies, 'o-', linewidth=2)
    for i, batch_size in enumerate(batch_sizes):
        plt.text(batch_size, latencies[i] + 0.01, f'{latencies[i]:.3f}s', ha='center')
    plt.xlabel('Batch Size')
    plt.ylabel('Latency (s)')
    plt.title('Latency vs Batch Size')
    plt.grid(True)
    
    # Memory vs Batch Size
    plt.subplot(2, 2, 3)
    plt.plot(batch_sizes, memories, 'o-', linewidth=2)
    for i, batch_size in enumerate(batch_sizes):
        plt.text(batch_size, memories[i] + 10, f'{memories[i]:.1f}MB', ha='center')
    plt.xlabel('Batch Size')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage vs Batch Size')
    plt.grid(True)
    
    # Efficiency: Throughput per Batch Size
    plt.subplot(2, 2, 4)
    efficiency = [t / b for t, b in zip(throughputs, batch_sizes)]
    plt.plot(batch_sizes, efficiency, 'o-', linewidth=2)
    for i, batch_size in enumerate(batch_sizes):
        plt.text(batch_size, efficiency[i] + 0.2, f'{efficiency[i]:.1f}', ha='center')
    plt.xlabel('Batch Size')
    plt.ylabel('Tokens/Second per Batch Unit')
    plt.title('Efficiency vs Batch Size')
    plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle(f"LLaMA 2 {args.model_size} Batch Size Analysis", fontsize=16, y=1.02)
    
    plot_file = os.path.join(args.output_dir, f"batch_size_analysis_{args.model_size}.png")
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    
    print(f"Batch size analysis plot saved to {plot_file}")

def analyze_memory_usage(model, tokenizer, prompt, args):
    """Perform detailed analysis of memory usage during inference."""
    print(f"Analyzing memory usage for prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
    
    memory_stats = {}
    
    # Initial state
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    memory_stats["initial"] = {
        "allocated": torch.cuda.memory_allocated() / (1024**2),
        "reserved": torch.cuda.memory_reserved() / (1024**2)
    }
    
    # After tokenization
    tokens = tokenizer.encode(prompt, bos=True, eos=False)
    tokens_tensor = torch.tensor(tokens).cuda().unsqueeze(0)
    memory_stats["after_tokenization"] = {
        "allocated": torch.cuda.memory_allocated() / (1024**2),
        "reserved": torch.cuda.memory_reserved() / (1024**2),
        "input_tokens": len(tokens)
    }
    
    # First forward pass
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        logits = model(tokens_tensor, 0)
    
    memory_stats["first_forward_pass"] = {
        "allocated": torch.cuda.memory_allocated() / (1024**2),
        "reserved": torch.cuda.memory_reserved() / (1024**2),
        "peak_allocated": torch.cuda.max_memory_allocated() / (1024**2)
    }
    
    # Generation steps (sample a few steps)
    generated_tokens = []
    memory_per_step = []
    
    with torch.no_grad():
        for i in range(min(5, args.max_gen_len)):  # Sample first 5 steps or less
            # Apply temperature scaling and sample
            curr_logits = logits[:, -1, :] / max(args.temperature, 1e-5)
            probs = torch.softmax(curr_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Save generated token
            generated_tokens.append(next_token.item())
            
            # Update input for next step
            tokens_tensor = torch.cat([tokens_tensor, next_token], dim=1)
            
            # Next forward pass
            torch.cuda.reset_peak_memory_stats()
            logits = model(tokens_tensor[:, -1:], i+1)
            
            # Record memory
            memory_per_step.append({
                "step": i,
                "allocated": torch.cuda.memory_allocated() / (1024**2),
                "reserved": torch.cuda.memory_reserved() / (1024**2),
                "peak_allocated": torch.cuda.max_memory_allocated() / (1024**2)
            })
    
    memory_stats["generation_steps"] = memory_per_step
    
    # Final state
    memory_stats["final"] = {
        "allocated": torch.cuda.memory_allocated() / (1024**2),
        "reserved": torch.cuda.memory_reserved() / (1024**2),
        "output_tokens": len(generated_tokens)
    }
    
    # Free memory
    del tokens_tensor
    del logits
    torch.cuda.empty_cache()
    gc.collect()
    
    # Print summary
    print("\nMemory Analysis Summary:")
    print(f"Initial allocated: {memory_stats['initial']['allocated']:.2f} MB")
    print(f"After tokenization: {memory_stats['after_tokenization']['allocated']:.2f} MB")
    print(f"First forward pass: {memory_stats['first_forward_pass']['allocated']:.2f} MB")
    print(f"First forward pass peak: {memory_stats['first_forward_pass']['peak_allocated']:.2f} MB")
    print(f"Final allocated: {memory_stats['final']['allocated']:.2f} MB")
    
    # Create a plot
    try:
        create_memory_analysis_plot(memory_stats, args)
    except Exception as e:
        print(f"Error creating memory analysis plot: {e}")
    
    # Save memory stats to file
    memory_file = os.path.join(args.output_dir, f"memory_analysis_{args.model_size}.json")
    with open(memory_file, 'w') as f:
        json.dump(memory_stats, f, indent=2)
    print(f"Memory analysis saved to {memory_file}")
    
    # Log to W&B if enabled
    if hasattr(args, 'use_wandb') and args.use_wandb:
        try:
            import wandb
            memory_plot_file = os.path.join(args.output_dir, f"memory_analysis_{args.model_size}.png")
            wandb.log({"memory_analysis": wandb.Image(memory_plot_file)})
            # Log key memory metrics
            wandb.log({
                "initial_memory": memory_stats['initial']['allocated'],
                "tokenization_memory": memory_stats['after_tokenization']['allocated'],
                "first_forward_memory": memory_stats['first_forward_pass']['allocated'],
                "peak_memory": memory_stats['first_forward_pass']['peak_allocated'],
                "final_memory": memory_stats['final']['allocated']
            })
        except ImportError:
            pass
    
    return memory_stats

def create_memory_analysis_plot(memory_stats, args):
    """Create visualization of memory usage during inference."""
    plt.figure(figsize=(12, 6))
    
    # Extract step data
    steps = [0]  # Initial point
    allocated = [memory_stats["initial"]["allocated"]]
    reserved = [memory_stats["initial"]["reserved"]]
    
    # Add tokenization point
    steps.append(1)
    allocated.append(memory_stats["after_tokenization"]["allocated"])
    reserved.append(memory_stats["after_tokenization"]["reserved"])
    
    # Add first forward pass
    steps.append(2)
    allocated.append(memory_stats["first_forward_pass"]["allocated"])
    reserved.append(memory_stats["first_forward_pass"]["reserved"])
    
    # Add generation steps
    for i, step in enumerate(memory_stats["generation_steps"]):
        steps.append(i + 3)  # Offset by 3 for initial, tokenization, and first pass
        allocated.append(step["allocated"])
        reserved.append(step["reserved"])
    
    # Plot memory usage
    plt.plot(steps, allocated, 'o-', label='Allocated Memory (MB)', linewidth=2)
    plt.plot(steps, reserved, 's-', label='Reserved Memory (MB)', linewidth=2)
    
    # Add peak memory from first forward pass
    plt.axhline(y=memory_stats["first_forward_pass"]["peak_allocated"], 
                color='r', linestyle='--', label='Peak Allocated (First Pass)')
    
    # Add labels for each point
    for i, s in enumerate(steps):
        plt.text(s, allocated[i] + 10, f'{allocated[i]:.1f}MB', ha='center')
    
    # Add step labels
    plt.xticks(steps, ['Initial', 'Tokenization', 'First Pass'] + 
               [f'Gen {i+1}' for i in range(len(memory_stats["generation_steps"]))])
    plt.xlabel('Inference Stage')
    plt.ylabel('Memory (MB)')
    plt.title(f'LLaMA 2 {args.model_size} Memory Usage During Inference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    os.makedirs(args.output_dir, exist_ok=True)
    plot_file = os.path.join(args.output_dir, f"memory_analysis_{args.model_size}.png")
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Memory analysis plot saved to {plot_file}")

if __name__ == "__main__":
    args = parse_args()
    print(f"Benchmarking LLaMA 2 {args.model_size}")
    
    # Update world size based on num_gpus argument
    if hasattr(args, 'num_gpus'):
        world_size = args.num_gpus
        os.environ["WORLD_SIZE"] = str(world_size)
    
    try:
        # Choose which benchmark to run based on command line flags
        if hasattr(args, 'run_comparison') and args.run_comparison:
            print("Running comparison of optimization techniques...")
            results = run_comparison_benchmarks(args)
        elif hasattr(args, 'analyze_memory') and args.analyze_memory:
            print("Running memory usage analysis...")
            # Load model once
            model, tokenizer = load_model(args.model_path, args.tokenizer_path, args.model_size, args)
            # Analyze memory on a single prompt
            memory_results = analyze_memory_usage(model, tokenizer, TEST_PROMPTS[0], args)
        elif hasattr(args, 'batch_sweep') and args.batch_sweep:
            print("Running batch size sweep...")
            results = batch_size_sweep(args, args.max_batch_sweep if hasattr(args, 'max_batch_sweep') else 8)
        else:
            # Run standard benchmark
            print("Running standard benchmark...")
            results = run_benchmarks(args)
        
        print("Benchmarking complete!")
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Close W&B if it was used
    if hasattr(args, 'use_wandb') and args.use_wandb:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass