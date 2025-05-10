"""
LLaMA 2 Bayesian Optimization for Inference Parameters (Enhanced Version)
"""

import os
import time
import json
import argparse
import sys
import gc
from pathlib import Path
import random
from tqdm import tqdm

# Version checks and dependency handling
print("Setting up dependencies...")
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    
    # Check numpy version compatibility 
    np_version = tuple(map(int, np.__version__.split('.')))
    if np_version[0] != 1 or np_version[1] < 23:
        print(f"Warning: NumPy version {np.__version__} may cause compatibility issues. Version 1.23.x is recommended.")

    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    import matplotlib.pyplot as plt
    
    # Import Bayesian Optimization libraries with proper error handling
    try:
        from bayes_opt import BayesianOptimization
        from bayes_opt.logger import JSONLogger
        from bayes_opt.event import Events
        from bayes_opt.util import load_logs, UtilityFunction
        print("Successfully imported bayes_opt")
    except ImportError as e:
        print(f"Error importing bayes_opt: {e}")
        print("Try installing dependencies in this order: numpy==1.23.5, scikit-learn==1.2.2, bayesian-optimization")
        sys.exit(1)
except ImportError as e:
    print(f"Error importing base libraries: {e}")
    sys.exit(1)

# Initialize PyTorch distributed environment
if not torch.distributed.is_initialized():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    
    try:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=1,
            rank=0
        )
        print("PyTorch distributed environment initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize PyTorch distributed environment: {e}")
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
try:
    try:
        from llama.llama.model import ModelArgs, Transformer
        from llama.llama.tokenizer import Tokenizer
        print("Imported LLaMA modules from llama.llama")
    except ImportError:
        try:
            from llama.model import ModelArgs, Transformer
            from llama.tokenizer import Tokenizer
            print("Imported LLaMA modules from llama")
        except ImportError:
            raise ImportError("Could not import LLaMA modules. Make sure the path is correct.")
except Exception as e:
    print(f"Fatal error: Failed to import LLaMA modules: {e}")
    sys.exit(1)

# Test prompts for optimization
OPTIMIZATION_PROMPTS = [
    "Explain the concept of machine learning in simple terms.",
    "What are the main differences between Python and JavaScript?",
    "How does blockchain technology work?",
    "What are some effective strategies for time management?",
    "Describe the process of photosynthesis in plants."
]

def parse_args():
    parser = argparse.ArgumentParser(description="Optimize LLaMA 2 inference parameters using Bayesian Optimization")
    parser.add_argument("--model_size", type=str, default="7B", choices=["7B", "13B"],
                        help="Size of LLaMA 2 model to benchmark")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model weights directory")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to the tokenizer model")
    parser.add_argument("--max_gen_len", type=int, default=30,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--n_iter", type=int, default=20,
                        help="Number of iterations for Bayesian optimization")
    parser.add_argument("--initial_points", type=int, default=5,
                        help="Number of initial random points for Bayesian optimization")
    parser.add_argument("--output_dir", type=str, default="optimization_results",
                        help="Directory to save results")
    parser.add_argument("--optimization_target", type=str, default="throughput", 
                        choices=["throughput", "latency", "balanced", "efficiency"],
                        help="Target metric to optimize for")
    parser.add_argument("--max_batch_size", type=int, default=1,
                        help="Maximum batch size")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--continue_optimization", action="store_true",
                        help="Continue optimization from previous logs")
    parser.add_argument("--run_grid_search", action="store_true",
                        help="Run grid search as a baseline comparison")
    parser.add_argument("--acquisition_function", type=str, default="ei",
                        choices=["ei", "ucb", "poi"],
                        help="Acquisition function for Bayesian optimization")
    parser.add_argument("--use_quantization", action="store_true",
                        help="Enable dynamic int8 quantization")
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
        
        # MEMORY OPTIMIZATION: Use smaller batch size and sequence length
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

    # Create and load model with memory optimizations
    try:
        print("Creating transformer model...")
        
        # MEMORY OPTIMIZATION: Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            print("Enabled flash attention")
        
        # Create model
        model = Transformer(params)
        print("Model created successfully")
        
        # Add quantization support if requested
        if hasattr(args, 'use_quantization') and args.use_quantization:
            print("Applying int8 quantization...")
            try:
                from torch.quantization import quantize_dynamic
                model = quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                print("Model quantized to int8")
            except Exception as e:
                print(f"Quantization failed: {e}. Continuing with original model.")
        
        # MEMORY OPTIMIZATION: Load model in half precision
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        
        # MEMORY OPTIMIZATION: Move to GPU with half precision
        print("Moving model to GPU with half precision...")
        model = model.half().cuda()
        model.eval()
        
        # More aggressive memory cleanup
        del checkpoint
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"Error creating/loading model: {e}")
        raise

    return model, tokenizer

def inference_step(model, tokenizer, prompt, temperature, top_p, max_gen_len, repetition_penalty=1.0):
    """Run inference with specified parameters."""
    # Tokenize the prompt
    tokens = tokenizer.encode(prompt, bos=True, eos=False)
    tokens = torch.tensor(tokens).cuda().unsqueeze(0)
    input_length = tokens.shape[1]

    # Clear CUDA cache before inference
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    # Setup timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
    # First forward pass timing (first token latency)
    first_token_start = torch.cuda.Event(enable_timing=True)
    first_token_end = torch.cuda.Event(enable_timing=True)
    first_token_start.record()

    # Generate output
    with torch.no_grad():
        generated_tokens = []
        
        for i in range(max_gen_len):
            # Process input through model
            if i == 0:
                logits = model(tokens, 0)
                first_token_end.record()
                torch.cuda.synchronize()
                first_token_time = first_token_start.elapsed_time(first_token_end) / 1000  # Convert to seconds
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
            
            # Sample from the filtered distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop if end of sequence token is generated
            if next_token.item() == tokenizer.eos_id:
                break
            
            # Append the generated token
            generated_tokens.append(next_token.item())
            
            # Update the input tensor for the next iteration
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Free memory after each step
            if i % 5 == 0:
                torch.cuda.empty_cache()

    # End timing
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds

    # Calculate metrics
    mem_peak = torch.cuda.max_memory_allocated()
    mem_used = mem_peak - mem_before
    output_length = len(generated_tokens)
    tokens_per_second = output_length / elapsed_time if elapsed_time > 0 else 0

    # Decode generated tokens
    output_text = tokenizer.decode([t for t in generated_tokens])

    return {
        "latency": elapsed_time,
        "first_token_time": first_token_time if 'first_token_time' in locals() else 0,
        "memory_used": mem_used / (1024 * 1024),  # Convert to MB
        "output_length": output_length,
        "tokens_per_second": tokens_per_second,
        "input_length": input_length,
        "total_tokens": input_length + output_length,
        "output_text": output_text
    }

def evaluate_quality(output_text, prompt):
    """Evaluate the quality of the generated text using heuristics."""
    # Simple heuristics
    quality_score = 0
    
    # Length - reward longer outputs up to a point
    if len(output_text) > 50:
        quality_score += min(1.0, len(output_text) / 200)
    
    # Repetition penalty - detect repeated n-grams
    words = output_text.split()
    if len(words) > 10:
        bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
        unique_bigrams = len(set(bigrams))
        bigram_diversity = unique_bigrams / len(bigrams)
        quality_score += bigram_diversity
    
    # Relevance to prompt - simple keyword matching
    prompt_words = set(prompt.lower().split())
    output_words = set(output_text.lower().split())
    common_words = prompt_words.intersection(output_words)
    if len(prompt_words) > 0:
        relevance = len(common_words) / len(prompt_words)
        quality_score += relevance
    
    # Normalize to 0-1 range
    return min(quality_score / 3.0, 1.0)

def objective_function(temperature, top_p, repetition_penalty=1.0, max_new_tokens=None, model=None, tokenizer=None, args=None):
    """Enhanced objective function for Bayesian optimization."""
    # Run inference for all test prompts
    results = []
    
    # Ensure parameters are within valid ranges
    temperature = max(0.05, min(2.0, temperature))
    top_p = max(0.1, min(1.0, top_p))
    repetition_penalty = max(1.0, min(1.5, repetition_penalty))
    
    # Use provided max_new_tokens if available, otherwise use args.max_gen_len
    gen_len = max_new_tokens if max_new_tokens is not None else args.max_gen_len
    
    print(f"Testing parameters: temperature={temperature:.3f}, top_p={top_p:.3f}, rep_penalty={repetition_penalty:.2f}, max_tokens={gen_len}")
    
    for prompt in OPTIMIZATION_PROMPTS:
        try:
            # Run the inference
            result = inference_step(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_gen_len=gen_len,
                repetition_penalty=repetition_penalty
            )
            
            # Calculate additional metrics
            result["quality_score"] = evaluate_quality(result["output_text"], prompt)
            result["efficiency"] = result["tokens_per_second"] / result["memory_used"] if result["memory_used"] > 0 else 0
            
            results.append(result)
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error evaluating prompt: {e}")
    
    # Calculate aggregate metrics
    if not results:
        return 0.0  # Return worst possible score if all evaluations failed
    
    avg_metrics = {
        "tokens_per_second": np.mean([r["tokens_per_second"] for r in results]),
        "latency": np.mean([r["latency"] for r in results]),
        "memory_used": np.mean([r["memory_used"] for r in results]),
        "quality_score": np.mean([r.get("quality_score", 0) for r in results]),
        "efficiency": np.mean([r.get("efficiency", 0) for r in results]),
        "first_token_time": np.mean([r.get("first_token_time", 0) for r in results])
    }
    
    # Calculate the final objective value based on optimization target
    if args.optimization_target == "throughput":
        objective_value = avg_metrics["tokens_per_second"]
    elif args.optimization_target == "latency":
        objective_value = -avg_metrics["latency"]  # Negative because we want to maximize
    elif args.optimization_target == "balanced":
        # Normalized metrics (assuming reasonable ranges for normalization)
        norm_throughput = min(avg_metrics["tokens_per_second"] / 20.0, 1.0)  # 20 tokens/sec is excellent
        norm_quality = avg_metrics["quality_score"]  # Already 0-1
        norm_memory = 1.0 - min(avg_metrics["memory_used"] / 2000.0, 1.0)  # Lower memory is better
        
        # Weighted combination
        objective_value = (0.5 * norm_throughput + 
                           0.3 * norm_quality + 
                           0.2 * norm_memory)
    elif args.optimization_target == "efficiency":
        objective_value = avg_metrics["efficiency"]
    else:
        objective_value = avg_metrics["tokens_per_second"]  # Default to throughput
    
    # Print summary
    print(f"Avg throughput: {avg_metrics['tokens_per_second']:.2f} tokens/sec, " +
          f"Avg latency: {avg_metrics['latency']:.3f}s, " + 
          f"Avg quality: {avg_metrics['quality_score']:.2f}, " +
          f"Objective value: {objective_value:.4f}")
    
    return objective_value

def run_grid_search(model, tokenizer, args):
    """Run a grid search as a baseline comparison to Bayesian optimization."""
    print("Running grid search for baseline comparison...")
    
    # Define grid points
    temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]
    top_ps = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    results = []
    
    # Create a progress bar for the grid search
    total_iterations = len(temperatures) * len(top_ps)
    with tqdm(total=total_iterations, desc="Grid Search Progress") as pbar:
        # Evaluate each combination
        for temp in temperatures:
            for top_p in top_ps:
                # Run evaluation
                metric = objective_function(temp, top_p, model=model, tokenizer=tokenizer, args=args)
                
                # Save result
                results.append({
                    "temperature": temp,
                    "top_p": top_p,
                    "metric": metric
                })
                
                pbar.update(1)
    
    # Find best result
    best_result = max(results, key=lambda x: x["metric"])
    
    print("\nGrid Search Results:")
    print(f"Best {args.optimization_target}: {best_result['metric']:.4f}")
    print(f"Best temperature: {best_result['temperature']:.4f}")
    print(f"Best top_p: {best_result['top_p']:.4f}")
    
    # Create visualization
    create_grid_search_plot(results, args)
    
    return best_result

def create_grid_search_plot(results, args):
    """Create a heatmap visualization of grid search results."""
    # Extract all unique temperature and top_p values
    temperatures = sorted(list(set([r["temperature"] for r in results])))
    top_ps = sorted(list(set([r["top_p"] for r in results])))
    
    # Create grid
    grid = np.zeros((len(temperatures), len(top_ps)))
    
    # Fill grid with results
    for result in results:
        i = temperatures.index(result["temperature"])
        j = top_ps.index(result["top_p"])
        grid[i, j] = result["metric"]
    
    # Adjust grid for latency (lower is better)
    if args.optimization_target == "latency":
        grid = -grid
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    
    # Add color bar
    cbar = plt.colorbar()
    if args.optimization_target == "throughput":
        cbar.set_label('Throughput (tokens/sec)')
    elif args.optimization_target == "latency":
        cbar.set_label('Latency (s)')
    else:
        cbar.set_label('Score')
    
    # Add labels
    plt.xticks(range(len(top_ps)), [f"{p:.1f}" for p in top_ps])
    plt.yticks(range(len(temperatures)), [f"{t:.1f}" for t in temperatures])
    plt.xlabel('Top-p')
    plt.ylabel('Temperature')
    plt.title(f'Grid Search Results for {args.optimization_target.capitalize()}')
    
    # Highlight the best point
    best_result = max(results, key=lambda x: x["metric"])
    best_i = temperatures.index(best_result["temperature"])
    best_j = top_ps.index(best_result["top_p"])
    plt.plot(best_j, best_i, 'r*', markersize=15)
    
    # Save plot
    plot_file = os.path.join(args.output_dir, f"grid_search_{args.optimization_target}.png")
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Grid search plot saved to {plot_file}")

def run_optimization(args):
    """Run Bayesian optimization to find optimal inference parameters."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model (only once for all optimization runs)
    model, tokenizer = load_model(args.model_path, args.tokenizer_path, args.model_size, args)
    
    # Define the search space (expanded to include more parameters)
    pbounds = {
        'temperature': (0.05, 2.0),          # Range for temperature parameter
        'top_p': (0.1, 1.0),                 # Range for top_p parameter
        'repetition_penalty': (1.0, 1.5),    # Range for repetition penalty
        'max_new_tokens': (10, 100)          # Range for max new tokens
    }
    
    # Define different acquisition functions
    acquisition_functions = {
        "ei": UtilityFunction(kind="ei", kappa=2.576, xi=0.0),  # Expected Improvement
        "ucb": UtilityFunction(kind="ucb", kappa=2.576),        # Upper Confidence Bound
        "poi": UtilityFunction(kind="poi", xi=0.0)              # Probability of Improvement
    }

    # Select the acquisition function based on parameters or optimization target
    if hasattr(args, 'acquisition_function') and args.acquisition_function in acquisition_functions:
        acq_function = acquisition_functions[args.acquisition_function]
    elif args.optimization_target == "throughput":
        acq_function = acquisition_functions["ei"]  # EI works well for maximization
    elif args.optimization_target == "latency":
        acq_function = acquisition_functions["ucb"]  # UCB can be more explorative
    else:
        acq_function = acquisition_functions["ei"]  # Default
    
    # Create optimizer
    optimizer = BayesianOptimization(
        f=lambda temperature, top_p, repetition_penalty, max_new_tokens: 
            objective_function(
                temperature, top_p, repetition_penalty, max_new_tokens, 
                model=model, tokenizer=tokenizer, args=args
            ),
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    # Initialize Weights & Biases if available
    try:
        import wandb
        use_wandb = True
        
        # Initialize W&B
        wandb.init(
            project="llama2-optimization",
            name=f"bayesian_opt_{args.optimization_target}",
            config={
                "model_size": args.model_size,
                "optimization_target": args.optimization_target,
                "parameter_bounds": pbounds,
                "n_iter": args.n_iter,
                "initial_points": args.initial_points,
                "acquisition_function": args.acquisition_function if hasattr(args, 'acquisition_function') else "ei"
            }
        )
        
        # Create a custom callback for W&B logging
        def wandb_callback(_):
            # Log the latest result
            if len(optimizer.res) > 0:
                latest = optimizer.res[-1]
                wandb_data = {
                    "iteration": len(optimizer.res) - 1,
                    "target_value": latest["target"],
                    "temperature": latest["params"]["temperature"],
                    "top_p": latest["params"]["top_p"],
                    "repetition_penalty": latest["params"]["repetition_penalty"],
                    "max_new_tokens": latest["params"]["max_new_tokens"]
                }
                wandb.log(wandb_data)
        
        # Subscribe the callback
        optimizer.subscribe(Events.OPTIMIZATION_STEP, wandb_callback)
        
    except ImportError:
        use_wandb = False
        print("Weights & Biases not installed. Install with 'pip install wandb' for experiment tracking.")
    
    # Setup logging
    log_path = os.path.join(args.output_dir, f"opt_logs_{args.optimization_target}.json")
    logger = JSONLogger(path=log_path)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    
    # Load previous logs if continuing optimization
    if args.continue_optimization and os.path.exists(log_path):
        print(f"Loading previous optimization logs from {log_path}")
        load_logs(optimizer, logs=[log_path])
        print(f"Loaded {len(optimizer.space.params)} previous points")
    
    # Run optimization
    print(f"Starting Bayesian optimization for {args.n_iter} iterations...")
    optimizer.maximize(
        init_points=args.initial_points,
        n_iter=args.n_iter,
        acq=acq_function
    )
    
    # Get the best parameters
    best_params = optimizer.max['params']
    best_score = optimizer.max['target']
    
    print("\n==================================")
    print("Optimization Results:")
    print(f"Best {args.optimization_target}: {best_score:.4f}")
    print(f"Best temperature: {best_params['temperature']:.4f}")
    print(f"Best top_p: {best_params['top_p']:.4f}")
    print(f"Best repetition_penalty: {best_params['repetition_penalty']:.4f}")
    print(f"Best max_new_tokens: {best_params['max_new_tokens']:.0f}")
    print("==================================")
    
    # Save the results to a file
    results = {
        'best_score': best_score,
        'best_params': best_params,
        'all_results': optimizer.res,
        'optimization_target': args.optimization_target,
        'model_size': args.model_size,
        'max_gen_len': args.max_gen_len,
    }
    
    results_file = os.path.join(args.output_dir, f"optimization_results_{args.optimization_target}.json")
    with open(results_file, 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        sanitized_results = json.loads(json.dumps(results, default=lambda o: float(o) if isinstance(o, np.float32) else o))
        json.dump(sanitized_results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Create plots for visualization
    create_optimization_plots(optimizer, args)
    
    # Perform validation with best parameters
    print("\nValidating best parameters...")
    validate_parameters(
        best_params['temperature'], 
        best_params['top_p'], 
        best_params['repetition_penalty'],
        int(best_params['max_new_tokens']),
        model, 
        tokenizer, 
        args
    )
    
    # Run grid search comparison if requested
    if hasattr(args, 'run_grid_search') and args.run_grid_search:
        print("\nComparing Bayesian optimization with grid search...")
        grid_search_result = run_grid_search(model, tokenizer, args)
        
        # Create comparison
        comparison = {
            "method": ["Bayesian Optimization", "Grid Search"],
            "temperature": [optimizer.max["params"]["temperature"], grid_search_result["temperature"]],
            "top_p": [optimizer.max["params"]["top_p"], grid_search_result["top_p"]],
            "performance": [optimizer.max["target"], grid_search_result["metric"]]
        }
        
        # Print comparison
        print("\nComparison of optimization methods:")
        print(f"{'Method':<25} {'Temperature':<15} {'Top-p':<15} {'Performance':<15}")
        print("-" * 70)
        for i in range(2):
            print(f"{comparison['method'][i]:<25} {comparison['temperature'][i]:<15.4f} {comparison['top_p'][i]:<15.4f} {comparison['performance'][i]:<15.4f}")
        
        # Create comparison plot
        plt.figure(figsize=(10, 6))
        
        # Bar chart comparing performance
        plt.bar(comparison["method"], comparison["performance"])
        plt.ylabel("Performance Metric")
        
        if args.optimization_target == "throughput":
            plt.ylabel("Throughput (tokens/sec)")
        elif args.optimization_target == "latency":
            plt.ylabel("Negative Latency (-seconds)")
        else:
            plt.ylabel("Combined Score")
        
        plt.title(f"Comparison of Optimization Methods for {args.optimization_target.capitalize()}")
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(args.output_dir, f"optimization_comparison_{args.optimization_target}.png")
        plt.savefig(plot_file)
        plt.close()
        
        print(f"Optimization comparison plot saved to {plot_file}")
    
    return best_params, best_score

def validate_parameters(temperature, top_p, repetition_penalty, max_new_tokens, model, tokenizer, args):
    """Validate the best parameters on a slightly different set of prompts."""
    validation_prompts = [
        "What is the significance of deep learning in modern AI?",
        "Explain the concept of quantum computing to a high school student.",
        "How do self-driving cars navigate and avoid obstacles?"
    ]
    
    print(f"Validating parameters: temperature={temperature:.4f}, top_p={top_p:.4f}, " +
          f"repetition_penalty={repetition_penalty:.4f}, max_tokens={max_new_tokens}")
    
    results = []
    for prompt in validation_prompts:
        print(f"\nPrompt: {prompt}")
        
        result = inference_step(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_new_tokens,
            repetition_penalty=repetition_penalty
        )
        
        # Calculate quality score
        quality = evaluate_quality(result["output_text"], prompt)
        result["quality_score"] = quality
        
        print(f"Generated {result['output_length']} tokens in {result['latency']:.2f}s")
        print(f"Throughput: {result['tokens_per_second']:.2f} tokens/sec")
        print(f"Quality score: {quality:.2f}")
        print(f"Response sample: {result['output_text'][:100]}...")
        
        results.append(result)
        
        # Clear cache between runs
        torch.cuda.empty_cache()
        gc.collect()
    
    # Calculate average metrics
    avg_latency = sum([r["latency"] for r in results]) / len(results)
    avg_throughput = sum([r["tokens_per_second"] for r in results]) / len(results)
    avg_quality = sum([r["quality_score"] for r in results]) / len(results)
    
    print("\nValidation Results:")
    print(f"Average latency: {avg_latency:.4f}s")
    print(f"Average throughput: {avg_throughput:.4f} tokens/sec")
    print(f"Average quality score: {avg_quality:.4f}")
    
    # Save validation results
    validation_results = {
        'temperature': temperature,
        'top_p': top_p,
        'repetition_penalty': repetition_penalty,
        'max_new_tokens': max_new_tokens,
        'avg_latency': float(avg_latency),
        'avg_throughput': float(avg_throughput),
        'avg_quality': float(avg_quality),
        'prompts': validation_prompts,
        'detailed_results': results
    }
    
    validation_file = os.path.join(args.output_dir, f"validation_results_{args.optimization_target}.json")
    with open(validation_file, 'w') as f:
        # Filter out non-serializable content like tensors
        sanitized_results = []
        for r in results:
            sr = {k: v for k, v in r.items() if k != 'tokens'}
            sanitized_results.append(sr)
        
        validation_results['detailed_results'] = sanitized_results
        json.dump(validation_results, f, indent=2)
    
    print(f"Validation results saved to {validation_file}")
    
    return avg_throughput, avg_latency, avg_quality

def create_optimization_plots(optimizer, args):
    """Create plots to visualize the optimization process."""
    try:
        # Extract data from optimizer
        iterations = range(len(optimizer.res))
        values = [res['target'] for res in optimizer.res]
        temperatures = [res['params']['temperature'] for res in optimizer.res]
        top_ps = [res['params']['top_p'] for res in optimizer.res]
        rep_penalties = [res['params']['repetition_penalty'] for res in optimizer.res]
        max_tokens = [res['params']['max_new_tokens'] for res in optimizer.res]
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        
        # Plot 1: Target value vs iteration
        axs[0, 0].plot(iterations, values, 'bo-')
        axs[0, 0].set_xlabel('Iteration')
        if args.optimization_target == 'throughput':
            axs[0, 0].set_ylabel('Throughput (tokens/sec)')
            axs[0, 0].set_title('Throughput vs Iteration')
        elif args.optimization_target == 'latency':
            # Convert back to positive values for plotting
            axs[0, 0].plot(iterations, [-v for v in values], 'bo-')
            axs[0, 0].set_ylabel('Latency (s)')
            axs[0, 0].set_title('Latency vs Iteration')
        else:
            axs[0, 0].set_ylabel('Objective Value')
            axs[0, 0].set_title(f'{args.optimization_target.capitalize()} vs Iteration')
        axs[0, 0].grid(True)
        
        # Plot 2: Temperature vs iteration
        axs[0, 1].plot(iterations, temperatures, 'ro-')
        axs[0, 1].set_xlabel('Iteration')
        axs[0, 1].set_ylabel('Temperature')
        axs[0, 1].set_title('Temperature vs Iteration')
        axs[0, 1].grid(True)
        
        # Plot 3: Top-p vs iteration
        axs[1, 0].plot(iterations, top_ps, 'go-')
        axs[1, 0].set_xlabel('Iteration')
        axs[1, 0].set_ylabel('Top-p')
        axs[1, 0].set_title('Top-p vs Iteration')
        axs[1, 0].grid(True)
        
        # Plot 4: Repetition penalty vs iteration
        axs[1, 1].plot(iterations, rep_penalties, 'mo-')
        axs[1, 1].set_xlabel('Iteration')
        axs[1, 1].set_ylabel('Repetition Penalty')
        axs[1, 1].set_title('Repetition Penalty vs Iteration')
        axs[1, 1].grid(True)
        
        # Plot 5: Max tokens vs iteration
        axs[2, 0].plot(iterations, max_tokens, 'co-')
        axs[2, 0].set_xlabel('Iteration')
        axs[2, 0].set_ylabel('Max New Tokens')
        axs[2, 0].set_title('Max New Tokens vs Iteration')
        axs[2, 0].grid(True)
        
        # Plot 6: Parameter space visualization (temp vs top_p)
        cm = plt.cm.get_cmap('viridis')
        sc = axs[2, 1].scatter(temperatures, top_ps, c=values, vmin=min(values), vmax=max(values), 
                            cmap=cm, s=100)
        axs[2, 1].set_xlabel('Temperature')
        axs[2, 1].set_ylabel('Top-p')
        if args.optimization_target == 'throughput':
            axs[2, 1].set_title('Parameter Space Exploration\n(color = throughput)')
        elif args.optimization_target == 'latency':
            axs[2, 1].set_title('Parameter Space Exploration\n(color = -latency)')
        else:
            axs[2, 1].set_title(f'Parameter Space\n(color = {args.optimization_target})')
        fig.colorbar(sc, ax=axs[2, 1])
        axs[2, 1].grid(True)
        
        plt.tight_layout()
        plot_file = os.path.join(args.output_dir, f"optimization_plots_{args.optimization_target}.png")
        plt.savefig(plot_file)
        plt.close()
        
        print(f"Optimization plots saved to {plot_file}")
        
        # Create additional parameter space visualizations
        plt.figure(figsize=(10, 8))
        plt.scatter(temperatures, rep_penalties, c=values, cmap='viridis', s=100)
        
        # Mark the best point
        best_idx = values.index(max(values))
        plt.scatter([temperatures[best_idx]], [rep_penalties[best_idx]], 
                    c='red', s=200, marker='*', edgecolors='black', linewidths=2)
        
        plt.xlabel('Temperature')
        plt.ylabel('Repetition Penalty')
        plt.colorbar(label='Performance')
        plt.title('Temperature vs Repetition Penalty with Best Point')
        plt.grid(True)
        temp_rep_plot_file = os.path.join(args.output_dir, f"temp_rep_space_{args.optimization_target}.png")
        plt.savefig(temp_rep_plot_file)
        plt.close()
        
        # Create top_p vs max_tokens visualization
        plt.figure(figsize=(10, 8))
        plt.scatter(top_ps, max_tokens, c=values, cmap='viridis', s=100)
        
        # Mark the best point
        plt.scatter([top_ps[best_idx]], [max_tokens[best_idx]], 
                    c='red', s=200, marker='*', edgecolors='black', linewidths=2)
        
        plt.xlabel('Top-p')
        plt.ylabel('Max New Tokens')
        plt.colorbar(label='Performance')
        plt.title('Top-p vs Max Tokens with Best Point')
        plt.grid(True)
        topp_tokens_plot_file = os.path.join(args.output_dir, f"topp_tokens_space_{args.optimization_target}.png")
        plt.savefig(topp_tokens_plot_file)
        plt.close()
        
    except Exception as e:
        print(f"Warning: Failed to create optimization plots: {e}")
        print("Continuing optimization process...")

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
            curr_logits = logits[:, -1, :] / max(0.8, 1e-5)  # Using default temperature of 0.8
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
    
    # Add labels
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
    try:
        args = parse_args()
        print(f"Optimizing LLaMA 2 {args.model_size} parameters for {args.optimization_target}")
        
        try:
            if hasattr(args, 'analyze_memory') and args.analyze_memory:
                # Load model
                model, tokenizer = load_model(args.model_path, args.tokenizer_path, args.model_size, args)
                # Analyze memory on a single prompt
                memory_results = analyze_memory_usage(model, tokenizer, OPTIMIZATION_PROMPTS[0], args)
            else:
                # Run standard optimization
                best_params, best_score = run_optimization(args)
                print("Optimization complete!")
                print(f"Best parameters: temperature={best_params['temperature']:.4f}, " +
                      f"top_p={best_params['top_p']:.4f}, " +
                      f"repetition_penalty={best_params['repetition_penalty']:.2f}, " +
                      f"max_new_tokens={int(best_params['max_new_tokens'])}")
                
                if args.optimization_target == 'throughput':
                    print(f"Best throughput: {best_score:.4f} tokens/sec")
                elif args.optimization_target == 'latency':
                    print(f"Best latency: {-best_score:.4f} seconds")
                else:
                    print(f"Best {args.optimization_target} score: {best_score:.4f}")
                
        except Exception as e:
            print(f"Optimization failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user. Exiting gracefully...")
        sys.exit(0)