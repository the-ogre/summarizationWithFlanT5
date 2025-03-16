#!/usr/bin/env python3
"""
Memory optimization utilities for LED model.
This script provides functions to help optimize memory usage
when working with large LED models on GPUs with limited memory.
"""

import gc
import torch

def estimate_model_memory(model_name="pszemraj/led-large-book-summary", precision="float16"):
    """
    Estimate how much memory a model will use when loaded.
    
    Args:
        model_name: Model identifier
        precision: Precision to use (float16, float32)
        
    Returns:
        Dictionary with memory estimates
    """
    # Rough model size estimates in GB
    model_sizes = {
        "pszemraj/led-large-book-summary": {
            "float32": 1.55,  # Approx 1.55 GB in full precision
            "float16": 0.78,  # Approx 0.78 GB in half precision
        },
        "pszemraj/led-base-book-summary": {
            "float32": 0.5,   # Approx 0.5 GB in full precision
            "float16": 0.25,  # Approx 0.25 GB in half precision
        }
    }
    
    # Get default values if model not found
    default_size = model_sizes.get(
        model_name, 
        {"float32": 1.5, "float16": 0.75}
    )
    
    # Get size for requested precision
    model_size = default_size.get(precision, default_size["float32"])
    
    # Estimate additional memory needed for processing
    activation_memory = model_size * 2  # Rough estimate for activations
    
    # Estimate total memory requirement
    total_estimate = model_size + activation_memory
    
    # Available GPU memory
    available_memory = 0
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        available_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # in GB
        
    return {
        "model_size_gb": model_size,
        "activation_memory_gb": activation_memory,
        "total_estimate_gb": total_estimate,
        "available_gpu_memory_gb": available_memory,
        "should_use_gpu": available_memory > total_estimate * 1.2,  # 20% buffer
        "recommended_precision": "float16" if available_memory < total_estimate * 2 else "float32",
    }

def clear_gpu_memory():
    """
    Clear CUDA cache and run garbage collection to free up GPU memory.
    Useful to call before loading a large model.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
def optimize_for_inference(model):
    """
    Apply inference-time optimizations to a model.
    
    Args:
        model: The PyTorch model to optimize
        
    Returns:
        Optimized model
    """
    # Set model to evaluation mode
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    return model
    
def get_recommended_batch_size(model_name, available_memory_gb=None):
    """
    Get recommended batch size based on model and available memory.
    
    Args:
        model_name: Name of the model
        available_memory_gb: Available GPU memory in GB (auto-detected if None)
        
    Returns:
        Recommended batch size
    """
    if available_memory_gb is None and torch.cuda.is_available():
        device = torch.cuda.current_device()
        available_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    
    # Default batch sizes based on available memory
    if available_memory_gb is None or available_memory_gb < 4:
        return 1
    elif available_memory_gb < 8:
        return 2
    elif available_memory_gb < 16:
        return 4
    else:
        return 8
        
if __name__ == "__main__":
    # When run as a script, print memory estimates
    estimates = estimate_model_memory()
    
    print("LED Model Memory Estimates:")
    print(f"Model size: {estimates['model_size_gb']:.2f} GB")
    print(f"Activation memory: {estimates['activation_memory_gb']:.2f} GB")
    print(f"Total estimate: {estimates['total_estimate_gb']:.2f} GB")
    
    if torch.cuda.is_available():
        print(f"Available GPU memory: {estimates['available_gpu_memory_gb']:.2f} GB")
        print(f"Should use GPU: {estimates['should_use_gpu']}")
        print(f"Recommended precision: {estimates['recommended_precision']}")
        print(f"Recommended batch size: {get_recommended_batch_size('pszemraj/led-large-book-summary')}")
    else:
        print("No GPU detected, using CPU for inference.")