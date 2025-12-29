#!/usr/bin/env python3
"""
QuanTA Adapter Loading and Inference Script

This script demonstrates how to:
1. Load a base model
2. Load a trained QuanTA adapter
3. Merge the adapter weights for efficient inference
4. Verify the adapter weights are properly loaded
5. Run a test inference

Usage:
    python load_quanta_for_inference.py --base_model /path/to/base/model --adapter /path/to/adapter/dir
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_quanta_adapter(base_model_path: str, adapter_path: str, device: str = "auto"):
    """
    Load a base model and apply a QuanTA adapter.
    
    Args:
        base_model_path: Path to the base model (e.g., Llama-3.1-8B)
        adapter_path: Path to the saved QuanTA adapter directory
        device: Device to load the model on ("auto", "cuda", "cpu")
    
    Returns:
        model: The model with QuanTA adapter applied
        tokenizer: The tokenizer
    """
    # Add the quanta package to path if needed
    # Uncomment and modify if quanta is not installed as a package
    # sys.path.insert(0, "/workspace/quanta/quanta")
    
    from quanta import PeftModel, PeftConfig
    
    print(f"Loading base model from: {base_model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    
    print(f"Loading QuanTA adapter from: {adapter_path}")
    
    # Load the adapter config to verify it's QuanTA
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path, "r") as f:
        adapter_config = json.load(f)
    
    print(f"Adapter config: {json.dumps(adapter_config, indent=2)}")
    
    if adapter_config.get("peft_type") != "QUANTA":
        raise ValueError(f"Expected QUANTA adapter, got: {adapter_config.get('peft_type')}")
    
    # Load the PEFT model with adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model, tokenizer


def verify_quanta_weights_loaded(model):
    """
    Verify that QuanTA weights are present and properly loaded in the model.
    
    Returns:
        dict: Summary of QuanTA weights found
    """
    quanta_params = {}
    total_quanta_params = 0
    
    for name, param in model.named_parameters():
        if "quanta_weights" in name:
            quanta_params[name] = {
                "shape": list(param.shape),
                "numel": param.numel(),
                "requires_grad": param.requires_grad,
                "mean": param.data.float().mean().item(),
                "std": param.data.float().std().item(),
            }
            total_quanta_params += param.numel()
    
    summary = {
        "num_quanta_layers": len(quanta_params),
        "total_quanta_params": total_quanta_params,
        "layers": quanta_params,
    }
    
    return summary


def merge_adapter_for_inference(model):
    """
    Merge the QuanTA adapter weights into the base model for faster inference.
    
    After merging, the model behaves like a regular model with the adapter
    weights baked in, which is more efficient for inference.
    """
    print("Setting model to eval mode (this triggers weight merging for QuanTA)...")
    model.eval()
    return model


def run_test_inference(model, tokenizer, prompt: str, max_new_tokens: int = 100):
    """
    Run a test inference to verify the model works.
    """
    print(f"\n{'='*60}")
    print("Running test inference...")
    print(f"{'='*60}")
    print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated output:\n{generated_text}")
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Load QuanTA adapter for inference")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to base model (e.g., /workspace/hf_cache/models/Llama-3.1-8B)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to the saved QuanTA adapter directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Solve this math problem step by step: What is 15 + 27?",
        help="Test prompt for inference",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate",
    )
    
    args = parser.parse_args()
    
    # 1. Load the model with adapter
    model, tokenizer = load_quanta_adapter(
        args.base_model,
        args.adapter,
        args.device,
    )
    
    # 2. Verify QuanTA weights are loaded
    print(f"\n{'='*60}")
    print("Verifying QuanTA weights are loaded...")
    print(f"{'='*60}")
    
    weight_summary = verify_quanta_weights_loaded(model)
    
    print(f"Number of QuanTA layers: {weight_summary['num_quanta_layers']}")
    print(f"Total QuanTA parameters: {weight_summary['total_quanta_params']:,}")
    
    if weight_summary['num_quanta_layers'] == 0:
        print("\n⚠️  WARNING: No QuanTA weights found! The adapter may not be loaded correctly.")
    else:
        print("\n✅ QuanTA weights successfully loaded!")
        print("\nFirst 5 QuanTA layers (sample):")
        for i, (name, info) in enumerate(list(weight_summary['layers'].items())[:5]):
            print(f"  {name}")
            print(f"    Shape: {info['shape']}, Mean: {info['mean']:.6f}, Std: {info['std']:.6f}")
    
    # 3. Merge for inference
    model = merge_adapter_for_inference(model)
    
    # 4. Run test inference
    run_test_inference(model, tokenizer, args.prompt, args.max_new_tokens)
    
    print(f"\n{'='*60}")
    print("Done! Model is ready for inference.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
