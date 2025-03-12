#!/usr/bin/env python
"""
Extracting activations with Wisent.

This script demonstrates how to extract activations from language models using
the Wisent library. Activations are the internal representations of a model at
different layers, which can be used for various purposes such as creating
control vectors or analyzing model behavior.
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

from wisent import WisentClient
from wisent.activations import ActivationExtractor


def main():
    """Extract activations from a language model and analyze them."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract activations from a language model")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1", help="Model name")
    parser.add_argument("--api-key", type=str, default=os.environ.get("WISENT_API_KEY"), help="Wisent API key")
    parser.add_argument("--api-url", type=str, default=os.environ.get("WISENT_API_URL", "https://api.wisent.ai"), help="Wisent API URL")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--prompt", type=str, default="Explain quantum computing to me.", help="Prompt for activation extraction")
    parser.add_argument("--layers", type=str, default="-1,-2,-3", help="Comma-separated list of layers to extract (e.g., '0,1,2' or '-1,-2,-3')")
    parser.add_argument("--tokens", type=str, default="-1", help="Comma-separated list of tokens to extract (e.g., '0,1,2' or '-1,-2,-3')")
    parser.add_argument("--upload", action="store_true", help="Upload activations to the Wisent backend")
    parser.add_argument("--visualize", action="store_true", help="Visualize the activations")
    parser.add_argument("--output", type=str, help="Output directory for visualization plots")
    
    args = parser.parse_args()
    
    # Check if API key is provided
    if not args.api_key:
        print("Error: API key is required. Set the WISENT_API_KEY environment variable or use --api-key.")
        sys.exit(1)
    
    # Parse layers and tokens
    layers = [int(layer.strip()) for layer in args.layers.split(",")]
    tokens = [int(token.strip()) for token in args.tokens.split(",")]
    
    # Initialize the client
    print(f"Initializing client with API key: {args.api_key[:5] if args.api_key else 'None'}...")
    client = WisentClient(api_key=args.api_key, base_url=args.api_url)
    
    # Print extraction parameters
    print(f"\nExtracting activations from model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Layers: {layers}")
    print(f"Tokens: {tokens}")
    print(f"Device: {args.device}")
    
    # Extract activations using the client API
    print("\n=== Extracting Activations using Client API ===\n")
    
    print("Extracting activations...")
    activations = client.activations.extract(
        model_name=args.model,
        prompt=args.prompt,
        layers=layers,
        tokens_to_extract=tokens,
        device=args.device
    )
    
    print(f"Extracted {len(activations.activations)} activations")
    print(f"Model: {activations.model_name}")
    print(f"Prompt: {activations.prompt}")
    print(f"Metadata: {activations.metadata}")
    
    # Print information about each activation
    print("\nActivation details:")
    for i, activation in enumerate(activations.activations):
        print(f"Activation {i+1}:")
        print(f"  Layer: {activation.layer}")
        print(f"  Token index: {activation.token_index}")
        print(f"  Token string: {activation.token_str}")
        
        # Convert to numpy array if it's a tensor
        values = activation.values
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        print(f"  Values shape: {values.shape}")
        print(f"  Values min: {values.min()}, max: {values.max()}, mean: {values.mean()}")
        print()
    
    # Using the ActivationExtractor directly
    print("\n=== Using the ActivationExtractor Directly ===\n")
    
    print("Creating an extractor...")
    extractor = ActivationExtractor(
        model_name=args.model,
        device=args.device
    )
    
    # Extract activations for multiple layers and tokens
    print("Extracting activations with custom parameters...")
    
    # Determine middle layer based on model architecture
    # This is a rough estimate and may need adjustment for different models
    if "mistral" in args.model.lower():
        middle_layer = 16  # Mistral-7B has 32 layers
    elif "llama" in args.model.lower():
        middle_layer = 16  # LLaMA-7B has 32 layers
    else:
        middle_layer = 12  # Default for many models
    
    custom_layers = [0, middle_layer, -1]  # First, middle, and last layers
    custom_tokens = [-10, -1]  # Extract last 10 tokens and final token
    
    print(f"Custom layers: {custom_layers}")
    print(f"Custom tokens: {custom_tokens}")
    
    activations_custom = extractor.extract(
        prompt=args.prompt,
        layers=custom_layers,
        tokens_to_extract=custom_tokens
    )
    
    print(f"Extracted {len(activations_custom.activations)} activations")
    
    # Visualize activations
    if args.visualize:
        print("\n=== Visualizing Activations ===\n")
        
        # Create output directory if specified
        if args.output:
            os.makedirs(args.output, exist_ok=True)
        
        # Select an activation to visualize
        activation = activations.activations[0]
        
        # Convert to numpy array if it's a tensor
        values = activation.values
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        # Plot the activation values
        plt.figure(figsize=(12, 6))
        plt.plot(values)
        plt.title(f"Activation values for layer {activation.layer}, token {activation.token_index}")
        plt.xlabel("Dimension")
        plt.ylabel("Activation value")
        plt.grid(True)
        
        if args.output:
            plt.savefig(os.path.join(args.output, "activation_values.png"))
            print(f"Saved activation values plot to {os.path.join(args.output, 'activation_values.png')}")
        else:
            plt.show()
        
        # Plot a histogram of the activation values
        plt.figure(figsize=(12, 6))
        plt.hist(values, bins=50)
        plt.title(f"Histogram of activation values for layer {activation.layer}, token {activation.token_index}")
        plt.xlabel("Activation value")
        plt.ylabel("Frequency")
        plt.grid(True)
        
        if args.output:
            plt.savefig(os.path.join(args.output, "activation_histogram.png"))
            print(f"Saved activation histogram plot to {os.path.join(args.output, 'activation_histogram.png')}")
        else:
            plt.show()
    
    # Upload activations to the Wisent backend
    if args.upload:
        print("\n=== Uploading Activations to the Wisent Backend ===\n")
        
        print("Uploading activations...")
        response = client.activations.upload(activations)
        
        print(f"Upload response: {response}")
        print(f"Batch ID: {response.get('id')}")
        
        # Retrieve the activations
        batch_id = response.get('id')
        if batch_id:
            print("\n=== Retrieving Activations from the Wisent Backend ===\n")
            
            print(f"Retrieving activations with batch ID: {batch_id}...")
            retrieved_activations = client.activations.get(batch_id)
            
            print(f"Retrieved {len(retrieved_activations.activations)} activations")
            print(f"Model: {retrieved_activations.model_name}")
            print(f"Prompt: {retrieved_activations.prompt}")
            print(f"Metadata: {retrieved_activations.metadata}")
        
        # List activation batches
        print("\n=== Listing Activation Batches ===\n")
        
        print(f"Listing activation batches for model: {args.model}...")
        batches = client.activations.list(model_name=args.model, limit=10)
        
        print(f"Found {len(batches)} activation batches:")
        for i, batch in enumerate(batches):
            print(f"Batch {i+1}:")
            print(f"  ID: {batch['id']}")
            print(f"  Prompt: {batch['prompt']}")
            print(f"  Created at: {batch.get('created_at', 'Unknown')}")
            print()


if __name__ == "__main__":
    main() 