#!/usr/bin/env python
"""
Basic usage of the Wisent library.

This script demonstrates the core functionality of the Wisent library for working
with control vectors and model inference.
"""

import argparse
import os
import sys

import torch

from wisent import WisentClient
from wisent.inference import Inferencer, InferenceConfig
from wisent.control_vector import ControlVectorManager


def main():
    """Demonstrate basic usage of the Wisent library."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Basic usage of the Wisent library")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1", help="Model name")
    parser.add_argument("--api-key", type=str, default=os.environ.get("WISENT_API_KEY"), help="Wisent API key")
    parser.add_argument("--api-url", type=str, default=os.environ.get("WISENT_API_URL", "https://api.wisent.ai"), help="Wisent API URL")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--prompt", type=str, default="Explain quantum computing to me.", help="Prompt for text generation")
    parser.add_argument("--local", action="store_true", help="Run local inference (requires downloading the model)")
    
    args = parser.parse_args()
    
    # Check if API key is provided
    if not args.api_key:
        print("Error: API key is required. Set the WISENT_API_KEY environment variable or use --api-key.")
        sys.exit(1)
    
    # Initialize the client
    print(f"Initializing client with API key: {args.api_key[:5] if args.api_key else 'None'}...")
    client = WisentClient(api_key=args.api_key, base_url=args.api_url)
    
    # Working with Control Vectors
    print("\n=== Working with Control Vectors ===\n")
    
    # List available control vectors
    print(f"Listing available control vectors for {args.model}...")
    vectors = client.control_vector.list(model=args.model)
    print(f"Found {len(vectors)} control vectors:")
    for i, vector in enumerate(vectors[:5]):  # Show only the first 5
        print(f"  {i+1}. {vector['name']}: {vector.get('description', 'No description')}")
    if len(vectors) > 5:
        print(f"  ... and {len(vectors) - 5} more")
    
    # Get a control vector
    if vectors:
        vector_name = vectors[0]["name"]
        print(f"\nGetting control vector: {vector_name}...")
        control_vector = client.control_vector.get(name=vector_name, model=args.model)
        print(f"Retrieved control vector: {control_vector.name}")
        print(f"Model: {control_vector.model_name}")
        print(f"Vector shape: {len(control_vector.values)}")
        print(f"Metadata: {control_vector.metadata}")
        
        # Combine control vectors if we have at least 2
        if len(vectors) >= 2:
            print("\nCombining control vectors...")
            vector_weights = {
                vectors[0]["name"]: 0.8,
                vectors[1]["name"]: 0.5
            }
            print(f"Vectors to combine: {vector_weights}")
            combined_vector = client.control_vector.combine(
                vectors=vector_weights,
                model=args.model
            )
            print(f"Combined vector name: {combined_vector.name}")
            print(f"Vector shape: {len(combined_vector.values)}")
            print(f"Metadata: {combined_vector.metadata}")
    else:
        print("No control vectors found for this model.")
        control_vector = None
    
    # Text Generation
    print("\n=== Text Generation ===\n")
    
    # Define a prompt
    prompt = args.prompt
    print(f"Prompt: {prompt}")
    
    # Generate text without control vectors
    print("\nGenerating text without control vectors...")
    response = client.inference.generate(
        model_name=args.model,
        prompt=prompt
    )
    print("\nGenerated text:")
    print(response.text)
    print(f"\nToken usage: {response.usage}")
    
    # Generate text with control vectors
    if vectors and len(vectors) >= 2:
        print("\nGenerating text with control vectors...")
        response_with_control = client.inference.generate_with_control(
            model_name=args.model,
            prompt=prompt,
            control_vectors=vector_weights,
            method="caa",
            scale=1.0
        )
        print("\nGenerated text with control vectors:")
        print(response_with_control.text)
        print(f"\nToken usage: {response_with_control.usage}")
        print(f"Metadata: {response_with_control.metadata}")
    
    # Local Inference
    if args.local and control_vector:
        print("\n=== Local Inference ===\n")
        
        print(f"Using device: {args.device}")
        
        # Initialize the control vector manager
        manager = ControlVectorManager(api_key=args.api_key, base_url=args.api_url)
        
        # Get a control vector
        print(f"Getting control vector: {vector_name}...")
        control_vector = manager.get(name=vector_name, model=args.model)
        
        # Initialize the inferencer
        print(f"Initializing inferencer with model: {args.model}...")
        inferencer = Inferencer(model_name=args.model, device=args.device)
        
        # Create an inference configuration
        config = InferenceConfig(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.0
        )
        
        # Generate text locally with the control vector
        print("Generating text locally with control vector...")
        response_local = inferencer.generate(
            prompt=prompt,
            control_vector=control_vector,
            method="caa",
            scale=1.0,
            config=config
        )
        
        print("\nGenerated text locally with control vector:")
        print(response_local.text)
        print(f"\nToken usage: {response_local.usage}")
        print(f"Metadata: {response_local.metadata}")


if __name__ == "__main__":
    main() 