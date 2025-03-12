#!/usr/bin/env python
"""
Creating and using custom control vectors with Wisent.

This script demonstrates how to create custom control vectors and use them for
text generation. Control vectors allow you to steer the output of language models
in specific directions.
"""

import argparse
import os
import sys

import torch
import numpy as np

from wisent import WisentClient
from wisent.activations import ActivationExtractor
from wisent.control_vector import ControlVector
from wisent.inference import Inferencer, InferenceConfig


def main():
    """Create custom control vectors and use them for text generation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Create and use custom control vectors")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1", help="Model name")
    parser.add_argument("--api-key", type=str, default=os.environ.get("WISENT_API_KEY"), help="Wisent API key")
    parser.add_argument("--api-url", type=str, default=os.environ.get("WISENT_API_URL", "https://api.wisent.ai"), help="Wisent API URL")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--prompt", type=str, default="Explain quantum computing to me.", help="Prompt for text generation")
    parser.add_argument("--desired-prompt", type=str, default="Write a concise summary of quantum computing.", help="Prompt for desired behavior")
    parser.add_argument("--undesired-prompt", type=str, default="Write a verbose and detailed explanation of quantum computing.", help="Prompt for undesired behavior")
    parser.add_argument("--method", type=str, default="caa", help="Method for applying control vectors (e.g., 'caa', 'add')")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling factor for control vectors")
    parser.add_argument("--upload", action="store_true", help="Upload custom control vector to the Wisent backend")
    
    args = parser.parse_args()
    
    # Check if API key is provided
    if not args.api_key:
        print("Error: API key is required. Set the WISENT_API_KEY environment variable or use --api-key.")
        sys.exit(1)
    
    # Initialize the client
    print(f"Initializing client with API key: {args.api_key[:5] if args.api_key else 'None'}...")
    client = WisentClient(api_key=args.api_key, base_url=args.api_url)
    
    # Creating a Custom Control Vector
    print("\n=== Creating a Custom Control Vector ===\n")
    
    # Method 1: Creating a Control Vector from Activations
    print("Method 1: Creating a Control Vector from Activations\n")
    
    print(f"Using device: {args.device}")
    
    # Define prompts for desired and undesired behavior
    desired_prompt = args.desired_prompt
    undesired_prompt = args.undesired_prompt
    
    print(f"Desired prompt: {desired_prompt}")
    print(f"Undesired prompt: {undesired_prompt}")
    
    # Create an extractor
    print("Creating an activation extractor...")
    extractor = ActivationExtractor(
        model_name=args.model,
        device=args.device
    )
    
    # Extract activations for the desired behavior
    print("Extracting activations for desired behavior...")
    desired_activations = extractor.extract(
        prompt=desired_prompt,
        layers=[-1],  # Extract from the last layer
        tokens_to_extract=[-1]  # Extract the last token
    )
    
    # Extract activations for the undesired behavior
    print("Extracting activations for undesired behavior...")
    undesired_activations = extractor.extract(
        prompt=undesired_prompt,
        layers=[-1],  # Extract from the last layer
        tokens_to_extract=[-1]  # Extract the last token
    )
    
    print(f"Extracted activations for desired behavior: {len(desired_activations.activations)}")
    print(f"Extracted activations for undesired behavior: {len(undesired_activations.activations)}")
    
    # Create the control vector
    print("\nCreating a control vector by subtracting undesired from desired activations...")
    
    # Get the activation values
    desired_values = desired_activations.activations[0].values
    undesired_values = undesired_activations.activations[0].values
    
    # Convert to tensors if they're not already
    if not isinstance(desired_values, torch.Tensor):
        desired_values = torch.tensor(desired_values)
    if not isinstance(undesired_values, torch.Tensor):
        undesired_values = torch.tensor(undesired_values)
    
    # Create the control vector by subtracting undesired from desired
    control_vector_values = desired_values - undesired_values
    
    # Normalize the vector
    control_vector_values = control_vector_values / torch.norm(control_vector_values)
    
    # Create a ControlVector object
    custom_control_vector = ControlVector(
        name="concise_custom",
        model_name=args.model,
        values=control_vector_values,
        metadata={
            "description": "Custom control vector for concise text",
            "created_from": {
                "desired": desired_prompt,
                "undesired": undesired_prompt
            }
        }
    )
    
    print(f"Created custom control vector: {custom_control_vector.name}")
    print(f"Vector shape: {len(custom_control_vector.values)}")
    print(f"Metadata: {custom_control_vector.metadata}")
    
    # Method 2: Combining Existing Control Vectors
    print("\nMethod 2: Combining Existing Control Vectors\n")
    
    # List available control vectors
    print(f"Listing available control vectors for {args.model}...")
    vectors = client.control_vector.list(model=args.model)
    
    if len(vectors) >= 2:
        # Get existing control vectors
        vector1_name = vectors[0]["name"]
        vector2_name = vectors[1]["name"]
        
        print(f"Getting control vectors: {vector1_name} and {vector2_name}...")
        vector1 = client.control_vector.get(name=vector1_name, model=args.model)
        vector2 = client.control_vector.get(name=vector2_name, model=args.model)
        
        # Combine them with weights
        vector_weights = {
            vector1_name: 0.7,
            vector2_name: 0.3
        }
        
        print(f"Combining vectors with weights: {vector_weights}...")
        combined_vector = client.control_vector.combine(
            vectors=vector_weights,
            model=args.model
        )
        
        print(f"Created combined control vector: {combined_vector.name}")
        print(f"Vector shape: {len(combined_vector.values)}")
        print(f"Metadata: {combined_vector.metadata}")
    else:
        print(f"Not enough control vectors available for {args.model} to combine.")
        combined_vector = None
    
    # Method 3: Creating a Control Vector from Scratch
    print("\nMethod 3: Creating a Control Vector from Scratch\n")
    
    # Get the dimension of the model's hidden states
    hidden_dim = len(custom_control_vector.values)
    
    # Create random values for demonstration
    print(f"Creating a random control vector with dimension {hidden_dim}...")
    random_values = torch.randn(hidden_dim)
    
    # Normalize the vector
    random_values = random_values / torch.norm(random_values)
    
    # Create a ControlVector object
    random_control_vector = ControlVector(
        name="random_custom",
        model_name=args.model,
        values=random_values,
        metadata={
            "description": "Random control vector for demonstration"
        }
    )
    
    print(f"Created random control vector: {random_control_vector.name}")
    print(f"Vector shape: {len(random_control_vector.values)}")
    print(f"Metadata: {random_control_vector.metadata}")
    
    # Using Custom Control Vectors for Text Generation
    print("\n=== Using Custom Control Vectors for Text Generation ===\n")
    
    # Initialize the inferencer
    print(f"Initializing inferencer with model: {args.model}...")
    inferencer = Inferencer(model_name=args.model, device=args.device)
    
    # Define a prompt
    prompt = args.prompt
    print(f"Prompt: {prompt}")
    
    # Create an inference configuration
    config = InferenceConfig(
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.0
    )
    
    # Generate text without control vector (baseline)
    print("\nGenerating text without control vector (baseline)...")
    response_baseline = inferencer.generate(
        prompt=prompt,
        config=config
    )
    
    print("\nGenerated text without control vector:")
    print(response_baseline.text)
    print(f"\nToken usage: {response_baseline.usage}")
    
    # Generate text with custom control vector
    print("\nGenerating text with custom control vector...")
    response_custom = inferencer.generate(
        prompt=prompt,
        control_vector=custom_control_vector,
        method=args.method,
        scale=args.scale,
        config=config
    )
    
    print("\nGenerated text with custom control vector:")
    print(response_custom.text)
    print(f"\nToken usage: {response_custom.usage}")
    print(f"Metadata: {response_custom.metadata}")
    
    # Experimenting with Different Scaling Factors
    print("\n=== Experimenting with Different Scaling Factors ===\n")
    
    scales = [0.5, 1.0, 2.0]
    
    for scale in scales:
        print(f"\nGenerating text with scale={scale}...")
        response = inferencer.generate(
            prompt=prompt,
            control_vector=custom_control_vector,
            method=args.method,
            scale=scale,
            config=config
        )
        
        print(f"\nGenerated text with scale={scale}:")
        print(response.text)
        print("\n" + "-"*80)
    
    # Comparing Different Control Vectors
    print("\n=== Comparing Different Control Vectors ===\n")
    
    if combined_vector:
        print("\nGenerating text with combined control vector...")
        response_combined = inferencer.generate(
            prompt=prompt,
            control_vector=combined_vector,
            method=args.method,
            scale=args.scale,
            config=config
        )
        
        print("\nGenerated text with combined control vector:")
        print(response_combined.text)
        print("\n" + "-"*80)
    
    print("\nGenerating text with random control vector...")
    response_random = inferencer.generate(
        prompt=prompt,
        control_vector=random_control_vector,
        method=args.method,
        scale=args.scale,
        config=config
    )
    
    print("\nGenerated text with random control vector:")
    print(response_random.text)
    
    # Uploading Custom Control Vectors to the Wisent Backend
    if args.upload:
        print("\n=== Uploading Custom Control Vector to the Wisent Backend ===\n")
        
        print("Uploading the custom control vector...")
        # Note: This is a placeholder - the actual API endpoint may differ
        response = client.http_client.post(
            "/control_vectors/upload",
            json_data=custom_control_vector.to_dict()
        )
        
        print(f"Upload response: {response}")


if __name__ == "__main__":
    main() 