#!/usr/bin/env python
"""
Creating and using custom control vectors with Wisent.

This script demonstrates three ways to obtain a control vector and how to use
each for text generation. Control vectors allow you to steer the output of
language models in specific directions. Vectors are built locally; the SDK
has no upload operation for them.
"""

import argparse
import os
import sys

import torch

from wisent import WisentClient
from wisent.activations import ActivationExtractor
from wisent.constants import DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P
from wisent.control_vector import ControlVector
from wisent.inference import Inferencer, InferenceConfig

# A short answer keeps the demo quick.
_DEMO_MAX_TOKENS = 100
# The weights the two catalog vectors are combined with, and the scales the
# custom vector is tried at.
_COMBINE_WEIGHTS = (0.7, 0.3)
_SCALES_TO_COMPARE = (0.5, 1.0, 2.0)
_RULE = "-" * 80


def parse_arguments():
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
    args = parser.parse_args()
    if not args.api_key:
        print("Error: API key is required. Set the WISENT_API_KEY environment variable or use --api-key.")
        sys.exit(1)
    return args


def vector_from_activations(args):
    """Method 1: the difference between the last-token activations of two prompts."""
    print("Method 1: Creating a Control Vector from Activations\n")
    print(f"Using device: {args.device}")
    print(f"Desired prompt: {args.desired_prompt}")
    print(f"Undesired prompt: {args.undesired_prompt}")

    print("Creating an activation extractor...")
    extractor = ActivationExtractor(model_name=args.model, device=args.device)

    print("Extracting activations for desired behavior...")
    desired_activations = extractor.extract(
        prompt=args.desired_prompt,
        layers=[-1],  # Extract from the last layer
        tokens_to_extract=[-1],  # Extract the last token
    )
    print("Extracting activations for undesired behavior...")
    undesired_activations = extractor.extract(
        prompt=args.undesired_prompt,
        layers=[-1],
        tokens_to_extract=[-1],
    )
    print(f"Extracted activations for desired behavior: {len(desired_activations.activations)}")
    print(f"Extracted activations for undesired behavior: {len(undesired_activations.activations)}")

    print("\nCreating a control vector by subtracting undesired from desired activations...")
    desired_values = torch.as_tensor(desired_activations.activations[0].values)
    undesired_values = torch.as_tensor(undesired_activations.activations[0].values)
    control_vector_values = desired_values - undesired_values
    control_vector_values = control_vector_values / torch.norm(control_vector_values)

    custom_control_vector = ControlVector(
        name="concise_custom",
        model_name=args.model,
        values=control_vector_values,
        metadata={
            "description": "Custom control vector for concise text",
            "created_from": {"desired": args.desired_prompt, "undesired": args.undesired_prompt},
        },
    )
    describe("custom", custom_control_vector)
    return custom_control_vector


def combined_from_catalog(client, args):
    """Method 2: two vectors the backend already holds, combined with weights.

    Returns None when the catalog holds fewer than two vectors for the model.
    """
    print("\nMethod 2: Combining Existing Control Vectors\n")
    print(f"Listing available control vectors for {args.model}...")
    vectors = client.control_vector.list(model=args.model)
    if len(vectors) < 2:
        print(f"Not enough control vectors available for {args.model} to combine.")
        return None

    vector1_name = vectors[0]["name"]
    vector2_name = vectors[1]["name"]
    print(f"Getting control vectors: {vector1_name} and {vector2_name}...")
    client.control_vector.get(name=vector1_name, model=args.model)
    client.control_vector.get(name=vector2_name, model=args.model)

    vector_weights = dict(zip((vector1_name, vector2_name), _COMBINE_WEIGHTS))
    print(f"Combining vectors with weights: {vector_weights}...")
    combined_vector = client.control_vector.combine(vectors=vector_weights, model=args.model)
    describe("combined", combined_vector)
    return combined_vector


def random_vector(args, hidden_dim):
    """Method 3: a normalized random vector of the model's hidden dimension."""
    print("\nMethod 3: Creating a Control Vector from Scratch\n")
    print(f"Creating a random control vector with dimension {hidden_dim}...")
    random_values = torch.randn(hidden_dim)
    random_values = random_values / torch.norm(random_values)
    random_control_vector = ControlVector(
        name="random_custom",
        model_name=args.model,
        values=random_values,
        metadata={"description": "Random control vector for demonstration"},
    )
    describe("random", random_control_vector)
    return random_control_vector


def describe(label, vector):
    print(f"Created {label} control vector: {vector.name}")
    print(f"Vector shape: {len(vector.values)}")
    print(f"Metadata: {vector.metadata}")


def generate_with_vectors(args, custom_control_vector, combined_vector, random_control_vector):
    """Generate the baseline, then with each vector, then across scales."""
    print("\n=== Using Custom Control Vectors for Text Generation ===\n")
    print(f"Initializing inferencer with model: {args.model}...")
    inferencer = Inferencer(model_name=args.model, device=args.device)
    print(f"Prompt: {args.prompt}")
    config = InferenceConfig(
        max_tokens=_DEMO_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        top_k=DEFAULT_TOP_K,
        repetition_penalty=1.0,
    )

    print("\nGenerating text without control vector (baseline)...")
    response_baseline = inferencer.generate(prompt=args.prompt, config=config)
    print("\nGenerated text without control vector:")
    print(response_baseline.text)
    print(f"\nToken usage: {response_baseline.usage}")

    print("\nGenerating text with custom control vector...")
    response_custom = inferencer.generate(
        prompt=args.prompt, control_vector=custom_control_vector, method=args.method, scale=args.scale, config=config
    )
    print("\nGenerated text with custom control vector:")
    print(response_custom.text)
    print(f"\nToken usage: {response_custom.usage}")
    print(f"Metadata: {response_custom.metadata}")

    print("\n=== Experimenting with Different Scaling Factors ===\n")
    for scale in _SCALES_TO_COMPARE:
        print(f"\nGenerating text with scale={scale}...")
        response = inferencer.generate(
            prompt=args.prompt, control_vector=custom_control_vector, method=args.method, scale=scale, config=config
        )
        print(f"\nGenerated text with scale={scale}:")
        print(response.text)
        print("\n" + _RULE)

    print("\n=== Comparing Different Control Vectors ===\n")
    if combined_vector is not None:
        print("\nGenerating text with combined control vector...")
        response_combined = inferencer.generate(
            prompt=args.prompt, control_vector=combined_vector, method=args.method, scale=args.scale, config=config
        )
        print("\nGenerated text with combined control vector:")
        print(response_combined.text)
        print("\n" + _RULE)

    print("\nGenerating text with random control vector...")
    response_random = inferencer.generate(
        prompt=args.prompt, control_vector=random_control_vector, method=args.method, scale=args.scale, config=config
    )
    print("\nGenerated text with random control vector:")
    print(response_random.text)


def main():
    """Create custom control vectors and use them for text generation."""
    args = parse_arguments()
    print(f"Initializing client with API key: {args.api_key[:5]}...")
    client = WisentClient(api_key=args.api_key, base_url=args.api_url)

    print("\n=== Creating a Custom Control Vector ===\n")
    custom_control_vector = vector_from_activations(args)
    combined_vector = combined_from_catalog(client, args)
    random_control_vector = random_vector(args, len(custom_control_vector.values))
    generate_with_vectors(args, custom_control_vector, combined_vector, random_control_vector)


if __name__ == "__main__":
    main()
