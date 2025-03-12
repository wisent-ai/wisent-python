#!/usr/bin/env python
"""
Extract activations from a model and optionally upload them to the Wisent backend.

This script demonstrates how to extract activations from a language model
and optionally upload them to the Wisent backend for further analysis or
to create control vectors.
"""

import argparse
import logging
import os
import sys
from typing import List

import torch

from wisent import WisentClient
from wisent.activations import ActivationExtractor


def parse_layers(layers_str: str) -> List[int]:
    """
    Parse a comma-separated string of layer indices into a list of integers.
    
    Args:
        layers_str: Comma-separated string of layer indices (e.g., "0,1,2" or "-1,-2,-3")
        
    Returns:
        List of layer indices as integers
    """
    return [int(layer.strip()) for layer in layers_str.split(",")]


def parse_tokens(tokens_str: str) -> List[int]:
    """
    Parse a comma-separated string of token indices into a list of integers.
    
    Args:
        tokens_str: Comma-separated string of token indices (e.g., "0,1,2" or "-1,-2,-3")
        
    Returns:
        List of token indices as integers
    """
    return [int(token.strip()) for token in tokens_str.split(",")]


def main():
    """Extract activations from a model and optionally upload them to the Wisent backend."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract activations from a model")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1", help="Model name")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to extract activations for")
    parser.add_argument("--layers", type=str, default="-1", help="Comma-separated list of layers to extract (e.g., '0,1,2' or '-1,-2,-3')")
    parser.add_argument("--tokens", type=str, default="-1", help="Comma-separated list of tokens to extract (e.g., '0,1,2' or '-1,-2,-3')")
    parser.add_argument("--api-key", type=str, default=os.environ.get("WISENT_API_KEY"), help="Wisent API key")
    parser.add_argument("--api-url", type=str, default=os.environ.get("WISENT_API_URL", "https://api.wisent.ai"), help="Wisent API URL")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--upload", action="store_true", help="Upload activations to the Wisent backend")
    
    args = parser.parse_args()
    
    # Parse layers and tokens
    layers = parse_layers(args.layers)
    tokens = parse_tokens(args.tokens)
    
    logger.info(f"Extracting activations from model: {args.model}")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Tokens: {tokens}")
    logger.info(f"Device: {args.device}")
    
    # Create an extractor
    extractor = ActivationExtractor(
        model_name=args.model,
        device=args.device
    )
    
    # Extract activations
    logger.info("Extracting activations...")
    activations = extractor.extract(
        prompt=args.prompt,
        layers=layers,
        tokens_to_extract=tokens
    )
    
    logger.info(f"Extracted {len(activations.activations)} activations")
    
    # Print information about each activation
    for i, activation in enumerate(activations.activations):
        logger.info(f"Activation {i+1}:")
        logger.info(f"  Layer: {activation.layer}")
        logger.info(f"  Token index: {activation.token_index}")
        logger.info(f"  Token string: {activation.token_str}")
        
        # Convert to numpy array if it's a tensor
        values = activation.values
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        logger.info(f"  Values shape: {values.shape}")
    
    # Upload activations if requested
    if args.upload:
        if not args.api_key:
            logger.error("API key is required for uploading activations. Set the WISENT_API_KEY environment variable or use --api-key.")
            sys.exit(1)
        
        logger.info("Uploading activations to the Wisent backend...")
        client = WisentClient(api_key=args.api_key, base_url=args.api_url)
        response = client.activations.upload(activations)
        
        logger.info(f"Upload response: {response}")
        logger.info(f"Batch ID: {response.get('id')}")


if __name__ == "__main__":
    main() 