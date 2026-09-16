#!/usr/bin/env python
"""
Generate text with control vectors.

This script demonstrates how to generate text using control vectors to steer
the output of language models in specific directions. It supports both local
inference and API-based inference.
"""

import argparse
import json
import logging
import os
import sys

import torch

from wisent import WisentClient
from wisent.control_vector import ControlVectorManager
from wisent.inference import Inferencer, InferenceConfig


def main():
    """Generate text with control vectors."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate text with control vectors")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1", help="Model name")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--vectors", type=str, required=True, help="JSON string of vector names and weights (e.g., '{\"helpful\": 0.8, \"concise\": 0.5}')")
    parser.add_argument("--method", type=str, default="caa", help="Method for applying control vectors (e.g., 'caa', 'add')")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling factor for control vectors")
    parser.add_argument("--api-key", type=str, default=os.environ.get("WISENT_API_KEY"), help="Wisent API key")
    parser.add_argument("--api-url", type=str, default=os.environ.get("WISENT_API_URL", "https://api.wisent.ai"), help="Wisent API URL")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--local", action="store_true", help="Use local inference instead of API")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Parse control vectors
    try:
        vectors = json.loads(args.vectors)
        if not isinstance(vectors, dict):
            logger.error("Vectors must be a dictionary mapping vector names to weights")
            sys.exit(1)
    except json.JSONDecodeError:
        logger.error("Invalid JSON for vectors. Format should be: '{\"vector_name\": weight}'")
        sys.exit(1)
    
    logger.info(f"Generating text with model: {args.model}")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Control vectors: {vectors}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Scale: {args.scale}")
    
    # Check if API key is provided
    if not args.api_key:
        logger.error("API key is required. Set the WISENT_API_KEY environment variable or use --api-key.")
        sys.exit(1)
    
    # Create inference config
    config = InferenceConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.0
    )
    
    try:
        # Generate text
        if args.local:
            # Local inference
            logger.info("Using local inference")
            
            # Initialize control vector manager
            manager = ControlVectorManager(api_key=args.api_key, base_url=args.api_url)
            
            # Get and combine vectors
            logger.info("Retrieving and combining control vectors...")
            combined_vector = manager.combine(vectors=vectors, model=args.model)
            
            # Initialize inferencer
            logger.info(f"Initializing inferencer with model: {args.model} on device: {args.device}")
            inferencer = Inferencer(model_name=args.model, device=args.device)
            
            # Generate
            logger.info("Generating text...")
            response = inferencer.generate(
                prompt=args.prompt,
                control_vector=combined_vector,
                method=args.method,
                scale=args.scale,
                config=config
            )
        else:
            # API inference
            logger.info("Using API inference")
            
            # Initialize client
            client = WisentClient(api_key=args.api_key, base_url=args.api_url)
            
            # Generate
            logger.info("Generating text...")
            response = client.inference.generate_with_control(
                model_name=args.model,
                prompt=args.prompt,
                control_vectors=vectors,
                method=args.method,
                scale=args.scale,
                config=config
            )
        
        # Print the response
        print("\n" + "=" * 40 + " GENERATED TEXT " + "=" * 40)
        print(response.text)
        print("=" * 100)
        
        # Print usage info
        print(f"Tokens: {response.usage}")
        
        # Print metadata
        if response.metadata:
            print("\nMetadata:")
            for key, value in response.metadata.items():
                print(f"  {key}: {value}")
        
        logger.info("Text generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 