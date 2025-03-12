#!/usr/bin/env python
"""
Simple chat application using the Wisent library.

This script demonstrates how to use the Wisent library to create a simple
chat application that uses control vectors to steer the model's responses.
"""

import argparse
import os
import sys
import json
from typing import Dict, List, Optional

import torch

from wisent import WisentClient
from wisent.control_vector import ControlVectorManager
from wisent.inference import Inferencer, InferenceConfig


def format_prompt(messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
    """
    Format a list of messages into a prompt for the model.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        system_prompt: Optional system prompt to prepend
        
    Returns:
        Formatted prompt string
    """
    formatted_prompt = ""
    
    # Add system prompt if provided
    if system_prompt:
        formatted_prompt += f"<s>[INST] {system_prompt} [/INST]</s>\n\n"
    
    # Add conversation history
    for i, message in enumerate(messages):
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            # If this is the first message or the previous message was from the assistant
            if i == 0 or messages[i-1]["role"] == "assistant":
                formatted_prompt += f"<s>[INST] {content} [/INST]"
            else:
                # Continue the previous instruction
                formatted_prompt += f"\n\n{content} [/INST]"
        elif role == "assistant":
            formatted_prompt += f" {content}</s>"
    
    return formatted_prompt


def main():
    """Run the chat application."""
    parser = argparse.ArgumentParser(description="Simple chat application using Wisent")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1", help="Model name")
    parser.add_argument("--api-key", type=str, default=os.environ.get("WISENT_API_KEY"), help="Wisent API key")
    parser.add_argument("--api-url", type=str, default=os.environ.get("WISENT_API_URL", "https://api.wisent.ai"), help="Wisent API URL")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--local", action="store_true", help="Use local inference instead of API")
    parser.add_argument("--vectors", type=str, default="{}", help="JSON string of vector names and weights")
    parser.add_argument("--method", type=str, default="caa", help="Method for applying control vectors")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling factor for control vectors")
    parser.add_argument("--system-prompt", type=str, default="You are a helpful, honest, and concise assistant.", help="System prompt")
    
    args = parser.parse_args()
    
    # Check if API key is provided
    if not args.api_key:
        print("Error: API key is required. Set the WISENT_API_KEY environment variable or use --api-key.")
        sys.exit(1)
    
    # Parse control vectors
    try:
        vectors = json.loads(args.vectors)
    except json.JSONDecodeError:
        print("Error: Invalid JSON for vectors. Format should be: '{\"vector_name\": weight}'")
        sys.exit(1)
    
    # Initialize client or inferencer
    if args.local:
        print(f"Initializing local inferencer with model: {args.model}")
        inferencer = Inferencer(model_name=args.model, device=args.device)
        
        # Initialize control vector manager if vectors are specified
        if vectors:
            print(f"Using control vectors: {vectors}")
            manager = ControlVectorManager(api_key=args.api_key, base_url=args.api_url)
            combined_vector = manager.combine(vectors=vectors, model=args.model)
        else:
            combined_vector = None
    else:
        print(f"Initializing API client with model: {args.model}")
        client = WisentClient(api_key=args.api_key, base_url=args.api_url)
    
    # Create inference config
    config = InferenceConfig(
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.0
    )
    
    # Initialize conversation history
    messages = []
    
    print("\nWelcome to the Wisent Chat! Type 'exit' to quit.\n")
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # Format the prompt
        prompt = format_prompt(messages, args.system_prompt)
        
        # Generate response
        if args.local:
            if combined_vector:
                response = inferencer.generate(
                    prompt=prompt,
                    control_vector=combined_vector,
                    method=args.method,
                    scale=args.scale,
                    config=config
                )
            else:
                response = inferencer.generate(
                    prompt=prompt,
                    config=config
                )
            
            generated_text = response.text
        else:
            if vectors:
                response = client.inference.generate_with_control(
                    model_name=args.model,
                    prompt=prompt,
                    control_vectors=vectors,
                    method=args.method,
                    scale=args.scale,
                    config=config
                )
            else:
                response = client.inference.generate(
                    model_name=args.model,
                    prompt=prompt,
                    config=config
                )
            
            generated_text = response.text
        
        # Add assistant message to history
        messages.append({"role": "assistant", "content": generated_text})
        
        # Print the response
        print(f"\nAssistant: {generated_text}\n")


if __name__ == "__main__":
    main() 