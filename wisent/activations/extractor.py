"""
Functionality for extracting activations from models.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.hooks import RemovableHandle
from transformers import AutoModelForCausalLM, AutoTokenizer

from wisent.activations.models import Activation, ActivationBatch, ActivationExtractorConfig

logger = logging.getLogger(__name__)


class ActivationExtractor:
    """
    Extracts activations from transformer models.
    
    Args:
        model_name: Name of the model to extract activations from
        config: Configuration for extraction
        device: Device to use for extraction
    """
    
    def __init__(
        self,
        model_name: str,
        config: Optional[ActivationExtractorConfig] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.config = config or ActivationExtractorConfig()
        
        if device:
            self.config.device = device
            
        self.device = self.config.device
        self.model = None
        self.tokenizer = None
        self._hooks = []
        self._activations = {}
        
        logger.info(f"Initializing ActivationExtractor for model {model_name} on {self.device}")
    
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        if self.model is None:
            logger.info(f"Loading model {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"Model loaded successfully")
    
    def _register_hooks(self, layers: List[int]) -> None:
        """
        Register hooks to capture activations from specified layers.
        
        Args:
            layers: List of layer indices to capture
        """
        self._remove_hooks()
        self._activations = {}
        
        # Get all transformer layers
        if hasattr(self.model, "transformer"):
            transformer_layers = self.model.transformer.h
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            transformer_layers = self.model.model.layers
        else:
            raise ValueError(f"Unsupported model architecture: {self.model_name}")
        
        num_layers = len(transformer_layers)
        
        # Resolve negative indices
        resolved_layers = []
        for layer in layers:
            if layer < 0:
                resolved_layer = num_layers + layer
            else:
                resolved_layer = layer
                
            if 0 <= resolved_layer < num_layers:
                resolved_layers.append(resolved_layer)
            else:
                logger.warning(f"Layer index {layer} out of range (0-{num_layers-1}), skipping")
        
        # Register hooks for each layer
        for layer_idx in resolved_layers:
            layer = transformer_layers[layer_idx]
            
            # Define hook function to capture activations
            def hook_fn(module, input, output, layer_idx=layer_idx):
                # For most models, output is a tuple with hidden states as the first element
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                if layer_idx not in self._activations:
                    self._activations[layer_idx] = []
                
                # Store a copy of the hidden states
                self._activations[layer_idx].append(hidden_states.detach())
            
            # Register hook on the output of the layer
            if hasattr(layer, "output"):
                handle = layer.output.register_forward_hook(
                    lambda module, input, output, layer_idx=layer_idx: hook_fn(module, input, output, layer_idx)
                )
            else:
                handle = layer.register_forward_hook(
                    lambda module, input, output, layer_idx=layer_idx: hook_fn(module, input, output, layer_idx)
                )
            
            self._hooks.append(handle)
            
        logger.info(f"Registered hooks for layers: {resolved_layers}")
    
    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def _get_token_indices(self, tokens_to_extract: List[int], total_tokens: int) -> List[int]:
        """
        Resolve token indices, handling negative indices.
        
        Args:
            tokens_to_extract: List of token indices to extract
            total_tokens: Total number of tokens
            
        Returns:
            List of resolved token indices
        """
        resolved_indices = []
        
        for idx in tokens_to_extract:
            if idx < 0:
                resolved_idx = total_tokens + idx
            else:
                resolved_idx = idx
                
            if 0 <= resolved_idx < total_tokens:
                resolved_indices.append(resolved_idx)
            else:
                logger.warning(f"Token index {idx} out of range (0-{total_tokens-1}), skipping")
        
        return resolved_indices
    
    def extract(
        self,
        prompt: str,
        layers: Optional[List[int]] = None,
        tokens_to_extract: Optional[List[int]] = None,
    ) -> ActivationBatch:
        """
        Extract activations from the model for a given prompt.
        
        Args:
            prompt: Input prompt
            layers: List of layers to extract activations from (default: from config)
            tokens_to_extract: List of token indices to extract (default: from config)
            
        Returns:
            Batch of activations
        """
        try:
            self._load_model()
            
            layers = layers or self.config.layers
            tokens_to_extract = tokens_to_extract or self.config.tokens_to_extract
            
            # Register hooks for the specified layers
            self._register_hooks(layers)
            
            # Tokenize the input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            
            # Get the total number of tokens
            total_tokens = input_ids.shape[1]
            
            # Resolve token indices
            token_indices = self._get_token_indices(tokens_to_extract, total_tokens)
            
            # Run the model to capture activations
            with torch.no_grad():
                self.model(input_ids)
            
            # Process captured activations
            activations = []
            
            for layer_idx, layer_activations in self._activations.items():
                # Layer activations should have shape [batch_size, seq_len, hidden_dim]
                hidden_states = layer_activations[0]
                
                # Get token strings for the specified indices
                token_strings = {}
                for token_idx in token_indices:
                    token_id = input_ids[0, token_idx].item()
                    token_strings[token_idx] = self.tokenizer.decode([token_id])
                
                # Extract activations for the specified tokens
                for token_idx in token_indices:
                    # Extract the activation for this token
                    token_activation = hidden_states[0, token_idx, :].cpu()
                    
                    # Create an Activation object
                    activation = Activation(
                        model_name=self.model_name,
                        layer=layer_idx,
                        token_index=token_idx,
                        values=token_activation,
                        token_str=token_strings.get(token_idx)
                    )
                    
                    activations.append(activation)
            
            # Clean up
            self._remove_hooks()
            
            # Create and return the batch
            return ActivationBatch(
                model_name=self.model_name,
                prompt=prompt,
                activations=activations,
                metadata={"total_tokens": total_tokens}
            )
            
        except Exception as e:
            logger.error(f"Error extracting activations: {str(e)}")
            self._remove_hooks()
            raise
    
    def __del__(self):
        """Clean up resources."""
        self._remove_hooks()
        
        # Free GPU memory
        if self.model is not None and hasattr(self.model, "to"):
            self.model = self.model.to("cpu")
            
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 