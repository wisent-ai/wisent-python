"""
Functionality for local inference with control vectors.
"""

import logging
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from wisent.control_vector.models import ControlVector
from wisent.inference.models import ControlVectorInferenceConfig, InferenceConfig, InferenceResponse

logger = logging.getLogger(__name__)


class ControlVectorHook:
    """
    Hook for applying control vectors during inference.
    
    Args:
        control_vector: Control vector to apply
        config: Configuration for applying the control vector
    """
    
    def __init__(
        self,
        control_vector: ControlVector,
        config: ControlVectorInferenceConfig,
    ):
        self.control_vector = control_vector
        self.config = config
        self.device = None
        self.vector_tensor = None
        self.hooks = []
    
    def register(self, model):
        """
        Register hooks on the model.
        
        Args:
            model: The model to register hooks on
        """
        self.device = next(model.parameters()).device
        self.vector_tensor = self.control_vector.to_tensor(self.device)
        
        # Get transformer layers
        if hasattr(model, "transformer"):
            transformer_layers = model.transformer.h
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            transformer_layers = model.model.layers
        else:
            raise ValueError(f"Unsupported model architecture: {model.__class__.__name__}")
        
        # Determine which layers to apply the control vector to
        num_layers = len(transformer_layers)
        layers = self.config.layers or [num_layers - 1]  # Default to last layer
        
        # Resolve negative indices
        resolved_layers = []
        for layer in layers:
            if layer < 0:
                resolved_layer = num_layers + layer
            else:
                resolved_layer = layer
                
            if 0 <= resolved_layer < num_layers:
                resolved_layers.append(resolved_layer)
        
        # Register hooks
        for layer_idx in resolved_layers:
            layer = transformer_layers[layer_idx]
            
            # Define hook function
            def hook_fn(module, input, output, layer_idx=layer_idx):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Apply the control vector
                if self.config.method == "caa":  # Context-Aware Addition
                    # Add the control vector to the hidden states
                    modified = hidden_states + self.vector_tensor * self.config.scale
                    
                    if isinstance(output, tuple):
                        return (modified,) + output[1:]
                    else:
                        return modified
                else:
                    logger.warning(f"Unsupported method: {self.config.method}, using original output")
                    return output
            
            # Register hook
            if hasattr(layer, "output"):
                handle = layer.output.register_forward_hook(
                    lambda module, input, output, layer_idx=layer_idx: hook_fn(module, input, output, layer_idx)
                )
            else:
                handle = layer.register_forward_hook(
                    lambda module, input, output, layer_idx=layer_idx: hook_fn(module, input, output, layer_idx)
                )
            
            self.hooks.append(handle)
    
    def remove(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class Inferencer:
    """
    Performs local inference with control vectors.
    
    Args:
        model_name: Name of the model
        device: Device to use for inference
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initializing Inferencer for model {model_name} on {self.device}")
    
    def _load_model(self):
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
    
    def generate(
        self,
        prompt: str,
        control_vector: Optional[ControlVector] = None,
        method: str = "caa",
        scale: float = 1.0,
        layers: Optional[List[int]] = None,
        config: Optional[InferenceConfig] = None,
    ) -> InferenceResponse:
        """
        Generate text using the model, optionally with a control vector.
        
        Args:
            prompt: Input prompt
            control_vector: Control vector to apply (optional)
            method: Method for applying the control vector
            scale: Scaling factor for the control vector
            layers: Layers to apply the control vector to
            config: Inference configuration
            
        Returns:
            Inference response
        """
        try:
            self._load_model()
            
            config = config or InferenceConfig()
            hook = None
            
            # Register control vector hook if provided
            if control_vector is not None:
                cv_config = ControlVectorInferenceConfig(
                    method=method,
                    scale=scale,
                    layers=layers,
                )
                hook = ControlVectorHook(control_vector, cv_config)
                hook.register(self.model)
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_length = inputs.input_ids.shape[1]
            
            # Configure generation
            generation_config = GenerationConfig(
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    generation_config=generation_config,
                )
            
            # Remove control vector hook if registered
            if hook is not None:
                hook.remove()
            
            # Decode output
            generated_text = self.tokenizer.decode(
                output_ids[0][prompt_length:], 
                skip_special_tokens=True
            )
            
            # Create response
            return InferenceResponse(
                text=generated_text,
                model=self.model_name,
                prompt=prompt,
                finish_reason="length",  # Simplified
                usage={
                    "prompt_tokens": prompt_length,
                    "completion_tokens": output_ids.shape[1] - prompt_length,
                    "total_tokens": output_ids.shape[1],
                },
                metadata={
                    "control_vector": control_vector.name if control_vector else None,
                    "method": method if control_vector else None,
                    "scale": scale if control_vector else None,
                }
            )
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            if hook is not None:
                hook.remove()
            raise
    
    def __del__(self):
        """Clean up resources."""
        # Free GPU memory
        if self.model is not None and hasattr(self.model, "to"):
            self.model = self.model.to("cpu")
            
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 