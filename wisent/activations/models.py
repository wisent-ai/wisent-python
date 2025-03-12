"""
Data models for model activations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel, Field


class Activation(BaseModel):
    """
    Represents a single activation from a model.
    
    Attributes:
        model_name: Name of the model
        layer: Layer index
        token_index: Token index
        values: Activation values
        token_str: String representation of the token (optional)
    """
    
    model_name: str
    layer: int
    token_index: int
    values: Union[List[float], np.ndarray, torch.Tensor]
    token_str: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API requests."""
        values = self.values
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        if isinstance(values, np.ndarray):
            values = values.tolist()
            
        return {
            "model_name": self.model_name,
            "layer": self.layer,
            "token_index": self.token_index,
            "values": values,
            "token_str": self.token_str,
        }


class ActivationBatch(BaseModel):
    """
    Represents a batch of activations from a model.
    
    Attributes:
        model_name: Name of the model
        prompt: Input prompt that generated the activations
        activations: List of activations
        metadata: Additional metadata (optional)
    """
    
    model_name: str
    prompt: str
    activations: List[Activation]
    metadata: Optional[Dict] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API requests."""
        return {
            "model_name": self.model_name,
            "prompt": self.prompt,
            "activations": [a.to_dict() for a in self.activations],
            "metadata": self.metadata or {},
        }


@dataclass
class ActivationExtractorConfig:
    """
    Configuration for activation extraction.
    
    Attributes:
        layers: List of layers to extract activations from
        tokens_to_extract: List of token indices to extract (negative indices count from the end)
        batch_size: Batch size for processing
        device: Device to use for extraction
    """
    
    layers: List[int] = field(default_factory=lambda: [-1])
    tokens_to_extract: List[int] = field(default_factory=lambda: [-1])
    batch_size: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu" 