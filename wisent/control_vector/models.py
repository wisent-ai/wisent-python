"""
Data models for control vectors.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel, Field


class ControlVector(BaseModel):
    """
    Represents a control vector for steering model outputs.
    
    Attributes:
        name: Name of the control vector
        model_name: Name of the model the vector is for
        values: Vector values
        metadata: Additional metadata
    """
    
    name: str
    model_name: str
    values: Union[List[float], np.ndarray, torch.Tensor]
    metadata: Optional[Dict] = Field(default_factory=dict)
    
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
            "name": self.name,
            "model_name": self.model_name,
            "values": values,
            "metadata": self.metadata or {},
        }
    
    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Convert values to a PyTorch tensor."""
        if isinstance(self.values, torch.Tensor):
            return self.values.to(device)
        elif isinstance(self.values, np.ndarray):
            return torch.tensor(self.values, device=device)
        else:
            return torch.tensor(self.values, device=device)


@dataclass
class ControlVectorConfig:
    """
    Configuration for control vector application.
    
    Attributes:
        scale: Scaling factor for the control vector
        method: Method for applying the control vector
        layers: Layers to apply the control vector to
    """
    
    scale: float = 1.0
    method: str = "caa"  # Context-Aware Addition
    layers: Optional[List[int]] = None 