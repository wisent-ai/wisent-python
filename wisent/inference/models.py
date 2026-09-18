"""
Data models for inference.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from wisent.constants import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P


class InferenceConfig(BaseModel):
    """
    Configuration for model inference.
    
    Attributes:
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty
        stop_sequences: Sequences that stop generation
    """
    
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    repetition_penalty: float = 1.0
    stop_sequences: Optional[List[str]] = None


class InferenceResponse(BaseModel):
    """
    Response from model inference.
    
    Attributes:
        text: Generated text
        model: Model used for generation
        prompt: Input prompt
        finish_reason: Reason generation stopped
        usage: Token usage information
        metadata: Additional metadata
    """
    
    text: str
    model: str
    prompt: str
    finish_reason: str = "length"
    usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    metadata: Dict = Field(default_factory=dict)


@dataclass
class ControlVectorInferenceConfig:
    """
    Configuration for inference with control vectors.
    
    Attributes:
        method: Method for applying control vectors
        scale: Scaling factor for control vectors
        layers: Layers to apply control vectors to
    """
    
    method: str = "caa"  # Context-Aware Addition
    scale: float = 1.0
    layers: Optional[List[int]] = None 