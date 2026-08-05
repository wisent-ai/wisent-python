"""
Client for interacting with the inference API.
"""

from typing import Callable, Dict, List, Optional, Union

from wisent.inference.models import InferenceConfig, InferenceResponse
from wisent.utils.auth import AuthManager
from wisent.utils.http import HTTPClient


class InferenceClient:
    """
    Client for interacting with the inference API.
    
    Args:
        auth_manager: Authentication manager
        base_url: Base URL for the API
        timeout: Request timeout in seconds
    """
    
    def __init__(
        self,
        auth_manager: AuthManager,
        base_url: str,
        timeout: int = 60,
        result_observer: Optional[Callable[[str, Dict], None]] = None,
    ):
        self.auth_manager = auth_manager
        self.http_client = HTTPClient(base_url, auth_manager.get_headers(), timeout)
        self._result_observer = result_observer

    def _notify_result(self, operation: str, result: Dict) -> None:
        """Keep onboarding bookkeeping from changing the API result contract."""
        if self._result_observer is None:
            return
        try:
            self._result_observer(operation, result)
        except Exception:
            # The result has already succeeded and parsed. A local onboarding
            # storage or delivery problem must not turn it into an API error.
            return
    
    def generate(
        self,
        model_name: str,
        prompt: str,
        config: Optional[InferenceConfig] = None,
    ) -> InferenceResponse:
        """
        Generate text using a model.
        
        Args:
            model_name: Name of the model
            prompt: Input prompt
            config: Inference configuration
            
        Returns:
            Inference response
        """
        config = config or InferenceConfig()
        
        data = self.http_client.post(
            "/inference/generate",
            json_data={
                "model": model_name,
                "prompt": prompt,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repetition_penalty": config.repetition_penalty,
                "stop_sequences": config.stop_sequences,
            }
        )
        
        result = InferenceResponse(**data)
        self._notify_result("inference.generate", data)
        return result
    
    def generate_with_control(
        self,
        model_name: str,
        prompt: str,
        control_vectors: Dict[str, float],
        method: str = "caa",
        scale: float = 1.0,
        config: Optional[InferenceConfig] = None,
    ) -> InferenceResponse:
        """
        Generate text using a model with control vectors.
        
        Args:
            model_name: Name of the model
            prompt: Input prompt
            control_vectors: Dictionary mapping vector names to weights
            method: Method for applying control vectors
            scale: Scaling factor for control vectors
            config: Inference configuration
            
        Returns:
            Inference response
        """
        config = config or InferenceConfig()
        
        data = self.http_client.post(
            "/inference/generate_with_control",
            json_data={
                "model": model_name,
                "prompt": prompt,
                "control_vectors": control_vectors,
                "method": method,
                "scale": scale,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repetition_penalty": config.repetition_penalty,
                "stop_sequences": config.stop_sequences,
            }
        )
        
        return InferenceResponse(**data) 