"""
Manager for working with control vectors.
"""

import logging
from typing import Dict, List, Optional, Union

import torch

from wisent.control_vector.models import ControlVector, ControlVectorConfig
from wisent.utils.auth import AuthManager
from wisent.utils.http import HTTPClient

logger = logging.getLogger(__name__)


class ControlVectorManager:
    """
    Manager for working with control vectors.
    
    Args:
        api_key: Wisent API key
        base_url: Base URL for the API
        timeout: Request timeout in seconds
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.wisent.ai",
        timeout: int = 60,
    ):
        self.auth = AuthManager(api_key)
        self.http_client = HTTPClient(base_url, self.auth.get_headers(), timeout)
        self.cache = {}  # Simple in-memory cache
    
    def get(self, name: str, model: str) -> ControlVector:
        """
        Get a control vector from the Wisent backend.
        
        Args:
            name: Name of the control vector
            model: Model name
            
        Returns:
            Control vector
        """
        cache_key = f"{name}:{model}"
        if cache_key in self.cache:
            logger.info(f"Using cached control vector: {name} for model {model}")
            return self.cache[cache_key]
        
        logger.info(f"Fetching control vector: {name} for model {model}")
        data = self.http_client.get(f"/control_vectors/{name}", params={"model": model})
        vector = ControlVector(**data)
        
        # Cache the result
        self.cache[cache_key] = vector
        
        return vector
    
    def list(
        self,
        model: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """
        List available control vectors from the Wisent backend.
        
        Args:
            model: Filter by model name
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of control vector metadata
        """
        params = {"limit": limit, "offset": offset}
        if model:
            params["model"] = model
            
        return self.http_client.get("/control_vectors", params=params)
    
    def combine(
        self,
        vectors: Dict[str, float],
        model: str,
    ) -> ControlVector:
        """
        Combine multiple control vectors with weights.
        
        Args:
            vectors: Dictionary mapping vector names to weights
            model: Model name
            
        Returns:
            Combined control vector
        """
        # Check if we can combine locally
        can_combine_locally = True
        local_vectors = {}
        
        for name in vectors.keys():
            cache_key = f"{name}:{model}"
            if cache_key not in self.cache:
                can_combine_locally = False
                break
            local_vectors[name] = self.cache[cache_key]
        
        if can_combine_locally:
            logger.info(f"Combining vectors locally for model {model}")
            return self._combine_locally(local_vectors, vectors, model)
        
        # Otherwise, use the API
        logger.info(f"Combining vectors via API for model {model}")
        data = self.http_client.post(
            "/control_vectors/combine",
            json_data={
                "vectors": vectors,
                "model": model,
            }
        )
        return ControlVector(**data)
    
    def _combine_locally(
        self,
        vectors: Dict[str, ControlVector],
        weights: Dict[str, float],
        model: str,
    ) -> ControlVector:
        """
        Combine vectors locally.
        
        Args:
            vectors: Dictionary mapping vector names to ControlVector objects
            weights: Dictionary mapping vector names to weights
            model: Model name
            
        Returns:
            Combined control vector
        """
        # Convert all vectors to tensors
        tensor_vectors = {}
        for name, vector in vectors.items():
            tensor_vectors[name] = vector.to_tensor()
        
        # Get the shape from the first vector
        first_vector = next(iter(tensor_vectors.values()))
        combined = torch.zeros_like(first_vector)
        
        # Combine vectors with weights
        for name, weight in weights.items():
            if name in tensor_vectors:
                combined += tensor_vectors[name] * weight
        
        # Create a new control vector
        vector_names = list(weights.keys())
        combined_name = f"combined_{'_'.join(vector_names)}"
        
        return ControlVector(
            name=combined_name,
            model_name=model,
            values=combined,
            metadata={
                "combined_from": {name: weight for name, weight in weights.items()},
            }
        ) 