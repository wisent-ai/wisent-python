"""
Client for interacting with the activations API.
"""

from typing import Dict, List, Optional, Union

from wisent.activations.extractor import ActivationExtractor
from wisent.activations.models import Activation, ActivationBatch
from wisent.utils.auth import AuthManager
from wisent.utils.http import HTTPClient


class ActivationsClient:
    """
    Client for interacting with the activations API.
    
    Args:
        auth_manager: Authentication manager
        base_url: Base URL for the API
        timeout: Request timeout in seconds
    """
    
    def __init__(self, auth_manager: AuthManager, base_url: str, timeout: int = 60):
        self.auth_manager = auth_manager
        self.http_client = HTTPClient(base_url, auth_manager.get_headers(), timeout)
    
    def extract(
        self,
        model_name: str,
        prompt: str,
        layers: Optional[List[int]] = None,
        tokens_to_extract: Optional[List[int]] = None,
        device: Optional[str] = None,
    ) -> ActivationBatch:
        """
        Extract activations from a model for a given prompt.
        
        Args:
            model_name: Name of the model
            prompt: Input prompt
            layers: List of layers to extract activations from (default: [-1])
            tokens_to_extract: List of token indices to extract (default: [-1])
            device: Device to use for extraction (default: "cuda" if available, else "cpu")
            
        Returns:
            Batch of activations
        """
        extractor = ActivationExtractor(model_name, device=device)
        return extractor.extract(prompt, layers, tokens_to_extract)
    
    def upload(self, batch: ActivationBatch) -> Dict:
        """
        Upload a batch of activations to the Wisent backend.
        
        Args:
            batch: Batch of activations
            
        Returns:
            Response from the API
        """
        return self.http_client.post("/activations/upload", json_data=batch.to_dict())
    
    def get(self, batch_id: str) -> ActivationBatch:
        """
        Get a batch of activations from the Wisent backend.
        
        Args:
            batch_id: ID of the batch
            
        Returns:
            Batch of activations
        """
        data = self.http_client.get(f"/activations/{batch_id}")
        return ActivationBatch(**data)
    
    def list(
        self,
        model_name: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """
        List activation batches from the Wisent backend.
        
        Args:
            model_name: Filter by model name
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of activation batch metadata
        """
        params = {"limit": limit, "offset": offset}
        if model_name:
            params["model_name"] = model_name
            
        return self.http_client.get("/activations", params=params) 