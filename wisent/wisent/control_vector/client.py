"""
Client for interacting with the control vector API.
"""

from typing import Dict, List, Optional, Union

from wisent.control_vector.models import ControlVector
from wisent.utils.auth import AuthManager
from wisent.utils.http import HTTPClient


class ControlVectorClient:
    """
    Client for interacting with the control vector API.
    
    Args:
        auth_manager: Authentication manager
        base_url: Base URL for the API
        timeout: Request timeout in seconds
    """
    
    def __init__(self, auth_manager: AuthManager, base_url: str, timeout: int = 60):
        self.auth_manager = auth_manager
        self.http_client = HTTPClient(base_url, auth_manager.get_headers(), timeout)
    
    def get(self, name: str, model: str) -> ControlVector:
        """
        Get a control vector from the Wisent backend.
        
        Args:
            name: Name of the control vector
            model: Model name
            
        Returns:
            Control vector
        """
        data = self.http_client.get(f"/control_vectors/{name}", params={"model": model})
        return ControlVector(**data)
    
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
        data = self.http_client.post(
            "/control_vectors/combine",
            json_data={
                "vectors": vectors,
                "model": model,
            }
        )
        return ControlVector(**data) 