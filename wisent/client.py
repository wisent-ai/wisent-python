"""
Main client class for interacting with the Wisent backend services.
"""

from typing import Dict, Optional

from wisent.activations import ActivationsClient
from wisent.control_vector import ControlVectorClient
from wisent.inference import InferenceClient
from wisent.utils.auth import AuthManager


class WisentClient:
    """
    Main client for interacting with the Wisent backend services.
    
    This client provides access to all Wisent API functionality through
    specialized sub-clients for different features.
    
    Args:
        api_key: Your Wisent API key
        base_url: The base URL for the Wisent API (default: https://api.wisent.ai)
        timeout: Request timeout in seconds (default: 60)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.wisent.ai",
        timeout: int = 60,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        
        # Initialize auth manager
        self.auth = AuthManager(api_key)
        
        # Initialize sub-clients
        self.activations = ActivationsClient(self.auth, base_url, timeout)
        self.control_vector = ControlVectorClient(self.auth, base_url, timeout)
        self.inference = InferenceClient(self.auth, base_url, timeout)
    
    def __repr__(self) -> str:
        return f"WisentClient(base_url='{self.base_url}')"
