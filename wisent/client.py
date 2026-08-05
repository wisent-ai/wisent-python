"""
Main client class for interacting with the Wisent backend services.
"""

from typing import Dict, Optional

from wisent.activations import ActivationsClient
from wisent.control_vector import ControlVectorClient
from wisent.inference import InferenceClient
from wisent.onboarding import FirstUseRuntime
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
        
        # Initialize the durable first-use runtime without advancing it.
        self.first_use = FirstUseRuntime()

        # Initialize sub-clients. Only a parsed result from the supported,
        # authenticated inference operation is allowed to complete first use.
        result_observer = (
            self.first_use._observe_api_result
            if isinstance(api_key, str) and bool(api_key.strip())
            else None
        )
        self.activations = ActivationsClient(self.auth, base_url, timeout)
        self.control_vector = ControlVectorClient(self.auth, base_url, timeout)
        self.inference = InferenceClient(
            self.auth,
            base_url,
            timeout,
            result_observer=result_observer,
        )
    
    def __repr__(self) -> str:
        return f"WisentClient(base_url='{self.base_url}')"
