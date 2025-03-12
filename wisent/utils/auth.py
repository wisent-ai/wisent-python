"""
Authentication utilities for the Wisent API.
"""

from typing import Dict


class AuthManager:
    """
    Manages authentication for Wisent API requests.
    
    Args:
        api_key: The Wisent API key
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def get_headers(self) -> Dict[str, str]:
        """
        Get the authentication headers for API requests.
        
        Returns:
            Dict containing the authentication headers
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        } 