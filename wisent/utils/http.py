"""
HTTP request utilities for the Wisent API.
"""

import json
from typing import Any, Dict, Optional, Union

import aiohttp
import requests
from requests.exceptions import RequestException


class APIError(Exception):
    """Exception raised for API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)


class HTTPClient:
    """
    HTTP client for making requests to the Wisent API.
    
    Args:
        base_url: The base URL for the API
        headers: Headers to include in all requests
        timeout: Request timeout in seconds
    """
    
    def __init__(self, base_url: str, headers: Dict[str, str], timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.headers = headers
        self.timeout = timeout
    
    def _build_url(self, endpoint: str) -> str:
        """Build the full URL for an API endpoint."""
        endpoint = endpoint.lstrip("/")
        return f"{self.base_url}/{endpoint}"
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a GET request to the API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Response data as a dictionary
            
        Raises:
            APIError: If the request fails
        """
        url = self._build_url(endpoint)
        try:
            response = requests.get(
                url, 
                headers=self.headers, 
                params=params, 
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            status_code = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
            response_data = None
            
            if hasattr(e, "response") and e.response is not None:
                try:
                    response_data = e.response.json()
                except (ValueError, AttributeError):
                    response_data = {"error": str(e)}
                    
            raise APIError(
                f"GET request to {url} failed: {str(e)}", 
                status_code=status_code,
                response=response_data
            ) from e
    
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None, json_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a POST request to the API.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json_data: JSON data
            
        Returns:
            Response data as a dictionary
            
        Raises:
            APIError: If the request fails
        """
        url = self._build_url(endpoint)
        try:
            response = requests.post(
                url, 
                headers=self.headers, 
                data=data, 
                json=json_data, 
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            status_code = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
            response_data = None
            
            if hasattr(e, "response") and e.response is not None:
                try:
                    response_data = e.response.json()
                except (ValueError, AttributeError):
                    response_data = {"error": str(e)}
                    
            raise APIError(
                f"POST request to {url} failed: {str(e)}", 
                status_code=status_code,
                response=response_data
            ) from e


class AsyncHTTPClient:
    """
    Asynchronous HTTP client for making requests to the Wisent API.
    
    Args:
        base_url: The base URL for the API
        headers: Headers to include in all requests
        timeout: Request timeout in seconds
    """
    
    def __init__(self, base_url: str, headers: Dict[str, str], timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.headers = headers
        self.timeout = timeout
    
    def _build_url(self, endpoint: str) -> str:
        """Build the full URL for an API endpoint."""
        endpoint = endpoint.lstrip("/")
        return f"{self.base_url}/{endpoint}"
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an asynchronous GET request to the API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Response data as a dictionary
            
        Raises:
            APIError: If the request fails
        """
        url = self._build_url(endpoint)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, 
                    headers=self.headers, 
                    params=params, 
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    return await response.json()
        except aiohttp.ClientError as e:
            status_code = getattr(response, "status", None) if 'response' in locals() else None
            response_data = None
            
            if 'response' in locals():
                try:
                    response_data = await response.json()
                except (ValueError, AttributeError):
                    response_data = {"error": str(e)}
                    
            raise APIError(
                f"Async GET request to {url} failed: {str(e)}", 
                status_code=status_code,
                response=response_data
            ) from e
    
    async def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None, json_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an asynchronous POST request to the API.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json_data: JSON data
            
        Returns:
            Response data as a dictionary
            
        Raises:
            APIError: If the request fails
        """
        url = self._build_url(endpoint)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    headers=self.headers, 
                    data=data, 
                    json=json_data, 
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    return await response.json()
        except aiohttp.ClientError as e:
            status_code = getattr(response, "status", None) if 'response' in locals() else None
            response_data = None
            
            if 'response' in locals():
                try:
                    response_data = await response.json()
                except (ValueError, AttributeError):
                    response_data = {"error": str(e)}
                    
            raise APIError(
                f"Async POST request to {url} failed: {str(e)}", 
                status_code=status_code,
                response=response_data
            ) from e 