"""
Tests for the WisentClient class.
"""

import unittest
from unittest.mock import MagicMock, patch

from wisent import WisentClient
from wisent.utils.auth import AuthManager
from wisent.utils.http import HTTPClient


class TestWisentClient(unittest.TestCase):
    """Tests for the WisentClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test_api_key"
        self.base_url = "https://test.api.wisent.ai"
        self.client = WisentClient(api_key=self.api_key, base_url=self.base_url)
    
    def test_init(self):
        """Test initialization of the client."""
        self.assertEqual(self.client.api_key, self.api_key)
        self.assertEqual(self.client.base_url, self.base_url)
        self.assertEqual(self.client.timeout, 60)
        
        # Test with custom timeout
        client = WisentClient(api_key=self.api_key, base_url=self.base_url, timeout=30)
        self.assertEqual(client.timeout, 30)
    
    def test_auth_manager(self):
        """Test that the auth manager is initialized correctly."""
        self.assertIsInstance(self.client.auth, AuthManager)
        self.assertEqual(self.client.auth.api_key, self.api_key)
    
    def test_sub_clients(self):
        """Test that sub-clients are initialized correctly."""
        self.assertIsNotNone(self.client.activations)
        self.assertIsNotNone(self.client.control_vector)
        self.assertIsNotNone(self.client.inference)
    
    def test_repr(self):
        """Test the string representation of the client."""
        self.assertEqual(repr(self.client), f"WisentClient(base_url='{self.base_url}')")


if __name__ == "__main__":
    unittest.main() 