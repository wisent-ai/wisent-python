"""
Tests for the control vector module.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from wisent.control_vector import ControlVector, ControlVectorClient, ControlVectorManager
from wisent.utils.auth import AuthManager


class TestControlVector(unittest.TestCase):
    """Tests for the ControlVector class."""
    
    def test_init(self):
        """Test initialization of a ControlVector."""
        # Test with list values
        values = [0.1, 0.2, 0.3]
        vector = ControlVector(
            name="test_vector",
            model_name="test_model",
            values=values,
            metadata={"test": "metadata"}
        )
        
        self.assertEqual(vector.name, "test_vector")
        self.assertEqual(vector.model_name, "test_model")
        self.assertEqual(vector.values, values)
        self.assertEqual(vector.metadata, {"test": "metadata"})
        
        # Test with numpy array
        values = np.array([0.1, 0.2, 0.3])
        vector = ControlVector(
            name="test_vector",
            model_name="test_model",
            values=values
        )
        
        self.assertTrue(np.array_equal(vector.values, values))
        self.assertEqual(vector.metadata, {})
        
        # Test with torch tensor
        values = torch.tensor([0.1, 0.2, 0.3])
        vector = ControlVector(
            name="test_vector",
            model_name="test_model",
            values=values
        )
        
        self.assertTrue(torch.equal(vector.values, values))
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        # Test with list values
        values = [0.1, 0.2, 0.3]
        vector = ControlVector(
            name="test_vector",
            model_name="test_model",
            values=values,
            metadata={"test": "metadata"}
        )
        
        expected = {
            "name": "test_vector",
            "model_name": "test_model",
            "values": values,
            "metadata": {"test": "metadata"},
        }
        
        self.assertEqual(vector.to_dict(), expected)
        
        # Test with numpy array
        values = np.array([0.1, 0.2, 0.3])
        vector = ControlVector(
            name="test_vector",
            model_name="test_model",
            values=values
        )
        
        expected = {
            "name": "test_vector",
            "model_name": "test_model",
            "values": values.tolist(),
            "metadata": {},
        }
        
        self.assertEqual(vector.to_dict(), expected)
        
        # Test with torch tensor
        values = torch.tensor([0.1, 0.2, 0.3])
        vector = ControlVector(
            name="test_vector",
            model_name="test_model",
            values=values
        )
        
        expected = {
            "name": "test_vector",
            "model_name": "test_model",
            "values": values.detach().cpu().numpy().tolist(),
            "metadata": {},
        }
        
        self.assertEqual(vector.to_dict(), expected)
    
    def test_to_tensor(self):
        """Test conversion to tensor."""
        # Test with list values
        values = [0.1, 0.2, 0.3]
        vector = ControlVector(
            name="test_vector",
            model_name="test_model",
            values=values
        )
        
        tensor = vector.to_tensor()
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.device.type, "cpu")
        self.assertTrue(torch.allclose(tensor, torch.tensor(values)))
        
        # Test with numpy array
        values = np.array([0.1, 0.2, 0.3])
        vector = ControlVector(
            name="test_vector",
            model_name="test_model",
            values=values
        )
        
        tensor = vector.to_tensor()
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.device.type, "cpu")
        self.assertTrue(torch.allclose(tensor, torch.tensor(values)))
        
        # Test with torch tensor
        values = torch.tensor([0.1, 0.2, 0.3])
        vector = ControlVector(
            name="test_vector",
            model_name="test_model",
            values=values
        )
        
        tensor = vector.to_tensor()
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.device.type, "cpu")
        self.assertTrue(torch.equal(tensor, values))
        
        # Test with device
        if torch.cuda.is_available():
            tensor = vector.to_tensor(device="cuda")
            self.assertEqual(tensor.device.type, "cuda")


class TestControlVectorClient(unittest.TestCase):
    """Tests for the ControlVectorClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager("test_api_key")
        self.base_url = "https://test.api.wisent.ai"
        self.client = ControlVectorClient(self.auth_manager, self.base_url)
    
    @patch("wisent.control_vector.client.HTTPClient")
    def test_get(self, mock_http_client_class):
        """Test getting a control vector."""
        # Mock the HTTP client
        mock_http_client = MagicMock()
        mock_http_client_class.return_value = mock_http_client
        
        # Mock the get method
        mock_response = {
            "name": "test_vector",
            "model_name": "test_model",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"test": "metadata"}
        }
        mock_http_client.get.return_value = mock_response
        
        # Create a client with the mocked HTTP client
        client = ControlVectorClient(self.auth_manager, self.base_url)
        
        # Call the method
        result = client.get("test_vector", "test_model")
        
        # Check that get was called correctly
        mock_http_client.get.assert_called_once_with(
            "/control_vectors/test_vector",
            params={"model": "test_model"}
        )
        
        # Check the result
        self.assertEqual(result.name, "test_vector")
        self.assertEqual(result.model_name, "test_model")
        self.assertEqual(result.values, [0.1, 0.2, 0.3])
        self.assertEqual(result.metadata, {"test": "metadata"})
    
    @patch("wisent.control_vector.client.HTTPClient")
    def test_list(self, mock_http_client_class):
        """Test listing control vectors."""
        # Mock the HTTP client
        mock_http_client = MagicMock()
        mock_http_client_class.return_value = mock_http_client
        
        # Mock the get method
        mock_response = [
            {"name": "vector1", "model_name": "test_model"},
            {"name": "vector2", "model_name": "test_model"}
        ]
        mock_http_client.get.return_value = mock_response
        
        # Create a client with the mocked HTTP client
        client = ControlVectorClient(self.auth_manager, self.base_url)
        
        # Call the method
        result = client.list(model="test_model", limit=10, offset=0)
        
        # Check that get was called correctly
        mock_http_client.get.assert_called_once_with(
            "/control_vectors",
            params={"model": "test_model", "limit": 10, "offset": 0}
        )
        
        # Check the result
        self.assertEqual(result, mock_response)
        
        # Test without model
        mock_http_client.get.reset_mock()
        result = client.list(limit=10, offset=0)
        
        # Check that get was called correctly
        mock_http_client.get.assert_called_once_with(
            "/control_vectors",
            params={"limit": 10, "offset": 0}
        )
    
    @patch("wisent.control_vector.client.HTTPClient")
    def test_combine(self, mock_http_client_class):
        """Test combining control vectors."""
        # Mock the HTTP client
        mock_http_client = MagicMock()
        mock_http_client_class.return_value = mock_http_client
        
        # Mock the post method
        mock_response = {
            "name": "combined_vector1_vector2",
            "model_name": "test_model",
            "values": [0.5, 0.7, 0.9],
            "metadata": {"combined_from": {"vector1": 0.5, "vector2": 0.5}}
        }
        mock_http_client.post.return_value = mock_response
        
        # Create a client with the mocked HTTP client
        client = ControlVectorClient(self.auth_manager, self.base_url)
        
        # Call the method
        result = client.combine(
            vectors={"vector1": 0.5, "vector2": 0.5},
            model="test_model"
        )
        
        # Check that post was called correctly
        mock_http_client.post.assert_called_once_with(
            "/control_vectors/combine",
            json_data={
                "vectors": {"vector1": 0.5, "vector2": 0.5},
                "model": "test_model",
            }
        )
        
        # Check the result
        self.assertEqual(result.name, "combined_vector1_vector2")
        self.assertEqual(result.model_name, "test_model")
        self.assertEqual(result.values, [0.5, 0.7, 0.9])
        self.assertEqual(result.metadata, {"combined_from": {"vector1": 0.5, "vector2": 0.5}})


class TestControlVectorManager(unittest.TestCase):
    """Tests for the ControlVectorManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test_api_key"
        self.base_url = "https://test.api.wisent.ai"
        self.manager = ControlVectorManager(api_key=self.api_key, base_url=self.base_url)
    
    @patch("wisent.control_vector.manager.HTTPClient")
    def test_get(self, mock_http_client_class):
        """Test getting a control vector."""
        # Mock the HTTP client
        mock_http_client = MagicMock()
        mock_http_client_class.return_value = mock_http_client
        
        # Mock the get method
        mock_response = {
            "name": "test_vector",
            "model_name": "test_model",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"test": "metadata"}
        }
        mock_http_client.get.return_value = mock_response
        
        # Create a manager with the mocked HTTP client
        manager = ControlVectorManager(api_key=self.api_key, base_url=self.base_url)
        
        # Call the method
        result = manager.get("test_vector", "test_model")
        
        # Check that get was called correctly
        mock_http_client.get.assert_called_once_with(
            "/control_vectors/test_vector",
            params={"model": "test_model"}
        )
        
        # Check the result
        self.assertEqual(result.name, "test_vector")
        self.assertEqual(result.model_name, "test_model")
        self.assertEqual(result.values, [0.1, 0.2, 0.3])
        self.assertEqual(result.metadata, {"test": "metadata"})
        
        # Check that the result is cached
        mock_http_client.get.reset_mock()
        result2 = manager.get("test_vector", "test_model")
        
        # Check that get was not called again
        mock_http_client.get.assert_not_called()
        
        # Check that the result is the same
        self.assertEqual(result2.name, "test_vector")
    
    @patch("wisent.control_vector.manager.HTTPClient")
    def test_combine_api(self, mock_http_client_class):
        """Test combining control vectors via API."""
        # Mock the HTTP client
        mock_http_client = MagicMock()
        mock_http_client_class.return_value = mock_http_client
        
        # Mock the post method
        mock_response = {
            "name": "combined_vector1_vector2",
            "model_name": "test_model",
            "values": [0.5, 0.7, 0.9],
            "metadata": {"combined_from": {"vector1": 0.5, "vector2": 0.5}}
        }
        mock_http_client.post.return_value = mock_response
        
        # Create a manager with the mocked HTTP client
        manager = ControlVectorManager(api_key=self.api_key, base_url=self.base_url)
        
        # Call the method
        result = manager.combine(
            vectors={"vector1": 0.5, "vector2": 0.5},
            model="test_model"
        )
        
        # Check that post was called correctly
        mock_http_client.post.assert_called_once_with(
            "/control_vectors/combine",
            json_data={
                "vectors": {"vector1": 0.5, "vector2": 0.5},
                "model": "test_model",
            }
        )
        
        # Check the result
        self.assertEqual(result.name, "combined_vector1_vector2")
        self.assertEqual(result.model_name, "test_model")
        self.assertEqual(result.values, [0.5, 0.7, 0.9])
        self.assertEqual(result.metadata, {"combined_from": {"vector1": 0.5, "vector2": 0.5}})
    
    @patch("wisent.control_vector.manager.HTTPClient")
    def test_combine_local(self, mock_http_client_class):
        """Test combining control vectors locally."""
        # Mock the HTTP client
        mock_http_client = MagicMock()
        mock_http_client_class.return_value = mock_http_client
        
        # Mock the get method for two vectors
        mock_http_client.get.side_effect = [
            {
                "name": "vector1",
                "model_name": "test_model",
                "values": [0.1, 0.2, 0.3],
                "metadata": {}
            },
            {
                "name": "vector2",
                "model_name": "test_model",
                "values": [0.4, 0.5, 0.6],
                "metadata": {}
            }
        ]
        
        # Create a manager with the mocked HTTP client
        manager = ControlVectorManager(api_key=self.api_key, base_url=self.base_url)
        
        # Pre-cache the vectors
        vector1 = manager.get("vector1", "test_model")
        vector2 = manager.get("vector2", "test_model")
        
        # Reset the mock
        mock_http_client.reset_mock()
        
        # Call the method
        result = manager.combine(
            vectors={"vector1": 0.5, "vector2": 0.5},
            model="test_model"
        )
        
        # Check that post was not called
        mock_http_client.post.assert_not_called()
        
        # Check the result
        self.assertEqual(result.name, "combined_vector1_vector2")
        self.assertEqual(result.model_name, "test_model")
        
        # Check that the values are combined correctly
        expected_values = torch.tensor([0.1, 0.2, 0.3]) * 0.5 + torch.tensor([0.4, 0.5, 0.6]) * 0.5
        self.assertTrue(torch.allclose(torch.tensor(result.values), expected_values))
        
        # Check the metadata
        self.assertEqual(result.metadata["combined_from"], {"vector1": 0.5, "vector2": 0.5})


if __name__ == "__main__":
    unittest.main() 