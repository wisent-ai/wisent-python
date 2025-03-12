"""
Tests for the activations module.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from wisent.activations import Activation, ActivationBatch, ActivationExtractor, ActivationsClient
from wisent.utils.auth import AuthManager


class TestActivation(unittest.TestCase):
    """Tests for the Activation class."""
    
    def test_init(self):
        """Test initialization of an Activation."""
        # Test with list values
        values = [0.1, 0.2, 0.3]
        activation = Activation(
            model_name="test_model",
            layer=0,
            token_index=1,
            values=values,
            token_str="test"
        )
        
        self.assertEqual(activation.model_name, "test_model")
        self.assertEqual(activation.layer, 0)
        self.assertEqual(activation.token_index, 1)
        self.assertEqual(activation.values, values)
        self.assertEqual(activation.token_str, "test")
        
        # Test with numpy array
        values = np.array([0.1, 0.2, 0.3])
        activation = Activation(
            model_name="test_model",
            layer=0,
            token_index=1,
            values=values
        )
        
        self.assertTrue(np.array_equal(activation.values, values))
        self.assertIsNone(activation.token_str)
        
        # Test with torch tensor
        values = torch.tensor([0.1, 0.2, 0.3])
        activation = Activation(
            model_name="test_model",
            layer=0,
            token_index=1,
            values=values
        )
        
        self.assertTrue(torch.equal(activation.values, values))
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        # Test with list values
        values = [0.1, 0.2, 0.3]
        activation = Activation(
            model_name="test_model",
            layer=0,
            token_index=1,
            values=values,
            token_str="test"
        )
        
        expected = {
            "model_name": "test_model",
            "layer": 0,
            "token_index": 1,
            "values": values,
            "token_str": "test",
        }
        
        self.assertEqual(activation.to_dict(), expected)
        
        # Test with numpy array
        values = np.array([0.1, 0.2, 0.3])
        activation = Activation(
            model_name="test_model",
            layer=0,
            token_index=1,
            values=values
        )
        
        expected = {
            "model_name": "test_model",
            "layer": 0,
            "token_index": 1,
            "values": values.tolist(),
            "token_str": None,
        }
        
        self.assertEqual(activation.to_dict(), expected)
        
        # Test with torch tensor
        values = torch.tensor([0.1, 0.2, 0.3])
        activation = Activation(
            model_name="test_model",
            layer=0,
            token_index=1,
            values=values
        )
        
        expected = {
            "model_name": "test_model",
            "layer": 0,
            "token_index": 1,
            "values": values.detach().cpu().numpy().tolist(),
            "token_str": None,
        }
        
        self.assertEqual(activation.to_dict(), expected)


class TestActivationBatch(unittest.TestCase):
    """Tests for the ActivationBatch class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.activation1 = Activation(
            model_name="test_model",
            layer=0,
            token_index=1,
            values=[0.1, 0.2, 0.3],
            token_str="test1"
        )
        
        self.activation2 = Activation(
            model_name="test_model",
            layer=1,
            token_index=2,
            values=[0.4, 0.5, 0.6],
            token_str="test2"
        )
    
    def test_init(self):
        """Test initialization of an ActivationBatch."""
        batch = ActivationBatch(
            model_name="test_model",
            prompt="test prompt",
            activations=[self.activation1, self.activation2],
            metadata={"test": "metadata"}
        )
        
        self.assertEqual(batch.model_name, "test_model")
        self.assertEqual(batch.prompt, "test prompt")
        self.assertEqual(batch.activations, [self.activation1, self.activation2])
        self.assertEqual(batch.metadata, {"test": "metadata"})
        
        # Test with default metadata
        batch = ActivationBatch(
            model_name="test_model",
            prompt="test prompt",
            activations=[self.activation1, self.activation2]
        )
        
        self.assertEqual(batch.metadata, {})
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        batch = ActivationBatch(
            model_name="test_model",
            prompt="test prompt",
            activations=[self.activation1, self.activation2],
            metadata={"test": "metadata"}
        )
        
        expected = {
            "model_name": "test_model",
            "prompt": "test prompt",
            "activations": [
                self.activation1.to_dict(),
                self.activation2.to_dict()
            ],
            "metadata": {"test": "metadata"},
        }
        
        self.assertEqual(batch.to_dict(), expected)


class TestActivationsClient(unittest.TestCase):
    """Tests for the ActivationsClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager("test_api_key")
        self.base_url = "https://test.api.wisent.ai"
        self.client = ActivationsClient(self.auth_manager, self.base_url)
    
    @patch("wisent.activations.client.ActivationExtractor")
    def test_extract(self, mock_extractor_class):
        """Test extraction of activations."""
        # Mock the extractor
        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor
        
        # Mock the extract method
        mock_batch = MagicMock()
        mock_extractor.extract.return_value = mock_batch
        
        # Call the method
        result = self.client.extract(
            model_name="test_model",
            prompt="test prompt",
            layers=[0, 1],
            tokens_to_extract=[1, 2],
            device="cpu"
        )
        
        # Check that the extractor was created correctly
        mock_extractor_class.assert_called_once_with("test_model", device="cpu")
        
        # Check that extract was called correctly
        mock_extractor.extract.assert_called_once_with("test prompt", [0, 1], [1, 2])
        
        # Check the result
        self.assertEqual(result, mock_batch)
    
    @patch("wisent.activations.client.HTTPClient")
    def test_upload(self, mock_http_client_class):
        """Test uploading of activations."""
        # Mock the HTTP client
        mock_http_client = MagicMock()
        mock_http_client_class.return_value = mock_http_client
        
        # Mock the post method
        mock_response = {"id": "test_batch_id"}
        mock_http_client.post.return_value = mock_response
        
        # Create a client with the mocked HTTP client
        client = ActivationsClient(self.auth_manager, self.base_url)
        
        # Create a batch to upload
        batch = ActivationBatch(
            model_name="test_model",
            prompt="test prompt",
            activations=[
                Activation(
                    model_name="test_model",
                    layer=0,
                    token_index=1,
                    values=[0.1, 0.2, 0.3],
                    token_str="test"
                )
            ]
        )
        
        # Call the method
        result = client.upload(batch)
        
        # Check that post was called correctly
        mock_http_client.post.assert_called_once_with(
            "/activations/upload",
            json_data=batch.to_dict()
        )
        
        # Check the result
        self.assertEqual(result, mock_response)
    
    @patch("wisent.activations.client.HTTPClient")
    def test_get(self, mock_http_client_class):
        """Test getting activations by ID."""
        # Mock the HTTP client
        mock_http_client = MagicMock()
        mock_http_client_class.return_value = mock_http_client
        
        # Mock the get method
        mock_response = {
            "model_name": "test_model",
            "prompt": "test prompt",
            "activations": [
                {
                    "model_name": "test_model",
                    "layer": 0,
                    "token_index": 1,
                    "values": [0.1, 0.2, 0.3],
                    "token_str": "test"
                }
            ],
            "metadata": {"test": "metadata"}
        }
        mock_http_client.get.return_value = mock_response
        
        # Create a client with the mocked HTTP client
        client = ActivationsClient(self.auth_manager, self.base_url)
        
        # Call the method
        result = client.get("test_batch_id")
        
        # Check that get was called correctly
        mock_http_client.get.assert_called_once_with("/activations/test_batch_id")
        
        # Check the result
        self.assertEqual(result.model_name, "test_model")
        self.assertEqual(result.prompt, "test prompt")
        self.assertEqual(len(result.activations), 1)
        self.assertEqual(result.activations[0].model_name, "test_model")
        self.assertEqual(result.activations[0].layer, 0)
        self.assertEqual(result.activations[0].token_index, 1)
        self.assertEqual(result.activations[0].values, [0.1, 0.2, 0.3])
        self.assertEqual(result.activations[0].token_str, "test")
        self.assertEqual(result.metadata, {"test": "metadata"})
    
    @patch("wisent.activations.client.HTTPClient")
    def test_list(self, mock_http_client_class):
        """Test listing activations."""
        # Mock the HTTP client
        mock_http_client = MagicMock()
        mock_http_client_class.return_value = mock_http_client
        
        # Mock the get method
        mock_response = [
            {"id": "batch1", "model_name": "test_model"},
            {"id": "batch2", "model_name": "test_model"}
        ]
        mock_http_client.get.return_value = mock_response
        
        # Create a client with the mocked HTTP client
        client = ActivationsClient(self.auth_manager, self.base_url)
        
        # Call the method
        result = client.list(model_name="test_model", limit=10, offset=0)
        
        # Check that get was called correctly
        mock_http_client.get.assert_called_once_with(
            "/activations",
            params={"model_name": "test_model", "limit": 10, "offset": 0}
        )
        
        # Check the result
        self.assertEqual(result, mock_response)
        
        # Test without model_name
        mock_http_client.get.reset_mock()
        result = client.list(limit=10, offset=0)
        
        # Check that get was called correctly
        mock_http_client.get.assert_called_once_with(
            "/activations",
            params={"limit": 10, "offset": 0}
        )


if __name__ == "__main__":
    unittest.main() 