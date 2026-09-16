"""
Tests for the activations module.
"""

import unittest

import numpy as np
import torch

from wisent.activations import Activation, ActivationBatch


class TestActivation(unittest.TestCase):
    """Tests for the Activation class."""
    
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


if __name__ == "__main__":
    unittest.main() 