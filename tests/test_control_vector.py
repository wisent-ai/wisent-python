"""
Tests for the control vector module.
"""

import unittest

import numpy as np
import torch

from wisent.control_vector import ControlVector


class TestControlVector(unittest.TestCase):
    """Tests for the ControlVector class."""
    
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


if __name__ == "__main__":
    unittest.main() 