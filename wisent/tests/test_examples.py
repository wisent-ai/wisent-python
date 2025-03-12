"""
Tests for the example scripts.

This module contains tests for the example scripts to ensure they work correctly.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
import numpy as np

# Add the examples directory to the path so we can import the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples')))

# Create mock modules
mock_wisent = MagicMock()
mock_wisent_inference = MagicMock()
mock_wisent_control_vector = MagicMock()
mock_wisent_activations = MagicMock()
mock_matplotlib_pyplot = MagicMock()

# Add the mock modules to sys.modules
sys.modules['wisent'] = mock_wisent
sys.modules['wisent.inference'] = mock_wisent_inference
sys.modules['wisent.control_vector'] = mock_wisent_control_vector
sys.modules['wisent.activations'] = mock_wisent_activations
sys.modules['matplotlib.pyplot'] = mock_matplotlib_pyplot


class TestBasicUsage(unittest.TestCase):
    """Tests for the basic_usage.py script."""
    
    def setUp(self):
        """Set up the test."""
        # Reset the mock modules
        mock_wisent.reset_mock()
        mock_wisent_inference.reset_mock()
        mock_wisent_control_vector.reset_mock()
        
        # Create mock objects
        self.mock_client = MagicMock()
        self.mock_control_vector_client = MagicMock()
        self.mock_inference_client = MagicMock()
        self.mock_client.control_vector = self.mock_control_vector_client
        self.mock_client.inference = self.mock_inference_client
        
        # Set up the WisentClient constructor
        mock_wisent.WisentClient = MagicMock(return_value=self.mock_client)
        
        # Set up the Inferencer and ControlVectorManager
        mock_wisent_inference.Inferencer = MagicMock()
        mock_wisent_inference.InferenceConfig = MagicMock()
        mock_wisent_control_vector.ControlVectorManager = MagicMock()
    
    def test_main_with_api_key(self):
        """Test the main function with an API key."""
        # Mock the command-line arguments
        with patch('sys.argv', ['basic_usage.py', '--api-key', 'test_api_key', '--prompt', 'Test prompt']):
            # Mock the control vector methods
            self.mock_control_vector_client.list.return_value = [
                {'name': 'vector1', 'description': 'Vector 1'},
                {'name': 'vector2', 'description': 'Vector 2'}
            ]
            
            mock_vector = MagicMock()
            mock_vector.name = 'vector1'
            mock_vector.model_name = 'test_model'
            mock_vector.values = torch.ones(10)
            mock_vector.metadata = {'description': 'Vector 1'}
            self.mock_control_vector_client.get.return_value = mock_vector
            
            mock_combined_vector = MagicMock()
            mock_combined_vector.name = 'combined_vector'
            mock_combined_vector.values = torch.ones(10)
            mock_combined_vector.metadata = {'description': 'Combined Vector'}
            self.mock_control_vector_client.combine.return_value = mock_combined_vector
            
            # Mock the inference methods
            mock_response = MagicMock()
            mock_response.text = 'Generated text'
            mock_response.usage = {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}
            mock_response.metadata = {'test': 'metadata'}
            self.mock_inference_client.generate.return_value = mock_response
            self.mock_inference_client.generate_with_control.return_value = mock_response
            
            # Mock sys.exit to prevent the test from exiting
            with patch('sys.exit'):
                # Mock print to avoid output during tests
                with patch('builtins.print'):
                    # Import the module
                    import basic_usage
                    
                    # Run the main function
                    basic_usage.main()
                    
                    # Check that the client was initialized with the correct API key
                    mock_wisent.WisentClient.assert_called_once_with(api_key='test_api_key', base_url='https://api.wisent.ai')
                    
                    # Check that the control vector methods were called
                    self.mock_control_vector_client.list.assert_called_once()
                    self.mock_control_vector_client.get.assert_called_once()
                    self.mock_control_vector_client.combine.assert_called_once()
                    
                    # Check that the inference methods were called
                    self.mock_inference_client.generate.assert_called_once()
                    self.mock_inference_client.generate_with_control.assert_called_once()
    
    def test_main_without_api_key(self):
        """Test the main function without an API key."""
        # Mock the command-line arguments
        with patch('sys.argv', ['basic_usage.py']):
            # Mock os.environ to not have the API key
            with patch.dict('os.environ', {}, clear=True):
                # Mock sys.exit to prevent the test from exiting
                mock_exit = MagicMock()
                with patch('sys.exit', mock_exit):
                    # Mock print to avoid output during tests
                    with patch('builtins.print'):
                        # Import the module
                        import basic_usage
                        
                        # Run the main function
                        basic_usage.main()
                        
                        # Check that sys.exit was called
                        mock_exit.assert_called_once_with(1)


class TestActivationExtraction(unittest.TestCase):
    """Tests for the activation_extraction.py script."""
    
    def setUp(self):
        """Set up the test."""
        # Reset the mock modules
        mock_wisent.reset_mock()
        mock_wisent_activations.reset_mock()
        mock_matplotlib_pyplot.reset_mock()
        
        # Create mock objects
        self.mock_client = MagicMock()
        self.mock_activations_client = MagicMock()
        self.mock_client.activations = self.mock_activations_client
        
        # Set up the WisentClient constructor
        mock_wisent.WisentClient = MagicMock(return_value=self.mock_client)
        
        # Set up the ActivationExtractor
        self.mock_extractor = MagicMock()
        mock_wisent_activations.ActivationExtractor = MagicMock(return_value=self.mock_extractor)
        
        # Set up matplotlib.pyplot
        mock_matplotlib_pyplot.figure = MagicMock()
        mock_matplotlib_pyplot.show = MagicMock()
    
    def test_main_with_api_key(self):
        """Test the main function with an API key."""
        # Mock the command-line arguments
        with patch('sys.argv', ['activation_extraction.py', '--api-key', 'test_api_key', '--prompt', 'Test prompt', '--visualize']):
            # Mock the activations methods
            mock_activation = MagicMock()
            mock_activation.layer = -1
            mock_activation.token_index = -1
            mock_activation.token_str = 'test'
            mock_activation.values = torch.ones(10)
            
            mock_activations_obj = MagicMock()
            mock_activations_obj.activations = [mock_activation]
            mock_activations_obj.model_name = 'test_model'
            mock_activations_obj.prompt = 'Test prompt'
            mock_activations_obj.metadata = {'test': 'metadata'}
            
            self.mock_activations_client.extract.return_value = mock_activations_obj
            self.mock_extractor.extract.return_value = mock_activations_obj
            
            # Mock sys.exit to prevent the test from exiting
            with patch('sys.exit'):
                # Mock print to avoid output during tests
                with patch('builtins.print'):
                    # Import the module
                    import activation_extraction
                    
                    # Run the main function
                    activation_extraction.main()
                    
                    # Check that the client was initialized with the correct API key
                    mock_wisent.WisentClient.assert_called_once_with(api_key='test_api_key', base_url='https://api.wisent.ai')
                    
                    # Check that the activations methods were called
                    self.mock_activations_client.extract.assert_called_once()
                    
                    # Check that the extractor was used
                    self.mock_extractor.extract.assert_called_once()
                    
                    # Check that matplotlib was used for visualization
                    mock_matplotlib_pyplot.figure.assert_called()
    
    def test_main_without_api_key(self):
        """Test the main function without an API key."""
        # Mock the command-line arguments
        with patch('sys.argv', ['activation_extraction.py']):
            # Mock os.environ to not have the API key
            with patch.dict('os.environ', {}, clear=True):
                # Mock sys.exit to prevent the test from exiting
                mock_exit = MagicMock()
                with patch('sys.exit', mock_exit):
                    # Mock print to avoid output during tests
                    with patch('builtins.print'):
                        # Import the module
                        import activation_extraction
                        
                        # Run the main function
                        activation_extraction.main()
                        
                        # Check that sys.exit was called
                        mock_exit.assert_called_once_with(1)


class TestCustomControlVectors(unittest.TestCase):
    """Tests for the custom_control_vectors.py script."""
    
    def setUp(self):
        """Set up the test."""
        # Reset the mock modules
        mock_wisent.reset_mock()
        mock_wisent_inference.reset_mock()
        mock_wisent_control_vector.reset_mock()
        mock_wisent_activations.reset_mock()
        
        # Create mock objects
        self.mock_client = MagicMock()
        self.mock_control_vector_client = MagicMock()
        self.mock_client.control_vector = self.mock_control_vector_client
        
        # Set up the WisentClient constructor
        mock_wisent.WisentClient = MagicMock(return_value=self.mock_client)
        
        # Set up the ControlVector class
        self.mock_vector = MagicMock()
        self.mock_vector.name = 'vector1'
        self.mock_vector.model_name = 'test_model'
        self.mock_vector.values = torch.ones(10)
        self.mock_vector.metadata = {'description': 'Vector 1'}
        mock_wisent_control_vector.ControlVector = MagicMock(return_value=self.mock_vector)
        
        # Set up the ActivationExtractor
        self.mock_extractor = MagicMock()
        mock_wisent_activations.ActivationExtractor = MagicMock(return_value=self.mock_extractor)
        
        # Set up the Inferencer
        self.mock_inferencer = MagicMock()
        mock_wisent_inference.Inferencer = MagicMock(return_value=self.mock_inferencer)
        mock_wisent_inference.InferenceConfig = MagicMock()
    
    def test_main_with_api_key(self):
        """Test the main function with an API key."""
        # Mock the command-line arguments
        with patch('sys.argv', ['custom_control_vectors.py', '--api-key', 'test_api_key', '--prompt', 'Test prompt']):
            # Mock the control vector methods
            self.mock_control_vector_client.list.return_value = [
                {'name': 'vector1', 'description': 'Vector 1'},
                {'name': 'vector2', 'description': 'Vector 2'}
            ]
            
            self.mock_control_vector_client.get.return_value = self.mock_vector
            
            mock_combined_vector = MagicMock()
            mock_combined_vector.name = 'combined_vector'
            mock_combined_vector.values = torch.ones(10)
            mock_combined_vector.metadata = {'description': 'Combined Vector'}
            self.mock_control_vector_client.combine.return_value = mock_combined_vector
            
            # Mock the activations
            mock_activation = MagicMock()
            mock_activation.layer = -1
            mock_activation.token_index = -1
            mock_activation.token_str = 'test'
            mock_activation.values = torch.ones(10)
            
            mock_activations_obj = MagicMock()
            mock_activations_obj.activations = [mock_activation]
            mock_activations_obj.model_name = 'test_model'
            mock_activations_obj.prompt = 'Test prompt'
            mock_activations_obj.metadata = {'test': 'metadata'}
            
            self.mock_extractor.extract.return_value = mock_activations_obj
            
            # Mock the inferencer
            mock_response = MagicMock()
            mock_response.text = 'Generated text'
            mock_response.usage = {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}
            mock_response.metadata = {'test': 'metadata'}
            self.mock_inferencer.generate.return_value = mock_response
            
            # Mock sys.exit to prevent the test from exiting
            with patch('sys.exit'):
                # Mock print to avoid output during tests
                with patch('builtins.print'):
                    # Import the module
                    import custom_control_vectors
                    
                    # Run the main function
                    custom_control_vectors.main()
                    
                    # Check that the client was initialized with the correct API key
                    mock_wisent.WisentClient.assert_called_once_with(api_key='test_api_key', base_url='https://api.wisent.ai')
                    
                    # Check that the control vector methods were called
                    self.mock_control_vector_client.list.assert_called_once()
                    self.mock_control_vector_client.get.assert_called()
                    self.mock_control_vector_client.combine.assert_called_once()
                    
                    # Check that the extractor was used
                    self.mock_extractor.extract.assert_called()
                    
                    # Check that the inferencer was used
                    self.mock_inferencer.generate.assert_called()
    
    def test_main_without_api_key(self):
        """Test the main function without an API key."""
        # Mock the command-line arguments
        with patch('sys.argv', ['custom_control_vectors.py']):
            # Mock os.environ to not have the API key
            with patch.dict('os.environ', {}, clear=True):
                # Mock sys.exit to prevent the test from exiting
                mock_exit = MagicMock()
                with patch('sys.exit', mock_exit):
                    # Mock print to avoid output during tests
                    with patch('builtins.print'):
                        # Import the module
                        import custom_control_vectors
                        
                        # Run the main function
                        custom_control_vectors.main()
                        
                        # Check that sys.exit was called
                        mock_exit.assert_called_once_with(1)


if __name__ == '__main__':
    unittest.main() 