"""
Tests for the inference module.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from wisent.control_vector import ControlVector
from wisent.inference import InferenceClient, InferenceConfig, InferenceResponse
from wisent.utils.auth import AuthManager


class TestInferenceConfig(unittest.TestCase):
    """Tests for the InferenceConfig class."""
    
    def test_init(self):
        """Test initialization of an InferenceConfig."""
        # Test with default values
        config = InferenceConfig()
        
        self.assertEqual(config.max_tokens, 256)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.top_k, 50)
        self.assertEqual(config.repetition_penalty, 1.0)
        self.assertIsNone(config.stop_sequences)
        
        # Test with custom values
        config = InferenceConfig(
            max_tokens=100,
            temperature=0.5,
            top_p=0.8,
            top_k=40,
            repetition_penalty=1.2,
            stop_sequences=[".", "!"]
        )
        
        self.assertEqual(config.max_tokens, 100)
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.top_p, 0.8)
        self.assertEqual(config.top_k, 40)
        self.assertEqual(config.repetition_penalty, 1.2)
        self.assertEqual(config.stop_sequences, [".", "!"])


class TestInferenceResponse(unittest.TestCase):
    """Tests for the InferenceResponse class."""
    
    def test_init(self):
        """Test initialization of an InferenceResponse."""
        # Test with required values
        response = InferenceResponse(
            text="Generated text",
            model="test_model",
            prompt="Test prompt"
        )
        
        self.assertEqual(response.text, "Generated text")
        self.assertEqual(response.model, "test_model")
        self.assertEqual(response.prompt, "Test prompt")
        self.assertEqual(response.finish_reason, "length")
        self.assertEqual(response.usage, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        self.assertEqual(response.metadata, {})
        
        # Test with all values
        response = InferenceResponse(
            text="Generated text",
            model="test_model",
            prompt="Test prompt",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            metadata={"test": "metadata"}
        )
        
        self.assertEqual(response.text, "Generated text")
        self.assertEqual(response.model, "test_model")
        self.assertEqual(response.prompt, "Test prompt")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage, {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})
        self.assertEqual(response.metadata, {"test": "metadata"})


class TestInferenceClient(unittest.TestCase):
    """Tests for the InferenceClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager("test_api_key")
        self.base_url = "https://test.api.wisent.ai"
        self.client = InferenceClient(self.auth_manager, self.base_url)
    
    @patch("wisent.inference.client.HTTPClient")
    def test_generate(self, mock_http_client_class):
        """Test generating text."""
        # Mock the HTTP client
        mock_http_client = MagicMock()
        mock_http_client_class.return_value = mock_http_client
        
        # Mock the post method
        mock_response = {
            "text": "Generated text",
            "model": "test_model",
            "prompt": "Test prompt",
            "finish_reason": "length",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "metadata": {}
        }
        mock_http_client.post.return_value = mock_response
        
        # Create a client with the mocked HTTP client
        client = InferenceClient(self.auth_manager, self.base_url)
        
        # Call the method with default config
        result = client.generate(
            model_name="test_model",
            prompt="Test prompt"
        )
        
        # Check that post was called correctly
        mock_http_client.post.assert_called_once_with(
            "/inference/generate",
            json_data={
                "model": "test_model",
                "prompt": "Test prompt",
                "max_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.0,
                "stop_sequences": None,
            }
        )
        
        # Check the result
        self.assertEqual(result.text, "Generated text")
        self.assertEqual(result.model, "test_model")
        self.assertEqual(result.prompt, "Test prompt")
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.usage, {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})
        self.assertEqual(result.metadata, {})
        
        # Test with custom config
        mock_http_client.post.reset_mock()
        config = InferenceConfig(
            max_tokens=100,
            temperature=0.5,
            top_p=0.8,
            top_k=40,
            repetition_penalty=1.2,
            stop_sequences=[".", "!"]
        )
        
        result = client.generate(
            model_name="test_model",
            prompt="Test prompt",
            config=config
        )
        
        # Check that post was called correctly
        mock_http_client.post.assert_called_once_with(
            "/inference/generate",
            json_data={
                "model": "test_model",
                "prompt": "Test prompt",
                "max_tokens": 100,
                "temperature": 0.5,
                "top_p": 0.8,
                "top_k": 40,
                "repetition_penalty": 1.2,
                "stop_sequences": [".", "!"],
            }
        )
    
    @patch("wisent.inference.client.HTTPClient")
    def test_generate_with_control(self, mock_http_client_class):
        """Test generating text with control vectors."""
        # Mock the HTTP client
        mock_http_client = MagicMock()
        mock_http_client_class.return_value = mock_http_client
        
        # Mock the post method
        mock_response = {
            "text": "Generated text",
            "model": "test_model",
            "prompt": "Test prompt",
            "finish_reason": "length",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "metadata": {"control_vectors": {"vector1": 0.5, "vector2": 0.5}}
        }
        mock_http_client.post.return_value = mock_response
        
        # Create a client with the mocked HTTP client
        client = InferenceClient(self.auth_manager, self.base_url)
        
        # Call the method with default config
        result = client.generate_with_control(
            model_name="test_model",
            prompt="Test prompt",
            control_vectors={"vector1": 0.5, "vector2": 0.5},
            method="caa",
            scale=1.0
        )
        
        # Check that post was called correctly
        mock_http_client.post.assert_called_once_with(
            "/inference/generate_with_control",
            json_data={
                "model": "test_model",
                "prompt": "Test prompt",
                "control_vectors": {"vector1": 0.5, "vector2": 0.5},
                "method": "caa",
                "scale": 1.0,
                "max_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.0,
                "stop_sequences": None,
            }
        )
        
        # Check the result
        self.assertEqual(result.text, "Generated text")
        self.assertEqual(result.model, "test_model")
        self.assertEqual(result.prompt, "Test prompt")
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.usage, {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})
        self.assertEqual(result.metadata, {"control_vectors": {"vector1": 0.5, "vector2": 0.5}})
        
        # Test with custom config
        mock_http_client.post.reset_mock()
        config = InferenceConfig(
            max_tokens=100,
            temperature=0.5,
            top_p=0.8,
            top_k=40,
            repetition_penalty=1.2,
            stop_sequences=[".", "!"]
        )
        
        result = client.generate_with_control(
            model_name="test_model",
            prompt="Test prompt",
            control_vectors={"vector1": 0.5, "vector2": 0.5},
            method="caa",
            scale=1.0,
            config=config
        )
        
        # Check that post was called correctly
        mock_http_client.post.assert_called_once_with(
            "/inference/generate_with_control",
            json_data={
                "model": "test_model",
                "prompt": "Test prompt",
                "control_vectors": {"vector1": 0.5, "vector2": 0.5},
                "method": "caa",
                "scale": 1.0,
                "max_tokens": 100,
                "temperature": 0.5,
                "top_p": 0.8,
                "top_k": 40,
                "repetition_penalty": 1.2,
                "stop_sequences": [".", "!"],
            }
        )


class TestInferencer(unittest.TestCase):
    """Tests for the Inferencer class."""
    
    @patch("wisent.inference.inferencer.AutoModelForCausalLM")
    @patch("wisent.inference.inferencer.AutoTokenizer")
    def test_init(self, mock_tokenizer_class, mock_model_class):
        """Test initialization of an Inferencer."""
        from wisent.inference import Inferencer
        
        # Create an inferencer
        inferencer = Inferencer(
            model_name="test_model",
            device="cpu"
        )
        
        # Check the attributes
        self.assertEqual(inferencer.model_name, "test_model")
        self.assertEqual(inferencer.device, "cpu")
        self.assertIsNone(inferencer.model)
        self.assertIsNone(inferencer.tokenizer)
        
        # Test with default device
        with patch("torch.cuda.is_available", return_value=True):
            inferencer = Inferencer(model_name="test_model")
            self.assertEqual(inferencer.device, "cuda")
        
        with patch("torch.cuda.is_available", return_value=False):
            inferencer = Inferencer(model_name="test_model")
            self.assertEqual(inferencer.device, "cpu")
    
    @patch("wisent.inference.inferencer.AutoModelForCausalLM")
    @patch("wisent.inference.inferencer.AutoTokenizer")
    def test_load_model(self, mock_tokenizer_class, mock_model_class):
        """Test loading a model."""
        from wisent.inference import Inferencer
        
        # Mock the model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Create an inferencer
        inferencer = Inferencer(
            model_name="test_model",
            device="cpu"
        )
        
        # Load the model
        inferencer._load_model()
        
        # Check that the model and tokenizer were loaded correctly
        mock_model_class.from_pretrained.assert_called_once_with(
            "test_model",
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        mock_tokenizer_class.from_pretrained.assert_called_once_with("test_model")
        
        # Check that the model and tokenizer were set
        self.assertEqual(inferencer.model, mock_model)
        self.assertEqual(inferencer.tokenizer, mock_tokenizer)
        
        # Check that loading again doesn't reload the model
        mock_model_class.from_pretrained.reset_mock()
        mock_tokenizer_class.from_pretrained.reset_mock()
        
        inferencer._load_model()
        
        mock_model_class.from_pretrained.assert_not_called()
        mock_tokenizer_class.from_pretrained.assert_not_called()
    
    @patch("wisent.inference.inferencer.AutoModelForCausalLM")
    @patch("wisent.inference.inferencer.AutoTokenizer")
    @patch("wisent.inference.inferencer.GenerationConfig")
    def test_generate(self, mock_generation_config_class, mock_tokenizer_class, mock_model_class):
        """Test generating text."""
        from wisent.inference import Inferencer, InferenceConfig
        
        # Mock the model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock the generation config
        mock_generation_config = MagicMock()
        mock_generation_config_class.return_value = mock_generation_config
        
        # Mock the tokenizer
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        mock_tokenizer.decode.return_value = "Generated text"
        
        # Mock the model
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        
        # Create an inferencer
        inferencer = Inferencer(
            model_name="test_model",
            device="cpu"
        )
        
        # Generate text
        result = inferencer.generate(
            prompt="Test prompt",
            config=InferenceConfig(
                max_tokens=100,
                temperature=0.5,
                top_p=0.8,
                top_k=40,
                repetition_penalty=1.2,
                stop_sequences=[".", "!"]
            )
        )
        
        # Check that the model was loaded
        mock_model_class.from_pretrained.assert_called_once()
        mock_tokenizer_class.from_pretrained.assert_called_once()
        
        # Check that the tokenizer was called correctly
        mock_tokenizer.assert_called_once_with("Test prompt", return_tensors="pt")
        
        # Check that the generation config was created correctly
        mock_generation_config_class.assert_called_once_with(
            max_new_tokens=100,
            temperature=0.5,
            top_p=0.8,
            top_k=40,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=mock_tokenizer.pad_token_id or mock_tokenizer.eos_token_id
        )
        
        # Check that the model was called correctly
        mock_model.generate.assert_called_once_with(
            mock_tokenizer.return_value["input_ids"],
            attention_mask=mock_tokenizer.return_value["attention_mask"],
            generation_config=mock_generation_config
        )
        
        # Check that the tokenizer decoded the output correctly
        mock_tokenizer.decode.assert_called_once_with(
            mock_model.generate.return_value[0][3:],
            skip_special_tokens=True
        )
        
        # Check the result
        self.assertEqual(result.text, "Generated text")
        self.assertEqual(result.model, "test_model")
        self.assertEqual(result.prompt, "Test prompt")
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.usage["prompt_tokens"], 3)
        self.assertEqual(result.usage["completion_tokens"], 2)
        self.assertEqual(result.usage["total_tokens"], 5)
        self.assertEqual(result.metadata, {"control_vector": None, "method": None, "scale": None})
    
    @patch("wisent.inference.inferencer.AutoModelForCausalLM")
    @patch("wisent.inference.inferencer.AutoTokenizer")
    @patch("wisent.inference.inferencer.GenerationConfig")
    @patch("wisent.inference.inferencer.ControlVectorHook")
    def test_generate_with_control(self, mock_hook_class, mock_generation_config_class, mock_tokenizer_class, mock_model_class):
        """Test generating text with a control vector."""
        from wisent.inference import Inferencer, InferenceConfig
        
        # Mock the model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock the generation config
        mock_generation_config = MagicMock()
        mock_generation_config_class.return_value = mock_generation_config
        
        # Mock the tokenizer
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        mock_tokenizer.decode.return_value = "Generated text"
        
        # Mock the model
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        
        # Mock the hook
        mock_hook = MagicMock()
        mock_hook_class.return_value = mock_hook
        
        # Create an inferencer
        inferencer = Inferencer(
            model_name="test_model",
            device="cpu"
        )
        
        # Create a control vector
        control_vector = ControlVector(
            name="test_vector",
            model_name="test_model",
            values=[0.1, 0.2, 0.3]
        )
        
        # Generate text with the control vector
        result = inferencer.generate(
            prompt="Test prompt",
            control_vector=control_vector,
            method="caa",
            scale=1.0,
            layers=[0, 1],
            config=InferenceConfig(
                max_tokens=100,
                temperature=0.5,
                top_p=0.8,
                top_k=40,
                repetition_penalty=1.2,
                stop_sequences=[".", "!"]
            )
        )
        
        # Check that the hook was created correctly
        mock_hook_class.assert_called_once()
        self.assertEqual(mock_hook_class.call_args[0][0], control_vector)
        self.assertEqual(mock_hook_class.call_args[0][1].method, "caa")
        self.assertEqual(mock_hook_class.call_args[0][1].scale, 1.0)
        self.assertEqual(mock_hook_class.call_args[0][1].layers, [0, 1])
        
        # Check that the hook was registered and removed
        mock_hook.register.assert_called_once_with(mock_model)
        mock_hook.remove.assert_called_once()
        
        # Check the result
        self.assertEqual(result.text, "Generated text")
        self.assertEqual(result.model, "test_model")
        self.assertEqual(result.prompt, "Test prompt")
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.usage["prompt_tokens"], 3)
        self.assertEqual(result.usage["completion_tokens"], 2)
        self.assertEqual(result.usage["total_tokens"], 5)
        self.assertEqual(result.metadata["control_vector"], "test_vector")
        self.assertEqual(result.metadata["method"], "caa")
        self.assertEqual(result.metadata["scale"], 1.0)


if __name__ == "__main__":
    unittest.main() 