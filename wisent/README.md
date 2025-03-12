# Wisent

A Python client library for interacting with the Wisent backend services. Wisent enables developers to control language model behavior using activation engineering.

## Installation

```bash
pip install wisent
```

Or install from source:

```bash
git clone https://github.com/wisent-ai/wisent.git
cd wisent
pip install -e .
```

## Features

- **Activations**: Extract and send model activations to the Wisent backend
- **Control Vectors**: Retrieve and apply control vectors for model inference
- **Inference**: Utilities for applying control vectors during inference
- **Utilities**: Helper functions for common tasks

## Quick Start

```python
from wisent import WisentClient

# Initialize the client
client = WisentClient(api_key="your_api_key", base_url="https://api.wisent.ai")

# Extract activations from a model and send to backend
activations = client.activations.extract(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    prompt="Tell me about quantum computing",
    layers=[0, 12, 24]
)

# Get a control vector from the backend
control_vector = client.control_vector.get(
    name="helpful",
    model="mistralai/Mistral-7B-Instruct-v0.1"
)

# Apply a control vector during inference
response = client.inference.generate_with_control(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    prompt="Tell me about quantum computing",
    control_vectors={"helpful": 0.8, "concise": 0.5}
)

# Print the response
print(response.text)
```

## Advanced Usage

### Extracting Activations

```python
from wisent.activations import ActivationExtractor

# Create an extractor
extractor = ActivationExtractor(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    device="cuda"
)

# Extract activations for a specific prompt
activations = extractor.extract(
    prompt="Tell me about quantum computing",
    layers=[0, 12, 24],
    tokens_to_extract=[-10, -1]  # Extract last 10 tokens and final token
)

# Send activations to the Wisent backend
from wisent import WisentClient
client = WisentClient(api_key="your_api_key")
client.activations.upload(activations)
```

### Working with Control Vectors

```python
from wisent.control_vector import ControlVectorManager

# Initialize the manager
manager = ControlVectorManager(api_key="your_api_key")

# Get a control vector
helpful_vector = manager.get("helpful", model="mistralai/Mistral-7B-Instruct-v0.1")

# Combine multiple vectors
combined_vector = manager.combine(
    vectors={
        "helpful": 0.8,
        "concise": 0.5
    },
    model="mistralai/Mistral-7B-Instruct-v0.1"
)

# Apply during inference
from wisent.inference import Inferencer
inferencer = Inferencer(model_name="mistralai/Mistral-7B-Instruct-v0.1")
response = inferencer.generate(
    prompt="Tell me about quantum computing",
    control_vector=combined_vector,
    method="caa"  # Context-Aware Addition
)
```

### Batch Processing

```python
# Extract activations for multiple prompts
prompts = [
    "Explain quantum computing",
    "What is machine learning?",
    "Tell me about neural networks"
]

results = []
for prompt in prompts:
    activations = extractor.extract(prompt=prompt, layers=[0, 12, 24])
    results.append(activations)

# Batch upload
client.activations.upload_batch(results)
```

## Supported Models

Wisent currently supports the following models:
- Mistral models (mistralai/Mistral-7B-Instruct-v0.1, mistralai/Mixtral-8x7B-Instruct-v0.1)
- Llama 2 models (meta-llama/Llama-2-7b-chat-hf, meta-llama/Llama-2-13b-chat-hf)
- Claude models (via API integration)
- GPT models (via API integration)

## Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- transformers 4.30.0 or higher

## Compatibility

The library has been tested on:
- Linux (Ubuntu 20.04+)
- macOS (11.0+)
- Windows 10/11

## Contributing

We welcome contributions to Wisent! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Documentation

For full documentation, visit [docs.wisent.ai](https://docs.wisent.ai).

## Support

For support, please:
- Check the [documentation](https://docs.wisent.ai)
- Open an issue on GitHub
- Contact us at support@wisent.ai

## License

MIT
