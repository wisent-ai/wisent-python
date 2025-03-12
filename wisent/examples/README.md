# Wisent Examples

This directory contains examples and tutorials for using the Wisent library.

## Command-Line Examples

### Extract Activations

Extract activations from a model and optionally upload them to the Wisent backend:

```bash
python extract_activations.py --model "mistralai/Mistral-7B-Instruct-v0.1" \
                             --prompt "Tell me about quantum computing" \
                             --layers -1,-2,-3 \
                             --tokens -1 \
                             --api-key "your_api_key" \
                             --upload
```

### Control Vector Inference

Generate text with control vectors:

```bash
python control_vector_inference.py --model "mistralai/Mistral-7B-Instruct-v0.1" \
                                  --prompt "Tell me about quantum computing" \
                                  --vectors '{"helpful": 0.8, "concise": 0.5}' \
                                  --method "caa" \
                                  --scale 1.0 \
                                  --api-key "your_api_key" \
                                  --local
```

### Simple Chat

A simple chat application that uses control vectors to steer the model's responses:

```bash
python simple_chat.py --model "mistralai/Mistral-7B-Instruct-v0.1" \
                     --api-key "your_api_key" \
                     --vectors '{"helpful": 0.8, "concise": 0.5}' \
                     --method "caa" \
                     --scale 1.0 \
                     --system-prompt "You are a helpful, honest, and concise assistant."
```

## Jupyter Notebooks

The `notebooks` directory contains Jupyter notebooks that demonstrate various aspects of the Wisent library:

- `basic_usage.ipynb`: Basic usage of the Wisent library for working with control vectors and model inference.
- `activation_extraction.ipynb`: How to extract activations from language models.
- `custom_control_vectors.ipynb`: How to create and use custom control vectors.

## Running the Examples

To run these examples, you'll need to:

1. Install the Wisent library:
   ```bash
   pip install wisent
   ```

2. Get an API key from [Wisent](https://wisent.ai).

3. Set the API key as an environment variable:
   ```bash
   export WISENT_API_KEY="your_api_key"
   ```

4. Run the examples as shown above.

## Additional Resources

- [Wisent Documentation](https://docs.wisent.ai)
- [Wisent GitHub Repository](https://github.com/wisent-ai/wisent) 