# Serving Reward Models with Modal

This project serves two Hugging Face reward models using Modal with NVIDIA L4 GPUs:
- `jasong03/compact-grpo-lora-qwen3-4b-reward-model`
- `jasong03/compact-grpo-lora-qwen3-4b-reward-function`

## Setup

1. Install Modal:
```bash
pip install modal
```

2. Set up Hugging Face token:

   **Option A: Using Modal Secrets (Recommended for production)**
   ```bash
   modal secret create hf-token HF_TOKEN=your_huggingface_token_here
   ```

   **Option B: Using environment variable (for local development)**
   ```bash
   # Create a .env file
   echo "HF_TOKEN=your_huggingface_token_here" > .env
   
   # Or export it directly
   export HF_TOKEN=your_huggingface_token_here
   ```

3. Authenticate with Modal:
```bash
modal token new
```

4. Deploy the app:
```bash
modal deploy app.py
```

## Usage

After deployment, you'll get two web endpoints. Replace `YOUR_USERNAME` with your Modal username in the URLs below.

### 1. Model Endpoint (adapter: jasong03/compact-grpo-lora-qwen3-4b-reward-model)
```bash
curl -X POST "https://YOUR_USERNAME--reward-models-serving-model.modal.run" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a helpful and accurate response to the question."}'
```

### 2. Reward Endpoint (adapter: jasong03/compact-grpo-lora-qwen3-4b-reward-function)
```bash
curl -X POST "https://YOUR_USERNAME--reward-models-serving-reward.modal.run" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a helpful and accurate response to the question."}'
```

Notes:
- Both endpoints accept JSON payload: `{ "text": "your prompt" }`
- Response is JSON: `{ "text": "generated text" }`

## Features

- Fixed base model: `unsloth/Qwen3-4B-Base`
- GPU acceleration with NVIDIA L4
- Hugging Face authentication built-in
- Two endpoints mapping to each LoRA adapter
- Simple text-in, text-out generation

## Notes

- The models are gated on Hugging Face, so ensure your token has access
- The base model is automatically detected from the adapter configuration
- Models are loaded in float16 for efficiency
- Container idle timeout is set to 300 seconds