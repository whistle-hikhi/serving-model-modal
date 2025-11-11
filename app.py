import modal
import os
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# Initialize Modal app
app = modal.App("reward-models-serving")

# Define the image with necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "accelerate>=0.30.0",
        "huggingface-hub>=0.20.0",
        "sentencepiece>=0.1.99",
        "protobuf>=4.25.0",
        "fastapi",
    )
)

# Hugging Face token from environment variable (for local development)
# In Modal, the token should be set as a secret named "hf-token"
HF_TOKEN = os.getenv("HF_TOKEN")

# Model IDs
REWARD_MODEL_ID = "jasong03/compact-grpo-lora-qwen3-4b-reward-model"
REWARD_FUNCTION_ID = "jasong03/compact-grpo-lora-qwen3-4b-reward-function"


@app.cls(
    image=image,
    gpu="L4",
    scaledown_window=300,
    secrets=[modal.Secret.from_name("hf-token")],
)
class TextModel:
    @modal.enter()
    def setup(self):
        """Initialize models when the class container starts."""
        # Get HF token from environment (set by Modal secret or env var)
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is required. Please set it in Modal secrets or environment.")
        
        # Login to Hugging Face
        login(token=hf_token)

        # Use fixed base model per requested process
        base_model_id = "unsloth/Qwen3-4B-Base"
        print(f"Base model: {base_model_id} | Adapters: {REWARD_MODEL_ID}, {REWARD_FUNCTION_ID}")
        
        # Load base model and tokenizer
        print("Loading base model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            token=hf_token,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype="auto",
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
        )
        
        # Load first adapter (reward model)
        print("Loading first adapter...")
        self.model_1 = PeftModel.from_pretrained(
            self.base_model,
            REWARD_MODEL_ID,
            torch_dtype="auto",
            token=hf_token,
        )
        self.model_1.eval()
        
        # Create a fresh base model instance for the second adapter
        print("Loading second adapter...")
        base_model_for_model_2 = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype="auto",
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
        )
        
        self.model_2 = PeftModel.from_pretrained(
            base_model_for_model_2,
            REWARD_FUNCTION_ID,
            torch_dtype="auto",
            token=hf_token,
        )
        self.model_2.eval()
        
        print("Models loaded successfully!")

    def _generate_text(self, model, text: str, max_new_tokens: int = 100) -> str:
        """Generate text using the model following the provided reference process."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Decode the whole sequence (as in the provided process)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

    @modal.method()
    def generate_model_1(self, text: str) -> str:
        """Generate text using first adapter."""
        return self._generate_text(self.model_1, text)

    @modal.method()
    def generate_model_2(self, text: str) -> str:
        """Generate text using second adapter."""
        return self._generate_text(self.model_2, text)


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def model(item: dict):
    """Web endpoint for adapter: jasong03/compact-grpo-lora-qwen3-4b-reward-model."""
    text = item.get("text", "")
    if not text:
        return {"error": "Text is required"}
    
    text_model = TextModel()
    result = text_model.generate_model_1.remote(text)
    return {"text": result}


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def reward(item: dict):
    """Web endpoint for adapter: jasong03/compact-grpo-lora-qwen3-4b-reward-function."""
    text = item.get("text", "")
    if not text:
        return {"error": "Text is required"}
    
    text_model = TextModel()
    result = text_model.generate_model_2.remote(text)
    return {"text": result}

