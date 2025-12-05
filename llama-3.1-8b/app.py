import modal
import os
from typing import Optional

"""
Serve `meta-llama/Llama-3.1-8B` on Modal using the transformers.pipeline API.

This is the basic local example:

    import transformers, torch
    model_id = "meta-llama/Llama-3.1-8B"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    pipeline("Hey how are you doing today?")

Below we wrap that into a Modal app so the model is loaded once and reused.

Model card: https://huggingface.co/meta-llama/Llama-3.1-8B
Make sure you’ve accepted the model license on Hugging Face and comply
with the Llama 3.1 Community License (e.g. “Built with Llama” attribution).
"""

app = modal.App("llama-3.1-8b-serving")

MODEL_ID = "meta-llama/Llama-3.1-8B"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "huggingface-hub>=0.20.0",
        "sentencepiece>=0.1.99",
        "protobuf>=4.25.0",
        "fastapi",
        "pydantic",
    )
)


@app.cls(
    image=image,
    gpu="A10G",  # choose a GPU type your Modal account supports
    secrets=[modal.Secret.from_name("hf-token")],  # exposes HF_TOKEN env var
    timeout=600,
)
class Llama31_8B:
    """Long-lived container that holds a text-generation pipeline."""

    @modal.enter()
    def load(self):
        import torch
        import transformers

        # If your secret is named "hf-token", Modal will set HF_TOKEN env var.
        hf_token: Optional[str] = os.getenv("HF_TOKEN")

        # Verify GPU is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GPU is required for this model.")

        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=MODEL_ID,
            model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "cuda:0"},
            token=hf_token,
        )

    @modal.method()
    def generate(
        self,
        prompt: str,
    ) -> str:
        """Generate text from a prompt using Llama 3.1 8B."""

        outputs = self.pipeline(
            prompt,
            # Use a large but practical upper bound for Llama 3.1 8B generations.
            # Actual usable max depends on prompt length vs model context window.
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        # transformers pipeline returns a list of dicts: [{"generated_text": "..."}]
        return outputs[0]["generated_text"]


llama = Llama31_8B()


@app.local_entrypoint()
def main():
    """Quick local test: `modal run app.py`."""
    prompt = "Hey how are you doing today?"
    result = llama.generate.remote(prompt)
    print(result)


# ---------- HTTP API (FastAPI on Modal) ----------


def _build_api():
    """Create FastAPI app that calls the Modal model class."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    fastapi_app = FastAPI(
        title="Llama 3.1 8B Text Generation API",
        description="Simple text-generation API powered by meta-llama/Llama-3.1-8B (Built with Llama).",
    )

    class GenerateRequest(BaseModel):
        prompt: str

    class GenerateResponse(BaseModel):
        completion: str

    @fastapi_app.get("/")
    def root():
        return {
            "message": "Llama 3.1 8B serving API (Built with Llama)",
            "model": MODEL_ID,
        }

    @fastapi_app.post("/generate", response_model=GenerateResponse)
    def generate(body: GenerateRequest) -> GenerateResponse:
        completion = llama.generate.remote(body.prompt)
        return GenerateResponse(completion=completion)

    return fastapi_app


@app.function(image=image)
@modal.asgi_app()
def serve():
    """
    ASGI entrypoint for Modal.

    Run locally:
        modal serve app.py

    Then call:
        curl -X POST http://localhost:8000/generate \\
          -H 'Content-Type: application/json' \\
          -d '{"prompt": "Hey how are you doing today?"}'
    """
    return _build_api()

