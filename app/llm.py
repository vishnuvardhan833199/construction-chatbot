# app/llm.py
from transformers import pipeline, set_seed
import os

# default model (small) - runs on CPU. You can change to a better local model if available.
DEFAULT_MODEL = os.environ.get("GEN_MODEL", "distilgpt2")

_generation_pipeline = None

def get_generator(model_name=None):
    global _generation_pipeline
    if _generation_pipeline is None or (model_name and model_name != _generation_pipeline.model.config._name_or_path):
        model = model_name or DEFAULT_MODEL
        # create text-generation pipeline
        _generation_pipeline = pipeline("text-generation", model=model, tokenizer=model, device=-1)
    return _generation_pipeline

def generate_answer(prompt: str, max_new_tokens: int = 200, temperature: float = 0.2, top_p: float = 0.95):
    gen = get_generator()
    set_seed(42)
    out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p, num_return_sequences=1)
    return out[0]["generated_text"]
