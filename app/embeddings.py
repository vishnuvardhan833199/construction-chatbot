# app/embeddings.py
from sentence_transformers import SentenceTransformer
import os

# Use a smaller model for speed (can override with ENV var EMBED_MODEL)
_EMBED_MODEL = os.environ.get("EMBED_MODEL", "paraphrase-MiniLM-L3-v2")
_model = None

def get_embed_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(_EMBED_MODEL)
    return _model

def embed_texts(texts):
    """
    texts: list[str]
    returns: numpy array of embeddings
    """
    model = get_embed_model()
    return model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True
    )
