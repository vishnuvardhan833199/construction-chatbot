# app/retriever.py
import os
import faiss
import json
import numpy as np
from app.embeddings import embed_texts
from pathlib import Path
import sqlite3

INDEX_PATH = Path("vector_store/faiss.index")
META_PATH = Path("vector_store/meta.json")

def ensure_vector_dir():
    Path("vector_store").mkdir(exist_ok=True)

def save_index(index, meta_list):
    ensure_vector_dir()
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)

def load_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        return None, None
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta_list = json.load(f)
    return index, meta_list

def build_index_from_docs(docs: list):
    """
    docs: list of dicts: {"id": <str>, "text": <str>, "source": <filename>}
    """
    texts = [d["text"] for d in docs]
    meta = [{"id": d["id"], "source": d.get("source",""), "text_snippet": d["text"][:400]} for d in docs]
    embs = embed_texts(texts)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via inner product after normalization
    # normalize vectors
    faiss.normalize_L2(embs)
    index.add(embs.astype(np.float32))
    save_index(index, meta)
    return index, meta

def query_index(query: str, top_k: int = 3):
    loaded = load_index()
    if loaded[0] is None:
        return []
    index, meta = loaded
    q_emb = embed_texts([query])
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb.astype(np.float32), top_k)
    results = []
    for idx in I[0]:
        if idx < len(meta):
            results.append(meta[idx])
    return results
