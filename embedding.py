import os
from sentence_transformers import SentenceTransformer

_model = None


def get_embedder():
    global _model
    if _model is None:
        local_snapshot = os.path.expanduser(
            "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
        )
        if os.path.isdir(local_snapshot):
            _model = SentenceTransformer(local_snapshot)
        else:
            try:
                _model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
            except Exception:
                _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def embed(text: str):
    return get_embedder().encode(text, normalize_embeddings=True)
