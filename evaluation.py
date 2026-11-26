import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def evaluate_rag(query: str, answer: str, contexts: list[str]) -> dict:
    """
    Lightweight embedding-similarity indicators.

    These are similarity-based proxies, not formal factuality guarantees
    or a substitute for human evaluation.
    """
    contexts = [item.strip() for item in (contexts or []) if item and item.strip()]
    if not query.strip() or not answer.strip() or not contexts:
        return {
            "Relevance Similarity": 0.0,
            "Context Alignment": 0.0,
            "Best Context Similarity": 0.0,
            "Context Coverage Proxy": 0.0,
            "Lowest Context Similarity": 0.0,
        }

    model = _get_model()
    query_emb = model.encode([query], normalize_embeddings=True)
    answer_emb = model.encode([answer], normalize_embeddings=True)
    context_emb = model.encode(contexts, normalize_embeddings=True)

    answer_context_scores = cosine_similarity(answer_emb, context_emb)[0]
    relevance = float(cosine_similarity(query_emb, answer_emb)[0][0])

    return {
        "Relevance Similarity": round(relevance, 3),
        "Context Alignment": round(float(np.mean(answer_context_scores)), 3),
        "Best Context Similarity": round(float(np.max(answer_context_scores)), 3),
        "Context Coverage Proxy": round(float(np.mean(answer_context_scores)), 3),
        "Lowest Context Similarity": round(float(np.min(answer_context_scores)), 3),
        "Note": "Embedding similarity indicators; not a factuality guarantee.",
    }
