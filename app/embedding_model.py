"""spaCy-backed word embedding service."""

import spacy


class EmbeddingModel:
    """Load a spaCy model once and expose its dense text vectors."""

    def __init__(self, model_name: str = "en_core_web_lg") -> None:
        self.nlp = spacy.load(model_name)

    def get_embedding(self, query_word: str) -> list[float]:
        """Return the embedding vector for a non-empty query word."""
        normalized_word = query_word.strip()
        if not normalized_word:
            raise ValueError("query_word must not be blank")

        document = self.nlp(normalized_word)
        if not document.has_vector or document.vector_norm == 0:
            raise ValueError(f"No embedding is available for '{normalized_word}'")

        return document.vector.astype(float).tolist()

