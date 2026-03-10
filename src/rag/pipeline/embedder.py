from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from rag.config import EmbeddingConfig


class SentenceTransformerEmbedder:
    """Embedder using sentence-transformers BGE-M3 model."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._config.model,
                cache_folder=str(self._config.cache_dir),
            )
        return self._model

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self._config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [vec.tolist() for vec in embeddings]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_batch([query])[0]

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

    @property
    def model_version(self) -> str:
        return self._config.model
