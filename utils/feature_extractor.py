"""
=============================================================================
MODULE: feature_extractor.py
PURPOSE: Feature Extraction Pipeline for Emotion Detection
DESCRIPTION:
    Implements two complementary feature representations:
      1. TF-IDF  – sparse bag-of-words with term weighting (baseline)
      2. Word2Vec – dense semantic embeddings averaged over sentence tokens

    Both extractors expose a unified sklearn-compatible fit / transform API
    so they can be hot-swapped in the training pipeline.
=============================================================================
"""

import logging
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    logging.warning("gensim not installed – Word2Vec extractor unavailable.")

logger = logging.getLogger(__name__)


# =============================================================================
# 1. TF-IDF Feature Extractor
# =============================================================================

class TFIDFExtractor:
    """
    TF-IDF vectoriser with L2 normalisation.

    Why TF-IDF?
    -----------
    • Captures term importance relative to the corpus – rare emotion-
      bearing words (e.g. "elated", "devastated") receive high weight.
    • Fast, interpretable, and memory-efficient (sparse matrix).
    • N-gram range (1, 2) captures bigrams like "not happy" or "very sad"
      that single tokens would miss.

    Parameters
    ----------
    max_features : int   – vocabulary size cap (keeps memory bounded)
    ngram_range  : tuple – (min_n, max_n) for n-gram extraction
    min_df       : int   – minimum document frequency (noise filter)
    max_df       : float – maximum document frequency (common-word filter)
    sublinear_tf : bool  – apply 1 + log(tf) instead of raw tf
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            analyzer="word",
            token_pattern=r"(?u)\b[a-z][a-z]+\b",  # alpha tokens only
        )
        self.normalizer = Normalizer(norm="l2")
        self._fitted = False
        logger.info(
            "TFIDFExtractor init | max_features=%d | ngram=%s",
            max_features,
            ngram_range,
        )

    def fit(self, texts: pd.Series | list) -> "TFIDFExtractor":
        """Fit the TF-IDF vocabulary on the training corpus."""
        logger.info("Fitting TF-IDF on %d documents …", len(texts))
        self.vectorizer.fit(texts)
        self._fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info("TF-IDF fitted | vocabulary size = %d", vocab_size)
        return self

    def transform(self, texts: pd.Series | list) -> np.ndarray:
        """Transform texts to normalised TF-IDF matrix."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X_sparse = self.vectorizer.transform(texts)
        X_dense = self.normalizer.transform(X_sparse)  # L2-normalise rows
        return X_dense

    def fit_transform(self, texts: pd.Series | list) -> np.ndarray:
        return self.fit(texts).transform(texts)

    @property
    def feature_names(self) -> list:
        """Return the list of feature (token / n-gram) names."""
        return self.vectorizer.get_feature_names_out().tolist()


# =============================================================================
# 2. Word2Vec Sentence Embedding Extractor
# =============================================================================

class Word2VecExtractor:
    """
    Train a Word2Vec model on the corpus and represent each sentence as
    the mean of its token embeddings.

    Why Word2Vec (mean pooling)?
    ----------------------------
    • Captures semantic similarity – synonyms cluster together in the
      embedding space, making the model robust to vocabulary variation.
    • Dense fixed-length representation compatible with all sklearn models.
    • Training on the domain corpus ensures emotion-specific contexts are
      captured (social media, reviews, etc.).

    Parameters
    ----------
    vector_size : int  – embedding dimensionality
    window      : int  – context window size
    min_count   : int  – minimum word frequency
    workers     : int  – parallel training threads
    epochs      : int  – training iterations
    sg          : int  – 0 = CBOW, 1 = Skip-gram (Skip-gram preferred for
                         rare emotion words)
    """

    def __init__(
        self,
        vector_size: int = 200,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 15,
        sg: int = 1,
    ):
        if not GENSIM_AVAILABLE:
            raise ImportError("Install gensim: pip install gensim")
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.sg = sg
        self.model: Word2Vec | None = None
        self._fitted = False
        logger.info(
            "Word2VecExtractor init | dim=%d | window=%d | sg=%d",
            vector_size,
            window,
            sg,
        )

    @staticmethod
    def _tokenize(texts) -> list[list[str]]:
        """Split pre-cleaned texts into token lists."""
        return [t.split() for t in texts]

    def fit(self, texts) -> "Word2VecExtractor":
        """Train Word2Vec on the corpus."""
        sentences = self._tokenize(texts)
        logger.info("Training Word2Vec on %d sentences …", len(sentences))
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            sg=self.sg,
            seed=42,
        )
        self._fitted = True
        vocab_size = len(self.model.wv.key_to_index)
        logger.info("Word2Vec trained | vocab = %d words", vocab_size)
        return self

    def _sentence_vector(self, tokens: list[str]) -> np.ndarray:
        """Average valid token embeddings; return zero vector if none found."""
        wv = self.model.wv
        vecs = [wv[t] for t in tokens if t in wv]
        if not vecs:
            return np.zeros(self.vector_size, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)

    def transform(self, texts) -> np.ndarray:
        """Return sentence embedding matrix (n_samples × vector_size)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        sentences = self._tokenize(texts)
        return np.vstack([self._sentence_vector(s) for s in sentences])

    def fit_transform(self, texts) -> np.ndarray:
        return self.fit(texts).transform(texts)

    def save(self, path: str) -> None:
        """Persist the Word2Vec model to disk."""
        if self.model:
            self.model.save(path)
            logger.info("Word2Vec model saved → %s", path)

    def load(self, path: str) -> "Word2VecExtractor":
        """Load a saved Word2Vec model from disk."""
        self.model = Word2Vec.load(path)
        self._fitted = True
        logger.info("Word2Vec model loaded ← %s", path)
        return self
