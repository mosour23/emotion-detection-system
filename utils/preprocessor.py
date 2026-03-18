"""
=============================================================================
MODULE: preprocessor.py
PURPOSE: Text Preprocessing Pipeline for Emotion Detection
DESCRIPTION:
    Handles all NLP preprocessing steps including tokenization,
    stop-word removal, lemmatization, emoji/special character handling.
    Designed to be modular and reusable across training and inference.
=============================================================================
"""

import re
import string
import logging
import unicodedata

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Ensure required NLTK resources are available
# ---------------------------------------------------------------------------
NLTK_RESOURCES = [
    ("tokenizers/punkt",        "punkt"),
    ("tokenizers/punkt_tab",    "punkt_tab"),
    ("corpora/stopwords",       "stopwords"),
    ("corpora/wordnet",         "wordnet"),
    ("corpora/omw-1.4",         "omw-1.4"),
]
for path, name in NLTK_RESOURCES:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name, quiet=True)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Emoji → text conversion dictionary (common emoticons + unicode ranges)
# ---------------------------------------------------------------------------
EMOJI_LEXICON = {
    ":)":  "happy",    ":-)": "happy",  "=)":  "happy",
    ":D":  "laughing", ":-D": "laughing",
    ":(" :  "sad",     ":-(": "sad",    "=(":  "sad",
    ":'(": "crying",   ";(":  "crying",
    ">:(": "angry",    ">:-(":"angry",
    ":o":  "surprised",":-o": "surprised",
    ":p":  "playful",  ":-p": "playful",
    ":-/": "confused", ":/":  "confused",
    "<3":  "love",     "</3": "heartbroken",
    "xd":  "laughing",
}

# Unicode emoji ranges that will be stripped after text-replacement pass
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & maps
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002500-\U00002BEF"   # misc chinese chars
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


class TextPreprocessor:
    """
    Complete NLP preprocessing pipeline.

    Steps (in order):
        1. Lower-case conversion
        2. Emoticon → text substitution
        3. Unicode emoji removal
        4. URL removal
        5. HTML tag stripping
        6. Special character / punctuation removal
        7. Number normalisation
        8. Tokenisation
        9. Stop-word removal
        10. Lemmatisation
        11. Short-token filtering
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_token_len: int = 2,
        custom_stopwords: list | None = None,
    ):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_token_len = min_token_len

        # Build stop-word set (English) + custom words
        self._stop_words = set(stopwords.words("english"))
        if custom_stopwords:
            self._stop_words.update(custom_stopwords)

        # Emotion-bearing words we never want removed
        _preserve = {"not", "no", "nor", "very", "too", "more", "most", "never"}
        self._stop_words -= _preserve

        self._lemmatizer = WordNetLemmatizer()
        logger.info("TextPreprocessor initialised.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _replace_emoticons(text: str) -> str:
        """Swap ASCII emoticons with their textual equivalents."""
        for emoticon, word in sorted(EMOJI_LEXICON.items(), key=lambda x: -len(x[0])):
            text = text.replace(emoticon, f" {word} ")
        return text

    @staticmethod
    def _remove_urls(text: str) -> str:
        return re.sub(r"https?://\S+|www\.\S+", " ", text)

    @staticmethod
    def _remove_html(text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text)

    @staticmethod
    def _normalise_numbers(text: str) -> str:
        """Replace standalone numbers with the token NUMBER."""
        return re.sub(r"\b\d+\b", "NUMBER", text)

    @staticmethod
    def _remove_special_chars(text: str) -> str:
        """Keep only alphabetic characters, spaces, and apostrophes."""
        text = re.sub(r"[^a-z\s']", " ", text)
        text = re.sub(r"'+", "'", text)          # collapse multiple apostrophes
        text = re.sub(r"\s+", " ", text).strip() # collapse whitespace
        return text

    def _lemmatise_token(self, token: str) -> str:
        """Lemmatise with both verb and noun forms (verb first for accuracy)."""
        lemma = self._lemmatizer.lemmatize(token, pos="v")
        if lemma == token:
            lemma = self._lemmatizer.lemmatize(token, pos="n")
        return lemma

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """
        Run the full preprocessing pipeline on a single text string.

        Parameters
        ----------
        text : str  – raw input text

        Returns
        -------
        str  – processed text (space-joined tokens)
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # Step 1 – lower-case
        text = text.lower()

        # Step 2 – emoticons → text
        text = self._replace_emoticons(text)

        # Step 3 – unicode emoji removal
        text = _EMOJI_PATTERN.sub(" ", text)

        # Step 4 – URLs
        text = self._remove_urls(text)

        # Step 5 – HTML tags
        text = self._remove_html(text)

        # Step 6 – numbers
        text = self._normalise_numbers(text)

        # Step 7 – special characters
        text = self._remove_special_chars(text)

        # Step 8 – tokenise
        tokens = word_tokenize(text)

        # Step 9 – stop-word removal
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words]

        # Step 10 – lemmatisation
        if self.lemmatize:
            tokens = [self._lemmatise_token(t) for t in tokens]

        # Step 11 – filter short tokens
        tokens = [t for t in tokens if len(t) >= self.min_token_len]

        return " ".join(tokens)

    def clean_series(self, series: pd.Series, verbose: bool = True) -> pd.Series:
        """
        Apply clean() element-wise to a pandas Series.

        Parameters
        ----------
        series  : pd.Series  – raw text column
        verbose : bool       – log progress every 5 000 rows

        Returns
        -------
        pd.Series – cleaned texts
        """
        results = []
        for i, text in enumerate(series, 1):
            results.append(self.clean(text))
            if verbose and i % 5_000 == 0:
                logger.info("Preprocessed %d / %d texts …", i, len(series))
        return pd.Series(results, index=series.index)
