"""
=============================================================================
MODULE: data_loader.py
PURPOSE: Dataset acquisition, validation, and splitting
DESCRIPTION:
    Loads the Emotions dataset (Kaggle / HuggingFace) or generates a rich
    synthetic corpus when the real dataset is unavailable (offline/CI).
    Provides an 80/20 stratified train-test split.
=============================================================================
"""

import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical emotion labels (6-class, aligned with the Emotions dataset)
# ---------------------------------------------------------------------------
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

EMOTION_COLOR_MAP = {
    "sadness":  "#5B8FDE",
    "joy":      "#F4D03F",
    "love":     "#E91E8C",
    "anger":    "#E74C3C",
    "fear":     "#8E44AD",
    "surprise": "#1ABC9C",
}


# =============================================================================
# Synthetic data generator (fallback / demo mode)
# =============================================================================

_TEMPLATES: dict[str, list[str]] = {
    "sadness": [
        "I feel so {adj1} and lonely today.",
        "Everything feels hopeless and I cannot stop {verb1}.",
        "I miss you so much it {verb2} inside.",
        "Life seems so {adj2} without you here.",
        "My heart is {adj3} and I don't know why.",
        "Tears keep {verb3} and I can't make them stop.",
        "I'm completely {adj4} after what happened.",
        "Nothing brings me joy anymore, just {adj1} emptiness.",
        "I cried myself to sleep feeling so {adj2}.",
        "Grief has left me {adj3} and exhausted.",
    ],
    "joy": [
        "This is the best day of my life, I am so {adj1}!",
        "I feel absolutely {adj1} and grateful for everything.",
        "Just got amazing news and I can't stop {verb1}!",
        "Life is beautiful and I am {adj2} to be alive.",
        "I passed my exams and I'm {adj1} beyond words!",
        "Dancing and {verb1} because everything is perfect.",
        "Today was {adj2} – I feel so blessed!",
        "My heart is full of {adj3} and I love it.",
        "Laughing and smiling because life is {adj1}.",
        "I am {adj2} with how things turned out!",
    ],
    "love": [
        "I love you more than words can express.",
        "You make me feel {adj1} every single day.",
        "My heart melts when you {verb1} at me.",
        "Being with you is the greatest feeling in the world.",
        "I am so {adj2} to have you in my life.",
        "Every moment with you is {adj1} and precious.",
        "You are my everything and I {verb2} you deeply.",
        "I feel so {adj3} whenever we are together.",
        "Your presence makes my heart feel {adj1} and warm.",
        "Loving you is the easiest and most {adj2} thing.",
    ],
    "anger": [
        "I am absolutely {adj1} about what just happened!",
        "This is completely unacceptable and I am {adj2}!",
        "How dare they do this – I am {adj1} beyond belief!",
        "I can't believe this injustice, it makes me {verb1}!",
        "Stop {verb2} me around, I am done with this!",
        "I am so {adj3} right now I can barely think.",
        "They crossed every line and I am {adj1}!",
        "This situation makes me want to {verb3} everything.",
        "I am {adj2} at the constant disrespect I receive.",
        "My blood is {verb1} because of their behaviour.",
    ],
    "fear": [
        "I am absolutely {adj1} about what might happen.",
        "The thought of it fills me with {adj2} dread.",
        "I cannot sleep because I am so {adj1}.",
        "My hands are {verb1} and I feel {adj3}.",
        "I'm {adj1} of the unknown ahead of me.",
        "A deep sense of {adj2} keeps overwhelming me.",
        "I don't feel safe and I'm {adj1} about it.",
        "Every shadow makes me feel {adj3} and anxious.",
        "Panic is {verb1} through my chest right now.",
        "I'm constantly {adj1} that something bad will happen.",
    ],
    "surprise": [
        "I cannot believe what just happened – totally {adj1}!",
        "Wow, that was completely {adj2} and unexpected!",
        "Nobody told me about this – I am {adj3}!",
        "That turn of events left me {adj1} and confused.",
        "I never saw that coming, what a {adj2} moment!",
        "My jaw dropped when I heard the {adj3} news.",
        "This is the most {adj1} thing I've ever experienced!",
        "I am in complete {adj2} at what just unfolded.",
        "Never expected that – truly {adj3} by the outcome.",
        "The announcement left everyone {adj1} and speechless.",
    ],
}

_FILL: dict[str, dict[str, list[str]]] = {
    "sadness": {
        "adj1": ["heartbroken", "devastated", "miserable", "desolate"],
        "adj2": ["bleak", "grey", "hollow", "meaningless"],
        "adj3": ["shattered", "broken", "crushed"],
        "adj4": ["defeated", "wrecked", "drained"],
        "verb1": ["crying", "weeping", "sobbing"],
        "verb2": ["aches", "hurts", "burns"],
        "verb3": ["falling", "flowing", "streaming"],
    },
    "joy": {
        "adj1": ["ecstatic", "thrilled", "overjoyed", "elated"],
        "adj2": ["wonderful", "amazing", "fantastic"],
        "adj3": ["happiness", "gratitude", "bliss"],
        "verb1": ["smiling", "laughing", "celebrating"],
    },
    "love": {
        "adj1": ["wonderful", "complete", "cherished"],
        "adj2": ["lucky", "blessed", "grateful"],
        "adj3": ["warm", "safe", "content"],
        "verb1": ["smile", "look", "laugh"],
        "verb2": ["adore", "cherish", "love"],
    },
    "anger": {
        "adj1": ["furious", "enraged", "livid"],
        "adj2": ["infuriated", "outraged", "seething"],
        "adj3": ["boiling", "blazing", "incensed"],
        "verb1": ["boil", "scream", "explode"],
        "verb2": ["pushing", "shoving", "dragging"],
        "verb3": ["smash", "destroy", "tear down"],
    },
    "fear": {
        "adj1": ["terrified", "petrified", "horrified"],
        "adj2": ["overwhelming", "crushing", "paralyzing"],
        "adj3": ["scared", "startled", "frozen"],
        "verb1": ["trembling", "pounding", "racing"],
    },
    "surprise": {
        "adj1": ["shocked", "stunned", "astonished"],
        "adj2": ["unexpected", "incredible", "jaw-dropping"],
        "adj3": ["amazed", "bewildered", "flabbergasted"],
    },
}


def _generate_synthetic_sample(emotion: str) -> str:
    """Fill a random template with random adjectives/verbs for an emotion."""
    template = random.choice(_TEMPLATES[emotion])
    fills = _FILL[emotion]
    text = template
    for key, options in fills.items():
        placeholder = "{" + key + "}"
        if placeholder in text:
            text = text.replace(placeholder, random.choice(options))
    return text


def generate_synthetic_dataset(
    samples_per_class: int = 800,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a balanced synthetic emotion dataset.

    Parameters
    ----------
    samples_per_class : int – number of samples per emotion label
    seed              : int – random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns ['text', 'label', 'emotion']
    """
    random.seed(seed)
    np.random.seed(seed)

    rows = []
    for idx, emotion in enumerate(EMOTION_LABELS):
        for _ in range(samples_per_class):
            rows.append({
                "text":    _generate_synthetic_sample(emotion),
                "label":   idx,
                "emotion": emotion,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(
        "Synthetic dataset generated | %d samples | %d classes",
        len(df),
        len(EMOTION_LABELS),
    )
    return df


# =============================================================================
# Real dataset loader (Kaggle / CSV)
# =============================================================================

def load_emotions_csv(path: str) -> pd.DataFrame:
    """
    Load the Emotions dataset from a local CSV file.

    Expected CSV columns: text, label  (label = 0–5 integer or emotion name)
    Kaggle dataset: dair-ai/emotion  (available on HuggingFace Datasets)

    Parameters
    ----------
    path : str – path to the CSV file

    Returns
    -------
    pd.DataFrame with columns ['text', 'label', 'emotion']
    """
    logger.info("Loading dataset from %s …", path)
    df = pd.read_csv(path)

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower()

    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")

    if "label" not in df.columns and "emotion" not in df.columns:
        raise ValueError("CSV must contain a 'label' or 'emotion' column.")

    # Build numeric label if only emotion string is present
    if "label" not in df.columns:
        label_map = {e: i for i, e in enumerate(EMOTION_LABELS)}
        df["label"] = df["emotion"].map(label_map)

    if "emotion" not in df.columns:
        label_map_inv = {i: e for i, e in enumerate(EMOTION_LABELS)}
        df["emotion"] = df["label"].map(label_map_inv)

    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    logger.info("Dataset loaded | %d samples", len(df))
    return df[["text", "label", "emotion"]]


# =============================================================================
# Train / Validation / Test split (3-way)
# =============================================================================

def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 3-way train-validation-test split.
    
    By default: 70% train, 10% validation, 20% test
    (70% + 10% + 20% = 100%)

    Parameters
    ----------
    df           : pd.DataFrame – full dataset
    test_size    : float        – fraction for test set (default 0.20)
    val_size     : float        – fraction for validation set (default 0.10)
                                  from original data
    random_state : int          – reproducibility seed

    Returns
    -------
    (train_df, val_df, test_df)
    """
    # First split: separate test set
    remaining_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    
    # Second split: separate validation from training
    # Adjust val_size relative to remaining data
    val_size_adjusted = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        remaining_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=remaining_df["label"],
    )
    
    logger.info(
        "3-way Split | train=%d (%.0f%%) | val=%d (%.0f%%) | test=%d (%.0f%%)",
        len(train_df),
        (len(train_df) / len(df)) * 100,
        len(val_df),
        (len(val_df) / len(df)) * 100,
        len(test_df),
        (len(test_df) / len(df)) * 100,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# =============================================================================
# HuggingFace Hub loader (dair-ai/emotion – the canonical rubric dataset)
# =============================================================================

def load_huggingface_emotion() -> pd.DataFrame:
    """
    Load the dair-ai/emotion dataset directly from HuggingFace Hub.
    Merges the official train, validation, and test splits into one
    DataFrame so the pipeline can re-split with stratified sampling.

    Requires
    --------
    pip install datasets

    Returns
    -------
    pd.DataFrame with columns ['text', 'label', 'emotion']
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace 'datasets' library not installed. "
            "Run: pip install datasets"
        )

    logger.info("Downloading dair-ai/emotion from HuggingFace Hub …")
    hf_ds = hf_load_dataset("dair-ai/emotion", trust_remote_code=True)

    # Combine all official splits – pipeline will re-split with stratification
    frames = []
    for split_name in ["train", "validation", "test"]:
        if split_name in hf_ds:
            split_df = hf_ds[split_name].to_pandas()
            frames.append(split_df)

    df = pd.concat(frames, ignore_index=True)

    # Map integer labels → emotion name strings
    # dair-ai/emotion label order: sadness=0, joy=1, love=2, anger=3, fear=4, surprise=5
    label_map = {i: e for i, e in enumerate(EMOTION_LABELS)}
    df["emotion"] = df["label"].map(label_map)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    logger.info(
        "HuggingFace dair-ai/emotion loaded | %d total samples | distribution:\n%s",
        len(df),
        df["emotion"].value_counts().to_string(),
    )
    return df[["text", "label", "emotion"]]


# =============================================================================
# Convenience loader – auto-selects real vs synthetic
# =============================================================================

def load_dataset(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load dataset with the following priority:

    1. Local CSV  – if `csv_path` is supplied and the file exists.
    2. HuggingFace Hub – dair-ai/emotion  (requires ``pip install datasets``).
    3. Synthetic fallback – generated in-memory for offline/CI runs.

    Parameters
    ----------
    csv_path : str | None – optional path to a local CSV file

    Returns
    -------
    pd.DataFrame with columns ['text', 'label', 'emotion']
    """
    # Priority 1: explicit CSV path
    if csv_path and Path(csv_path).exists():
        return load_emotions_csv(csv_path)

    # Priority 2: HuggingFace Hub (dair-ai/emotion)
    try:
        return load_huggingface_emotion()
    except Exception as exc:
        logger.warning(
            "HuggingFace dataset load failed (%s) – falling back to synthetic data.",
            exc,
        )

    # Priority 3: synthetic fallback
    logger.warning(
        "Using synthetic dataset (4 800 samples). "
        "For production quality, supply the real dair-ai/emotion dataset."
    )
    return generate_synthetic_dataset(samples_per_class=800)
