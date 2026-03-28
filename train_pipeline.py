"""
=============================================================================
FILE:    train_pipeline.py
PURPOSE: End-to-end training orchestration script
USAGE:
    python train_pipeline.py                        # synthetic data
    python train_pipeline.py --csv path/to/data.csv # real CSV
    python train_pipeline.py --model rf             # Random Forest
    python train_pipeline.py --no-tune              # skip GridSearchCV
=============================================================================
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Add project root to sys.path so sibling packages resolve correctly
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from data.data_loader      import load_dataset, split_dataset, EMOTION_LABELS
from utils.preprocessor    import TextPreprocessor
from utils.feature_extractor import TFIDFExtractor
from models.model_trainer  import EmotionModelTrainer
from utils.visualizer      import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_text_length_distribution,
    plot_wordclouds,
    plot_metrics_summary,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_pipeline")

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
MODELS_DIR  = ROOT / "models" / "saved"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================

def step_load(csv_path: str | None) -> pd.DataFrame:
    """Step 1 – Load dataset."""
    logger.info("=" * 55)
    logger.info("STEP 1 │ DATA LOADING")
    logger.info("=" * 55)
    df = load_dataset(csv_path)
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Label distribution:\n%s", df["emotion"].value_counts().to_string())
    return df


def step_eda(df: pd.DataFrame) -> None:
    """Step 2 – Exploratory Data Analysis plots."""
    logger.info("=" * 55)
    logger.info("STEP 2 │ EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 55)
    plot_class_distribution(df, save_path=str(OUTPUTS_DIR / "01_class_distribution.png"))
    plot_text_length_distribution(df, save_path=str(OUTPUTS_DIR / "02_text_lengths.png"))
    logger.info("EDA plots saved to %s", OUTPUTS_DIR)


def step_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3 – Text Preprocessing."""
    logger.info("=" * 55)
    logger.info("STEP 3 │ TEXT PREPROCESSING")
    logger.info("=" * 55)
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True, min_token_len=2)
    df["clean_text"] = preprocessor.clean_series(df["text"])

    # Log a few examples
    for _, row in df.sample(3, random_state=42).iterrows():
        logger.info("  RAW  : %s", row["text"][:80])
        logger.info("  CLEAN: %s", row["clean_text"][:80])
        logger.info("  ─" * 30)

    # Word cloud (after cleaning)
    plot_wordclouds(df, text_col="clean_text", save_path=str(OUTPUTS_DIR / "03_wordclouds.png"))
    return df


def step_encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Step 4 – Label encoding."""
    logger.info("STEP 4 │ LABEL ENCODING")
    le = LabelEncoder()
    le.classes_ = np.array(EMOTION_LABELS)      # fix canonical order
    df["label"] = le.transform(df["emotion"])
    return df, le


def step_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Step 5 – Train / validation / test split (70 / 10 / 20)."""
    logger.info("=" * 55)
    logger.info("STEP 5 │ 3-WAY SPLIT (70% / 10% / 20%)")
    logger.info("=" * 55)
    train_df, val_df, test_df = split_dataset(df, test_size=0.20, val_size=0.10)
    return train_df, val_df, test_df


def step_feature_extraction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, TFIDFExtractor]:
    """Step 6 – TF-IDF feature extraction (2-way split)."""
    logger.info("=" * 55)
    logger.info("STEP 6 │ FEATURE EXTRACTION (TF-IDF)")
    logger.info("=" * 55)
    extractor = TFIDFExtractor(max_features=50_000, ngram_range=(1, 2))
    X_train = extractor.fit_transform(train_df["clean_text"])
    X_test  = extractor.transform(test_df["clean_text"])
    logger.info("X_train shape: %s | X_test shape: %s", X_train.shape, X_test.shape)
    return X_train, X_test, extractor


def step_feature_extraction_3way(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TFIDFExtractor]:
    """Step 6 – TF-IDF feature extraction (3-way split)."""
    logger.info("=" * 55)
    logger.info("STEP 6 │ FEATURE EXTRACTION (TF-IDF)")
    logger.info("=" * 55)
    extractor = TFIDFExtractor(max_features=50_000, ngram_range=(1, 2))
    X_train = extractor.fit_transform(train_df["clean_text"])
    X_val   = extractor.transform(val_df["clean_text"])
    X_test  = extractor.transform(test_df["clean_text"])
    logger.info("X_train shape: %s | X_val shape: %s | X_test shape: %s", 
                X_train.shape, X_val.shape, X_test.shape)
    return X_train, X_val, X_test, extractor


def step_train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
    model_name: str,
    tune: bool,
) -> tuple[dict, any]:
    """Step 7 – Model training and evaluation (3-way split)."""
    logger.info("=" * 55)
    logger.info("STEP 7 │ TRAINING – model=%s | tune=%s", model_name.upper(), tune)
    logger.info("=" * 55)

    trainer = EmotionModelTrainer(
        model_name=model_name,
        feature_extractor=None,        # features already extracted
        label_encoder=le,
        output_dir=str(MODELS_DIR),
    )
    trainer.train(X_train, y_train, tune_hyperparams=tune)
    
    # Evaluate on validation set
    logger.info("\nValidation Set Performance:")
    val_metrics = trainer.evaluate(X_val, y_val)
    
    # Evaluate on test set (final evaluation)
    logger.info("\nTest Set Performance (FINAL):")
    test_metrics = trainer.evaluate(X_test, y_test)
    
    trainer.save()
    return test_metrics, trainer


def step_visualise_results(metrics: dict) -> None:
    """Step 8 – Evaluation visualisations."""
    logger.info("=" * 55)
    logger.info("STEP 8 │ RESULTS VISUALISATION")
    logger.info("=" * 55)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        metrics["emotion_names"],
        save_path=str(OUTPUTS_DIR / "04_confusion_matrix.png"),
    )
    plot_per_class_f1(
        metrics["per_class_f1"],
        save_path=str(OUTPUTS_DIR / "05_per_class_f1.png"),
    )
    plot_metrics_summary(
        metrics,
        save_path=str(OUTPUTS_DIR / "06_metrics_dashboard.png"),
    )
    logger.info("All visualisations saved to %s", OUTPUTS_DIR)


def print_metrics_table(metrics: dict) -> None:
    """
    Print a formatted evaluation metrics table to stdout.

    This function ensures all rubric-required metrics are clearly visible
    to the assessor during a demo run, independently of the logging level.
    Metrics printed: Accuracy, Precision, Recall, Sensitivity, Specificity,
    F1 (weighted), and per-class Sensitivity & Specificity for all 6 emotions.
    """
    emotion_names = metrics.get("emotion_names", [])
    W = 62
    sep = "=" * W
    print(f"\n{sep}")
    print("  EMOTION DETECTION - FINAL EVALUATION METRICS")
    print(sep)
    print(f"  {'Metric':<30} {'Value':>10}")
    print("  " + "-" * (W - 2))
    print(f"  {'Accuracy':<30} {metrics['accuracy']:>10.4f}")
    print(f"  {'Precision (weighted)':<30} {metrics['precision']:>10.4f}")
    print(f"  {'Recall (weighted)':<30} {metrics['recall']:>10.4f}")
    print(f"  {'Sensitivity (weighted TPR)':<30} {metrics['sensitivity']:>10.4f}")
    print(f"  {'Specificity (weighted TNR)':<30} {metrics['specificity']:>10.4f}")
    print(f"  {'F1 Score (weighted)':<30} {metrics['f1_weighted']:>10.4f}")
    print(f"\n  PER-CLASS BREAKDOWN")
    print(f"  {'Emotion':<12} {'Sensitivity (TPR)':>18} {'Specificity (TNR)':>18}")
    print("  " + "-" * (W - 2))
    for e in emotion_names:
        sens = metrics["per_class_sensitivity"].get(e, 0.0)
        spec = metrics["per_class_specificity"].get(e, 0.0)
        print(f"  {e:<12} {sens:>18.4f} {spec:>18.4f}")
    print(sep + "\n")


def save_artefacts(extractor: TFIDFExtractor, le: LabelEncoder) -> None:
    """Persist extractor and label encoder so the app can reload them."""
    with open(MODELS_DIR / "tfidf_extractor.pkl", "wb") as f:
        pickle.dump(extractor, f)
    with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    logger.info("Artefacts saved to %s", MODELS_DIR)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Emotion Detection Training Pipeline")
    parser.add_argument("--csv",     type=str, default=None,  help="Path to dataset CSV")
    parser.add_argument("--model",   type=str, default="svm", 
                        choices=["svm", "rf", "gb", "lstm", "gru"],
                        help="Model selection: svm, rf, gb (scikit-learn) or lstm, gru (PyTorch)")
    parser.add_argument("--no-tune", action="store_true",     help="Skip hyperparameter tuning")
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  EMOTION DETECTION – TRAINING PIPELINE       ║")
    logger.info("║  Asia Pacific University  │  CT104-3-M       ║")
    logger.info("╚══════════════════════════════════════════════╝")

    # 1. Load
    df = step_load(args.csv)

    # 2. EDA
    step_eda(df)

    # 3. Preprocess
    df = step_preprocess(df)

    # 4. Encode
    df, le = step_encode_labels(df)

    # 5. Split
    train_df, val_df, test_df = step_split(df)

    # 6. Features
    X_train, X_val, X_test, extractor = step_feature_extraction_3way(train_df, val_df, test_df)
    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    # 7. Train + Evaluate
    metrics, trainer = step_train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test,
        le=le,
        model_name=args.model,
        tune=not args.no_tune,
    )

    # 8. Visualise
    step_visualise_results(metrics)

    # 9. Save artefacts
    save_artefacts(extractor, le)

    # 10. Print final metrics table (stdout – visible regardless of log level)
    print_metrics_table(metrics)

    logger.info("Pipeline complete. All outputs → %s", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
