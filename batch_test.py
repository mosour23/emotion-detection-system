"""
=============================================================================
SCRIPT: batch_test.py
PURPOSE: Batch emotion prediction on test sample files
USAGE: python batch_test.py --model svm
=============================================================================
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.preprocessor import TextPreprocessor
from utils.feature_extractor import TFIDFExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
MODELS_SAVED_DIR = ROOT / "models" / "saved"
TEST_SAMPLES_DIR = ROOT / "test_samples"

# Emotion classes
EMOTION_CLASSES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😊",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


def load_artifacts():
    """Load trained model, TF-IDF extractor, and label encoder."""
    try:
        with open(MODELS_SAVED_DIR / "svm_classifier.pkl", "rb") as f:
            classifier = pickle.load(f)
        
        with open(MODELS_SAVED_DIR / "tfidf_extractor.pkl", "rb") as f:
            extractor = pickle.load(f)
        
        with open(MODELS_SAVED_DIR / "label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        
        logger.info("✓ Artifacts loaded successfully")
        return classifier, extractor, le
    except FileNotFoundError as e:
        logger.error(f"✗ Artifact not found: {e}")
        logger.error("Run 'python train_pipeline.py' first to train the model")
        sys.exit(1)


def batch_predict(classifier, extractor, preprocessor, texts):
    """
    Make predictions on batch of texts.
    
    Parameters
    ----------
    classifier : fitted SVM classifier
    extractor : fitted TFIDFExtractor
    preprocessor : TextPreprocessor
    texts : list of strings
    
    Returns
    -------
    list of predicted emotion labels
    """
    # Preprocess
    cleaned = [preprocessor.clean(t) for t in texts]
    
    # Extract features
    X = extractor.transform(cleaned)
    
    # Predict
    predictions = classifier.predict(X)
    
    # Get probabilities
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(X)
    else:
        probabilities = None
    
    return predictions, probabilities


def read_test_file(filepath):
    """Read test file and return list of sentences."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
        return sentences
    except FileNotFoundError:
        logger.warning(f"Test file not found: {filepath}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Batch emotion prediction on test samples")
    parser.add_argument(
        "--emotion",
        type=str,
        choices=EMOTION_CLASSES + ["all"],
        default="all",
        help="Which emotion class test file to evaluate"
    )
    parser.add_argument(
        "--show-proba",
        action="store_true",
        help="Show class probabilities"
    )
    args = parser.parse_args()
    
    # Load artifacts
    logger.info("Loading trained model and artifacts...")
    classifier, extractor, le = load_artifacts()
    
    # Setup preprocessor
    preprocessor = TextPreprocessor()
    
    # Determine which test files to use
    if args.emotion == "all":
        test_files = {e: TEST_SAMPLES_DIR / f"{e}_samples.txt" for e in EMOTION_CLASSES}
    else:
        test_files = {args.emotion: TEST_SAMPLES_DIR / f"{args.emotion}_samples.txt"}
    
    # Process each test file
    for emotion, filepath in test_files.items():
        sentences = read_test_file(filepath)
        if not sentences:
            continue
        
        logger.info(f"\n{'='*70}")
        logger.info(f"BATCH TEST: {emotion.upper()} {EMOTION_EMOJIS.get(emotion, '')}")
        logger.info(f"{'='*70}")
        
        # Make predictions
        predictions, probabilities = batch_predict(classifier, extractor, preprocessor, sentences)
        pred_labels = le.inverse_transform(predictions)
        
        # Display results
        correct = sum(1 for p in pred_labels if p == emotion)
        accuracy = correct / len(sentences) * 100 if sentences else 0
        
        logger.info(f"Accuracy on {emotion}: {accuracy:.1f}% ({correct}/{len(sentences)})\n")
        
        # Show predictions
        for i, (sentence, pred_label, pred_idx) in enumerate(zip(sentences, pred_labels, predictions), 1):
            emoji = EMOTION_EMOJIS.get(pred_label, "")
            status = "✓" if pred_label == emotion else "✗"
            
            logger.info(f"{status} [{i}] {sentence}")
            logger.info(f"    → Predicted: {pred_label.upper()} {emoji}")
            
            if args.show_proba and probabilities is not None:
                proba_dict = {le.inverse_transform([j])[0]: f"{probabilities[i-1][j]:.4f}" 
                             for j in range(len(EMOTION_CLASSES))}
                logger.info(f"    Probabilities: {proba_dict}")
            logger.info("")
    
    logger.info("=" * 70)
    logger.info("Batch testing complete!")


if __name__ == "__main__":
    main()
