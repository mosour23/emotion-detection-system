"""
=============================================================================
MODULE: model_trainer.py
PURPOSE: Model Selection, Training, Hyperparameter Tuning, and Evaluation
DESCRIPTION:
    Implements and compares three classifiers:
      1. Support Vector Machine (SVM)  – strong text baseline
      2. Random Forest                 – ensemble, interpretable
      3. Gradient Boosting (XGBoost)   – high-performance ensemble

    Each model is wrapped in a scikit-learn Pipeline so preprocessing and
    classification are a single callable object.

    Algorithm justification (SVM as primary):
    -----------------------------------------
    • SVM with an RBF or linear kernel consistently outperforms simpler
      models on high-dimensional TF-IDF text features.
    • It maximises the margin between emotion classes, reducing over-fitting
      even with limited data.
    • Its dual formulation remains efficient for sparse feature matrices.
    • Random Forest provides a complementary ensemble baseline, while
      XGBoost adds gradient boosting for potential accuracy gains.
=============================================================================
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    multilabel_confusion_matrix,
)

# Optional deep learning models
try:
    from .lstm_classifier import PyTorchEmotionClassifier
    DL_AVAILABLE = True
except (ImportError, RuntimeError):
    DL_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Model definitions
# =============================================================================

def _build_svm() -> CalibratedClassifierCV:
    """
    Linear SVM wrapped in Platt scaling for probability estimates.
    LinearSVC is much faster than SVC(kernel='linear') on large corpora.
    """
    return CalibratedClassifierCV(
        LinearSVC(max_iter=5_000, random_state=42, C=1.0, class_weight="balanced"),
        cv=3,
    )


def _build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
def _build_gradient_boosting() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    )


def _build_lstm():
    """Build LSTM emotion classifier (if PyTorch available)."""
    if not DL_AVAILABLE:
        raise RuntimeError(
            "PyTorch not available. Install with: pip install torch"
        )
    return PyTorchEmotionClassifier(
        model_type="lstm",
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        epochs=20,
        batch_size=32,
    )


def _build_gru():
    """Build GRU emotion classifier (if PyTorch available)."""
    if not DL_AVAILABLE:
        raise RuntimeError(
            "PyTorch not available. Install with: pip install torch"
        )
    return PyTorchEmotionClassifier(
        model_type="gru",
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        epochs=20,
        batch_size=32,
    )


MODEL_REGISTRY: dict[str, Any] = {
    "svm":  _build_svm,
    "rf":   _build_random_forest,
    "gb":   _build_gradient_boosting,
}

# Add deep learning models if PyTorch is available
if DL_AVAILABLE:
    MODEL_REGISTRY["lstm"] = _build_lstm
    MODEL_REGISTRY["gru"] = _build_gru

# Hyperparameter grids for GridSearchCV
PARAM_GRIDS: dict[str, dict] = {
    "svm": {
        "classifier__estimator__C": [0.1, 1.0, 5.0, 10.0],
    },
    "rf": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth":    [None, 20, 40],
        "classifier__min_samples_split": [2, 5],
    },
    "gb": {
        "classifier__n_estimators":  [100, 150],
        "classifier__learning_rate": [0.05, 0.1, 0.2],
        "classifier__max_depth":     [3, 5],
    },
    "lstm": {},  # No GridSearch for neural networks
    "gru":  {},  # No GridSearch for neural networks
}


# =============================================================================
# Trainer class
# =============================================================================

class EmotionModelTrainer:
    """
    End-to-end trainer for emotion classification models.

    Parameters
    ----------
    model_name   : str  – key in MODEL_REGISTRY ('svm', 'rf', 'gb')
    feature_extractor : fitted TFIDFExtractor or Word2VecExtractor
    label_encoder     : fitted sklearn LabelEncoder
    output_dir   : str  – directory to save trained artefacts
    """

    def __init__(
        self,
        model_name: str,
        feature_extractor,
        label_encoder: LabelEncoder,
        output_dir: str = "models",
    ):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_REGISTRY)}")

        self.model_name = model_name
        self.extractor = feature_extractor
        self.le = label_encoder
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.classifier = MODEL_REGISTRY[model_name]()
        self.best_params: dict = {}
        self._trained = False

        logger.info("EmotionModelTrainer | model=%s", model_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract(self, texts) -> np.ndarray:
        """Transform cleaned texts to feature matrix."""
        return self.extractor.transform(texts)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune_hyperparams: bool = True,
        cv_folds: int = 5,
        scoring: str = "f1_macro",
    ) -> "EmotionModelTrainer":
        """
        Train the classifier, optionally with GridSearchCV tuning.
        
        Note: Deep learning models (LSTM/GRU) skip hyperparameter tuning
        and are trained directly with preset hyperparameters.

        Parameters
        ----------
        X_train          : np.ndarray – feature matrix (training)
        y_train          : np.ndarray – integer labels (training)
        tune_hyperparams : bool       – run GridSearchCV if True (sklearn models only)
        cv_folds         : int        – number of CV folds
        scoring          : str        – optimisation metric

        Returns
        -------
        self
        """
        logger.info("Starting training | model=%s | tune=%s", self.model_name, tune_hyperparams)
        t0 = time.time()

        # Deep learning models: manual hyperparameter sweep
        # (GridSearchCV is not compatible with custom PyTorch wrappers)
        if self.model_name in ["lstm", "gru"]:
            from sklearn.model_selection import train_test_split as _tts
            from sklearn.metrics import accuracy_score as _acc

            # ── Hyperparameter candidates ──────────────────────────────────
            HIDDEN_SIZES  = [64, 128]
            DROPOUT_RATES = [0.2, 0.3]
            # ──────────────────────────────────────────────────────────────

            # Hold-out 10% of training data for sweep validation
            X_tr, X_hv, y_tr, y_hv = _tts(
                X_train, y_train,
                test_size=0.10,
                random_state=42,
                stratify=y_train,
            )

            best_val_acc  = -1.0
            best_config   = {"hidden_size": 128, "dropout": 0.3}  # safe default

            logger.info(
                "Starting %s hyperparameter sweep | hidden_sizes=%s | dropouts=%s",
                self.model_name.upper(), HIDDEN_SIZES, DROPOUT_RATES,
            )

            for hs in HIDDEN_SIZES:
                for dr in DROPOUT_RATES:
                    sweep_clf = PyTorchEmotionClassifier(
                        model_type=self.model_name,
                        input_size=int(X_tr.shape[1]),
                        hidden_size=hs,
                        num_classes=self.classifier.num_classes,
                        dropout=dr,
                        epochs=10,    # reduced for sweep speed
                        batch_size=32,
                    )
                    sweep_clf.fit(X_tr, y_tr)
                    hv_acc = _acc(y_hv, sweep_clf.predict(X_hv))
                    logger.info(
                        "  [sweep] hidden=%3d | dropout=%.1f | val_acc=%.4f",
                        hs, dr, hv_acc,
                    )
                    if hv_acc > best_val_acc:
                        best_val_acc = hv_acc
                        best_config  = {"hidden_size": hs, "dropout": dr}

            self.best_params = best_config
            logger.info(
                "Best %s config: %s | val_acc=%.4f – re-fitting on full training set …",
                self.model_name.upper(), best_config, best_val_acc,
            )

            # Re-train with best config on the FULL training set and full epochs
            self.classifier = PyTorchEmotionClassifier(
                model_type=self.model_name,
                input_size=int(X_train.shape[1]),
                hidden_size=best_config["hidden_size"],
                num_classes=self.classifier.num_classes,
                dropout=best_config["dropout"],
                epochs=20,
                batch_size=32,
            )
            self.classifier.fit(X_train, y_train)
        elif tune_hyperparams:
            grid = PARAM_GRIDS.get(self.model_name, {})
            if grid:
                # Wrap classifier in a trivial pipeline to use param_grid syntax
                pipe = Pipeline([("classifier", self.classifier)])
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                search = GridSearchCV(
                    pipe,
                    param_grid=grid,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=1,
                    refit=True,
                )
                search.fit(X_train, y_train)
                self.classifier = search.best_estimator_.named_steps["classifier"]
                self.best_params = search.best_params_
                logger.info("Best params found: %s", self.best_params)
                logger.info("Best CV %s = %.4f", scoring, search.best_score_)
            else:
                logger.warning("No param grid for '%s'; fitting with defaults.", self.model_name)
                self.classifier.fit(X_train, y_train)
        else:
            self.classifier.fit(X_train, y_train)

        elapsed = time.time() - t0
        self._trained = True
        logger.info("Training complete | %.1f seconds", elapsed)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return integer class predictions."""
        if not self._trained:
            raise RuntimeError("Model not yet trained – call train() first.")
        return self.classifier.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probability estimates.
        Falls back to one-hot if classifier lacks predict_proba.
        """
        if not self._trained:
            raise RuntimeError("Model not trained.")
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X)
        # Fallback: one-hot from hard labels
        preds = self.predict(X)
        n_classes = len(self.le.classes_)
        result = np.zeros((len(preds), n_classes))
        result[np.arange(len(preds)), preds] = 1.0
        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True,
    ) -> dict:
        """
        Compute comprehensive evaluation metrics.

        Returns
        -------
        dict with keys: accuracy, precision, recall, f1, confusion_matrix,
                        classification_report, per_class_metrics
        """
        y_pred = self.predict(X_test)
        emotion_names = self.le.classes_.tolist()

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred,        average="weighted", zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=emotion_names,
            zero_division=0,
        )

        # Per-class F1
        per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
        per_class = {
            name: round(float(score), 4)
            for name, score in zip(emotion_names, per_class_f1)
        }

        # Calculate Sensitivity (Recall) and Specificity per class
        # Using multilabel_confusion_matrix for per-class metrics
        mcm = multilabel_confusion_matrix(y_test, y_pred)
        
        per_class_sensitivity = {}
        per_class_specificity = {}
        
        for i, emotion in enumerate(emotion_names):
            tn, fp, fn, tp = mcm[i].ravel()
            
            # Sensitivity (True Positive Rate / Recall)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            per_class_sensitivity[emotion] = round(float(sensitivity), 4)
            
            # Specificity (True Negative Rate)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            per_class_specificity[emotion] = round(float(specificity), 4)

        # Calculate class-frequency weighted average sensitivity and specificity
        # (consistent with sklearn's average="weighted" convention)
        class_counts = np.bincount(y_test, minlength=len(emotion_names))
        class_weights = class_counts / class_counts.sum()
        weighted_sensitivity = sum(
            per_class_sensitivity[e] * w for e, w in zip(emotion_names, class_weights)
        )
        weighted_specificity = sum(
            per_class_specificity[e] * w for e, w in zip(emotion_names, class_weights)
        )

        metrics = {
            "accuracy":                   round(acc,  4),
            "precision":                  round(prec, 4),
            "recall":                     round(rec,  4),
            "sensitivity":                round(weighted_sensitivity, 4),
            "specificity":                round(weighted_specificity, 4),
            "f1_weighted":                round(f1,   4),
            "confusion_matrix":           cm.tolist(),
            "classification_report":      report,
            "per_class_f1":               per_class,
            "per_class_sensitivity":      per_class_sensitivity,
            "per_class_specificity":      per_class_specificity,
            "emotion_names":              emotion_names,
        }

        if verbose:
            logger.info("=" * 60)
            logger.info("EVALUATION RESULTS – %s", self.model_name.upper())
            logger.info("=" * 60)
            logger.info("Accuracy     : %.4f", acc)
            logger.info("Precision    : %.4f", prec)
            logger.info("Recall       : %.4f", rec)
            logger.info("Sensitivity  : %.4f", weighted_sensitivity)
            logger.info("Specificity  : %.4f", weighted_specificity)
            logger.info("F1 (wtd)     : %.4f", f1)
            logger.info("\n%s", report)
            logger.info("\nPER-CLASS SENSITIVITY:")
            for emotion, sens in per_class_sensitivity.items():
                logger.info("  %s: %.4f", emotion, sens)
            logger.info("\nPER-CLASS SPECIFICITY:")
            for emotion, spec in per_class_specificity.items():
                logger.info("  %s: %.4f", emotion, spec)

        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> str:
        """Pickle the trained classifier to disk."""
        if not self._trained:
            raise RuntimeError("Cannot save an untrained model.")
        path = self.output_dir / f"{self.model_name}_classifier.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.classifier, f)
        logger.info("Classifier saved → %s", path)
        return str(path)

    def load(self, path: str) -> "EmotionModelTrainer":
        """Load a saved classifier from disk."""
        with open(path, "rb") as f:
            self.classifier = pickle.load(f)
        self._trained = True
        logger.info("Classifier loaded ← %s", path)
        return self
