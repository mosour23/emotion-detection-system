# 🧠 Emotion Detection in Textual Data
### CT104-3-M Pattern Recognition | Asia Pacific University (APU)

---

## 📁 Project Structure

```
PR-System/
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset loading, 3-way split, synthetic generation
│   ├── emotion_data.csv        # Sample emotion dataset
│   └── get_data.py             # Data acquisition utilities
│
├── utils/
│   ├── __init__.py
│   ├── preprocessor.py         # NLP preprocessing pipeline
│   ├── feature_extractor.py    # TF-IDF & Word2Vec extractors
│   ├── visualizer.py           # All chart/plot functions
│   └── cv_visualizer.py        # Cross-validation visualization
│
├── models/
│   ├── __init__.py
│   ├── model_trainer.py        # ML training: SVM / RF / GB with metrics
│   ├── lstm_classifier.py      # Deep Learning: LSTM & GRU models
│   └── saved/                  # Trained artefacts (auto-created)
│       ├── tfidf_extractor.pkl
│       ├── label_encoder.pkl
│       └── svm_classifier.pkl
│
├── app/
│   ├── __init__.py
│   └── streamlit_app.py        # Streamlit web interface for inference
│
├── test_samples/               # Pre-made test samples for batch testing
│   ├── anger_samples.txt
│   ├── fear_samples.txt
│   ├── joy_samples.txt
│   ├── love_samples.txt
│   ├── sadness_samples.txt
│   ├── surprise_samples.txt
│   └── README.md
│
├── outputs/                    # Auto-created visualisation PNGs
│   ├── 01_class_distribution.png
│   ├── 02_text_lengths.png
│   ├── 03_wordclouds.png
│   ├── 04_confusion_matrix.png
│   ├── 05_per_class_f1.png
│   └── 06_metrics_dashboard.png
│
├── train_pipeline.py           # End-to-end CLI training script
├── batch_test.py               # Batch emotion prediction from test samples
├── requirements.txt
├── IMPLEMENTATION_SUMMARY.md   # Details on enhancements & new features
├── GITHUB_PUBLISHING_GUIDE.md  # Guide for GitHub publishing
├── JOURNAL_ARTICLE_TEMPLATE.md # Template for academic papers
└── README.md
```

---

## ⚙️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model

#### Machine Learning Models:
```bash
# With synthetic dataset (no CSV needed – great for testing)
python train_pipeline.py

# With your own CSV (must have 'text' and 'label'/'emotion' columns)
python train_pipeline.py --csv path/to/emotions.csv

# Choose a different classifier
python train_pipeline.py --model svm      # Support Vector Machine (default)
python train_pipeline.py --model rf       # Random Forest
python train_pipeline.py --model gb       # Gradient Boosting

# Skip hyperparameter tuning for faster runs
python train_pipeline.py --no-tune
```

#### Deep Learning Models (LSTM/GRU):
```bash
# Train LSTM classifier
python train_pipeline.py --model lstm

# Train GRU classifier
python train_pipeline.py --model gru
```

#### Batch Testing:
```bash
# Run batch predictions on test samples
python batch_test.py --model svm
python batch_test.py --model lstm
```
→ Processes all sample files in `test_samples/` directory

### 3. Launch the Web Interface
```bash
streamlit run app/streamlit_app.py
```
Open your browser at **http://localhost:8501**

---

## 🎯 System Architecture

### Data Pipeline (3-Way Split)
```
Raw Text
   │
   ▼
┌─────────────────────────────────────────────────┐
│         DATASET SPLITTING (3-way)               │
│  Train: 70% | Validation: 10% | Test: 20%      │
│         (Stratified split by emotion)           │
└──────────────────┬──────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    TRAINING         VALIDATION              TEST
    (70%)            (10%)                  (20%)
                    (used for                (held-out
                  hyperparameter tuning)     evaluation)
```

### Processing Pipeline (Both ML & DL)
```
Raw Text
   │
   ▼
┌─────────────────────────────────┐
│  PREPROCESSING (preprocessor.py)│
│  • Lowercasing                  │
│  • Emoticon → text              │
│  • URL / HTML removal           │
│  • Special char removal         │
│  • Tokenisation (NLTK)          │
│  • Stop-word removal            │
│  • Lemmatisation (WordNet)      │
└─────────────┬───────────────────┘
              │ Clean Text
         ┌────┴────┐
         ▼         ▼
    ┌─────────┐ ┌──────────────┐
    │   ML    │ │     DL       │
    │ Track   │ │   Track      │
    └─────────┘ └──────────────┘
         │          │
         ▼          ▼
   ┌─────────────────────────┐    ┌───────────────────────┐
   │ FEATURE EXTRACTION      │    │ EMBEDDING LAYER       │
   │ (TF-IDF / Word2Vec)     │    │ (Learned representations)
   │ • Vocabulary: 50,000    │    │ • Sequence length: 100|
   │ • N-grams: (1, 2)       │    │ • Embedding dim: 128  │
   │ • Sublinear TF          │    │ • Dropout: 0.5        │
   │ • L2 normalization      │    └──────────┬────────────┘
   └──────────┬──────────────┘                │
              │                              ▼
              │                    ┌────────────────────┐
              │                    │ LSTM/GRU Layer(s)  │
              │                    │ • Hidden units: 128│
              │                    │ • Bidirectional    │
              │                    │ • Dropout: 0.5     │
              │                    └────────┬───────────┘
              │                             │
              ▼                             ▼
         ┌──────────────────────────────────────┐
         │  CLASSIFIER HEAD (Dense Layers)      │
         │  • Dense(256) + ReLU + Dropout(0.3)  │
         │  • Dense(6) + Softmax                │
         └────────────────┬─────────────────────┘
                          │
                          ▼
             Emotion: sadness / joy / love /
                anger / fear / surprise
```

### ML Track (SVM/RF/GB)
| Component | Details |
|-----------|---------|
| **Preprocessor** | NLTK + WordNet lemmatization |
| **Extractor** | TF-IDF (50k vocab) + Optional Word2Vec |
| **Classifier** | Linear SVM (default), Random Forest, Gradient Boosting |
| **CV Strategy** | 5-fold stratified GridSearchCV |
| **Calibration** | Platt scaling for probability estimates |

### DL Track (LSTM/GRU)
| Component | Details |
|-----------|---------|
| **Preprocessor** | Same as ML track (NLTK vectorization) |
| **Embedding** | 128-dim learned word embeddings |
| **Encoder** | Bidirectional LSTM or GRU (128 units) |
| **Classifier** | Dense layers (256 → 6 outputs) with Dropout |
| **Training** | Adam optimizer, categorical cross-entropy |
| **Regularization** | Batch normalization + Dropout (0.3-0.5) |

---

## 📊 Evaluation Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Accuracy** | Overall correct predictions / total predictions | TP+TN / TP+TN+FP+FN |
| **Precision** | Weighted average: TP / (TP + FP) per class | Σ(wi·Pi) |
| **Recall** | Weighted average: TP / (TP + FN) per class | Σ(wi·Ri) |
| **F1 Score** | Weighted harmonic mean of Precision & Recall | 2·P·R / P+R |
| **Sensitivity** | True Positive Rate (recall per class) | TP / (TP + FN) |
| **Specificity** | True Negative Rate | TN / (TN + FP) |
| **Confusion Matrix** | 6×6 table of actual vs. predicted emotions | Per-class breakdown |

**Per-Class Reporting:** Each metric is computed per emotion and then weighted-averaged by support.

### Example Output
```
OVERALL METRICS:
  Accuracy     : 0.9583
  Precision    : 0.9583
  Recall       : 0.9583
  F1 Score     : 0.9583
  Sensitivity  : 0.9583
  Specificity  : 0.9965

PER-CLASS SENSITIVITY (True Positive Rate):
  sadness : 0.9500
  joy     : 1.0000
  love    : 0.9167
  anger   : 0.9667
  fear    : 0.9333
  surprise: 0.9500

PER-CLASS SPECIFICITY (True Negative Rate):
  sadness : 0.9938
  joy     : 1.0000
  love    : 0.9969
  ...
```

---

## 📚 Additional Documentation

This project includes several supplementary documents:

| Document | Purpose |
|----------|---------|
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Details on critical enhancements: 3-way split, sensitivity/specificity metrics, LSTM/GRU models |
| [GITHUB_PUBLISHING_GUIDE.md](GITHUB_PUBLISHING_GUIDE.md) | Step-by-step guide for publishing to GitHub |
| [JOURNAL_ARTICLE_TEMPLATE.md](JOURNAL_ARTICLE_TEMPLATE.md) | Template for academic paper submissions |

---

## 🔧 Advanced Features

### Multi-Model Comparison
The system supports both traditional ML and Deep Learning approaches:

**Machine Learning (Fast, Interpretable):**
- Linear SVM (default) – O(n·d) complexity
- Random Forest – Ensemble, feature importance
- Gradient Boosting – Sequential error correction

**Deep Learning (Higher Accuracy on Large Datasets):**
- LSTM – Bidirectional, captures long-range dependencies
- GRU – Similar to LSTM, faster training

### Cross-Validation Visualization
```python
# Generates cross-validation fold analysis
from utils.cv_visualizer import CVVisualizer
cv_viz = CVVisualizer(model, X, y)
cv_viz.plot_folds()
```

---

## 🌐 Dataset

The system is compatible with the **dair-ai/emotion** dataset (HuggingFace):

```bash
# Download via HuggingFace Datasets
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('dair-ai/emotion')
ds['train'].to_csv('data/emotions_train.csv', index=False)
ds['test'].to_csv('data/emotions_test.csv', index=False)
"
python train_pipeline.py --csv data/emotions_train.csv
```

**Emotion Classes:** sadness, joy, love, anger, fear, surprise (6 classes)

### Test Samples
Pre-made test samples are provided in the `test_samples/` directory:
- `sadness_samples.txt` – 10 sadness examples
- `joy_samples.txt` – 10 joy examples
- `love_samples.txt` – 10 love examples
- `anger_samples.txt` – 10 anger examples
- `fear_samples.txt` – 10 fear examples
- `surprise_samples.txt` – 10 surprise examples

Use these for quick validation:
```bash
python batch_test.py --model svm
```

---

## 🔄 Extending the System

### Add a New ML Classifier
```python
# In models/model_trainer.py
from sklearn.naive_bayes import ComplementNB

MODEL_REGISTRY["cnb"] = lambda: ComplementNB()
PARAM_GRIDS["cnb"] = {"classifier__alpha": [0.1, 0.5, 1.0]}
```

### Add a New DL Architecture
```python
# In models/lstm_classifier.py
class TransformerEmotionClassifier:
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        # Implement transformer-based emotion detection
        pass
```

### Switch to Word2Vec Features
```python
from utils.feature_extractor import Word2VecExtractor

extractor = Word2VecExtractor(vector_size=200, sg=1)
X_train = extractor.fit_transform(train_df["clean_text"])
X_test  = extractor.transform(test_df["clean_text"])
```

### Modify Preprocessing Pipeline
```python
from utils.preprocessor import TextPreprocessor

preprocessor = TextPreprocessor(
    lowercase=True,
    remove_urls=True,
    remove_stopwords=False,  # Keep stopwords
    lemmatize=False,         # Skip lemmatization
)
```

---

## 📋 CLO & Assessment Alignment

| CLO | Description | Implementation |
|-----|-------------|-----------------|
| **CLO1** | Pattern recognition algorithms on real-world data | ML models (SVM/RF/GB) + DL models (LSTM/GRU) trained on emotion corpus |
| **CLO2** | Supervised classification for pattern detection | Multi-class text classification with 3-way train/val/test split |
| **CLO3** | Performance analysis & model evaluation | Comprehensive metrics: Accuracy, Precision, Recall, F1, Sensitivity, Specificity |
| **CLO4** | Comparative algorithm analysis | ML vs. DL comparison, multiple classifier options in ML track |
| **CLO5** | Data preprocessing & feature engineering | NLTK preprocessing, TF-IDF extraction, Word2Vec embeddings |
| **CLO6** | Visualization & interpretation | Confusion matrices, per-class metrics, class distribution, word clouds, cross-validation plots |

### Key Features Demonstrating Mastery
✅ **3-Way Data Split** – Proper train/validation/test separation prevents overfitting  
✅ **Sensitivity & Specificity Metrics** – Beyond basic accuracy; per-class TP/TN rates  
✅ **Both ML & DL Approaches** – SVM/RF/GB vs. LSTM/GRU comparison  
✅ **Batch Processing** – Practical `batch_test.py` script for production-like workflows  
✅ **Web Interface** – Streamlit app for interactive inference  
✅ **Comprehensive Documentation** – Implementation summary + publishing guide  

---

## 🚀 Quick Reference

| Task | Command |
|------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Train ML model (SVM) | `python train_pipeline.py` |
| Train DL model (LSTM) | `python train_pipeline.py --model lstm` |
| Batch test on samples | `python batch_test.py --model svm` |
| Launch web interface | `streamlit run app/streamlit_app.py` |
| Train with custom CSV | `python train_pipeline.py --csv data/my_emotions.csv` |
| Skip hyperparameter tuning | `python train_pipeline.py --no-tune` |

---

*Asia Pacific University (APU) · CT104-3-M Pattern Recognition · 2024*
