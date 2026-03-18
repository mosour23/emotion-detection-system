# 🧠 Emotion Detection in Textual Data
### CT104-3-M Pattern Recognition | Asia Pacific University (APU)

---

## 📁 Project Structure

```
emotion_detection/
│
├── data/
│   ├── __init__.py
│   └── data_loader.py          # Dataset loading, splitting, synthetic generation
│
├── utils/
│   ├── __init__.py
│   ├── preprocessor.py         # NLP preprocessing pipeline
│   ├── feature_extractor.py    # TF-IDF & Word2Vec extractors
│   └── visualizer.py           # All chart/plot functions
│
├── models/
│   ├── __init__.py
│   ├── model_trainer.py        # SVM / RF / GB training + evaluation
│   └── saved/                  # Trained artefacts (auto-created)
│       ├── tfidf_extractor.pkl
│       ├── label_encoder.pkl
│       └── svm_classifier.pkl
│
├── app/
│   ├── __init__.py
│   └── streamlit_app.py        # Streamlit web interface
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
├── requirements.txt
└── README.md
```

---

## ⚙️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
# With synthetic dataset (no CSV needed – great for testing)
python train_pipeline.py

# With your own CSV (must have 'text' and 'label'/'emotion' columns)
python train_pipeline.py --csv path/to/emotions.csv

# Choose a different classifier
python train_pipeline.py --model rf       # Random Forest
python train_pipeline.py --model gb       # Gradient Boosting

# Skip hyperparameter tuning for faster runs
python train_pipeline.py --no-tune
```

### 3. Launch the Web Interface
```bash
streamlit run app/streamlit_app.py
```
Open your browser at **http://localhost:8501**

---

## 🎯 System Architecture

### Data Flow
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
              ▼
┌─────────────────────────────────┐
│  FEATURE EXTRACTION (TF-IDF)    │
│  • Vocabulary: 50 000 tokens    │
│  • N-grams: (1, 2)              │
│  • Sublinear TF weighting       │
│  • L2 row normalisation         │
└─────────────┬───────────────────┘
              │ Feature Matrix  (n × 50 000)
              ▼
┌─────────────────────────────────┐
│  CLASSIFIER (Linear SVM)        │
│  • CalibratedClassifierCV       │
│  • C tuned via GridSearchCV     │
│  • 5-fold stratified CV         │
│  • Platt scaling (probabilities)│
└─────────────┬───────────────────┘
              │ Prediction + Probabilities
              ▼
        Emotion Label
   sadness / joy / love /
   anger / fear / surprise
```

### Why Linear SVM?
| Property | Benefit for Text Classification |
|----------|---------------------------------|
| Max-margin classifier | Robust to outliers in high-dim TF-IDF space |
| L2 regularisation | Prevents over-fitting on sparse features |
| Linear kernel | O(n·d) complexity – scales to millions of samples |
| Platt calibration | Outputs calibrated class probabilities |
| Proven track record | Consistently top-performing baseline for NLP |

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions / total predictions |
| **Precision** | Weighted average: TP / (TP + FP) per class |
| **Recall** | Weighted average: TP / (TP + FN) per class |
| **F1 Score** | Weighted harmonic mean of Precision & Recall |
| **Confusion Matrix** | 6×6 table of actual vs. predicted emotions |

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

**Emotion Classes:** sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5)

---

## 🔄 Extending the System

### Add a New Classifier
```python
# In models/model_trainer.py
from sklearn.naive_bayes import ComplementNB

MODEL_REGISTRY["cnb"] = lambda: ComplementNB()
PARAM_GRIDS["cnb"] = {"classifier__alpha": [0.1, 0.5, 1.0]}
```

### Switch to Word2Vec Features
```python
from utils.feature_extractor import Word2VecExtractor

extractor = Word2VecExtractor(vector_size=200, sg=1)
X_train = extractor.fit_transform(train_df["clean_text"])
X_test  = extractor.transform(test_df["clean_text"])
```

---

## 📋 CLO Alignment

| CLO | Description | Evidence in Code |
|-----|-------------|-----------------|
| CLO1 | Pattern recognition algorithms on real-world data | SVM on emotion text corpus |
| CLO2 | Supervised classification for pattern detection | LinearSVC + GridSearchCV |
| CLO3 | Performance analysis of ML algorithms | Accuracy, P, R, F1, CM |

---

*Asia Pacific University (APU) · CT104-3-M Pattern Recognition*
