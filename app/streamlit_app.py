"""
=============================================================================
FILE:    app/streamlit_app.py
PURPOSE: Streamlit web interface for real-time emotion classification
USAGE:
    streamlit run app/streamlit_app.py
DESCRIPTION:
    Provides a polished, interactive UI where a user can:
      • Type or paste any text
      • See the predicted emotion with confidence probabilities
      • View a live probability bar chart
      • Optionally run the full training pipeline from the sidebar
=============================================================================
"""

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup – resolve project root so imports work regardless of cwd
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.preprocessor    import TextPreprocessor
from data.data_loader      import EMOTION_LABELS, EMOTION_COLOR_MAP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = ROOT / "models" / "saved"

EMOTION_EMOJI = {
    "sadness":  "😢",
    "joy":      "😊",
    "love":     "❤️",
    "anger":    "😡",
    "fear":     "😨",
    "surprise": "😲",
}

EMOTION_BG = {
    "sadness":  "#D6E4FF",
    "joy":      "#FFFDE7",
    "love":     "#FCE4EC",
    "anger":    "#FFEBEE",
    "fear":     "#EDE7F6",
    "surprise": "#E0F7F4",
}

EMOTION_TIPS = {
    "sadness":  "🌧  It's okay to feel sad. Reach out to someone you trust.",
    "joy":      "☀️  Wonderful! Keep spreading that positive energy.",
    "love":     "💞  Love is a beautiful feeling – cherish it.",
    "anger":    "🔥  Take a deep breath. Emotions are valid, and so is calm.",
    "fear":     "🛡  Fear is natural. Break challenges into small steps.",
    "surprise": "🎉  Life is full of surprises – embrace the unexpected!",
}

DEMO_TEXTS = [
    "I feel absolutely devastated after losing my job today.",
    "This is the best day ever! I passed all my exams with flying colours!",
    "I love spending time with my family more than anything in the world.",
    "I am absolutely furious about the way they treated my friend!",
    "The thunderstorm outside is making me feel terrified and anxious.",
    "I can't believe it – they threw me a surprise birthday party!",
]


# =============================================================================
# Model loading (cached so it runs only once per session)
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_artefacts():
    """
    Load TF-IDF extractor, label encoder, and classifier from disk.
    Returns None for each if the artefacts are not yet trained.
    """
    extractor = le = classifier = None
    try:
        with open(MODELS_DIR / "tfidf_extractor.pkl",  "rb") as f:
            extractor = pickle.load(f)
        with open(MODELS_DIR / "label_encoder.pkl",    "rb") as f:
            le = pickle.load(f)
        # Load whichever classifier is available (prefer svm)
        for name in ["svm", "rf", "gb"]:
            clf_path = MODELS_DIR / f"{name}_classifier.pkl"
            if clf_path.exists():
                with open(clf_path, "rb") as f:
                    classifier = pickle.load(f)
                break
    except FileNotFoundError:
        pass
    return extractor, le, classifier


@st.cache_resource(show_spinner=False)
def get_preprocessor():
    return TextPreprocessor(remove_stopwords=True, lemmatize=True, min_token_len=2)


# =============================================================================
# Prediction logic
# =============================================================================

def predict_emotion(text: str, extractor, le, classifier) -> tuple[str, dict]:
    """
    Predict the emotion of `text`.

    Returns
    -------
    (predicted_emotion, {emotion: probability})
    """
    preprocessor = get_preprocessor()
    clean = preprocessor.clean(text)
    if not clean.strip():
        return "unknown", {e: 0.0 for e in EMOTION_LABELS}

    X = extractor.transform([clean])
    proba = classifier.predict_proba(X)[0]
    label_idx = int(np.argmax(proba))
    emotion   = le.classes_[label_idx]
    prob_dict = {le.classes_[i]: float(p) for i, p in enumerate(proba)}
    return emotion, prob_dict


# =============================================================================
# UI helpers
# =============================================================================

def render_emotion_card(emotion: str, proba: dict) -> None:
    """Render the coloured result card."""
    conf = proba.get(emotion, 0.0)
    emoji = EMOTION_EMOJI.get(emotion, "🤔")
    bg    = EMOTION_BG.get(emotion, "#F5F5F5")
    color = EMOTION_COLOR_MAP.get(emotion, "#555555")
    tip   = EMOTION_TIPS.get(emotion, "")

    st.markdown(
        f"""
        <div style="
            background:{bg};
            border-left: 6px solid {color};
            border-radius: 10px;
            padding: 24px 28px;
            margin-top: 20px;
        ">
            <h2 style="color:{color}; margin:0 0 6px 0;">
                {emoji} &nbsp; {emotion.capitalize()}
            </h2>
            <p style="font-size:1.05rem; color:#444; margin:0 0 8px 0;">
                Confidence: <strong>{conf:.1%}</strong>
            </p>
            <p style="font-size:0.95rem; color:#666; margin:0;">{tip}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_bars(proba: dict, emotion: str) -> None:
    """Render individual probability bars per emotion."""
    st.markdown("#### Emotion Probabilities")
    sorted_proba = sorted(proba.items(), key=lambda x: -x[1])
    for emo, prob in sorted_proba:
        color = EMOTION_COLOR_MAP.get(emo, "#888")
        is_top = (emo == emotion)
        weight = "bold" if is_top else "normal"
        bar_html = f"""
        <div style="display:flex; align-items:center; margin:5px 0;">
            <div style="width:90px; font-size:0.85rem; font-weight:{weight};">
                {EMOTION_EMOJI.get(emo,'')} {emo}
            </div>
            <div style="flex:1; background:#E8E8E8; border-radius:6px; height:18px; margin:0 10px;">
                <div style="
                    width:{prob*100:.1f}%;
                    background:{color};
                    height:100%;
                    border-radius:6px;
                    transition: width 0.5s ease;
                "></div>
            </div>
            <div style="width:45px; font-size:0.8rem; text-align:right; font-weight:{weight};">
                {prob:.1%}
            </div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)


# =============================================================================
# PAGE LAYOUT
# =============================================================================

def main():
    st.set_page_config(
        page_title="Emotion Detector | APU CT104-3-M",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Global CSS ──
    st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
        .stTextArea textarea { font-size: 1rem; }
        h1 { font-size: 2rem !important; }
        .stButton > button {
            background: #3B82F6; color: white;
            border-radius: 8px; border: none;
            padding: 0.5rem 1.5rem; font-size: 1rem;
        }
        .stButton > button:hover { background: #2563EB; }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        st.markdown("## 🧠")
    with col_title:
        st.markdown(
            "# Emotion Detection in Textual Data\n"
            "**CT104-3-M Pattern Recognition  |  Asia Pacific University (APU)**"
        )
    st.divider()

    # ── Load artefacts ──
    with st.spinner("Loading model artefacts …"):
        extractor, le, classifier = load_artefacts()

    model_ready = all(x is not None for x in [extractor, le, classifier])

    # ─────────────────────────────
    # SIDEBAR
    # ─────────────────────────────
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/en/thumb/a/a6/"
            "Asia_Pacific_University_of_Technology_%26_Innovation_logo.svg/"
            "200px-Asia_Pacific_University_of_Technology_%26_Innovation_logo.svg.png",
            use_container_width=True,
        )
        st.header("⚙️  System Info")
        if model_ready:
            st.success("✅ Model loaded and ready")
        else:
            st.warning(
                "⚠️  No trained model found.\n\n"
                "Run the training pipeline first:\n\n"
                "```bash\npython train_pipeline.py\n```"
            )

        st.divider()
        st.header("📖  About")
        st.markdown(
            """
            **Algorithm:** Linear SVM + TF-IDF (1–2 grams)

            **Emotions detected:**
            - 😢 Sadness
            - 😊 Joy
            - ❤️ Love
            - 😡 Anger
            - 😨 Fear
            - 😲 Surprise

            **Pipeline:**
            1. Tokenisation
            2. Stop-word removal
            3. Lemmatisation
            4. TF-IDF (50k features)
            5. Linear SVM + Platt scaling
            """
        )
        st.divider()
        st.caption("CT104-3-M | Pattern Recognition | APU")

    # ─────────────────────────────
    # MAIN CONTENT
    # ─────────────────────────────

    if not model_ready:
        st.error(
            "The trained model artefacts were not found. "
            "Please run `python train_pipeline.py` from the project root "
            "to train and save the model, then restart the app."
        )
        st.stop()

    tab_predict, tab_batch, tab_about = st.tabs(
        ["🔮 Predict", "📋 Batch Analysis", "📚 System Architecture"]
    )

    # ─────────────── TAB 1: PREDICT ───────────────
    with tab_predict:
        st.markdown("### Enter text to classify its emotion")

        # Demo text selector
        demo_label = st.selectbox(
            "Or choose a demo sentence:",
            ["(type your own below) …"] + DEMO_TEXTS,
            index=0,
        )

        default_val = "" if demo_label.startswith("(") else demo_label
        user_text   = st.text_area(
            "Your text:",
            value=default_val,
            height=130,
            placeholder="Type or paste any sentence here …",
        )

        col_btn, col_clear = st.columns([3, 1])
        with col_btn:
            predict_clicked = st.button("🔮  Detect Emotion", use_container_width=True)
        with col_clear:
            clear_clicked = st.button("🗑  Clear", use_container_width=True)

        if clear_clicked:
            st.rerun()

        if predict_clicked:
            if not user_text.strip():
                st.warning("Please enter some text first.")
            else:
                with st.spinner("Analysing …"):
                    time.sleep(0.3)   # tiny UX pause
                    emotion, proba = predict_emotion(user_text, extractor, le, classifier)

                col_result, col_chart = st.columns([1, 1])
                with col_result:
                    render_emotion_card(emotion, proba)
                with col_chart:
                    render_probability_bars(proba, emotion)

                # Cleaned text expander
                with st.expander("🔧 Preprocessed Text"):
                    pp = get_preprocessor()
                    st.code(pp.clean(user_text))

    # ─────────────── TAB 2: BATCH ───────────────
    with tab_batch:
        st.markdown("### Analyse multiple sentences at once")
        st.markdown(
            "Enter one sentence per line (or paste comma-separated lines). "
            "Results will be displayed as an interactive table."
        )
        batch_input = st.text_area(
            "Sentences (one per line):",
            height=200,
            placeholder="I feel so happy today!\nI am really scared about the exam.\nThis is absolutely unacceptable!",
        )
        if st.button("🔮  Analyse All"):
            lines = [l.strip() for l in batch_input.splitlines() if l.strip()]
            if not lines:
                st.warning("Please enter at least one sentence.")
            else:
                results = []
                with st.spinner(f"Classifying {len(lines)} sentence(s) …"):
                    for sentence in lines:
                        emo, proba = predict_emotion(sentence, extractor, le, classifier)
                        results.append({
                            "Sentence": sentence[:80] + ("…" if len(sentence) > 80 else ""),
                            "Emotion":  f"{EMOTION_EMOJI.get(emo,'')} {emo}",
                            "Confidence": f"{proba.get(emo, 0):.1%}",
                        })
                import pandas as pd
                st.dataframe(pd.DataFrame(results), use_container_width=True)

    # ─────────────── TAB 3: ARCHITECTURE ───────────────
    with tab_about:
        st.markdown("""
## System Architecture

### 1 ▸ Data Preprocessing
| Step | Technique | Rationale |
|------|-----------|-----------|
| Lowercasing | `str.lower()` | Normalise vocabulary |
| Emoticon → text | Dictionary substitution | Preserve emotional signals |
| URL / HTML removal | Regex | Reduce noise |
| Number normalisation | Regex (`NUMBER` token) | Reduce sparsity |
| Special char removal | Regex | Clean alphabet space |
| Tokenisation | NLTK `word_tokenize` | PTB-style tokens |
| Stop-word removal | NLTK corpus – preserved negations | Remove uninformative tokens |
| Lemmatisation | NLTK `WordNetLemmatizer` | Reduce morphological variants |

### 2 ▸ Feature Extraction – TF-IDF
- **Vocabulary size:** 50 000 tokens
- **N-gram range:** (1, 2) – unigrams + bigrams capture "not happy", "very sad"
- **Term weighting:** sublinear TF (`1 + log(tf)`) to down-weight high-frequency tokens
- **L2 normalisation:** rows normalised to unit length for SVM efficiency

### 3 ▸ Model – Linear SVM
**Why SVM?**
- Maximises margin between emotion classes in high-dimensional TF-IDF space
- Regularisation (`C`) prevents overfitting on sparse text features
- `LinearSVC` solves the dual in O(n·features) – scales to 100 k+ samples
- Calibrated with Platt scaling to yield class probabilities

### 4 ▸ Hyperparameter Optimisation
`GridSearchCV` with 5-fold stratified cross-validation over:
```
C ∈ {0.1, 1.0, 5.0, 10.0}
scoring = macro-averaged F1
```

### 5 ▸ Evaluation Framework
| Metric | Formula |
|--------|---------|
| Accuracy | TP+TN / N |
| Precision (weighted) | Σ wᵢ·(TPᵢ / (TPᵢ + FPᵢ)) |
| Recall (weighted) | Σ wᵢ·(TPᵢ / (TPᵢ + FNᵢ)) |
| F1 (weighted) | Σ wᵢ·(2·Pᵢ·Rᵢ / (Pᵢ+Rᵢ)) |
| Confusion Matrix | 6 × 6 count / normalised |

### 6 ▸ Scalability Notes
- TF-IDF uses a sparse CSR matrix – memory scales with `O(n_docs × avg_tokens)`
- `LinearSVC` dual solver is near-linear in dataset size
- Preprocessing is batched and stateless – trivially parallelisable
- Artefacts (vectorizer, classifier) are pickle-serialised for instant reloads
""")


if __name__ == "__main__":
    main()
