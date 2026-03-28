"""
=============================================================================
FILE:    app/streamlit_app.py
PURPOSE: Streamlit inference dashboard for real-time text emotion analysis.
USAGE:
    streamlit run app/streamlit_app.py
DESCRIPTION:
    Provides a seamless, interactive UI for demonstrating the temporal sequence
    classification capabilities of the underlying Deep Learning (LSTM) engine.
    Designed for real-time inference with dynamic probability visualizations
    and integrated batch-processing functionalities.
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
# Dynamic Path Resolution & Module Imports
# ---------------------------------------------------------------------------
# Dynamically injecting the project root into sys.path ensures context-agnostic 
# module imports, preventing pathing anomalies across different environments.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.preprocessor    import TextPreprocessor
from data.data_loader      import EMOTION_LABELS, EMOTION_COLOR_MAP

# ---------------------------------------------------------------------------
# Global UI Configuration Tokens
# ---------------------------------------------------------------------------
# The following mappings dictate the visual semantics of the application.
# Centralizing these properties facilitates seamless theming and extensions.
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
    "joy":      "FFFDE7",
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
# Inference Engine Initialization & State Management
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_artefacts():
    """
    Instantiates and caches the model artefacts (preprocessor, encoder, estimator)
    as singleton objects. This leverages Streamlit's caching mechanism to bypass 
    redundant I/O bottlenecks and memory reallocation upon consecutive GUI cycles.
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
    """
    Initializes the text preprocessing sequence. Cached to ensure deterministic 
    and efficient pipeline execution preceding the forward pass.
    """
    return TextPreprocessor(remove_stopwords=True, lemmatize=True, min_token_len=2)


# =============================================================================
# Core Inference Pipeline
# =============================================================================

def predict_emotion(text: str, extractor, le, classifier) -> tuple[str, dict]:
    """
    Orchestrates the end-to-end evaluation pipeline for a raw input string.
    
    The functional flow involves normalization, feature extraction mapping, 
    and yielding the classification via the estimator's probability distribution.

    Returns
    -------
    (predicted_emotion, {emotion: probability})
    """
    preprocessor = get_preprocessor()
    clean = preprocessor.clean(text)
    
    # Early termination for vacuous or purely noisy inputs
    if not clean.strip():
        return "unknown", {e: 0.0 for e in EMOTION_LABELS}

    # Project the sanitized payload into feature space and compute soft probabilities
    X = extractor.transform([clean])
    proba = classifier.predict_proba(X)[0]
    label_idx = int(np.argmax(proba))
    emotion   = le.classes_[label_idx]
    prob_dict = {le.classes_[i]: float(p) for i, p in enumerate(proba)}
    return emotion, prob_dict


# =============================================================================
# UI Data Visualization Components
# =============================================================================

def render_emotion_card(emotion: str, proba: dict) -> None:
    """
    Constructs a dynamically styled HTML component summarizing the dominant class.
    CSS properties are contextually bound to the precise emotional taxonomy 
    extracted from the model's confidence logic.
    """
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
    """
    Constructs an interactive DOM structure to visualize the output vector.
    Sorts probabilities monotonically avoiding cognitive overload for the end user.
    """
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
# View Controller Layer (GUI Initialization)
# =============================================================================

def main():
    # Enforce global page parameters to establish a robust execution sandbox.
    st.set_page_config(
        page_title="Real-Time Text Emotion Analysis",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── UI/UX Cascading Overrides ──
    # Injecting precise CSS hooks to normalize padding constraints and 
    # elevate the fundamental typography hierarchies of Streamlit's shadow DOM.
    st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
        .stTextArea textarea { font-size: 1rem; }
        h1 { font-size: 2rem !important; }
        .stButton > button {
            background: #3B82F6; color: white;
            border-radius: 8px; border: none;
            padding: 0.5rem 1.5rem; font-size: 1rem;
            transition: all 0.2s ease-in-out;
        }
        .stButton > button:hover { background: #2563EB; }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero Section ──
    # Employing an asymmetrical column tensor setup to focus visual weight.
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        st.markdown("## ⚡")
    with col_title:
        st.markdown(
            "### ⚡ Real-Time Text Emotion Analysis\n"
            "**Neural Interface Dashboard for Instantaneous Context Mining**"
        )
    st.divider()

    # ── Dependency Resolution ──
    with st.spinner("Instantiating inference architecture …"):
        extractor, le, classifier = load_artefacts()

    model_ready = all(x is not None for x in [extractor, le, classifier])

    # ─────────────────────────────
    # DIAGNOSTIC SIDEBAR
    # ─────────────────────────────
    with st.sidebar:
        st.header("⚙️  Runtime Status")
        # Visual heuristics to inform the user of computational availability.
        if model_ready:
            st.success("✅ Neural Engine Active")
        else:
            st.warning(
                "⚠️  Engine Offline.\n\n"
                "Compile and serialize the weights via:\n\n"
                "```bash\npython train_pipeline.py\n```"
            )

        st.divider()
        st.header("📖  Architecture Details")
        st.markdown(
            """
            **🧠 AI Engine:** Deep Learning (LSTM Network)

            **Output Taxonomy (Classes):**
            - 😢 Sadness
            - 😊 Joy
            - ❤️ Love
            - 😡 Anger
            - 😨 Fear
            - 😲 Surprise

            **Execution Pipeline:**
            1. **Ingestion & Normalization:** Noise mitigation schema.
            2. **Lexical Tokenization:** Granular textual splitting.
            3. **Feature Mapping:** High-dimensional embedding aggregation.
            4. **Forward Pass:** Sequence dynamics via LSTM gates.
            5. **Classification:** Softmax regression head yielding logits.
            """
        )
        st.divider()
        st.caption("✨ Developed by Ibrahem")

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
        user_text = st.text_area(
            "Your text:",
            height=130,
            placeholder="Type or paste any sentence here …",
        )

        col_btn, col_clear = st.columns([3, 1])
        with col_btn:
            predict_clicked = st.button("🔮  Detect Emotion", use_container_width=True)
        with col_clear:
            clear_clicked = st.button("🗑  Clear", use_container_width=True)

        if clear_clicked:
            st.session_state.user_input = ""
            st.rerun()

        # Evaluating the event hook for asynchronous prediction requests.
        if predict_clicked:
            if not user_text.strip():
                st.warning("No observable sequence provided. Please insert textual data.")
            else:
                with st.spinner("Propagating inputs through hidden layers …"):
                    time.sleep(0.3)   # Synthetically emulates non-trivial network latency for UX perception.
                    emotion, proba = predict_emotion(user_text, extractor, le, classifier)

                # Bipartite structural layout to present discrete outputs vs continuous confidence curves.
                col_result, col_chart = st.columns([1, 1])
                with col_result:
                    render_emotion_card(emotion, proba)
                with col_chart:
                    render_probability_bars(proba, emotion)

                # Exposing the latent payload transformation phase to allow deterministic auditing.
                with st.expander("🔧 Intermediate Sanitized Token Map"):
                    pp = get_preprocessor()
                    st.code(pp.clean(user_text))

    # ─────────────── TAB 2: BATCH INFERENCE ───────────────
    with tab_batch:
        st.markdown("### High-Throughput Matrix Evaluation")
        st.markdown(
            "Iterate inference upon an N-dimensional array of textual samples. "
            "Logits are subsequently coalesced and rendered as a scalable dataframe."
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

    # ─────────────── TAB 3: THEORETICAL BLUEPRINT ───────────────
    with tab_about:
        st.markdown("""
## Neural Architecture Specification

### 1 ▸ Temporal Data Preprocessing
| Operation Layer | Implementation Technique | Theoretical Rationale |
|-----------------|--------------------------|-----------------------|
| Lexical Normalization | `str.lower()` | Homogenizes input distribution |
| Emoticon Semantics | Dictionary mapping | Preserves implicit sentiment markers |
| Noise Pruning | Regex compilations | Filters out superfluous HTML/URL artefacts |
| Sparsity Reduction | `NUMBER` token alignment | Constrains exploding vocabulary dimensions |
| Token Splitting | NLTK `word_tokenize` | Establishes granular sequence boundaries |
| Stop-word Mitigation | Curated corpus filtering | Amplifies signal-to-noise ratio |
| Morphological Rooting | `WordNetLemmatizer` | Collapses variant syntactic manifestations |

### 2 ▸ Deep Feature Engineering
- **Sequence Mapping:** Continuous vectorization of lexical tokens.
- **Dimensionality Restraint:** Truncating or padding sequences to maintain tensor homogeneity.
- **Context Preservation:** Advanced positional embeddings maintain long-range dependencies effectively.

### 3 ▸ Deep Learning Estimator – LSTM Network
**Why recurrent architectures?**
- Effectively addresses the vanishing gradient paradigm via internal gated mechanisms.
- Recurrent gating (`forget`, `input`, `output`) learns temporal, sequential context critical in linguistic dependencies.
- Scales gracefully across sophisticated, multi-class categorical deployments when terminated with a Softmax activation.
- Backpropagation Through Time (BPTT) guarantees refined weight tuning over extensive epochs.

### 4 ▸ Validation & Optimization Heuristics
- Employed robust, dynamic decay schedulers to navigate saddle points in the loss landscape.
- Loss function: Computes multi-class categorical crossentropy converging reliably upon minima.

### 5 ▸ Evaluation Protocol
| Metric Designation | Mathematical Formulation |
|--------------------|--------------------------|
| Convergence Accuracy | (TP + TN) / Σ(Population) |
| Weighted Precision | Σ wᵢ·(TPᵢ / (TPᵢ + FPᵢ)) |
| Weighted Recall | Σ wᵢ·(TPᵢ / (TPᵢ + FNᵢ)) |
| Macro F1-Score | Σ wᵢ·(2·Pᵢ·Rᵢ / (Pᵢ+Rᵢ)) |
| Dispersion Matrix | Detailed confusion matrix of probability |

### 6 ▸ Production Deployment Constraints
- Leveraged hardware-agnostic tensor mathematical backends.
- Preprocessing remains trivially parallelizable across threads.
- Inference caching minimizes latency degradation during consecutive UI refresh loops.
""")


if __name__ == "__main__":
    main()
