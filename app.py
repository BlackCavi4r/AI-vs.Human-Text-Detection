import io
import json
import pickle
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, hstack

from llm_explanations import (
    LLM_OPTIONS,
    format_llm_outputs_for_report,
    generate_llm_explanation,
    model_label,
)
from project_utils import (
    clean_text,
    compute_linguistic_features,
    create_deep_model,
    linguistic_feature_names,
    summarize_text_stats,
    texts_to_sequences,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

ML_MODEL_FILES = {
    "SVM": MODELS_DIR / "svm_model.pkl",
    "Decision Tree": MODELS_DIR / "decision_tree_model.pkl",
    "AdaBoost": MODELS_DIR / "adaboost_model.pkl",
}
DL_MODEL_FILES = {
    "FNN": MODELS_DIR / "fnn_model.pt",
    "LSTM": MODELS_DIR / "lstm_model.pt",
    "CNN": MODELS_DIR / "cnn_model.pt",
}
ALL_MODELS = list(ML_MODEL_FILES.keys()) + list(DL_MODEL_FILES.keys())


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1 / (1 + np.exp(-x))


@st.cache_resource
def load_ml_artifacts():
    artifacts = {}
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    scaler_path = MODELS_DIR / "linguistic_scaler.pkl"
    names_path = MODELS_DIR / "feature_names.pkl"
    if not vectorizer_path.exists() or not scaler_path.exists():
        return None
    artifacts["vectorizer"] = joblib.load(vectorizer_path)
    artifacts["scaler"] = joblib.load(scaler_path)
    if names_path.exists():
        with open(names_path, "rb") as f:
            artifacts["feature_names"] = pickle.load(f)
    else:
        artifacts["feature_names"] = list(artifacts["vectorizer"].get_feature_names_out()) + linguistic_feature_names()
    artifacts["models"] = {}
    for name, path in ML_MODEL_FILES.items():
        if path.exists():
            artifacts["models"][name] = joblib.load(path)
    return artifacts


@st.cache_resource
def load_deep_artifacts():
    import torch

    vocab_path = MODELS_DIR / "deep_vocab.pkl"
    if not vocab_path.exists():
        return None
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    loaded = {"vocab": vocab, "models": {}}
    for display_name, path in DL_MODEL_FILES.items():
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location="cpu")
        model_name = checkpoint.get("model_name", display_name.lower())
        config = checkpoint.get("config", {})
        model = create_deep_model(model_name, vocab_size=checkpoint["vocab_size"], config=config)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        loaded["models"][display_name] = {
            "model": model,
            "max_len": checkpoint.get("max_len", 180),
            "config": config,
        }
    return loaded


@st.cache_data
def load_model_metrics():
    csv_path = REPORTS_DIR / "model_comparison.csv"
    json_path = REPORTS_DIR / "model_metrics.json"
    metrics_df = None
    metrics_json = None
    if csv_path.exists():
        metrics_df = pd.read_csv(csv_path)
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            metrics_json = json.load(f)
    return metrics_df, metrics_json


def build_ml_features(text: str, artifacts):
    text = clean_text(text)
    X_tfidf = artifacts["vectorizer"].transform([text])
    X_ling = artifacts["scaler"].transform(compute_linguistic_features([text]))
    return hstack([X_tfidf, csr_matrix(X_ling)], format="csr")


def predict_ml(text: str, model_name: str, artifacts):
    model = artifacts["models"][model_name]
    X = build_ml_features(text, artifacts)
    if hasattr(model, "predict_proba"):
        prob_ai = float(model.predict_proba(X)[0, 1])
    elif hasattr(model, "decision_function"):
        prob_ai = float(sigmoid(model.decision_function(X))[0])
    else:
        pred = int(model.predict(X)[0])
        prob_ai = 0.75 if pred == 1 else 0.25
    label = "AI-written" if prob_ai >= 0.5 else "Human-written"
    confidence = prob_ai if prob_ai >= 0.5 else 1 - prob_ai
    return label, prob_ai, confidence, X


def predict_deep(text: str, model_name: str, artifacts):
    import torch

    model_bundle = artifacts["models"][model_name]
    vocab = artifacts["vocab"]
    max_len = model_bundle["max_len"]
    sequence = texts_to_sequences([text], vocab, max_len=max_len)
    X = torch.tensor(sequence, dtype=torch.long)
    with torch.no_grad():
        logits = model_bundle["model"](X)
        prob_ai = float(torch.sigmoid(logits).cpu().numpy()[0])
    label = "AI-written" if prob_ai >= 0.5 else "Human-written"
    confidence = prob_ai if prob_ai >= 0.5 else 1 - prob_ai
    return label, prob_ai, confidence


def explain_ml_prediction(text: str, model_name: str, artifacts, X):
    feature_names = artifacts["feature_names"]
    model = artifacts["models"].get(model_name)
    if model is None:
        return pd.DataFrame()

    values = X.toarray().ravel()
    explanation_rows = []

    if hasattr(model, "coef_"):
        weights = model.coef_.ravel()
        contributions = values * weights
        top_idx = np.argsort(np.abs(contributions))[-10:][::-1]
        for idx in top_idx:
            if idx < len(feature_names) and values[idx] != 0:
                direction = "AI" if contributions[idx] > 0 else "Human"
                explanation_rows.append({
                    "feature": feature_names[idx],
                    "value": round(float(values[idx]), 4),
                    "impact": round(float(contributions[idx]), 4),
                    "pushes_toward": direction,
                })
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        active_importance = values * importances
        top_idx = np.argsort(active_importance)[-10:][::-1]
        for idx in top_idx:
            if idx < len(feature_names) and values[idx] != 0:
                explanation_rows.append({
                    "feature": feature_names[idx],
                    "value": round(float(values[idx]), 4),
                    "importance": round(float(importances[idx]), 4),
                    "note": "Important feature used by the tree-based model",
                })
    return pd.DataFrame(explanation_rows)


def simple_text_explanation(text: str):
    stats = summarize_text_stats(text)
    rows = [
        {"signal": "Word count", "value": int(stats["word_count"])},
        {"signal": "Average sentence length", "value": round(stats["avg_sentence_length"], 2)},
        {"signal": "Vocabulary richness", "value": round(stats["vocab_richness"], 3)},
        {"signal": "Punctuation ratio", "value": round(stats["punctuation_ratio"], 3)},
        {"signal": "Readability score", "value": round(stats["flesch_reading_ease"], 2)},
    ]
    return pd.DataFrame(rows)


def extract_text_from_upload(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        except Exception as exc:
            st.error(f"Could not read PDF file: {exc}")
            return ""

    if name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as exc:
            st.error(f"Could not read Word file: {exc}")
            return ""

    st.warning("Unsupported file type. Please upload PDF, DOCX, or TXT.")
    return ""


def run_all_model_predictions(text, ml_artifacts, dl_artifacts):
    rows = []
    for name in ALL_MODELS:
        try:
            if name in ML_MODEL_FILES and ml_artifacts and name in ml_artifacts["models"]:
                label, prob_ai, confidence, _ = predict_ml(text, name, ml_artifacts)
            elif name in DL_MODEL_FILES and dl_artifacts and name in dl_artifacts["models"]:
                label, prob_ai, confidence = predict_deep(text, name, dl_artifacts)
            else:
                continue
            rows.append({
                "model": name,
                "prediction": label,
                "ai_detection_score": round(prob_ai, 4),
                "human_score": round(1 - prob_ai, 4),
                "confidence_in_result": round(confidence, 4),
            })
        except Exception as exc:
            rows.append({"model": name, "prediction": f"Error: {exc}", "ai_probability": None, "confidence": None})
    return pd.DataFrame(rows)


def create_report(text, selected_model, prediction_label, prob_ai, confidence, comparison_df, stats_df, llm_outputs=None):
    lines = []
    lines.append("AI vs Human Text Detection Report")
    lines.append("=" * 36)
    lines.append(f"Selected model: {selected_model}")
    lines.append(f"Prediction: {prediction_label}")
    lines.append(f"AI detection score: {prob_ai:.4f}")
    lines.append(f"Human score: {1 - prob_ai:.4f}")
    lines.append(f"Confidence in predicted label: {confidence:.4f}")
    lines.append("")
    lines.append("Text statistics")
    lines.append("-" * 16)
    for _, row in stats_df.iterrows():
        lines.append(f"{row.iloc[0]}: {row.iloc[1]}")
    lines.append("")
    lines.append("Side-by-side model predictions")
    lines.append("-" * 32)
    if comparison_df is not None and not comparison_df.empty:
        lines.append(comparison_df.to_string(index=False))
    else:
        lines.append("No model comparison available.")
    llm_report = format_llm_outputs_for_report(llm_outputs or {})
    if llm_report:
        lines.append("")
        lines.append(llm_report)
    lines.append("")
    lines.append("Analyzed text preview")
    lines.append("-" * 21)
    lines.append(clean_text(text)[:1200])
    return "\n".join(lines)


def analysis_signature(text, selected_model, prediction_label, prob_ai, confidence):
    raw = f"{selected_model}|{prediction_label}|{prob_ai:.6f}|{confidence:.6f}|{clean_text(text)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


st.set_page_config(page_title="AI vs Human Text Detector", layout="wide")

st.title("AI vs. Human Text Detection")
st.caption(
    "Paste text or upload a document, then compare how different models judge whether the writing looks human-written or AI-written."
)

ml_artifacts = load_ml_artifacts()
dl_artifacts = load_deep_artifacts()
metrics_df, metrics_json = load_model_metrics()

if not ml_artifacts and not dl_artifacts:
    st.error("No trained models were found. Run `python train_models.py` from the project folder first.")
    st.stop()

available_models = []
if ml_artifacts:
    available_models.extend(list(ml_artifacts["models"].keys()))
if dl_artifacts:
    available_models.extend(list(dl_artifacts["models"].keys()))

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload a PDF, Word document, or TXT file", type=["pdf", "docx", "txt"])
    selected_model = st.selectbox("Choose the main classifier", available_models)
    st.divider()
    st.markdown("**Labels**")
    st.write("0 = Human-written text")
    st.write("1 = AI-written text")
    st.caption("The labels come from the training dataset. The app reports probabilities, not final proof.")

uploaded_text = extract_text_from_upload(uploaded)
manual_text = st.text_area(
    "Paste text directly",
    value=uploaded_text,
    height=220,
    placeholder="Paste an essay, paragraph, or document section here...",
)
text = clean_text(manual_text)

if len(text.strip()) < 20:
    st.info("Add a few sentences first. Very short text does not give the models enough evidence.")
    st.stop()

left, right = st.columns([1, 1])

with left:
    st.subheader("Prediction")
    if selected_model in ML_MODEL_FILES:
        label, prob_ai, confidence, X_single = predict_ml(text, selected_model, ml_artifacts)
    else:
        label, prob_ai, confidence = predict_deep(text, selected_model, dl_artifacts)
        X_single = None

    st.metric("Result", label)
    st.metric("AI detection score", f"{prob_ai * 100:.2f}%")
    st.metric("Human score", f"{(1 - prob_ai) * 100:.2f}%")
    st.metric("Confidence in result", f"{confidence * 100:.2f}%")
    st.progress(float(confidence))
    st.caption(
        "Confidence means how strongly the selected model leans toward its predicted label. "
        "The AI detection score is the number to watch when asking how AI-like the text looks."
    )

with right:
    st.subheader("Text statistics")
    stats_df = simple_text_explanation(text)
    st.dataframe(stats_df, hide_index=True, width="stretch")

st.subheader("Explanation")
if selected_model in ML_MODEL_FILES and X_single is not None:
    explanation_df = explain_ml_prediction(text, selected_model, ml_artifacts, X_single)
    if not explanation_df.empty:
        st.dataframe(explanation_df, hide_index=True, width="stretch")
    else:
        st.write("This model does not expose detailed feature weights. The text statistics above provide the main explanation signals.")
else:
    st.write(
        "Deep learning models are harder to explain directly, so the app uses the text statistics, "
        "the model comparison, and the LLM explanation section below to make the result easier to understand."
    )

st.subheader("Side-by-side model comparison on this text")
current_comparison = run_all_model_predictions(text, ml_artifacts, dl_artifacts)
st.dataframe(current_comparison, hide_index=True, width="stretch")

st.subheader("LLM explanation")
st.caption(
    "Project 2 extension: use one Hugging Face LLM for a quick explanation, "
    "or run all three LLMs to compare how they explain the same prediction."
)

llm_signature = analysis_signature(text, selected_model, label, prob_ai, confidence)
stored_llm = st.session_state.get("llm_analysis", {})
if stored_llm.get("signature") != llm_signature:
    current_llm_outputs = {}
else:
    current_llm_outputs = stored_llm.get("outputs", {})

llm_mode = st.radio(
    "LLM analysis mode",
    ["Fast: selected LLM", "Full: compare all LLMs"],
    horizontal=True,
)
selected_llm = st.selectbox(
    "Choose LLM",
    list(LLM_OPTIONS.keys()),
    format_func=model_label,
    disabled=llm_mode == "Full: compare all LLMs",
)

if st.button("Generate LLM explanation", type="primary"):
    target_models = list(LLM_OPTIONS.keys()) if llm_mode == "Full: compare all LLMs" else [selected_llm]
    generated_outputs = {}
    for model_id in target_models:
        with st.spinner(f"Loading and running {LLM_OPTIONS[model_id]['display_name']}..."):
            try:
                generated_outputs[model_id] = generate_llm_explanation(
                    model_id=model_id,
                    text=text,
                    classifier_name=selected_model,
                    prediction_label=label,
                    prob_ai=prob_ai,
                    confidence=confidence,
                    stats_df=stats_df,
                    comparison_df=current_comparison,
                )
            except Exception as exc:
                generated_outputs[model_id] = f"Could not generate explanation with this LLM: {exc}"
    current_llm_outputs = generated_outputs
    st.session_state["llm_analysis"] = {
        "signature": llm_signature,
        "outputs": current_llm_outputs,
    }

if current_llm_outputs:
    for model_id, explanation in current_llm_outputs.items():
        with st.expander(model_label(model_id), expanded=True):
            st.write(explanation)
else:
    st.info("Click the button above to generate an LLM explanation. The first run may take longer while the model downloads.")

st.subheader("Training-set model comparison")
if metrics_df is not None:
    show_cols = [c for c in ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "training_seconds"] if c in metrics_df.columns]
    st.dataframe(metrics_df[show_cols], hide_index=True, width="stretch")
else:
    st.write("No saved training metrics found yet.")

with st.expander("Project notes"):
    st.write(
        "The ML models use TF-IDF plus writing-style features. The deep learning models use tokenized text sequences. "
        "The LLMs do not make the final classification; they explain the classifier result in a more readable way. "
        "Because AI detection is probabilistic, the output should be treated as a signal, not as proof."
    )

report_text = create_report(text, selected_model, label, prob_ai, confidence, current_comparison, stats_df, current_llm_outputs)
st.download_button(
    label="Download analysis report",
    data=report_text,
    file_name="ai_human_detection_report.txt",
    mime="text/plain",
)
