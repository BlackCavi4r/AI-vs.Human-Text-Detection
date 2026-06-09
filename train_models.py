import argparse
import json
import pickle
import random
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from gensim.models import Word2Vec

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from project_utils import (
    build_vocab,
    clean_text,
    compute_linguistic_features,
    create_deep_model,
    linguistic_feature_names,
    texts_to_sequences,
    tokenize,
    average_word_vectors,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "training_data" / "train_data_with_labels.xlsx"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EMBEDDING_DIR = MODELS_DIR / "embedding_model"

RANDOM_STATE = 42


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1 / (1 + np.exp(-x))


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    expected = {"text", "label"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset must contain columns {expected}. Missing: {missing}")
    df = df[["text", "label"]].copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].copy()
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])].copy()
    return df.reset_index(drop=True)


def make_feature_matrix(texts, tfidf_vectorizer=None, ling_scaler=None, fit=False, max_features=5000):
    if fit:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words="english",
            sublinear_tf=True,
        )
        X_tfidf = tfidf_vectorizer.fit_transform(texts)
        ling = compute_linguistic_features(texts)
        ling_scaler = StandardScaler()
        X_ling = ling_scaler.fit_transform(ling)
    else:
        X_tfidf = tfidf_vectorizer.transform(texts)
        ling = compute_linguistic_features(texts)
        X_ling = ling_scaler.transform(ling)
    X = hstack([X_tfidf, csr_matrix(X_ling)], format="csr")
    return X, tfidf_vectorizer, ling_scaler


def evaluate_from_scores(y_true, scores, threshold=0.5):
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    except Exception:
        metrics["roc_auc"] = None
    return y_pred, metrics


def save_confusion_matrix_plot(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Human", "AI"])
    ax.set_yticklabels(["Human", "AI"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = FIGURES_DIR / f"confusion_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return str(out.relative_to(PROJECT_ROOT))


def save_roc_plot(y_true, scores, model_name):
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIGURES_DIR / f"roc_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return str(out.relative_to(PROJECT_ROOT))


def feature_comparison(X_train_text, X_test_text, y_train, y_test, tfidf_vectorizer, X_train_ling, X_test_ling, fast=False):
    rows = []

    # TF-IDF only
    X_train_tfidf = tfidf_vectorizer.transform(X_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)
    clf.fit(X_train_tfidf, y_train)
    probs = clf.predict_proba(X_test_tfidf)[:, 1]
    _, m = evaluate_from_scores(y_test, probs)
    rows.append({"feature_set": "TF-IDF", **m})

    # Linguistic features only
    clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)
    clf.fit(X_train_ling, y_train)
    probs = clf.predict_proba(X_test_ling)[:, 1]
    _, m = evaluate_from_scores(y_test, probs)
    rows.append({"feature_set": "Linguistic Features", **m})

    # Word2Vec average embeddings
    vector_size = 40 if fast else 75
    tokenized_train = [tokenize(t) for t in X_train_text]
    tokenized_test = [tokenize(t) for t in X_test_text]
    w2v_cap = 350 if fast else 700
    w2v_training_tokens = tokenized_train[: min(w2v_cap, len(tokenized_train))]
    w2v = Word2Vec(
        sentences=w2v_training_tokens,
        vector_size=vector_size,
        window=5,
        min_count=2,
        workers=1,
        sg=0,
        epochs=2 if fast else 4,
        seed=RANDOM_STATE,
    )
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    w2v.save(str(EMBEDDING_DIR / "word2vec.model"))
    X_train_w2v = average_word_vectors(tokenized_train, w2v.wv, vector_size)
    X_test_w2v = average_word_vectors(tokenized_test, w2v.wv, vector_size)
    clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)
    clf.fit(X_train_w2v, y_train)
    probs = clf.predict_proba(X_test_w2v)[:, 1]
    _, m = evaluate_from_scores(y_test, probs)
    rows.append({"feature_set": "Word2Vec Average Embeddings", **m})

    result = pd.DataFrame(rows).sort_values("f1", ascending=False)
    result.to_csv(REPORTS_DIR / "feature_comparison.csv", index=False)
    return result


def train_ml_models(X_train, X_test, y_train, y_test, fast=False):
    metrics = {}
    model_paths = {}
    scores_by_model = {}

    # SVM
    svm_grid = GridSearchCV(
        LinearSVC(random_state=RANDOM_STATE, class_weight="balanced", dual=False, max_iter=5000),
        param_grid={"C": ([1.0] if fast else [0.5, 1.0, 2.0])},
        scoring="f1",
        cv=(2 if fast else 3),
        n_jobs=1,
    )
    t0 = time.time()
    svm_grid.fit(X_train, y_train)
    svm = svm_grid.best_estimator_
    decision_scores = svm.decision_function(X_test)
    probs = sigmoid(decision_scores)
    y_pred, m = evaluate_from_scores(y_test, probs)
    m.update({"best_params": svm_grid.best_params_, "training_seconds": round(time.time() - t0, 2)})
    metrics["SVM"] = m
    model_paths["SVM"] = "models/svm_model.pkl"
    scores_by_model["SVM"] = probs
    joblib.dump(svm, MODELS_DIR / "svm_model.pkl")
    metrics["SVM"]["confusion_matrix_plot"] = save_confusion_matrix_plot(y_test, y_pred, "SVM")
    metrics["SVM"]["roc_plot"] = save_roc_plot(y_test, probs, "SVM")

    # Decision Tree
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        param_grid={"max_depth": ([12] if fast else [10, 20, None]), "min_samples_split": ([2] if fast else [2, 10])},
        scoring="f1",
        cv=(2 if fast else 3),
        n_jobs=1,
    )
    t0 = time.time()
    dt_grid.fit(X_train, y_train)
    dt = dt_grid.best_estimator_
    probs = dt.predict_proba(X_test)[:, 1]
    y_pred, m = evaluate_from_scores(y_test, probs)
    m.update({"best_params": dt_grid.best_params_, "training_seconds": round(time.time() - t0, 2)})
    metrics["Decision Tree"] = m
    model_paths["Decision Tree"] = "models/decision_tree_model.pkl"
    scores_by_model["Decision Tree"] = probs
    joblib.dump(dt, MODELS_DIR / "decision_tree_model.pkl")
    metrics["Decision Tree"]["confusion_matrix_plot"] = save_confusion_matrix_plot(y_test, y_pred, "Decision Tree")
    metrics["Decision Tree"]["roc_plot"] = save_roc_plot(y_test, probs, "Decision Tree")

    # AdaBoost
    ada_grid = GridSearchCV(
        AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE), random_state=RANDOM_STATE),
        param_grid={"n_estimators": ([25] if fast else [50, 100]), "learning_rate": ([0.8] if fast else [0.5, 1.0])},
        scoring="f1",
        cv=(2 if fast else 3),
        n_jobs=1,
    )
    t0 = time.time()
    ada_grid.fit(X_train, y_train)
    ada = ada_grid.best_estimator_
    probs = ada.predict_proba(X_test)[:, 1]
    y_pred, m = evaluate_from_scores(y_test, probs)
    m.update({"best_params": ada_grid.best_params_, "training_seconds": round(time.time() - t0, 2)})
    metrics["AdaBoost"] = m
    model_paths["AdaBoost"] = "models/adaboost_model.pkl"
    scores_by_model["AdaBoost"] = probs
    joblib.dump(ada, MODELS_DIR / "adaboost_model.pkl")
    metrics["AdaBoost"]["confusion_matrix_plot"] = save_confusion_matrix_plot(y_test, y_pred, "AdaBoost")
    metrics["AdaBoost"]["roc_plot"] = save_roc_plot(y_test, probs, "AdaBoost")

    return metrics, model_paths, scores_by_model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def predict_deep_scores(model, loader, device):
    model.eval()
    scores = []
    y_true = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            scores.extend(probs.tolist())
            y_true.extend(yb.numpy().tolist())
    return np.array(y_true), np.array(scores)


def deep_loader(sequences, labels, batch_size=128, shuffle=False):
    X = torch.tensor(sequences, dtype=torch.long)
    y = torch.tensor(np.asarray(labels), dtype=torch.float32)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def deep_candidate_settings(fast=False):
    """Small tuning space for FNN, LSTM, and CNN."""
    if fast:
        return {
            "fnn": [
                {"trial": "fnn_fast", "display": "FNN", "max_len": 80, "epochs": 6,
                 "config": {"embedding_dim": 32, "hidden_dim": 64, "dropout": 0.3, "lr": 0.001}},
            ],
            "lstm": [
                {"trial": "lstm_fast", "display": "LSTM", "max_len": 50, "epochs": 8,
                 "config": {"embedding_dim": 32, "hidden_dim": 48, "dropout": 0.3, "lr": 0.001}},
            ],
            "cnn": [
                {"trial": "cnn_fast", "display": "CNN", "max_len": 80, "epochs": 6,
                 "config": {"embedding_dim": 32, "num_filters": 48, "filter_sizes": (3, 4, 5), "dropout": 0.3, "lr": 0.001}},
            ],
        }

    return {
        "fnn": [
            {"trial": "fnn_a", "display": "FNN", "max_len": 120, "epochs": 15,
             "config": {"embedding_dim": 64, "hidden_dim": 128, "dropout": 0.3, "lr": 0.001}},
            {"trial": "fnn_b", "display": "FNN", "max_len": 150, "epochs": 12,
             "config": {"embedding_dim": 96, "hidden_dim": 160, "dropout": 0.4, "lr": 0.0008}},
        ],
        "lstm": [
            {"trial": "lstm_a", "display": "LSTM", "max_len": 60, "epochs": 20,
             "config": {"embedding_dim": 64, "hidden_dim": 64, "dropout": 0.3, "lr": 0.001}},
            {"trial": "lstm_b", "display": "LSTM", "max_len": 90, "epochs": 16,
             "config": {"embedding_dim": 64, "hidden_dim": 96, "dropout": 0.4, "lr": 0.0008}},
        ],
        "cnn": [
            {"trial": "cnn_a", "display": "CNN", "max_len": 120, "epochs": 15,
             "config": {"embedding_dim": 64, "num_filters": 64, "filter_sizes": (3, 4, 5), "dropout": 0.3, "lr": 0.001}},
            {"trial": "cnn_b", "display": "CNN", "max_len": 150, "epochs": 12,
             "config": {"embedding_dim": 96, "num_filters": 96, "filter_sizes": (2, 3, 4, 5), "dropout": 0.4, "lr": 0.0008}},
        ],
    }


def train_deep_config(model_name, item, train_texts, y_train, eval_texts, y_eval, vocab, device, criterion):
    X_train_seq = texts_to_sequences(train_texts, vocab, max_len=item["max_len"])
    X_eval_seq = texts_to_sequences(eval_texts, vocab, max_len=item["max_len"])
    train_loader = deep_loader(X_train_seq, y_train, batch_size=128, shuffle=True)
    eval_loader = deep_loader(X_eval_seq, y_eval, batch_size=128, shuffle=False)

    config = item["config"]
    model = create_deep_model(model_name, vocab_size=len(vocab), config=config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 0.001))

    t0 = time.time()
    losses = []
    for _ in range(item["epochs"]):
        losses.append(train_one_epoch(model, train_loader, optimizer, criterion, device))

    y_true, scores = predict_deep_scores(model, eval_loader, device)
    y_pred, metrics = evaluate_from_scores(y_true, scores)
    metrics["training_seconds"] = round(time.time() - t0, 2)
    metrics["train_loss_last"] = float(losses[-1]) if losses else None
    return model, y_true, scores, y_pred, metrics


def train_deep_models(train_texts, y_train, test_texts, y_test, max_len=None, epochs=None, deep_trials=2, max_vocab_size=10000, fast=False):
    vocab = build_vocab(train_texts, max_vocab_size=max_vocab_size, min_freq=2)
    with open(MODELS_DIR / "deep_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    candidates = deep_candidate_settings(fast=fast)
    if epochs is not None:
        for model_candidates in candidates.values():
            for item in model_candidates:
                item["epochs"] = epochs
    if max_len is not None:
        for model_candidates in candidates.values():
            for item in model_candidates:
                item["max_len"] = max_len

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()
    deep_metrics = {}
    tuning_rows = []
    scores_by_model = {}

    # Validation split for the deep learning tuning step.
    tune_train_texts, val_texts, tune_y_train, y_val = train_test_split(
        np.asarray(train_texts),
        np.asarray(y_train),
        test_size=0.2,
        stratify=np.asarray(y_train),
        random_state=RANDOM_STATE,
    )

    for model_name, model_candidates in candidates.items():
        if deep_trials is not None and deep_trials > 0:
            model_candidates = model_candidates[:deep_trials]

        display_name = model_candidates[0]["display"]
        print(f"Tuning {display_name}...", flush=True)
        best_item = None
        best_val_f1 = -1.0

        for item in model_candidates:
            _, _, _, _, val_metrics = train_deep_config(
                model_name,
                item,
                tune_train_texts,
                tune_y_train,
                val_texts,
                y_val,
                vocab,
                device,
                criterion,
            )
            row = {
                "model": item["display"],
                "trial": item["trial"],
                "split": "validation",
                "selected": False,
                "max_len": item["max_len"],
                "epochs": item["epochs"],
                **item["config"],
            }
            row.update({f"validation_{k}": val_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]})
            tuning_rows.append(row)

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_item = item

        print(f"Training final {display_name} with selected settings...", flush=True)
        final_model, y_true, scores, y_pred, test_metrics = train_deep_config(
            model_name,
            best_item,
            train_texts,
            y_train,
            test_texts,
            y_test,
            vocab,
            device,
            criterion,
        )

        selected_config = best_item["config"]
        test_metrics.update({
            "best_params": {"max_len": best_item["max_len"], "epochs": best_item["epochs"], **selected_config},
            "validation_f1_for_selected_config": float(best_val_f1),
            "confusion_matrix_plot": save_confusion_matrix_plot(y_true, y_pred, display_name),
            "roc_plot": save_roc_plot(y_true, scores, display_name),
        })

        torch.save(
            {
                "model_name": model_name,
                "vocab_size": len(vocab),
                "config": selected_config,
                "state_dict": final_model.cpu().state_dict(),
                "max_len": best_item["max_len"],
            },
            MODELS_DIR / f"{model_name}_model.pt",
        )

        deep_metrics[display_name] = test_metrics
        scores_by_model[display_name] = scores

        selected_row = {
            "model": display_name,
            "trial": best_item["trial"],
            "split": "test_selected_final_model",
            "selected": True,
            "max_len": best_item["max_len"],
            "epochs": best_item["epochs"],
            **selected_config,
        }
        selected_row.update({f"test_{k}": test_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]})
        selected_row["validation_f1_for_selected_config"] = best_val_f1
        tuning_rows.append(selected_row)

    pd.DataFrame(tuning_rows).to_csv(REPORTS_DIR / "deep_tuning_results.csv", index=False)
    return deep_metrics, scores_by_model

def save_overall_roc(y_test, scores_by_model):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, scores in scores_by_model.items():
        try:
            fpr, tpr, _ = roc_curve(y_test, scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")
        except Exception:
            continue
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = FIGURES_DIR / "roc_model_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return str(out.relative_to(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="smaller run while testing")
    parser.add_argument("--full-data", action="store_true", help="use the full Excel file instead of a balanced sample")
    parser.add_argument("--rows-per-class", type=int, default=None, help="optional cap per label")
    parser.add_argument("--epochs", type=int, default=None, help="override DL epochs")
    parser.add_argument("--deep-trials", type=int, default=2, help="number of DL configs to test per model during tuning")
    args = parser.parse_args()

    set_seed()
    try:
        torch.set_num_threads(2)
    except Exception:
        pass
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    EMBEDDING_DIR.mkdir(exist_ok=True)

    max_features = 1500 if args.fast else 5000
    max_vocab_size = 4000 if args.fast else 10000
    epochs = args.epochs

    df = load_dataset(DATA_PATH)
    if not args.full_data:
        cap = args.rows_per_class if args.rows_per_class is not None else (600 if args.fast else 1250)
        pieces = []
        for label in [0, 1]:
            group = df[df["label"] == label]
            pieces.append(group.sample(n=min(cap, len(group)), random_state=RANDOM_STATE))
        df = pd.concat(pieces).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Loaded dataset: {df.shape[0]} rows")
    print(df["label"].value_counts().sort_index())

    # Save EDA summary.
    eda_summary = {
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().to_dict().items()},
        "average_words": float(np.mean([len(tokenize(t)) for t in df["text"]])),
        "median_words": float(np.median([len(tokenize(t)) for t in df["text"]])),
        "note": "Saved run uses a balanced sample unless --full-data is used.",
    }
    with open(REPORTS_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)

    # Basic balance chart.
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    counts = df["label"].value_counts().sort_index()
    ax.bar(["Human (0)", "AI (1)"], counts.values)
    ax.set_title("Class Balance")
    ax.set_ylabel("Number of samples")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=180)
    plt.close(fig)

    train_texts, test_texts, y_train, y_test = train_test_split(
        df["text"].values,
        df["label"].values,
        test_size=0.2,
        stratify=df["label"].values,
        random_state=RANDOM_STATE,
    )

    X_train, tfidf_vectorizer, ling_scaler = make_feature_matrix(
        train_texts, fit=True, max_features=max_features
    )
    X_test, _, _ = make_feature_matrix(
        test_texts, tfidf_vectorizer=tfidf_vectorizer, ling_scaler=ling_scaler, fit=False
    )

    joblib.dump(tfidf_vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(ling_scaler, MODELS_DIR / "linguistic_scaler.pkl")

    feature_names = list(tfidf_vectorizer.get_feature_names_out()) + linguistic_feature_names()
    with open(MODELS_DIR / "feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    print("Training traditional ML models...")
    ml_metrics, model_paths, scores_by_model = train_ml_models(X_train, X_test, y_train, y_test, fast=args.fast)

    print("Training deep learning models...")
    deep_metrics, deep_scores = train_deep_models(
        train_texts,
        y_train,
        test_texts,
        y_test,
        max_len=None,
        epochs=epochs,
        deep_trials=(1 if args.fast else args.deep_trials),
        max_vocab_size=max_vocab_size,
        fast=args.fast,
    )
    scores_by_model.update(deep_scores)

    print("Running feature comparison...")
    X_train_ling = ling_scaler.transform(compute_linguistic_features(train_texts))
    X_test_ling = ling_scaler.transform(compute_linguistic_features(test_texts))
    feature_result = feature_comparison(
        train_texts,
        test_texts,
        y_train,
        y_test,
        tfidf_vectorizer,
        X_train_ling,
        X_test_ling,
        fast=args.fast,
    )
    print(feature_result)

    all_metrics = {**ml_metrics, **deep_metrics}
    for model_name, m in all_metrics.items():
        print(model_name, {k: v for k, v in m.items() if k in ["accuracy", "precision", "recall", "f1", "roc_auc"]})

    comparison_df = pd.DataFrame([
        {"model": model_name, **metrics}
        for model_name, metrics in all_metrics.items()
    ])
    comparison_df = comparison_df.sort_values("f1", ascending=False)
    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    all_metrics["_overall"] = {
        "roc_comparison_plot": save_overall_roc(y_test, scores_by_model),
        "label_definition": {"0": "Human-written text", "1": "AI-written text"},
        "test_size": int(len(y_test)),
        "fast_training_mode": bool(args.fast),
        "full_data": bool(args.full_data),
        "rows_used_for_saved_run": int(df.shape[0]),
    }
    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print("Training complete.")
    print(f"Saved models to: {MODELS_DIR}")
    print(f"Saved reports to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
