---
title: AI vs Human Text Detection
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Project 2 | AI vs. Human Text Detection

This app looks at a piece of writing and estimates whether it is more likely to be human-written or AI-written. It continues my Project 1 machine learning work and adds three Hugging Face language models for Project 2. The original classifiers still make the prediction, and the LLMs help explain the result in plain language.

The app supports pasted text and uploaded `.txt`, `.pdf`, and `.docx` files. After the user chooses a classifier, the app shows the predicted label, AI score, human score, confidence, text statistics, model comparison, and optional LLM explanations.

## Problem

AI-generated writing is becoming common in school, work, and online communication. A detector should not be treated as final proof, but it can help a reader compare writing signals and understand why a text may look machine-generated. My goal was to build a practical demo that combines traditional machine learning, deep learning, and LLM explanations in one web app.

Dataset labels:

- `0` = Human-written text
- `1` = AI-written text

## What I Added for Project 2

Project 2 required at least two LLMs, so I added three small Hugging Face instruction/chat models:

1. `Qwen/Qwen2.5-0.5B-Instruct`
   - Explains the selected classifier result.
2. `HuggingFaceTB/SmolLM2-360M-Instruct`
   - Gives a second writing-style review.
3. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
   - Explains the same result in plain English for non-technical users.

The LLM section has two modes:

- **Fast: selected LLM** - runs one LLM for a quicker explanation.
- **Full: compare all LLMs** - runs all three LLMs so the explanations can be compared.

The LLMs only load when the user clicks **Generate LLM explanation**. This keeps the normal prediction flow fast and avoids loading large models before they are needed.

## Models in the App

Traditional machine learning:

1. SVM
2. Decision Tree
3. AdaBoost

Deep learning:

4. FNN
5. LSTM
6. CNN for text

Large language models:

7. Qwen2.5 0.5B Instruct
8. SmolLM2 360M Instruct
9. TinyLlama 1.1B Chat

## Dataset and Features

The training data is stored in `data/training_data/train_data_with_labels.xlsx`.

The traditional models use:

- TF-IDF word and phrase features
- Linguistic features such as word count, sentence length, vocabulary richness, punctuation ratio, uppercase ratio, digit ratio, and readability

The deep learning models use tokenized word sequences saved with the trained PyTorch model files.

## Saved Results

These are the saved results from `reports/model_comparison.csv`:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| SVM | 0.938 | 0.936 | 0.940 | 0.938 | 0.989 |
| AdaBoost | 0.914 | 0.888 | 0.948 | 0.917 | 0.976 |
| FNN | 0.876 | 0.870 | 0.884 | 0.877 | 0.952 |
| CNN | 0.866 | 0.865 | 0.868 | 0.866 | 0.942 |
| Decision Tree | 0.844 | 0.812 | 0.896 | 0.852 | 0.836 |
| LSTM | 0.756 | 0.750 | 0.768 | 0.759 | 0.792 |

For demos, I usually start with SVM because it had the strongest saved score.

## How the App Works

1. The user pastes text or uploads a document.
2. The app extracts and cleans the text.
3. The selected classifier predicts whether the writing looks AI-written or human-written.
4. The app shows the AI score, human score, confidence, text statistics, and side-by-side model comparison.
5. If the user requests it, one selected LLM or all three LLMs generate a short explanation of the result.
6. The user can download a text report.

The LLM explanation is not used as the final prediction. It is there to help the user understand the classifier output.

## How to Run Locally

```bash
cd ai_human_detection_project
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The first LLM run can take longer because the model files are downloaded and cached.

## Hugging Face Spaces Deployment

This repository is prepared for a Hugging Face Docker Space that runs the Streamlit app on port `7860`.

Steps:

1. Go to `https://huggingface.co/spaces`.
2. Create a new Space.
3. Select **Docker** as the SDK.
4. Upload or sync this project.
5. Make sure the Space includes:
   - `Dockerfile`
   - `app.py`
   - `llm_explanations.py`
   - `project_utils.py`
   - `requirements.txt`
   - `README.md`
   - `models/`
   - `reports/`
   - `sample_docs/`
6. Wait for the build to finish.
7. Test text input, file upload, classifier prediction, and both LLM analysis modes.

Public Space link:

```text
https://huggingface.co/spaces/Caviar22/AI-vs-Human-Text-Detection
```

## Files and Folders

```text
ai_human_detection_project/
|-- app.py
|-- llm_explanations.py
|-- project_utils.py
|-- train_models.py
|-- Dockerfile
|-- requirements.txt
|-- requirements-training.txt
|-- README.md
|-- models/
|-- data/
|-- notebooks/
|-- reports/
|-- sample_docs/
`-- tests/
```

## What I Learned

This project helped me see the difference between prediction and explanation. The SVM model gave the strongest classification score, but the LLMs made the result easier to understand for a normal user. I also learned that AI detection is not absolute. The best version of the app is honest about uncertainty: it gives a score, compares models, shows writing signals, and explains why the result may or may not be reliable.

## Important Note

This app is a class project and should not be used as final evidence that a person did or did not use AI. The prediction is probabilistic, and the LLM explanation is only a supporting interpretation of the classifier output.
