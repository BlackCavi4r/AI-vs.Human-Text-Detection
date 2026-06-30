---
title: AI vs Human Text Detection
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# AI vs. Human Text Detection

This project started as my Project 1 classifier for AI vs. human-written text. For Project 2, I kept that work and turned it into a working Streamlit app with document upload, side-by-side model comparison, Hugging Face LLM explanations, and a public Hugging Face Spaces deployment.

The app reviews a writing sample and estimates whether it looks more human-written or AI-generated. The result is probability-based, so I treat it as a helpful signal instead of final proof.

Public app: https://huggingface.co/spaces/Caviar22/AI-vs-Human-Text-Detection

## Project Sequence

**Project 1 foundation**

- Built the AI vs. human text detection pipeline.
- Trained traditional machine learning models: SVM, Decision Tree, and AdaBoost.
- Trained deep learning models: FNN, LSTM, and CNN.
- Compared models using accuracy, precision, recall, F1, and ROC AUC.
- Saved the trained models, metrics, plots, and project reports.

**Project 2 extension**

- Turned the detector into a Streamlit web app.
- Added direct text input and document upload for TXT, PDF, and DOCX files.
- Added side-by-side model comparison for the same input text.
- Added three Hugging Face LLMs to explain the classifier result in plain language.
- Deployed the final app on Hugging Face Spaces.

## Model Results

The strongest saved result came from SVM, so that is the model I would use first in a demo.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| SVM | 0.938 | 0.936 | 0.940 | 0.938 | 0.989 |
| AdaBoost | 0.914 | 0.888 | 0.948 | 0.917 | 0.976 |
| FNN | 0.876 | 0.870 | 0.884 | 0.877 | 0.952 |
| CNN | 0.866 | 0.865 | 0.868 | 0.866 | 0.942 |
| Decision Tree | 0.844 | 0.812 | 0.896 | 0.852 | 0.836 |
| LSTM | 0.756 | 0.750 | 0.768 | 0.759 | 0.792 |

## LLMs Added for Project 2

The LLMs do not make the final prediction. Their job is to explain the classifier result so the user can understand the writing signals behind it.

- `Qwen/Qwen2.5-0.5B-Instruct` - structured explanation of the selected classifier result.
- `HuggingFaceTB/SmolLM2-360M-Instruct` - second writing-style review.
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` - plain-English explanation for non-technical users.

The app has two LLM modes:

- **Fast: selected LLM** - runs one selected LLM.
- **Full: compare all LLMs** - runs all three LLMs for comparison.

## Dataset and Features

The training data is stored in `data/training_data/train_data_with_labels.xlsx`.

Labels:

- `0` = human-written text
- `1` = AI-written text

The traditional models use TF-IDF features plus writing-style features such as word count, sentence length, vocabulary richness, punctuation ratio, uppercase ratio, digit ratio, and readability. The deep learning models use tokenized text sequences.

## How the App Works

1. A user pastes text or uploads a document.
2. The app extracts and cleans the text.
3. The selected classifier predicts whether the text looks AI-written or human-written.
4. The app shows AI score, human score, confidence, text statistics, and model comparison.
5. The user can generate an LLM explanation.
6. The user can download a text report.

## Run Locally

```powershell
cd Project2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The first LLM run can take longer because the model files may need to download and cache.

## Main Files

- `app.py` - Streamlit app.
- `project_utils.py` - model loading, preprocessing, document reading, and prediction helpers.
- `llm_explanations.py` - Hugging Face LLM explanation logic.
- `train_models.py` - training pipeline from Project 1.
- `models/` - saved trained models.
- `reports/` - saved metrics, plots, and comparison files.
- `sample_docs/` - sample text files for testing.

## What I Learned

The biggest lesson from this sequence was that prediction and explanation are different parts of the problem. Project 1 helped me compare which classifiers worked best, and Project 2 made the result easier to use by adding document upload, model comparison, LLM explanations, and a web interface. I also learned that AI detection needs careful wording because the output is a probability, not a guaranteed answer.

## Important Note

This project is for learning and demonstration. AI detection can be wrong, especially on short or heavily edited text. The app should be used as a supporting tool, not as final evidence.
