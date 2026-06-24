# Project 2 | AI vs. Human Text Detection with LLM Explanations

https://ai-vs-human-text-detection-ocwenwd28inzfhzhpkxajg.streamlit.app/

This project detects whether a text sample looks human-written or AI-written. It continues my Project 1 machine learning app and extends it for Project 2 by adding Hugging Face Large Language Models (LLMs) that explain the classifier result.

The app lets a user paste text or upload a PDF, Word document, or TXT file. Then the user can choose one of the trained models and get a prediction, an AI score, a human score, text statistics, model comparison, LLM-generated explanation, and a downloadable report.

Labels used in the dataset:

- `0` = Human-written text
- `1` = AI-written text

## Project 2 LLM extension

The original Project 1 classifiers remain part of the app. Project 2 adds two Hugging Face LLMs:

1. `Qwen/Qwen2.5-0.5B-Instruct` - creates a structured explanation of the selected classifier result.
2. `HuggingFaceTB/SmolLM2-360M-Instruct` - provides a second-opinion writing-style review.

The Streamlit interface includes two LLM modes:

- **Fast: selected LLM** - runs one selected LLM for a quicker explanation.
- **Full: compare both LLMs** - runs both integrated LLMs so the user can compare explanations.

The LLMs are loaded only after the user clicks **Generate LLM explanation**, so the regular classifier prediction still loads quickly.

## Folder structure

```text
ai_human_detection_project/
├── app.py
├── train_models.py
├── project_utils.py
├── requirements.txt
├── README.md
├── models/
│   ├── svm_model.pkl
│   ├── decision_tree_model.pkl
│   ├── adaboost_model.pkl
│   ├── fnn_model.pt
│   ├── lstm_model.pt
│   ├── cnn_model.pt
│   ├── tfidf_vectorizer.pkl
│   ├── deep_vocab.pkl
│   ├── linguistic_scaler.pkl
│   ├── feature_names.pkl
│   └── embedding_model/
│       └── word2vec.model
├── data/
│   ├── training_data/
│   │   └── train_data_with_labels.xlsx
│   └── test_data/
├── notebooks/
│   └── project1_notebook.ipynb
├── reports/
│   ├── model_comparison.csv
│   ├── feature_comparison.csv
│   ├── deep_tuning_results.csv
│   └── figures/
└── sample_docs/
    ├── human_sample.txt
    └── ai_sample.txt
```

## How to run the app

Open the project folder first:

```bash
cd ai_human_detection_project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Or activate it on Windows CMD:

```bash
.venv\Scripts\activate.bat
```

Install the libraries:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

The first LLM explanation may take longer because the Hugging Face model files must be downloaded and cached.

## Hugging Face Spaces deployment

This app is ready for a Hugging Face Spaces Streamlit deployment:

1. Create a new Space at `https://huggingface.co/spaces`.
2. Choose **Streamlit** as the SDK.
3. Upload the source files, `models/`, `reports/`, `sample_docs/`, `app.py`, `project_utils.py`, `llm_explanations.py`, `requirements.txt`, and `README.md`.
4. Wait for the Space build to install dependencies from `requirements.txt`.
5. Open the public Space link and test text input, document upload, classifier prediction, and both LLM explanation modes.

## How to retrain the models

The saved models are already included in the `models/` folder. To retrain them:

```bash
python train_models.py
```

For a faster testing run:

```bash
python train_models.py --fast
```

The normal run uses a balanced sample from the Excel file so it does not take forever on a regular laptop. To force the full dataset:

```bash
python train_models.py --full-data
```

## Models included

Traditional machine learning:

1. SVM
2. Decision Tree
3. AdaBoost

Deep learning:

4. FNN
5. LSTM
6. CNN for text

The deep learning models are saved as PyTorch `.pt` files.

LLM models:

7. Qwen2.5 0.5B Instruct (`Qwen/Qwen2.5-0.5B-Instruct`)
8. SmolLM2 360M Instruct (`HuggingFaceTB/SmolLM2-360M-Instruct`)

## Model tuning

For the traditional ML models, I used `GridSearchCV` and selected the best setup by F1-score:

- **SVM:** `C = [0.5, 1.0, 2.0]`
- **Decision Tree:** `max_depth = [10, 20, None]`, `min_samples_split = [2, 10]`
- **AdaBoost:** `n_estimators = [50, 100]`, `learning_rate = [0.5, 1.0]`

For the deep learning models, I used a small manual tuning grid instead of a huge search because these models take longer to train on a normal laptop. The tuning checks practical values for sequence length, embedding size, hidden units or filters, dropout, learning rate, and epochs.

The main deep learning configurations are documented in:

```text
reports/tuning_plan.csv
reports/deep_tuning_results.csv
reports/tuning_notes.md
```

The saved Streamlit app loads the selected model files from the `models/` folder.

## Features used

I used three groups of features:

1. **TF-IDF** — word and phrase frequency.
2. **Word2Vec average embeddings** — average word vectors trained from the dataset.
3. **Linguistic features** — word count, sentence length, vocabulary richness, punctuation, uppercase ratio, digit ratio, and readability.

The traditional ML models use TF-IDF plus linguistic features. The deep learning models use tokenized word sequences.

## Saved model results

These are the results from the saved run in `reports/model_comparison.csv`:

| model         |   accuracy |   precision |   recall |    f1 |   roc_auc |
|:--------------|-----------:|------------:|---------:|------:|----------:|
| SVM           |      0.938 |       0.936 |    0.94  | 0.938 |     0.989 |
| AdaBoost      |      0.914 |       0.888 |    0.948 | 0.917 |     0.976 |
| FNN           |      0.876 |       0.87  |    0.884 | 0.877 |     0.952 |
| CNN           |      0.866 |       0.865 |    0.868 | 0.866 |     0.942 |
| Decision Tree |      0.844 |       0.812 |    0.896 | 0.852 |     0.836 |
| LSTM          |      0.756 |       0.75  |    0.768 | 0.759 |     0.792 |

## Notes on the app output

The app shows three main numbers:

- **AI detection score:** estimated probability that the text is AI-written.
- **Human score:** estimated probability that the text is human-written.
- **Confidence in result:** how confident the selected model is in the final label.

A text can be predicted as Human and still have a high confidence score. That does not mean it is AI. The AI detection score is the number to look at for that.

The LLM explanations are meant to make the result easier to understand. They do not replace the classifier score. They can also be wrong, so the final app presents them as supporting explanations instead of absolute proof.

## Test files

I included two files in `sample_docs/`:

- `human_sample.txt`
- `ai_sample.txt`

For the demo video, I would test with the SVM model first because it has the best saved score.

## My design choices

I started with TF-IDF because it is simple and strong for text classification. I added linguistic features because they make the prediction easier to explain. For the deep learning part, I used PyTorch because the model files are easy to save and load in the Streamlit app.

For Project 2, I added two smaller Hugging Face LLMs because they are practical for a public demo and can run within normal Hugging Face Spaces limits. The LLMs explain why the classifier may have predicted AI-written or human-written text, point out writing signals, and remind the user that AI detection is probabilistic.

I would not treat this app as a perfect AI detector. It is better as a comparison project that shows how different ML, DL, and LLM components behave on the same classification task.
