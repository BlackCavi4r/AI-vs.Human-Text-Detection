# Project 1 — AI vs. Human Text Detection

This is my Project 1 app for detecting whether a text sample looks human-written or AI-written.

The app lets a user paste text or upload a PDF, Word document, or TXT file. Then the user can choose one of the trained models and get a prediction, an AI score, a human score, text statistics, a simple explanation, model comparison, and a downloadable report.

Labels used in the dataset:

- `0` = Human-written text
- `1` = AI-written text

## Folder structure

```text
ai_human_detection_project/
├── app.py
├── train_models.py
├── project_utils.py
├── requirements.txt
├── README.md
├── models/
├── data/
│   ├── training_data/
│   └── test_data/
├── notebooks/
│   └── project1_notebook.ipynb
├── reports/
│   └── figures/
└── sample_docs/
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

## Test files

I included two files in `sample_docs/`:

- `human_sample.txt`
- `ai_sample.txt`

For the demo video, I would test with the SVM model first because it has the best saved score.

## My design choices

I started with TF-IDF because it is simple and strong for text classification. I added linguistic features because they make the prediction easier to explain. For the deep learning part, I used PyTorch because the model files are easy to save and load in the Streamlit app.

I would not treat this app as a perfect AI detector. It is better as a comparison project that shows how different ML and DL models behave on the same classification task.
