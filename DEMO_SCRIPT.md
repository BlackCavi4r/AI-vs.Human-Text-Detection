# Project 2 Demo Script

Target length: 5 to 10 minutes.

## 1. Opening

Hi, my project is called AI vs. Human Text Detection. The goal is to take a writing sample and estimate whether it looks more like human-written text or AI-written text. This is not meant to be final proof. It is a decision-support tool that shows model scores, writing signals, and LLM explanations.

## 2. Project 1 Foundation

For Project 1, I trained several classifiers:

- SVM
- Decision Tree
- AdaBoost
- FNN
- LSTM
- CNN

The best saved model was SVM, with an F1 score of about 0.938 and ROC AUC of about 0.989. The traditional models use TF-IDF and linguistic features. The deep learning models use tokenized text sequences.

## 3. Input Options

The app lets the user either paste text directly or upload a document. It supports TXT, PDF, and Word DOCX files. For the demo, I will start with the sample AI text or paste a paragraph directly into the text box.

## 4. Classifier Prediction

After text is added, the app shows:

- The predicted label
- AI detection score
- Human score
- Confidence in the predicted label
- Text statistics

The AI detection score is the main number for how AI-like the text appears. Confidence is how strongly the selected model believes its final label.

## 5. Model Comparison

The app also runs the same text through the other saved models. This is useful because it shows whether the models agree or disagree. If several models point in the same direction, the result is more convincing. If they disagree, that tells the user to be more careful.

## 6. Project 2 LLM Extension

For Project 2, I added two Hugging Face LLMs:

- Qwen2.5 0.5B Instruct
- SmolLM2 360M Instruct

The app has two modes:

- Fast mode runs one selected LLM.
- Full mode runs both LLMs so the explanations can be compared.

The LLMs do not replace the classifier. They explain why the classifier may have made the prediction, which writing signals matter, and why the user should treat the result as probabilistic.

## 7. Downloadable Report

After the prediction and optional LLM explanation, the app can download a text report. The report includes the selected model, prediction, scores, text statistics, model comparison, LLM explanation, and a preview of the analyzed text.

## 8. What I Learned

The main thing I learned is that prediction and explanation are different. The SVM model gave the strongest score, but the LLMs made the result easier to understand. I also learned that AI detection should be presented carefully. The app should not say "this is definitely AI" or "this is definitely human." It should show evidence and uncertainty.

## 9. Closing

This completes my Project 2 app. It keeps the Project 1 classifiers, adds two meaningful LLM integrations, provides a Streamlit interface, and is prepared for deployment on Hugging Face Spaces.
