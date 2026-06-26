# Project 2 Submission Checklist

Use this checklist before submitting on Canvas.

## Required by the Assignment

- [x] Continue Project 1 idea.
- [x] Keep the original machine/deep learning pipeline working.
- [x] Add at least two Hugging Face LLMs. This version includes three.
- [x] Use Streamlit or Gradio.
- [x] Support direct text input.
- [x] Support document upload.
- [x] Show classifier prediction.
- [x] Show model scores and explanation signals.
- [x] Add LLM-generated explanation.
- [x] Include `requirements.txt`.
- [x] Include `README.md`.
- [x] Prepare for Hugging Face Spaces deployment.
- [x] Add the final public Hugging Face Spaces URL after deployment.
- [ ] Record and submit a 5-10 minute demo video.

## Files to Include in the Canvas ZIP

- `app.py`
- `llm_explanations.py`
- `project_utils.py`
- `train_models.py`
- `Dockerfile`
- `requirements.txt`
- `requirements-training.txt`
- `README.md`
- `SUBMISSION_CHECKLIST.md`
- `models/`
- `reports/`
- `sample_docs/`
- `data/`
- `notebooks/`
- `tests/`

Do not include:

- `.venv/`
- `.git/`
- `.ruff_cache/`
- `__pycache__/`
- local Hugging Face cache files

## Deployment Notes

The Hugging Face CLI is not currently logged in on this machine. To deploy from the terminal, log in first:

```powershell
.\.venv\Scripts\hf.exe auth login
```

Then create a Docker Space from the Hugging Face website or with the CLI. The Docker Space runs the Streamlit app on port `7860`.

Suggested CLI flow:

```powershell
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\hf.exe repo create Caviar22/AI-vs-Human-Text-Detection --repo-type space --space_sdk docker --exist-ok
.\.venv\Scripts\hf.exe upload Caviar22/AI-vs-Human-Text-Detection . . --repo-type space --commit-message "Deploy Project 2 Streamlit app" --exclude ".git/*" ".venv/*" ".ruff_cache/*" "__pycache__/*" "*.pyc" "BlackCavi4r_AI-vs.Human-Text-Detection.mp4" "outputs/*"
```

Expected public URL:

```text
https://huggingface.co/spaces/Caviar22/AI-vs-Human-Text-Detection
```
