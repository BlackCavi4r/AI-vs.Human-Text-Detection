# Project 2 Submission Checklist

Use this checklist before submitting on Canvas.

## Required by the Assignment

- [x] Continue Project 1 idea.
- [x] Keep the original machine/deep learning pipeline working.
- [x] Add at least two Hugging Face LLMs.
- [x] Use Streamlit or Gradio.
- [x] Support direct text input.
- [x] Support document upload.
- [x] Show classifier prediction.
- [x] Show model scores and explanation signals.
- [x] Add LLM-generated explanation.
- [x] Include `requirements.txt`.
- [x] Include `README.md`.
- [x] Prepare for Hugging Face Spaces deployment.
- [ ] Add the final public Hugging Face Spaces URL after deployment.
- [ ] Record and submit a 5-10 minute demo video.

## Files to Include in the Canvas ZIP

- `app.py`
- `llm_explanations.py`
- `project_utils.py`
- `train_models.py`
- `requirements.txt`
- `requirements-training.txt`
- `README.md`
- `DEMO_SCRIPT.md`
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

Then create a Streamlit Space from the Hugging Face website or with the CLI. After the Space is live, update the README with the public Space URL.
