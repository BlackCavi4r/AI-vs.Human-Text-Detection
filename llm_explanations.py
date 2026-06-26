from __future__ import annotations

from functools import lru_cache
from typing import Mapping


LLM_OPTIONS = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "display_name": "Qwen2.5 0.5B Instruct",
        "task": "text-generation",
        "purpose": "Structured explanation of the classifier result",
    },
    "HuggingFaceTB/SmolLM2-360M-Instruct": {
        "display_name": "SmolLM2 360M Instruct",
        "task": "text-generation",
        "purpose": "Second-opinion writing-style review",
    },
}


def trim_text_for_prompt(text: object, max_chars: int = 1800) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "\n\n[Text was truncated.]"


def _format_stats(stats_df) -> str:
    if stats_df is None or getattr(stats_df, "empty", True):
        return "No text statistics were available."

    lines = []
    for _, row in stats_df.iterrows():
        signal = row.get("signal", row.iloc[0])
        value = row.get("value", row.iloc[1] if len(row) > 1 else "")
        lines.append(f"- {signal}: {value}")
    return "\n".join(lines)


def _format_model_comparison(comparison_df, max_rows: int = 6) -> str:
    if comparison_df is None or getattr(comparison_df, "empty", True):
        return "No side-by-side model comparison was available."

    lines = []
    for _, row in comparison_df.head(max_rows).iterrows():
        model = row.get("model", "Unknown model")
        prediction = row.get("prediction", "Unknown prediction")
        score = row.get("ai_detection_score", row.get("ai_probability", "n/a"))
        if isinstance(score, float):
            score_text = f"{score:.2%}"
        else:
            score_text = str(score)
        lines.append(f"- {model}: {prediction} (AI score: {score_text})")
    return "\n".join(lines)


def build_llm_prompt(
    text: str,
    classifier_name: str,
    prediction_label: str,
    prob_ai: float,
    confidence: float,
    stats_df,
    comparison_df,
) -> str:
    text_preview = trim_text_for_prompt(text)
    stats_summary = _format_stats(stats_df)
    comparison_summary = _format_model_comparison(comparison_df)

    return f"""Explain this AI vs. human text detection result in 3 short bullet points for a non-technical user.

Classifier result:
- Selected classifier: {classifier_name}
- Prediction: {prediction_label}
- AI detection score: {prob_ai:.2%}
- Human score: {1 - prob_ai:.2%}
- Confidence in predicted label: {confidence:.2%}

Text statistics:
{stats_summary}

Other model predictions:
{comparison_summary}

Text sample to review. Do not quote it back:
{text_preview}

Include:
1. Why the classifier may have predicted this label.
2. Which writing signals support or weaken the result.
3. A short caution that AI detectors are probabilistic and can be wrong.
Do not copy the scores, statistics, or text sample back to the user. Use them only as evidence.
Keep the response under 180 words.

Answer with exactly 3 bullets:"""


@lru_cache(maxsize=2)
def load_llm_pipeline(model_id: str):
    if model_id not in LLM_OPTIONS:
        raise ValueError(f"Unsupported LLM model: {model_id}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
    except ImportError as exc:
        raise RuntimeError(
            "LLM dependencies are not installed. Install transformers and sentencepiece from requirements.txt."
        ) from exc

    option = LLM_OPTIONS[model_id]
    task = option["task"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if task == "text-generation":
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return pipeline(task, model=model, tokenizer=tokenizer, device=-1)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return pipeline(task, model=model, tokenizer=tokenizer, device=-1)


def _render_generation_prompt(model_id: str, prompt: str, pipe) -> str:
    if LLM_OPTIONS[model_id]["task"] != "text-generation":
        return prompt

    tokenizer = getattr(pipe, "tokenizer", None)
    chat_template = getattr(tokenizer, "chat_template", None)
    if not tokenizer or not chat_template:
        return prompt

    messages = [
        {
            "role": "system",
            "content": "You explain text classification results clearly and cautiously.",
        },
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_llm_explanation(
    model_id: str,
    text: str,
    classifier_name: str,
    prediction_label: str,
    prob_ai: float,
    confidence: float,
    stats_df,
    comparison_df,
    max_new_tokens: int = 180,
) -> str:
    pipe = load_llm_pipeline(model_id)
    prompt = build_llm_prompt(
        text=text,
        classifier_name=classifier_name,
        prediction_label=prediction_label,
        prob_ai=prob_ai,
        confidence=confidence,
        stats_df=stats_df,
        comparison_df=comparison_df,
    )
    rendered_prompt = _render_generation_prompt(model_id, prompt, pipe)
    task = LLM_OPTIONS[model_id]["task"]

    if task == "text-generation":
        tokenizer = getattr(pipe, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
        result = pipe(
            rendered_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            return_full_text=False,
            pad_token_id=pad_token_id,
        )
    else:
        result = pipe(rendered_prompt, max_new_tokens=max_new_tokens, do_sample=False)

    if not result:
        return "The selected LLM did not return an explanation."

    generated = result[0].get("generated_text") or result[0].get("summary_text") or ""
    return generated.strip() or "The selected LLM returned an empty explanation."


def format_llm_outputs_for_report(outputs: Mapping[str, str]) -> str:
    if not outputs:
        return ""

    lines = ["LLM explanations", "-" * 16]
    for model_id, explanation in outputs.items():
        lines.append(model_id)
        lines.append("~" * len(model_id))
        lines.append(str(explanation).strip() or "No explanation was generated.")
        lines.append("")
    return "\n".join(lines).rstrip()


def model_label(model_id: str) -> str:
    option = LLM_OPTIONS.get(model_id, {})
    display = option.get("display_name", model_id)
    purpose = option.get("purpose", "")
    return f"{display} - {purpose}" if purpose else display
