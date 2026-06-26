import unittest

import pandas as pd

from llm_explanations import (
    LLM_OPTIONS,
    build_llm_prompt,
    format_llm_outputs_for_report,
    trim_text_for_prompt,
)


class LLMExplanationTests(unittest.TestCase):
    def test_three_llm_options_are_available(self):
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", LLM_OPTIONS)
        self.assertIn("HuggingFaceTB/SmolLM2-360M-Instruct", LLM_OPTIONS)
        self.assertIn("TinyLlama/TinyLlama-1.1B-Chat-v1.0", LLM_OPTIONS)
        self.assertEqual(len(LLM_OPTIONS), 3)

    def test_trim_text_for_prompt_short_text_is_unchanged(self):
        text = "This is a short paragraph."

        result = trim_text_for_prompt(text, max_chars=100)

        self.assertEqual(result, text)

    def test_trim_text_for_prompt_long_text_keeps_limit_and_marks_truncation(self):
        text = "a" * 250

        result = trim_text_for_prompt(text, max_chars=80)

        self.assertLessEqual(len(result), 120)
        self.assertIn("Text was truncated", result)
        self.assertTrue(result.startswith("a" * 80))

    def test_build_llm_prompt_includes_prediction_scores_and_stats(self):
        stats_df = pd.DataFrame(
            [
                {"signal": "Word count", "value": 120},
                {"signal": "Vocabulary richness", "value": 0.72},
            ]
        )
        comparison_df = pd.DataFrame(
            [
                {"model": "SVM", "prediction": "AI-written", "ai_detection_score": 0.91},
                {"model": "CNN", "prediction": "Human-written", "ai_detection_score": 0.42},
            ]
        )

        prompt = build_llm_prompt(
            text="The sample text to evaluate.",
            classifier_name="SVM",
            prediction_label="AI-written",
            prob_ai=0.91,
            confidence=0.91,
            stats_df=stats_df,
            comparison_df=comparison_df,
        )

        self.assertIn("SVM", prompt)
        self.assertIn("AI-written", prompt)
        self.assertIn("91.00%", prompt)
        self.assertIn("Word count: 120", prompt)
        self.assertIn("CNN: Human-written", prompt)
        self.assertIn("The sample text to evaluate.", prompt)

    def test_format_llm_outputs_for_report_includes_each_model_output(self):
        outputs = {
            "Qwen/Qwen2.5-0.5B-Instruct": "Explanation two.",
            "HuggingFaceTB/SmolLM2-360M-Instruct": "Explanation three.",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "Explanation four.",
        }

        report = format_llm_outputs_for_report(outputs)

        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", report)
        self.assertIn("Explanation two.", report)
        self.assertIn("HuggingFaceTB/SmolLM2-360M-Instruct", report)
        self.assertIn("Explanation three.", report)
        self.assertIn("TinyLlama/TinyLlama-1.1B-Chat-v1.0", report)
        self.assertIn("Explanation four.", report)


if __name__ == "__main__":
    unittest.main()
