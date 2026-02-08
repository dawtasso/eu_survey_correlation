"""LLM-based judge for evaluating survey-vote match quality using Ollama + Mistral."""

import json
import re

import ollama
import pandas as pd
from loguru import logger
from tqdm import tqdm

DEFAULT_MODEL = "mistral"

JUDGE_PROMPT = """\
You are an expert analyst evaluating whether a European Parliament vote is \
thematically related to an EU survey question.

SURVEY QUESTION:
{question_text}

VOTE SUMMARY:
{vote_summary}

Evaluate whether this vote is genuinely related to the survey question's topic.
Respond ONLY with valid JSON (no markdown, no extra text):
{{"score": <1-10>, "explanation": "<one sentence>", "go": <true or false>}}

Rules:
- score 1-3: unrelated topics
- score 4-6: loosely related (same broad domain but different focus)
- score 7-10: clearly related (vote directly addresses the survey topic)
- "go" should be true only if score >= 7
"""


class MatchJudge:
    """Evaluate survey-vote match pairs using a local LLM via Ollama."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        logger.info(f"Initializing MatchJudge with model '{model}'")
        # Verify the model is available
        try:
            ollama.show(model)
            logger.success(f"Model '{model}' is available")
        except ollama.ResponseError:
            logger.warning(
                f"Model '{model}' not found locally. "
                f"Run 'ollama pull {model}' to download it."
            )
            raise

    def judge_pair(self, question_text: str, vote_summary: str) -> dict:
        """
        Judge a single survey question / vote summary pair.

        Returns:
            dict with keys: score (int), explanation (str), go (bool).
            On failure: score=-1, explanation="parse error", go=False.
        """
        prompt = JUDGE_PROMPT.format(
            question_text=question_text,
            vote_summary=vote_summary,
        )

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            content = response["message"]["content"].strip()
            return self._parse_response(content)
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return {"score": -1, "explanation": f"llm error: {e}", "go": False}

    def _parse_response(self, content: str) -> dict:
        """Parse LLM JSON response, with fallback regex extraction."""
        # Try direct JSON parse
        try:
            result = json.loads(content)
            return self._validate_result(result)
        except json.JSONDecodeError:
            pass

        # Fallback: extract JSON from markdown code blocks or surrounding text
        json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return self._validate_result(result)
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse LLM response: {content[:200]}")
        return {"score": -1, "explanation": "parse error", "go": False}

    def _validate_result(self, result: dict) -> dict:
        """Ensure result has the expected keys and types."""
        score = int(result.get("score", -1))
        explanation = str(result.get("explanation", ""))
        go = bool(result.get("go", False))
        return {"score": score, "explanation": explanation, "go": go}

    def judge_dataframe(
        self,
        df: pd.DataFrame,
        question_col: str = "question_text",
        summary_col: str = "vote_summary",
    ) -> pd.DataFrame:
        """
        Judge all pairs in a DataFrame.

        Adds columns: llm_score, llm_explanation, llm_go.

        Args:
            df: DataFrame with match pairs.
            question_col: Column name for survey question text.
            summary_col: Column name for vote summary text.

        Returns:
            Copy of df with added LLM judgment columns.
        """
        results = []
        logger.info(f"Judging {len(df)} match pairs with '{self.model}'...")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Judging pairs"):
            judgment = self.judge_pair(
                question_text=str(row[question_col]),
                vote_summary=str(row[summary_col]),
            )
            results.append(judgment)

        judgments_df = pd.DataFrame(results)
        result_df = df.copy().reset_index(drop=True)
        result_df["llm_score"] = judgments_df["score"]
        result_df["llm_explanation"] = judgments_df["explanation"]
        result_df["llm_go"] = judgments_df["go"]

        n_go = result_df["llm_go"].sum()
        logger.success(
            f"Judging complete: {n_go}/{len(result_df)} pairs marked as 'go' "
            f"(avg score: {result_df['llm_score'].mean():.1f})"
        )
        return result_df

