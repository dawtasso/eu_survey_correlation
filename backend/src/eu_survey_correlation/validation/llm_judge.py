"""LLM-based judge for evaluating survey-vote match quality using Ollama + Mistral."""

import json
import re
from pathlib import Path

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
        output_path: str | Path | None = None,
        already_judged: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Judge all pairs in a DataFrame.

        Adds columns: llm_score, llm_explanation, llm_go.

        Args:
            df: DataFrame with match pairs.
            question_col: Column name for survey question text.
            summary_col: Column name for vote summary text.
            output_path: If provided, save results to this CSV after each judgment.
            already_judged: DataFrame of previously judged pairs to prepend to output.

        Returns:
            Copy of df with added LLM judgment columns.
        """
        logger.info(f"Judging {len(df)} match pairs with '{self.model}'...")

        # Prepare output path
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize results list with already judged pairs
        judged_rows = []
        if already_judged is not None and not already_judged.empty:
            judged_rows = already_judged.to_dict("records")

        df_reset = df.reset_index(drop=True)

        for idx, row in tqdm(
            df_reset.iterrows(), total=len(df_reset), desc="Judging pairs"
        ):
            judgment = self.judge_pair(
                question_text=str(row[question_col]),
                vote_summary=str(row[summary_col]),
            )

            # Create row with judgment columns
            row_dict = row.to_dict()
            row_dict["llm_score"] = judgment["score"]
            row_dict["llm_explanation"] = judgment["explanation"]
            row_dict["llm_go"] = judgment["go"]
            judged_rows.append(row_dict)

            # Save incrementally after each judgment
            if output_path:
                pd.DataFrame(judged_rows).to_csv(output_path, index=False)

        result_df = pd.DataFrame(judged_rows)

        # Filter to only newly judged rows for stats
        new_judgments = result_df.tail(len(df_reset))
        n_go = new_judgments["llm_go"].sum()
        logger.success(
            f"Judging complete: {n_go}/{len(new_judgments)} pairs marked as 'go' "
            f"(avg score: {new_judgments['llm_score'].mean():.1f})"
        )
        return result_df
