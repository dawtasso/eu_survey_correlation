"""Consolidates all LLM-based text simplification (survey questions and vote summaries)."""

import asyncio
import json
from pathlib import Path

import ollama
import pandas as pd
from loguru import logger
from ollama import AsyncClient
from tqdm import tqdm

DEFAULT_MODEL = "mistral"


class Simplifier:
    """Simplify text via a local LLM (Ollama) with JSON-file caching."""

    SURVEY_QUESTION_PROMPT = """\
Rewrite this survey question into a short, clear, easy-to-read question.

Rules:
- Remove any question ID prefix (e.g. "QA2.1", "QB3_2", "Q1_1")
- Remove "Base: ..." filters and technical instructions
- Remove "MULTIPLE ANSWERS POSSIBLE" and similar instructions
- Remove "For each of the following..." preambles when possible — keep only the specific item
- Remove CARD references (e.g. "CARD 5", "STILL CARD 8")
- Remove interviewer instructions (e.g. "READ OUT", "ASK ALL", "ASK IF")
- Remove scale descriptions (e.g. "using a scale from 0 to 10")
- Keep the core meaning intact
- Output ONLY the rewritten question, nothing else
- If the question asks about a specific sub-item (after a colon or dash), include both the main topic and the sub-item
- Maximum 1-2 sentences
- Do NOT answer the question, just rewrite it

Examples:
Input: "QA2.1. To what extent do you agree or disagree that each of the following actors is doing enough to ensure that the green transition is fair? :-Your regional or local authorities"
Output: Are your regional or local authorities doing enough to ensure a fair green transition?

Input: "QB6.4. What is your opinion on each of the following statements? Please tell for each statement, whether you are for it or against it. :-A common defence and security policy among EU Member States"
Output: Are you for or against a common defence and security policy among EU Member States?

Input: "Q5_3 To what extent do you agree or disagree with each of the following statements? Everyone should get vaccinated against COVID-19 | Base: All"
Output: Should everyone get vaccinated against COVID-19?

Now rewrite this question:
{text}"""

    VOTE_SUMMARY_PROMPT = """\
Rewrite this European Parliament vote summary into a short, clear description.

Rules:
- Maximum 2 sentences
- Focus on WHAT was voted on and the outcome (passed/rejected)
- Remove procedural details (vote counts, amendment numbers, article references)
- Remove group names (EPP, S&D, etc.) unless essential
- Keep the core policy topic clear
- Output ONLY the rewritten summary, nothing else

Vote summary:
{text}"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_path: Path | None = None,
    ):
        self.model = model
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path) as f:
                self._cache = json.load(f)
            logger.info(
                f"Loaded {len(self._cache)} cached simplifications from {self.cache_path}"
            )

    def _save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)

    def simplify(self, text: str, prompt_template: str) -> str:
        """Simplify a single text using the given prompt template.

        The template must contain a ``{text}`` placeholder.
        Results are cached by input text.
        """
        if not text or text == "nan" or len(text.strip()) < 10:
            return text

        if text in self._cache:
            return self._cache[text]

        prompt = prompt_template.format(text=text)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            result = response["message"]["content"].strip()
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            result = text[:200]

        self._cache[text] = result
        return result

    async def _simplify_one(
        self,
        text: str,
        prompt_template: str,
        client: AsyncClient,
        semaphore: asyncio.Semaphore,
    ) -> str:
        if not text or text == "nan" or len(text.strip()) < 10:
            return text
        if text in self._cache:
            return self._cache[text]

        prompt = prompt_template.format(text=text[:1500])
        async with semaphore:
            try:
                response = await client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.0},
                )
                result = response["message"]["content"].strip()
                if result.startswith('"') and result.endswith('"'):
                    result = result[1:-1]
            except Exception as e:
                logger.error(f"Ollama call failed: {e}")
                result = text[:200]

        self._cache[text] = result
        return result

    def simplify_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        output_column: str,
        prompt_template: str,
        save_path: Path | None = None,
        concurrency: int = 4,
    ) -> pd.DataFrame:
        """Simplify a text column in a DataFrame using async concurrency.

        Runs up to *concurrency* Ollama requests in parallel. Cached results
        are returned immediately without consuming a slot.

        Args:
            df: Source DataFrame (not modified in-place).
            text_column: Column containing the text to simplify.
            output_column: Column name for the simplified text.
            prompt_template: Prompt with ``{text}`` placeholder.
            save_path: Optional CSV path to save the result.
            concurrency: Max parallel Ollama requests (default 4).

        Returns:
            DataFrame with the new *output_column* added.
        """
        df = df.copy()
        texts = [str(v) for v in df[text_column]]

        # Identify which texts actually need LLM calls
        to_compute = [
            t
            for t in texts
            if t not in self._cache and t and t != "nan" and len(t.strip()) >= 10
        ]
        cached = len(texts) - len(to_compute)
        if cached:
            logger.info(f"Cache hit for {cached}/{len(texts)} rows")
        if not to_compute:
            df[output_column] = [self._cache.get(t, t) for t in texts]
            logger.success(f"All {len(df)} rows served from cache")
            if save_path:
                df.to_csv(save_path, index=False)
            return df

        logger.info(
            f"Simplifying {len(to_compute)} texts (concurrency={concurrency})..."
        )

        async def _run() -> None:
            client = AsyncClient()
            sem = asyncio.Semaphore(concurrency)
            pbar = tqdm(total=len(to_compute), desc="Simplifying")

            async def _process(text: str) -> None:
                await self._simplify_one(text, prompt_template, client, sem)
                pbar.update(1)

            await asyncio.gather(*[_process(t) for t in to_compute])
            pbar.close()

        # Run the event loop (works in scripts and notebooks via nest_asyncio)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            loop.run_until_complete(_run())
        else:
            asyncio.run(_run())

        # Map results from cache (all texts should be cached now)
        df[output_column] = [self._cache.get(t, t) for t in texts]

        if save_path:
            df.to_csv(save_path, index=False)
        self._save_cache()

        logger.success(
            f"Simplified {len(df)} rows ({len(to_compute)} new, {cached} cached)"
        )
        return df
