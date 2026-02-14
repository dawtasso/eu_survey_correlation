"""End-to-end classification pipeline orchestrator.

Flow:
    1. Classify ALL surveys → keep opinion_forward
    2. Classify ALL votes   → keep substantive
    3. Semantic match filtered surveys × filtered votes
    4. Validate matched pairs → valid / not_valid
    5. Enrich with dates
    6. Save enriched_matches.csv
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from .classifier import LLMClient, PairValidator, SurveyClassifier, VoteClassifier
from .dates import enrich_with_dates
from .matcher import SemanticMatcher

DEFAULT_MODEL = "mistral"


class ClassificationPipeline:
    """Orchestrates the full classify → match → validate pipeline."""

    def __init__(
        self,
        output_dir: Path = Path("data/classification"),
        model: str = DEFAULT_MODEL,
        surveys_path: Path = Path("data/surveys/all_survey_questions.csv"),
        votes_path: Path = Path("data/votes/vote_summaries.csv"),
        metadata_path: Path = Path("data/surveys/distributions_metadata.json"),
        votes_csv_path: Path = Path("data/votes/votes.csv"),
        top_k: int = 5,
        threshold: float = 0.5,
    ):
        self.output_dir = Path(output_dir)
        self.surveys_path = Path(surveys_path)
        self.votes_path = Path(votes_path)
        self.metadata_path = Path(metadata_path)
        self.votes_csv_path = Path(votes_csv_path)
        self.top_k = top_k
        self.threshold = threshold

        # Lazy-init LLM client and sub-components
        self._llm: LLMClient | None = None
        self._model = model

        self.survey_classifier = SurveyClassifier(self._get_llm(), self.output_dir)
        self.vote_classifier = VoteClassifier(self._get_llm(), self.output_dir)
        self.pair_validator = PairValidator(self._get_llm(), self.output_dir)
        self.matcher = SemanticMatcher(
            top_k=self.top_k, threshold=self.threshold
        )

    def _get_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(model=self._model)
        return self._llm

    def run(
        self,
        *,
        resume: bool = False,
        limit: int | None = None,
        stage: str = "all",
    ) -> pd.DataFrame:
        """Run the pipeline (or a specific stage).

        Parameters
        ----------
        resume : Skip already-processed items in each stage.
        limit : Process at most N items per stage (for testing).
        stage : One of "all", "questions", "votes", "match", "pairs".

        Returns
        -------
        The final enriched DataFrame (empty if only running a partial stage).
        """
        # ── Stage 1: Classify surveys ──────────────────────────────────
        if stage in ("all", "questions"):
            surveys_df = pd.read_csv(self.surveys_path)
            logger.info(f"Loaded {len(surveys_df)} surveys from {self.surveys_path}")
            self.survey_classifier.classify(surveys_df, resume=resume, limit=limit)

        if stage == "questions":
            return self.survey_classifier.load_results()

        # ── Stage 2: Classify votes ────────────────────────────────────
        if stage in ("all", "votes"):
            votes_df = pd.read_csv(self.votes_path)
            logger.info(f"Loaded {len(votes_df)} votes from {self.votes_path}")
            self.vote_classifier.classify(votes_df, resume=resume, limit=limit)

        if stage == "votes":
            return self.vote_classifier.load_results()

        # ── Stage 3: Semantic matching on filtered subsets ─────────────
        if stage in ("all", "match"):
            forward_qs = self.survey_classifier.get_forward_looking()
            substantive_vs = self.vote_classifier.get_substantive()

            logger.info(
                f"Matching {len(forward_qs)} opinion_forward surveys "
                f"× {len(substantive_vs)} substantive votes"
            )

            matches_df = self.matcher.match(forward_qs, substantive_vs)
            matches_path = self.output_dir / "filtered_matches.csv"
            self.matcher.save_matches(matches_df, matches_path)

        if stage == "match":
            return pd.read_csv(self.output_dir / "filtered_matches.csv")

        # ── Stage 4: Validate matched pairs ────────────────────────────
        if stage in ("all", "pairs"):
            matches_path = self.output_dir / "filtered_matches.csv"
            if not matches_path.exists():
                logger.error(
                    f"No matches at {matches_path}. Run --stage match first."
                )
                return pd.DataFrame()

            matches_df = pd.read_csv(matches_path)
            if matches_df.empty:
                logger.warning("No candidate pairs to validate (empty matches file)")
                return pd.DataFrame()

            logger.info(f"Loaded {len(matches_df)} candidate pairs for validation")
            validated = self.pair_validator.validate(
                matches_df, resume=resume, limit=limit
            )

        # ── Stage 5: Enrich with dates and merge everything ────────────
        validated = self.pair_validator.load_results()

        # Merge question + vote classification metadata
        q_results = self.survey_classifier.load_results()
        v_results = self.vote_classifier.load_results()

        enriched = validated.merge(
            q_results[["question_en", "question_type", "question_type_explanation"]],
            on="question_en",
            how="left",
        )
        enriched = enriched.merge(
            v_results[["vote_id", "vote_type", "vote_type_explanation"]],
            on="vote_id",
            how="left",
        )

        # Add dates
        enriched = enrich_with_dates(
            enriched, self.metadata_path, self.votes_csv_path
        )

        # Save final output
        final_path = self.output_dir / "enriched_matches.csv"
        enriched.to_csv(final_path, index=False)
        logger.success(f"Pipeline complete → {len(enriched)} rows saved to {final_path}")

        self._log_summary(enriched)
        return enriched

    @staticmethod
    def _log_summary(df: pd.DataFrame) -> None:
        logger.info("─── Pipeline Summary ───")
        logger.info(f"Total pairs: {len(df)}")

        if "pair_valid" in df.columns:
            n_valid = df["pair_valid"].sum()
            logger.info(f"Valid pairs: {n_valid}/{len(df)}")

        if "survey_before_vote" in df.columns:
            n_temporal = df["survey_before_vote"].sum()
            logger.info(f"Survey before vote: {n_temporal}/{len(df)}")

        if "similarity_score" in df.columns:
            logger.info(
                f"Similarity: mean={df['similarity_score'].mean():.3f}, "
                f"median={df['similarity_score'].median():.3f}"
            )
