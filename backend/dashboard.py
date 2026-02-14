"""Streamlit dashboard for exploring EU survey-vote matches.

Run with: streamlit run backend/dashboard.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="EU Survey-Vote Correlation",
    page_icon="\U0001f1ea\U0001f1fa",
    layout="wide",
)

DATA_DIR = Path("data/classification")

# ── Data loaders ─────────────────────────────────────────────────────────────


@st.cache_data
def load_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "question_en" in df.columns and "question_text" not in df.columns:
        df = df.rename(columns={"question_en": "question_text"})
    if "summary" in df.columns and "vote_summary" not in df.columns:
        df = df.rename(columns={"summary": "vote_summary"})
    for col in ("survey_date", "vote_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "pair_valid" in df.columns:
        df["pair_valid"] = df["pair_valid"].map(
            {True: True, False: False, "True": True, "False": False}
        )
    return df


# ── Load all available datasets ──────────────────────────────────────────────

surveys_df = load_csv(str(DATA_DIR / "questions_classified.csv"))
votes_df = load_csv(str(DATA_DIR / "votes_classified.csv"))
pairs_df = load_csv(str(DATA_DIR / "pairs_validated.csv"))
enriched_df = load_csv(str(DATA_DIR / "enriched_matches.csv"))


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_surveys, tab_votes, tab_pairs, tab_overview, tab_detail = st.tabs(
    [
        "Survey Classification",
        "Vote Classification",
        "Pair Validation",
        "Overview",
        "Pair Detail",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Survey Classification
# ══════════════════════════════════════════════════════════════════════════════

with tab_surveys:
    st.header("Survey Question Classification")

    if surveys_df is None or surveys_df.empty:
        st.warning(
            "No survey classifications found. Run:\n\n"
            "```bash\nuv run python backend/scripts/run_classification.py --stage questions\n```"
        )
    else:
        # Metrics row
        total = len(surveys_df)
        n_forward = (surveys_df["question_type"] == "opinion_forward").sum()
        n_not = (surveys_df["question_type"] == "not_forward").sum()
        n_err = total - n_forward - n_not

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total classified", total)
        c2.metric("Opinion forward", n_forward)
        c3.metric("Not forward", n_not)
        c4.metric("Errors", n_err)

        # Pie chart + bar chart side by side
        col_pie, col_bar = st.columns(2)

        with col_pie:
            counts = surveys_df["question_type"].value_counts().reset_index()
            counts.columns = ["type", "count"]
            fig = px.pie(
                counts,
                values="count",
                names="type",
                title="Classification Breakdown",
                color="type",
                color_discrete_map={
                    "opinion_forward": "#2ecc71",
                    "not_forward": "#e74c3c",
                    "error": "#95a5a6",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_bar:
            # Explanation word frequency — top reasons for not_forward
            not_fwd = surveys_df[surveys_df["question_type"] == "not_forward"]
            if not not_fwd.empty and "question_type_explanation" in not_fwd.columns:
                st.subheader("Why not_forward?")
                # Show a sample of explanations
                explanations = not_fwd["question_type_explanation"].dropna().value_counts().head(10)
                fig = px.bar(
                    x=explanations.values,
                    y=explanations.index,
                    orientation="h",
                    labels={"x": "Count", "y": "Explanation"},
                    title="Top reasons for not_forward",
                )
                fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

        # Filter and browse
        st.subheader("Browse questions")
        type_filter = st.radio(
            "Show",
            ["All", "opinion_forward", "not_forward"],
            horizontal=True,
            key="survey_type_filter",
        )
        display = surveys_df.copy()
        if type_filter != "All":
            display = display[display["question_type"] == type_filter]

        st.dataframe(
            display,
            use_container_width=True,
            height=500,
            column_config={
                "question_text": st.column_config.TextColumn(
                    "Question", width="large"
                ),
                "question_type": st.column_config.TextColumn("Type", width="small"),
                "question_type_explanation": st.column_config.TextColumn(
                    "Explanation", width="medium"
                ),
            },
        )

        # Detail viewer
        st.subheader("Inspect a question")
        idx = st.number_input(
            "Row index",
            0,
            len(display) - 1,
            0,
            key="survey_detail_idx",
        )
        row = display.iloc[idx]
        st.markdown(f"**Question:** {row['question_text']}")
        if row["question_type"] == "opinion_forward":
            st.success(f"**{row['question_type']}** — {row.get('question_type_explanation', '')}")
        else:
            st.error(f"**{row['question_type']}** — {row.get('question_type_explanation', '')}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Vote Classification
# ══════════════════════════════════════════════════════════════════════════════

with tab_votes:
    st.header("EP Vote Classification")

    if votes_df is None or votes_df.empty:
        st.warning(
            "No vote classifications found. Run:\n\n"
            "```bash\nuv run python backend/scripts/run_classification.py --stage votes\n```"
        )
    else:
        total = len(votes_df)
        n_subst = (votes_df["vote_type"] == "substantive").sum()
        n_proc = (votes_df["vote_type"] == "procedural").sum()
        n_err = total - n_subst - n_proc

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total classified", total)
        c2.metric("Substantive", n_subst)
        c3.metric("Procedural", n_proc)
        c4.metric("Errors", n_err)

        col_pie, col_bar = st.columns(2)

        with col_pie:
            counts = votes_df["vote_type"].value_counts().reset_index()
            counts.columns = ["type", "count"]
            fig = px.pie(
                counts,
                values="count",
                names="type",
                title="Classification Breakdown",
                color="type",
                color_discrete_map={
                    "substantive": "#3498db",
                    "procedural": "#f39c12",
                    "error": "#95a5a6",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_bar:
            proc = votes_df[votes_df["vote_type"] == "procedural"]
            if not proc.empty and "vote_type_explanation" in proc.columns:
                st.subheader("Why procedural?")
                explanations = proc["vote_type_explanation"].dropna().value_counts().head(10)
                fig = px.bar(
                    x=explanations.values,
                    y=explanations.index,
                    orientation="h",
                    labels={"x": "Count", "y": "Explanation"},
                    title="Top reasons for procedural",
                )
                fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

        # Filter and browse
        st.subheader("Browse votes")
        type_filter = st.radio(
            "Show",
            ["All", "substantive", "procedural"],
            horizontal=True,
            key="vote_type_filter",
        )
        display = votes_df.copy()
        if type_filter != "All":
            display = display[display["vote_type"] == type_filter]

        st.dataframe(
            display,
            use_container_width=True,
            height=500,
            column_config={
                "vote_summary": st.column_config.TextColumn(
                    "Vote summary", width="large"
                ),
                "vote_id": st.column_config.NumberColumn("Vote ID"),
                "vote_type": st.column_config.TextColumn("Type", width="small"),
                "vote_type_explanation": st.column_config.TextColumn(
                    "Explanation", width="medium"
                ),
            },
        )

        # Detail viewer
        st.subheader("Inspect a vote")
        idx = st.number_input(
            "Row index",
            0,
            len(display) - 1,
            0,
            key="vote_detail_idx",
        )
        row = display.iloc[idx]
        st.markdown(f"**Vote {row.get('vote_id', '')}:** {row.get('vote_summary', '')}")
        if row["vote_type"] == "substantive":
            st.success(f"**{row['vote_type']}** — {row.get('vote_type_explanation', '')}")
        else:
            st.warning(f"**{row['vote_type']}** — {row.get('vote_type_explanation', '')}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Pair Validation
# ══════════════════════════════════════════════════════════════════════════════

with tab_pairs:
    st.header("Pair Validation Results")

    if pairs_df is None or pairs_df.empty:
        st.warning(
            "No pair validations found. Run:\n\n"
            "```bash\nuv run python backend/scripts/run_classification.py --stage pairs\n```"
        )
    else:
        total = len(pairs_df)
        n_valid = pairs_df["pair_valid"].sum() if "pair_valid" in pairs_df.columns else 0
        n_invalid = total - n_valid

        c1, c2, c3 = st.columns(3)
        c1.metric("Total validated", total)
        c2.metric("Valid", int(n_valid))
        c3.metric("Not valid", int(n_invalid))

        col_pie, col_bar = st.columns(2)

        with col_pie:
            counts = pd.DataFrame(
                {"verdict": ["Valid", "Not valid"], "count": [int(n_valid), int(n_invalid)]}
            )
            fig = px.pie(
                counts,
                values="count",
                names="verdict",
                title="Validation Breakdown",
                color="verdict",
                color_discrete_map={"Valid": "#2ecc71", "Not valid": "#e74c3c"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_bar:
            invalid = pairs_df[pairs_df["pair_valid"] == False]  # noqa: E712
            if not invalid.empty and "pair_explanation" in invalid.columns:
                st.subheader("Why not valid?")
                explanations = invalid["pair_explanation"].dropna().value_counts().head(10)
                fig = px.bar(
                    x=explanations.values,
                    y=explanations.index,
                    orientation="h",
                    labels={"x": "Count", "y": "Explanation"},
                    title="Top rejection reasons",
                )
                fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

        # Similarity distribution by validity
        if "similarity_score" in pairs_df.columns:
            st.subheader("Similarity score by validity")
            fig = px.histogram(
                pairs_df,
                x="similarity_score",
                color="pair_valid",
                nbins=30,
                barmode="overlay",
                opacity=0.7,
                color_discrete_map={True: "#2ecc71", False: "#e74c3c"},
                labels={"pair_valid": "Valid?"},
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        # Filter and browse
        st.subheader("Browse pairs")
        validity_filter = st.radio(
            "Show",
            ["All", "Valid", "Not valid"],
            horizontal=True,
            key="pair_validity_filter",
        )
        display = pairs_df.copy()
        if validity_filter == "Valid":
            display = display[display["pair_valid"] == True]  # noqa: E712
        elif validity_filter == "Not valid":
            display = display[display["pair_valid"] == False]  # noqa: E712

        q_col = "question_text" if "question_text" in display.columns else "question_en"
        v_col = "vote_summary" if "vote_summary" in display.columns else "summary"

        display_cols = [
            c
            for c in [q_col, v_col, "pair_valid", "pair_explanation", "similarity_score"]
            if c in display.columns
        ]

        st.dataframe(
            display[display_cols],
            use_container_width=True,
            height=500,
            column_config={
                q_col: st.column_config.TextColumn("Survey question", width="large"),
                v_col: st.column_config.TextColumn("Vote summary", width="large"),
                "pair_valid": st.column_config.CheckboxColumn("Valid?"),
                "similarity_score": st.column_config.NumberColumn(
                    "Similarity", format="%.3f"
                ),
            },
        )

        # Detail viewer
        st.subheader("Inspect a pair")
        idx = st.number_input(
            "Row index",
            0,
            max(len(display) - 1, 0),
            0,
            key="pair_detail_idx",
        )
        if len(display) > 0:
            row = display.iloc[idx]
            left, right = st.columns(2)
            with left:
                st.markdown(f"**Survey:** {row.get(q_col, '')}")
            with right:
                st.markdown(f"**Vote:** {row.get(v_col, '')}")
            if row.get("pair_valid"):
                st.success(f"**VALID** — {row.get('pair_explanation', '')}")
            else:
                st.error(f"**NOT VALID** — {row.get('pair_explanation', '')}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Overview (enriched matches with dates)
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.header("Enriched Matches Overview")

    if enriched_df is None or enriched_df.empty:
        st.warning(
            "No enriched matches found. Run the full pipeline:\n\n"
            "```bash\nuv run python backend/scripts/run_classification.py --resume\n```"
        )
    else:
        # Sidebar-like filters inline
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            if "question_type" in enriched_df.columns:
                q_types = sorted(enriched_df["question_type"].dropna().unique())
                sel_q = st.multiselect("Question type", q_types, default=q_types, key="ov_qt")
                enriched_df = enriched_df[enriched_df["question_type"].isin(sel_q)]
        with col_f2:
            if "vote_type" in enriched_df.columns:
                v_types = sorted(enriched_df["vote_type"].dropna().unique())
                sel_v = st.multiselect("Vote type", v_types, default=v_types, key="ov_vt")
                enriched_df = enriched_df[enriched_df["vote_type"].isin(sel_v)]
        with col_f3:
            if "pair_valid" in enriched_df.columns:
                val_choice = st.radio(
                    "Validity", ["All", "Valid", "Not valid"], horizontal=True, key="ov_val"
                )
                if val_choice == "Valid":
                    enriched_df = enriched_df[enriched_df["pair_valid"] == True]  # noqa: E712
                elif val_choice == "Not valid":
                    enriched_df = enriched_df[enriched_df["pair_valid"] == False]  # noqa: E712

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total pairs", len(enriched_df))
        if "pair_valid" in enriched_df.columns:
            valid_count = enriched_df["pair_valid"].sum()
            c2.metric("Valid", int(valid_count) if pd.notna(valid_count) else 0)
        if "survey_before_vote" in enriched_df.columns:
            c3.metric("Survey before vote", int(enriched_df["survey_before_vote"].sum()))
        if "similarity_score" in enriched_df.columns:
            c4.metric("Avg similarity", f"{enriched_df['similarity_score'].mean():.3f}")

        # Timeline scatter
        if "survey_date" in enriched_df.columns and "vote_date" in enriched_df.columns:
            dated = enriched_df.dropna(subset=["survey_date", "vote_date"]).copy()
            if not dated.empty:
                st.subheader("Timeline: Survey date vs Vote date")
                color_col = None
                if "pair_valid" in dated.columns and dated["pair_valid"].notna().any():
                    dated["validity"] = dated["pair_valid"].map(
                        {True: "Valid", False: "Not valid"}
                    ).fillna("N/A")
                    color_col = "validity"

                hover_cols = [
                    c for c in ["question_text", "vote_summary", "similarity_score"]
                    if c in dated.columns
                ]
                fig = px.scatter(
                    dated,
                    x="survey_date",
                    y="vote_date",
                    color=color_col,
                    hover_data=hover_cols,
                    color_discrete_map={
                        "Valid": "#2ecc71",
                        "Not valid": "#e74c3c",
                        "N/A": "#95a5a6",
                    },
                    opacity=0.7,
                )
                fig.add_shape(
                    type="line",
                    x0=dated["survey_date"].min(),
                    y0=dated["survey_date"].min(),
                    x1=dated["vote_date"].max(),
                    y1=dated["vote_date"].max(),
                    line=dict(color="gray", dash="dash"),
                )
                fig.update_layout(xaxis_title="Survey date", yaxis_title="Vote date", height=500)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Points above the dashed line = survey before vote (good).")

        # Table
        st.subheader("All enriched pairs")
        display_cols = [
            c
            for c in [
                "question_text",
                "vote_summary",
                "question_type",
                "vote_type",
                "pair_valid",
                "pair_explanation",
                "similarity_score",
                "survey_date",
                "vote_date",
            ]
            if c in enriched_df.columns
        ]
        st.dataframe(
            enriched_df[display_cols],
            use_container_width=True,
            height=500,
            column_config={
                "question_text": st.column_config.TextColumn("Survey", width="large"),
                "vote_summary": st.column_config.TextColumn("Vote", width="large"),
                "similarity_score": st.column_config.NumberColumn("Sim", format="%.3f"),
                "survey_date": st.column_config.DateColumn("Survey date"),
                "vote_date": st.column_config.DateColumn("Vote date"),
                "pair_valid": st.column_config.CheckboxColumn("Valid?"),
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Pair Detail
# ══════════════════════════════════════════════════════════════════════════════

with tab_detail:
    st.header("Pair Detail Inspector")

    source = enriched_df if enriched_df is not None and not enriched_df.empty else pairs_df
    if source is None or source.empty:
        st.info("No data available yet.")
    else:
        q_col = "question_text" if "question_text" in source.columns else "question_en"
        v_col = "vote_summary" if "vote_summary" in source.columns else "summary"

        pair_idx = st.number_input(
            "Pair index",
            min_value=0,
            max_value=len(source) - 1,
            value=0,
            step=1,
            key="detail_idx",
        )
        row = source.iloc[pair_idx]

        left, right = st.columns(2)

        with left:
            st.subheader("Survey Question")
            st.markdown(f"**{row.get(q_col, 'N/A')}**")
            if "file_name" in row.index:
                st.caption(f"Source: {row['file_name']}")
            if "survey_date" in row.index and pd.notna(row.get("survey_date")):
                st.caption(f"Date: {row['survey_date'].strftime('%Y-%m-%d')}")
            if "question_type" in row.index:
                if row["question_type"] == "opinion_forward":
                    st.success(f"Type: {row['question_type']}")
                else:
                    st.warning(f"Type: {row['question_type']}")
                if "question_type_explanation" in row.index:
                    st.caption(row["question_type_explanation"])

        with right:
            st.subheader("EP Vote")
            st.markdown(f"**{row.get(v_col, 'N/A')}**")
            if "vote_id" in row.index:
                st.caption(f"Vote ID: {row['vote_id']}")
            if "vote_date" in row.index and pd.notna(row.get("vote_date")):
                st.caption(f"Date: {row['vote_date'].strftime('%Y-%m-%d')}")
            if "vote_type" in row.index:
                if row["vote_type"] == "substantive":
                    st.success(f"Type: {row['vote_type']}")
                else:
                    st.warning(f"Type: {row['vote_type']}")
                if "vote_type_explanation" in row.index:
                    st.caption(row["vote_type_explanation"])

        st.markdown("---")

        col_sim, col_valid = st.columns(2)
        if "similarity_score" in row.index:
            col_sim.metric("Similarity", f"{row['similarity_score']:.3f}")
        if "pair_valid" in row.index and pd.notna(row.get("pair_valid")):
            if row["pair_valid"]:
                col_valid.metric("Verdict", "VALID")
                st.success(f"**Valid**: {row.get('pair_explanation', '')}")
            else:
                col_valid.metric("Verdict", "NOT VALID")
                st.error(f"**Not valid**: {row.get('pair_explanation', '')}")
        else:
            col_valid.metric("Verdict", "---")

        if (
            "survey_date" in row.index
            and "vote_date" in row.index
            and pd.notna(row.get("survey_date"))
            and pd.notna(row.get("vote_date"))
        ):
            delta = (row["vote_date"] - row["survey_date"]).days
            if delta > 0:
                st.caption(f"Survey published **{delta} days before** the vote.")
            else:
                st.caption(f"Vote happened **{abs(delta)} days before** the survey.")
