"""Streamlit app to review labeled survey-vote pairs.

Usage:
    uv run streamlit run streamlits/review_pairs.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

DATA_PATH = "data/training/labeled_pairs.csv"

st.set_page_config(page_title="Pair Review", layout="wide")
st.title("Survey-Vote Pair Review")


def load_data():
    return pd.read_csv(DATA_PATH)


if st.sidebar.button("Reload data"):
    st.cache_data.clear()

df = load_data()

# --- Sidebar filters ---
st.sidebar.header("Filters")

labels = ["all"] + sorted(df["llm_label"].dropna().unique().tolist())
selected_label = st.sidebar.selectbox("Label", labels)

sample_types = ["all"] + sorted(df["sample_type"].dropna().unique().tolist())
selected_type = st.sidebar.selectbox("Sample type", sample_types)

score_range = st.sidebar.slider(
    "LLM score range", 0.0, 1.0, (0.0, 1.0), step=0.05
)

show_flagged = st.sidebar.checkbox("Flagged only")

# Apply filters
filtered = df.copy()
if selected_label != "all":
    filtered = filtered[filtered["llm_label"] == selected_label]
if selected_type != "all":
    filtered = filtered[filtered["sample_type"] == selected_type]
filtered = filtered[
    (filtered["llm_score"] >= score_range[0])
    & (filtered["llm_score"] <= score_range[1])
]
if show_flagged:
    filtered = filtered[filtered["llm_flagged"] == True]  # noqa: E712

# --- Summary stats ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total pairs", len(filtered))
col2.metric("Avg score", f"{filtered['llm_score'].mean():.2f}" if len(filtered) else "—")
col3.metric("Valid", (filtered["llm_label"] == "valid").sum() if len(filtered) else 0)
col4.metric("Flagged", filtered["llm_flagged"].sum() if len(filtered) else 0)

# --- Label distribution chart ---
label_counts = filtered["llm_label"].value_counts()
if not label_counts.empty:
    colors = {
        "valid": "#2ecc71",
        "retrospective": "#9b59b6",
        "indirect": "#e67e22",
        "unrelated": "#e74c3c",
    }
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(
        label_counts.values,
        labels=label_counts.index,
        colors=[colors.get(l, "#95a5a6") for l in label_counts.index],
        autopct="%1.0f%%",
        startangle=90,
    )
    ax.set_title("Label distribution")
    st.pyplot(fig)

st.divider()

# --- Pair cards ---
if filtered.empty:
    st.info("No pairs match the current filters.")
else:
    for idx, row in filtered.iterrows():
        label_emoji = {
            "valid": "\u2705",
            "retrospective": "\u23ea",
            "indirect": "\u2194\ufe0f",
            "unrelated": "\u274c",
        }.get(row["llm_label"], "\u2753")

        label_color = {
            "valid": "green",
            "retrospective": "violet",
            "indirect": "orange",
            "unrelated": "red",
        }.get(row["llm_label"], "gray")

        with st.container(border=True):
            top_left, top_right = st.columns([4, 1])
            with top_left:
                st.markdown(
                    f"{label_emoji} :**{label_color}[{row['llm_label'].upper()}]** · "
                    f"Score: **{row['llm_score']:.2f}** · "
                    f"Cosine: {row['similarity_score']:.3f} · "
                    f"`{row['sample_type']}`"
                    + (" · :red[FLAGGED]" if row.get("llm_flagged") else "")
                )
            with top_right:
                st.caption(f"vote {row['vote_id']}")

            left, right = st.columns(2)
            with left:
                st.markdown("**Survey question**")
                st.write(row["question_text"])
            with right:
                st.markdown("**Vote summary**")
                st.write(row["vote_summary"][:500] + ("..." if len(str(row["vote_summary"])) > 500 else ""))

            st.caption(f"LLM explanation: {row['llm_explanation']}")
