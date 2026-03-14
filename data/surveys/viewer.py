"""Quick Streamlit viewer for Volume B answer distributions."""

import pandas as pd
import streamlit as st

CSV_PATH = "/Users/ugo/Documents/MH2D_projets/dawta/eu_survey_correlation/data/surveys/volume_b_answer_distributions.csv"

st.set_page_config(page_title="Volume B Viewer", layout="wide")
st.title("Eurobarometer Volume B - Answer Distributions")


@st.cache_data
def load():
    df = pd.read_csv(CSV_PATH)
    df["q_key"] = df["sheet_id"] + " | " + df["file_name"]
    return df


df = load()

# --- sidebar filters ---
st.sidebar.header("Filters")
search = st.sidebar.text_input("Search questions", "")

demo_options = ["all"] + sorted(df["demographic_type"].unique().tolist())
demo_filter = st.sidebar.selectbox("Demographic type", demo_options)

show_summary = st.sidebar.checkbox("Include summary rows (Total Agree/Disagree)", False)

# --- filter data ---
if search:
    mask = df["question_clean"].str.contains(search, case=False, na=False) | df[
        "sheet_id"
    ].str.contains(search, case=False, na=False)
    df = df[mask]

if demo_filter != "all":
    df = df[df["demographic_type"] == demo_filter]

if not show_summary:
    df = df[~df["is_summary"]]

# --- stats ---
n_questions = df["q_key"].nunique()
n_files = df["file_name"].nunique()
st.sidebar.markdown(
    f"**{n_questions}** questions | **{n_files}** files | **{len(df)}** rows"
)

# --- question selector ---
questions = df.drop_duplicates("q_key")[["q_key", "question_clean"]].sort_values(
    "q_key"
)
q_labels = {
    row.q_key: f"{row.q_key}  —  {str(row.question_clean)[:120]}"
    for row in questions.itertuples()
}

selected = st.selectbox(
    "Select question", list(q_labels.keys()), format_func=lambda k: q_labels[k]
)

qdf = df[df["q_key"] == selected]

st.subheader(qdf["question_clean"].iloc[0] if not qdf.empty else "")
st.caption(f"Sheet: `{qdf['sheet_id'].iloc[0]}` | File: `{qdf['file_name'].iloc[0]}`")

# --- demographic tabs ---
available_demos = sorted(qdf["demographic_type"].unique().tolist())
if not available_demos:
    st.warning("No data for this question with current filters.")
    st.stop()

tabs = st.tabs([d.replace("_", " ").title() for d in available_demos])

VALUE_LABELS = {
    "eu27": "EU27",
    "poor": "Poor (bills often)",
    "medium": "Medium (bills sometimes)",
    "rich": "Rich (bills rarely)",
    "working_class": "Working class",
    "lower_middle": "Lower middle",
    "middle": "Middle class",
    "upper_middle": "Upper middle",
    "upper": "Upper class",
    "male": "Male",
    "female": "Female",
    "self_employed": "Self-employed",
    "manager": "Manager",
    "employee": "Employee",
    "other_white_collar": "Other white collar",
    "manual_worker": "Manual worker",
    "house_person": "House person",
    "unemployed": "Unemployed",
    "retired": "Retired",
    "student": "Student",
    "not_working": "Not working",
}

for tab, demo_type in zip(tabs, available_demos):
    with tab:
        subset = qdf[qdf["demographic_type"] == demo_type].copy()
        subset["demo_label"] = subset["demographic_value"].map(
            lambda v: VALUE_LABELS.get(v, v)
        )

        # Pivot for chart: answers as rows, demographic values as columns
        pivot = subset.pivot_table(
            index="answer_label", columns="demo_label", values="pct", aggfunc="first"
        )
        # Reorder columns nicely
        col_order = [
            VALUE_LABELS.get(v, v) for v in subset["demographic_value"].unique()
        ]
        pivot = pivot[[c for c in col_order if c in pivot.columns]]

        st.bar_chart(pivot, horizontal=True, height=max(250, len(pivot) * 50))

        # Data table
        with st.expander("Data table"):
            table = subset[
                ["answer_label", "demo_label", "count", "pct", "total_base"]
            ].copy()
            table.columns = ["Answer", "Group", "Count", "%", "Base"]
            table["Count"] = table["Count"].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) else "-"
            )
            table["Base"] = table["Base"].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) else "-"
            )
            table["%"] = table["%"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
            st.dataframe(table, use_container_width=True, hide_index=True)
