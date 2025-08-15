import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./geo_tracker.db")
engine = create_engine(DATABASE_URL)

st.set_page_config(page_title="GEO Tracker", layout="wide")
st.title("GEO Tracker Dashboard")

@st.cache_data(ttl=300)
def load_metrics() -> pd.DataFrame:
    try:
        return pd.read_sql("SELECT * FROM metrics_daily", engine)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_calls() -> pd.DataFrame:
    q = """
        SELECT mc.id, r.run_key, p.topic, p.intent,
               mc.provider, mc.model, mc.model_version, mc.is_mock,
               mc.latency_ms, mc.input_tokens, mc.output_tokens,
               mc.status, mc.refusal_flag, mc.raw_answer
        FROM model_calls mc
        JOIN runs r     ON mc.run_id  = r.id
        JOIN prompts p  ON mc.prompt_ref = p.id
        ORDER BY mc.id DESC
    """
    try:
        return pd.read_sql(q, engine)
    except Exception:
        return pd.DataFrame()

metrics = load_metrics()

if metrics.empty:
    st.info("No metrics yet. Run the panel script first.")
    st.stop()

# Sidebar filters
with st.sidebar:
    all_providers = sorted(metrics["provider"].dropna().unique().tolist())
    all_topics    = sorted(metrics["topic"].dropna().unique().tolist())

    providers = st.multiselect("Providers", all_providers, default=all_providers)
    topics    = st.multiselect("Topics", all_topics, default=all_topics)
    hide_mocks = st.checkbox("Hide mock calls", value=True)

# Filtered metrics
df = metrics[(metrics["provider"].isin(providers)) & (metrics["topic"].isin(topics))]

st.subheader("Daily Metrics")
st.dataframe(df, use_container_width=True)

st.subheader("GEO score by provider and topic")
if not df.empty:
    pivot = (
        df.pivot_table(
            index=["provider", "model", "topic"],
            values=["geo_score", "presence_rate", "share_of_voice", "avg_sentiment", "grounding_quality", "placement_quality"],
            aggfunc="mean"
        )
        .reset_index()
    )
    if not pivot.empty:
        st.bar_chart(pivot, x="provider", y="geo_score", use_container_width=True)
        st.subheader("Presence vs Share of Voice")
        st.scatter_chart(pivot, x="presence_rate", y="share_of_voice", use_container_width=True)
    else:
        st.caption("No data for the selected filters.")
else:
    st.caption("No data for the selected filters.")

st.subheader("Raw calls")
calls = load_calls()
if not calls.empty:
    # Apply same provider/topic filters to calls
    calls = calls[(calls["provider"].isin(providers)) & (calls["topic"].isin(topics))]

    # Hide mocks if requested
    if hide_mocks and "is_mock" in calls.columns:
        calls = calls[calls["is_mock"] == 0]

    st.dataframe(
        calls[[
            "run_key","topic","provider","model","model_version","is_mock",
            "latency_ms","input_tokens","output_tokens","status","refusal_flag"
        ]],
        use_container_width=True
    )

    # Inspect a specific answer
    if not calls.empty:
        selected = st.selectbox("Inspect answer by ID", options=calls["id"].tolist())
        raw = calls.loc[calls["id"] == selected, "raw_answer"].values[0]
        st.text_area("Raw answer", value=raw, height=300)
else:
    st.caption("No call records yet.")
