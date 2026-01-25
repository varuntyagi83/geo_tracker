# streamlit_app.py
import os, json, time, sqlite3, datetime
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import altair as alt

from config import OPENAI_DEFAULT_MODEL, GEMINI_DEFAULT_MODEL
from gsheets import read_prompts_dataframe
from run import execute_all

DB_PATH = os.getenv("DB_PATH", "geo_tracker.db")

st.set_page_config(page_title="GEO Tracker", layout="wide")
st.title("🌍 GEO Tracker — LLM Monitoring")

# ---------------- Utilities ----------------
@st.cache_data(show_spinner=False)
def load_tables(db_path: str):
    if not os.path.exists(db_path):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    con = sqlite3.connect(db_path)
    try:
        runs = pd.read_sql_query("SELECT * FROM runs ORDER BY run_ts DESC", con)
        responses = pd.read_sql_query("SELECT * FROM responses ORDER BY id DESC", con)
        metrics = pd.read_sql_query("SELECT * FROM metrics ORDER BY id DESC", con)
    finally:
        con.close()
    return runs, responses, metrics

def parse_sources(cell) -> List[Dict[str, Any]]:
    if cell in (None, "", "null"):
        return []
    try:
        data = json.loads(cell)
        if isinstance(data, list):
            out, seen = [], set()
            for it in data:
                if isinstance(it, dict):
                    url = (it.get("url") or "").strip()
                    title = it.get("title")
                else:
                    url, title = str(it).strip(), None
                if url and url not in seen:
                    out.append({"url": url, "title": title})
                    seen.add(url)
            return out
    except Exception:
        pass
    return []

def linkify_sources(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return ""
    parts = []
    for i, s in enumerate(sources, 1):
        url = s.get("url")
        title = s.get("title") or f"source {i}"
        parts.append(f"[{title}]({url})")
    return " • ".join(parts)

# ---------------- Sidebar Controls ----------------
with st.sidebar:
    st.header("Controls")
    mode = st.radio("Knowledge Source", options=["internal", "provider_web"], index=0)
    providers = st.multiselect("Providers", options=["openai", "gemini"], default=["openai"])
    openai_model = st.text_input("OpenAI Model", value=OPENAI_DEFAULT_MODEL)
    gemini_model = st.text_input("Gemini Model", value=GEMINI_DEFAULT_MODEL)

    market = st.text_input("Market (country)", value="DE", help="ISO 3166 alpha 2, for example DE")
    lang   = st.text_input("Language", value="de", help="IETF tag, for example de or en")

    raw = st.checkbox("Send raw question", value=False)
    request_timeout = st.slider("Request timeout (seconds)", 10, 120, 60)
    max_retries = st.slider("Max retries", 0, 3, 1)
    sleep_ms = st.slider("Sleep between rows (ms)", 0, 1000, 0)
    auto = st.slider("Auto refresh (seconds)", 0, 300, 0)

    st.divider()
    st.subheader("Run Prompts Now")
    if st.button("Execute Now"):
        with st.status("Running prompts...", expanded=True) as status:
            df_prompts = read_prompts_dataframe()
            st.write(f"Loaded {len(df_prompts)} prompt(s) from Sheet.")
            if "openai" in providers:
                execute_all(
                    "openai",
                    openai_model,
                    mode,
                    market=market,
                    lang=lang,
                    raw=raw,
                    request_timeout=request_timeout,
                    max_retries=max_retries,
                    sleep_ms=sleep_ms,
                )
                st.write("OpenAI done.")
            if "gemini" in providers:
                execute_all(
                    "gemini",
                    gemini_model,
                    mode,
                    market=market,
                    lang=lang,
                    raw=raw,
                    request_timeout=request_timeout,
                    max_retries=max_retries,
                    sleep_ms=sleep_ms,
                )
                st.write("Gemini done.")
            status.update(label="Run complete ✅", state="complete")

# ---------------- Load and prep data ----------------
runs, responses, metrics = load_tables(DB_PATH)

if runs.empty:
    st.info("No data yet. Execute prompts to see results.")
    st.code(f"DB_PATH={DB_PATH}")
    if auto and auto > 0:
        time.sleep(auto)
        st.experimental_rerun()
    st.stop()

df = runs.merge(responses, left_on="id", right_on="run_id", how="left", suffixes=("", "_resp"))
df = df.merge(metrics, left_on="id", right_on="run_id", how="left", suffixes=("", "_met"))

# Types
if "run_ts" in df.columns:
    df["run_ts"] = pd.to_datetime(df["run_ts"], errors="coerce")

# Backward compatible columns expected by older charts
# presence_rate := presence
if "presence" in df.columns and "presence_rate" not in df.columns:
    df["presence_rate"] = df["presence"]
# trustworthiness := trust_authority (mirror for older visuals)
if "trust_authority" in df.columns and "trustworthiness" not in df.columns:
    df["trustworthiness"] = df["trust_authority"]

# Provider sources
df["sources_list"] = df.get("provider_sources", "").apply(parse_sources)
df["sources_count"] = df["sources_list"].apply(lambda x: len(x) if isinstance(x, list) else 0)
df["sources"] = df["sources_list"].apply(linkify_sources)

# Numeric coercion
for col in [
    "presence_rate",
    "sentiment",
    "trustworthiness",
    "trust_authority",
    "trust_sunday",
    "latency_ms",
    "tokens_in",
    "tokens_out",
    "cost_usd",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------- Filters ----------------
st.subheader("Results")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    filt_provider = st.selectbox("Filter: Provider", ["(all)"] + sorted(df["provider"].dropna().unique().tolist()))
with col2:
    filt_mode = st.selectbox("Filter: Mode", ["(all)"] + sorted(df["mode"].dropna().unique().tolist()))
with col3:
    filt_category = st.text_input("Filter: Category contains", "")
with col4:
    filt_prompt = st.text_input("Filter: Prompt ID equals", "")
with col5:
    since_days = st.number_input("Since days", min_value=0, value=7, step=1)

mask = pd.Series(True, index=df.index)
if filt_provider != "(all)":
    mask &= df["provider"].eq(filt_provider)
if filt_mode != "(all)":
    mask &= df["mode"].eq(filt_mode)
if filt_category.strip():
    mask &= df["category"].fillna("").str.contains(filt_category.strip(), case=False, regex=False)
if filt_prompt.strip():
    mask &= df["prompt_id"].fillna("").eq(filt_prompt.strip())
if since_days and since_days > 0 and "run_ts" in df.columns:
    since_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=int(since_days))
    mask &= df["run_ts"] >= pd.to_datetime(since_dt)

v = df[mask].copy()

# ---------------- Key Metrics Summary (per provider) ----------------
st.markdown("### Key Metrics Summary per Provider")
providers_in_data = sorted(v["provider"].dropna().unique())

for prov in providers_in_data:
    v_p = v[v["provider"] == prov]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(f"{prov} – Runs", int(v_p["id"].nunique()))
    with c2:
        st.metric(f"{prov} – Avg Sentiment", f"{v_p['sentiment'].mean(skipna=True):.2f}")
    with c3:
        st.metric(f"{prov} – Avg Presence", f"{v_p['presence_rate'].mean(skipna=True):.0%}" if "presence_rate" in v_p else "0%")
    with c4:
        st.metric(f"{prov} – Avg Trust (Authority)", f"{v_p['trust_authority'].mean(skipna=True):.2f}" if "trust_authority" in v_p else "0.00")
    with c5:
        st.metric(f"{prov} – Sources/Resp", f"{v_p['sources_count'].mean(skipna=True):.2f}")

st.divider()

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Overview", "📈 Trends", "⚖️ Provider Comparison", "🔎 Trust & Sources", "⚙️ Operations", "🚨 Alerts"]
)

# ---------------- Overview ----------------
with tab1:
    st.subheader("Prompt Catalog")
    try:
        df_prompts = read_prompts_dataframe()
        st.dataframe(df_prompts, width="stretch")
    except Exception as e:
        st.error(f"Could not load prompts: {e}")

    st.markdown("### Latest by Prompt")
    latest = v.sort_values("run_ts", ascending=False).groupby(["provider","prompt_id"]).head(1)

    show_cols = [
        "run_ts","provider","model","mode","market","lang",
        "prompt_id","question",
        "presence_rate","sentiment","trust_authority","trust_sunday",
        "latency_ms","tokens_in","tokens_out","sources_count"
    ]
    show_cols = [c for c in show_cols if c in latest.columns]
    st.dataframe(latest[show_cols], width="stretch")

    st.markdown("### Details")
    for _, r in latest.iterrows():
        title = f"{r.get('run_ts')} • {r.get('provider')} • {r.get('prompt_id')} • {r.get('category')}"
        with st.expander(title):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**Metadata**")
                meta = {
                    "provider": r.get("provider"),
                    "model": r.get("model"),
                    "mode": r.get("mode"),
                    "market": r.get("market"),
                    "lang": r.get("lang"),
                }
                st.write({k: v for k, v in meta.items() if v is not None})

                st.markdown("**Question**")
                st.code(r.get("question") or "")

                st.markdown("**Prompt sent**")
                st.code(r.get("prompt_text") or "")

            with c2:
                st.markdown("**Response**")
                st.write(r.get("response_text") or "")
                if r.get("sources"):
                    st.markdown(f"**Sources**  \n{r['sources']}")
                st.markdown("**Metrics**")
                st.write({
                    "presence": r.get("presence_rate"),
                    "sentiment": r.get("sentiment"),
                    "trust_authority": r.get("trust_authority"),
                    "trust_sunday": r.get("trust_sunday"),
                    "latency_ms": r.get("latency_ms"),
                    "tokens_in": r.get("tokens_in"),
                    "tokens_out": r.get("tokens_out"),
                })

# ---------------- Trends ----------------
with tab2:
    st.subheader("Sentiment Trends")
    trend = (
        v.groupby([pd.Grouper(key="run_ts", freq="D"), "category"])
        .agg(avg_sentiment=("sentiment","mean"))
        .reset_index()
    )
    if not trend.empty:
        chart = alt.Chart(trend).mark_line(point=True).encode(
            x="run_ts:T",
            y=alt.Y("avg_sentiment:Q", scale=alt.Scale(domain=[-1, 1])),
            color="category:N",
            tooltip=["run_ts:T","category:N","avg_sentiment:Q"]
        )
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Presence Rate Heatmap")
    heat = (
        v.groupby([pd.Grouper(key="run_ts", freq="D"), "category"])
        .agg(presence=("presence_rate","mean"))
        .reset_index()
    )
    if not heat.empty:
        heatmap = alt.Chart(heat).mark_rect().encode(
            x="run_ts:T",
            y="category:N",
            color=alt.Color("presence:Q", scale=alt.Scale(domain=[0, 1])),
            tooltip=["run_ts:T","category:N","presence:Q"]
        )
        st.altair_chart(heatmap, use_container_width=True)


# ---------------- Provider Comparison ----------------
with tab3:
    st.subheader("Provider Comparison")
    if "trust_authority" in v.columns and "trust_sunday" in v.columns:
        comp = v.groupby(["provider","category"]).agg(
            sentiment=("sentiment","mean"),
            presence=("presence_rate","mean"),
            trust_authority=("trust_authority","mean"),
            trust_sunday=("trust_sunday","mean"),
        ).reset_index()

        st.markdown("**Sentiment**")
        bar_sent = alt.Chart(comp).mark_bar().encode(
            x="category:N", y="sentiment:Q", color="provider:N",
            tooltip=["provider","category","sentiment"]
        )
        st.altair_chart(bar_sent, use_container_width=True)

        st.markdown("**Presence**")
        bar_pres = alt.Chart(comp).mark_bar().encode(
            x="category:N", y="presence:Q", color="provider:N",
            tooltip=["provider","category","presence"]
        )
        st.altair_chart(bar_pres, use_container_width=True)

        st.markdown("**Trust (Authority vs Sunday-owned)**")
        comp_melt = comp.melt(
            id_vars=["provider","category"],
            value_vars=["trust_authority","trust_sunday"],
            var_name="trust_type", value_name="trust_value"
        )
        bar_trust = alt.Chart(comp_melt).mark_bar().encode(
            x="category:N",
            y="trust_value:Q",
            color="provider:N",
            column=alt.Column("trust_type:N"),
            tooltip=["provider","category","trust_type","trust_value"]
        )
        st.altair_chart(bar_trust, use_container_width=True)

# ---------------- Trust & Sources ----------------
with tab4:
    st.subheader("Trustworthiness by Source Domain")
    domains = []
    for _, r in v.iterrows():
        for s in r.get("sources_list") or []:
            url = s.get("url") or ""
            domain = url.split("/")[2] if "://" in url else url
            domains.append({"domain": domain, "trust_authority": r.get("trust_authority", 0)})
    if domains:
        dom_df = pd.DataFrame(domains)
        dom_agg = dom_df.groupby("domain").trust_authority.mean().reset_index().sort_values("trust_authority", ascending=False).head(15)
        bar = alt.Chart(dom_agg).mark_bar().encode(
            x="trust_authority:Q",
            y=alt.Y("domain:N", sort="-x")
        )
        st.altair_chart(bar, use_container_width=True)

    st.subheader("Sentiment Distribution")
    hist = alt.Chart(v).mark_bar().encode(
        alt.X("sentiment:Q", bin=alt.Bin(maxbins=20), title="Sentiment"),
        y="count()"
    )
    st.altair_chart(hist, use_container_width=True)

# ---------------- Operations ----------------
with tab5:
    st.subheader("Latency Over Time")
    lat = v.groupby([pd.Grouper(key="run_ts", freq="D"), "provider"]).latency_ms.mean().reset_index()
    if not lat.empty:
        chart = alt.Chart(lat).mark_line(point=True).encode(
            x="run_ts:T", y="latency_ms:Q", color="provider:N"
        )
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Recency Coverage (Runs per Category)")
    rec = v.groupby("category").id.count().reset_index().rename(columns={"id": "runs"})
    bar = alt.Chart(rec).mark_bar().encode(x="category:N", y="runs:Q")
    st.altair_chart(bar, use_container_width=True)

# ---------------- Alerts ----------------
with tab6:
    st.subheader("Alerts & Flags")
    alerts = []
    for _, r in v.iterrows():
        s = r.get("sentiment")
        ta = r.get("trust_authority")
        p = r.get("presence_rate")

        if pd.notna(s) and s < -0.3:
            alerts.append(f"Negative sentiment: {r.get('category')} ({s:.2f})")
        if pd.notna(ta) and ta < 0.5:
            alerts.append(f"Low authority-trust: {r.get('category')} ({ta:.2f})")
        if pd.notna(p) and p == 0 and pd.notna(s) and s > 0.2:
            alerts.append(f"Missed brand presence: {r.get('category')} (positive sentiment {s:.2f}, presence 0)")

    if alerts:
        for a in alerts:
            st.write("• " + a)
    else:
        st.success("No alerts.")

# ---------------- Auto Refresh ----------------
if auto and auto > 0:
    time.sleep(auto)
    st.experimental_rerun()
