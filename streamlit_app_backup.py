# --- Backup DB bootstrap (forces SQLite path to the backup file) ---
import os
import sqlite3
from pathlib import Path

# Point to the backup DB sitting next to this file
_BACKUP_DB = Path(__file__).parent / "geo_tracker_backup_20250827_231709.db"

# Provide DB_PATH for modules that read it (no KeyError if unset)
os.environ.setdefault("DB_PATH", str(_BACKUP_DB))

# Monkey-patch sqlite3.connect so any import (e.g., db.py) uses the backup DB
# even if it ignores DB_PATH. Keeps same kwargs and adds a safe default.
_original_connect = sqlite3.connect
def _connect_override(*args, **kwargs):
    kwargs.setdefault("check_same_thread", False)
    return _original_connect(os.environ.get("DB_PATH", str(_BACKUP_DB)), **kwargs)
sqlite3.connect = _connect_override
# -------------------------------------------------------------------

import streamlit as st
import pandas as pd
import json, datetime, time
import altair as alt
from config import OPENAI_DEFAULT_MODEL, GEMINI_DEFAULT_MODEL
from gsheets import read_prompts_dataframe
from db import fetch_joined
from run import execute_all

st.set_page_config(page_title="GEO Tracker", layout="wide")
st.title("🌍 GEO Tracker — LLM Monitoring")

# ---------------- Sidebar Controls ----------------
with st.sidebar:
    st.header("Controls")
    mode = st.radio("Knowledge Source", options=["internal","web"], index=0)
    providers = st.multiselect("Providers", options=["openai","gemini"], default=["openai"])
    openai_model = st.text_input("OpenAI Model", value=OPENAI_DEFAULT_MODEL)
    gemini_model = st.text_input("Gemini Model", value=GEMINI_DEFAULT_MODEL)

    # NEW: market & language inputs (threaded to run.py)
    market = st.text_input("Market (country)", value="DE", help="ISO-3166 alpha-2, e.g., DE, FR, US")
    lang   = st.text_input("Language", value="de", help="IETF tag, e.g., de, de-DE, en")

    auto = st.slider("Auto refresh (seconds)", 0, 300, 0)
    st.caption("Tip: run the CLI runner on a schedule for continuous logging.")

    st.divider()
    st.subheader("Run Prompts Now")
    if st.button("Execute Now"):
        with st.status("Running prompts...", expanded=True) as status:
            df_prompts = read_prompts_dataframe()
            st.write(f"Loaded {len(df_prompts)} prompt(s) from Sheet.")
            if "openai" in providers:
                execute_all("openai", openai_model, mode, market=market, lang=lang)
                st.write("OpenAI done.")
            if "gemini" in providers:
                execute_all("gemini", gemini_model, mode, market=market, lang=lang)
                st.write("Gemini done.")
            status.update(label="Run complete ✅", state="complete")

# ---------------- Filters ----------------
st.subheader("Results")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    filt_provider = st.selectbox("Filter: Provider", ["(all)","openai","gemini"])
with col2:
    filt_mode = st.selectbox("Filter: Mode", ["(all)","internal","web"])
with col3:
    filt_category = st.text_input("Filter: Category contains", "")
with col4:
    filt_prompt = st.text_input("Filter: Prompt ID equals", "")
with col5:
    since_days = st.number_input("Since days", min_value=0, value=7, step=1)

provider = None if filt_provider=="(all)" else filt_provider
category = filt_category.strip() or None
prompt_id = filt_prompt.strip() or None
mode_filter = None if filt_mode=="(all)" else filt_mode
since_iso = None
if since_days and since_days > 0:
    since_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=since_days)
    since_iso = since_dt.isoformat()

rows = fetch_joined(
    provider=provider,
    category=None if not category else category,
    prompt_id=prompt_id,
    mode=mode_filter,
    since_iso=since_iso
)

if category:
    rows = [r for r in rows if category.lower() in (r.get("category") or "").lower()]

if not rows:
    st.info("No runs yet. Execute prompts to see results.")
    if auto and auto > 0:
        time.sleep(auto)
        st.experimental_rerun()
    st.stop()

df = pd.DataFrame(rows)
df["run_ts"] = pd.to_datetime(df["run_ts"])

# Coerce metrics to numeric (important with None semantics)
for col in ["presence_rate", "sentiment", "trustworthiness", "trust_authority", "trust_sunday",
            "latency_ms", "tokens_in", "tokens_out", "cost_usd"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Extract market/lang from runs.extra_json for display
def _from_extra(row, key):
    try:
        d = json.loads(row.get("extra_json") or "{}")
        return d.get(key)
    except Exception:
        return None

if "extra_json" in df.columns:
    df["market"] = df.apply(lambda r: _from_extra(r, "market"), axis=1)
    df["lang"]   = df.apply(lambda r: _from_extra(r, "lang"), axis=1)
else:
    df["market"] = None
    df["lang"] = None

# ---------------- Key Metrics Summary (per provider) ----------------
st.markdown("### 🔑 Key Metrics Summary (per LLM)")

# Presence toggle + scope
colA, colB = st.columns([1, 2])
with colA:
    show_presence = st.checkbox(
        "Include Presence in summary",
        value=False,
        help="Presence averages can be misleading unless you only include runs where presence metric applies."
    )
with colB:
    presence_scope = "Only runs with presence metric"
    if show_presence:
        presence_scope = st.radio(
            "Presence scope",
            options=["All runs", "Only runs with presence metric"],
            index=1,
            horizontal=True,
            help="Choose whether to average presence across all runs or only those where presence was actually evaluated."
        )

# Sentiment scope toggle (brand-present only vs all)
colS1, colS2 = st.columns([1, 2])
with colS1:
    limit_sentiment = st.checkbox(
        "Sentiment: brand-present only (recommended)",
        value=True,
        help="When checked, average sentiment is computed only over rows where the brand is mentioned. "
             "Uncheck to treat missing sentiment as neutral = 0.0."
    )
with colS2:
    st.caption(
        "Tip: brand-present sentiment reflects attitude toward Sunday Natural only, "
        "while the all-runs view mixes in neutral rows where the brand is not mentioned."
    )

# Build summary with both toggles applied
summary_rows = []
providers_in_data = sorted(df["provider"].dropna().unique())

for prov in providers_in_data:
    df_p = df[df["provider"] == prov].copy()

    # Sentiment aggregation
    if limit_sentiment:
        avg_sentiment = df_p["sentiment"].mean(skipna=True)
    else:
        avg_sentiment = df_p["sentiment"].fillna(0.0).mean()

    # Trust aggregation
    avg_trust = df_p["trustworthiness"].mean(skipna=True)

    # Presence aggregation
    if show_presence:
        if presence_scope == "Only runs with presence metric":
            mask = df_p["presence_rate"].notna()
            avg_presence = df_p.loc[mask, "presence_rate"].mean(skipna=True)
        else:
            avg_presence = df_p["presence_rate"].mean(skipna=True)
    else:
        avg_presence = None

    summary_rows.append({
        "provider": prov,
        "avg_sentiment": avg_sentiment,
        "avg_trust": avg_trust,
        "avg_presence": avg_presence
    })

summary = pd.DataFrame(summary_rows)

# Render metric cards once
for _, row in summary.iterrows():
    p = row["provider"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(f"{p} – Avg Sentiment", f"{(row['avg_sentiment'] if pd.notna(row['avg_sentiment']) else 0):.2f}")
    with c2:
        st.metric(f"{p} – Avg Trustworthiness", f"{(row['avg_trust'] if pd.notna(row['avg_trust']) else 0):.2f}")
    with c3:
        if show_presence:
            val = row["avg_presence"]
            st.metric(f"{p} – Avg Presence", f"{(val if pd.notna(val) else 0):.0%}")

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

    st.markdown("### Aggregations")
    agg1 = df.groupby(["provider","category"]).agg({
        "presence_rate":"mean",
        "sentiment":"mean",
        "trustworthiness":"mean",
        "run_id":"count"
    }).rename(columns={"run_id":"runs"}).reset_index()
    st.dataframe(agg1, width="stretch")

    st.markdown("### Latest by Prompt")
    latest = df.sort_values("run_ts", ascending=False).groupby(["provider","prompt_id"]).head(1)
    show_cols = [
        "run_ts","provider","model","mode","prompt_id","category",
        "market","lang",
        "presence_rate","sentiment","trustworthiness","latency_ms","tokens_in","tokens_out"
    ]
    show_cols = [c for c in show_cols if c in latest.columns]
    st.dataframe(latest[show_cols], width="stretch")

    st.markdown("### Details")
    for _, r in latest.iterrows():
        with st.expander(f"{r['run_ts']} • {r['provider']} • {r['prompt_id']} • {r['category']}"):
            c1, c2 = st.columns([1,1])
            with c1:
                st.markdown("**Metadata**")
                meta = {
                    "provider": r.get("provider"),
                    "model": r.get("model"),
                    "mode": r.get("mode"),
                    "market": r.get("market"),
                    "lang": r.get("lang"),
                }
                st.write({k:v for k,v in meta.items() if v is not None})

                st.markdown("**Question**")
                st.code(r["question"])
                st.markdown("**Prompt sent**")
                st.code(r["prompt_text"])
                if r.get("web_query"):
                    st.markdown("**Web query**")
                    st.code(r["web_query"])
                if r.get("web_sources_json"):
                    try:
                        sources = json.loads(r["web_sources_json"])
                        if sources:
                            st.markdown("**Sources**")
                            for i,s in enumerate(sources, start=1):
                                st.write(f"[{i}] {s.get('title','')} — {s.get('url','')}")
                    except Exception:
                        pass
            with c2:
                st.markdown("**Response**")
                st.write(r.get("response_text") or "")
                st.markdown("**Metrics**")
                st.write({
                    "presence_rate": r.get("presence_rate"),
                    "sentiment": r.get("sentiment"),
                    "trustworthiness": r.get("trustworthiness"),
                    "latency_ms": r.get("latency_ms"),
                    "tokens_in": r.get("tokens_in"),
                    "tokens_out": r.get("tokens_out"),
                })

# ---------------- Trends ----------------
with tab2:
    st.subheader("Sentiment Trends")
    trend = (
        df.groupby([pd.Grouper(key="run_ts", freq="D"), "category"])
        .agg(avg_sentiment=("sentiment","mean"))
        .reset_index()
    )
    chart = (
        alt.Chart(trend)
        .mark_line(point=True)
        .encode(
            x="run_ts:T",
            y=alt.Y("avg_sentiment:Q", scale=alt.Scale(domain=[-1,1])),
            color="category:N",
            tooltip=["run_ts:T","category:N","avg_sentiment:Q"]
        )
    )
    st.altair_chart(chart, width="stretch")

    st.subheader("Presence Rate Heatmap")
    heat = (
        df.groupby([pd.Grouper(key="run_ts", freq="D"), "category"])
        .agg(presence=("presence_rate","mean"))
        .reset_index()
    )
    heatmap = (
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x="run_ts:T",
            y="category:N",
            color=alt.Color("presence:Q", scale=alt.Scale(domain=[0,1])),
            tooltip=["run_ts:T","category:N","presence:Q"]
        )
    )
    st.altair_chart(heatmap, width="stretch")

# ---------------- Provider Comparison ----------------
with tab3:
    st.subheader("Provider Comparison")

    trust_authority_col = "trust_authority" if "trust_authority" in df.columns else None
    trust_sunday_col = "trust_sunday" if "trust_sunday" in df.columns else None

    if trust_authority_col and trust_sunday_col:
        comp = df.groupby(["provider","category"]).agg(
            sentiment=("sentiment","mean"),
            presence=("presence_rate","mean"),
            trust_authority=(trust_authority_col,"mean"),
            trust_sunday=(trust_sunday_col,"mean"),
        ).reset_index()

        st.markdown("**Sentiment**")
        bar_sent = (
            alt.Chart(comp).mark_bar().encode(
                x="category:N", y="sentiment:Q", color="provider:N",
                tooltip=["provider","category","sentiment"]
            )
        )
        st.altair_chart(bar_sent, width="stretch")

        st.markdown("**Presence**")
        bar_pres = (
            alt.Chart(comp).mark_bar().encode(
                x="category:N", y="presence:Q", color="provider:N",
                tooltip=["provider","category","presence"]
            )
        )
        st.altair_chart(bar_pres, width="stretch")

        st.markdown("**Trust (Authority vs Sunday-owned)**")
        comp_melt = comp.melt(
            id_vars=["provider","category"],
            value_vars=["trust_authority","trust_sunday"],
            var_name="trust_type", value_name="trust_value"
        )
        bar_trust = (
            alt.Chart(comp_melt).mark_bar().encode(
                x="category:N",
                y="trust_value:Q",
                color="provider:N",
                column=alt.Column("trust_type:N", header=alt.Header(title=None)),
                tooltip=["provider","category","trust_type","trust_value"]
            )
        )
        st.altair_chart(bar_trust, width="stretch")

    else:
        comp = df.groupby(["provider","category"]).agg(
            sentiment=("sentiment","mean"),
            presence=("presence_rate","mean"),
            trust=("trustworthiness","mean")
        ).reset_index()

        for metric in ["sentiment","presence","trust"]:
            st.markdown(f"**{metric.capitalize()}**")
            bar = (
                alt.Chart(comp)
                .mark_bar()
                .encode(x="category:N", y=f"{metric}:Q", color="provider:N",
                        tooltip=["provider","category",metric])
            )
            st.altair_chart(bar, width="stretch")

# ---------------- Trust & Sources ----------------
with tab4:
    st.subheader("Trustworthiness by Source Domain")
    domains = []
    for _, r in df.iterrows():
        if r.get("web_sources_json"):
            try:
                sources = json.loads(r["web_sources_json"])
                for s in sources:
                    url = s.get("url") or ""
                    domain = url.split("/")[2] if "://" in url else url
                    domains.append({"domain": domain, "trust": r.get("trustworthiness", 0)})
            except Exception:
                pass
    if domains:
        dom_df = pd.DataFrame(domains)
        dom_agg = (
            dom_df.groupby("domain").trust.mean()
            .reset_index()
            .sort_values("trust", ascending=False)
            .head(15)
        )
        bar = alt.Chart(dom_agg).mark_bar().encode(
            x="trust:Q",
            y=alt.Y("domain:N", sort="-x")
        )
        st.altair_chart(bar, width="stretch")

    st.subheader("Sentiment Distribution")
    hist = alt.Chart(df).mark_bar().encode(
        alt.X("sentiment:Q", bin=alt.Bin(maxbins=20), title="Sentiment"),
        y="count()"
    )
    st.altair_chart(hist, width="stretch")

# ---------------- Operations ----------------
with tab5:
    st.subheader("Latency Over Time")
    lat = df.groupby([pd.Grouper(key="run_ts", freq="D"), "provider"]).latency_ms.mean().reset_index()
    chart = alt.Chart(lat).mark_line(point=True).encode(
        x="run_ts:T",
        y="latency_ms:Q",
        color="provider:N"
    )
    st.altair_chart(chart, width="stretch")

    st.subheader("Recency Coverage (Runs per Category)")
    rec = df.groupby("category").run_id.count().reset_index().rename(columns={"run_id": "runs"})
    bar = alt.Chart(rec).mark_bar().encode(
        x="category:N",
        y="runs:Q"
    )
    st.altair_chart(bar, width="stretch")

# ---------------- Alerts ----------------
with tab6:
    st.subheader("Alerts & Flags")
    alerts = []
    for _, r in df.iterrows():
        s = r.get("sentiment")
        t = r.get("trustworthiness")
        p = r.get("presence_rate")

        if pd.notna(s) and s < -0.3:
            alerts.append(f"⚠️ Negative sentiment: {r['category']} ({s:.2f})")

        if pd.notna(t) and t < 0.5:
            alerts.append(f"⚠️ Low trustworthiness: {r['category']} ({t:.2f})")

        if pd.notna(p) and p == 0 and pd.notna(s) and s > 0.2:
            alerts.append(
                f"⚠️ Missed opportunity: {r['category']} (positive sentiment {s:.2f}, no brand mention)"
            )

    if alerts:
        for a in alerts:
            st.write(a)
    else:
        st.success("No alerts. All clear ✅")

# ---------------- Auto Refresh ----------------
if auto and auto > 0:
    time.sleep(auto)
    st.experimental_rerun()
