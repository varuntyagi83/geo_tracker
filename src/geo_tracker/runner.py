from __future__ import annotations
import os, json, argparse, datetime as dt
from pathlib import Path
from typing import List, Dict, Any
from .config import SETTINGS
from .registry import PromptRegistry
from .storage import (
    init_db, get_session,
    Prompt, Run, ModelCall, ExtractedMention, Sentiment, Grounding
)
from .extractors import (
    extract_mentions, placement_quality, grounding_quality, sentiment_score
)
from .metrics import (
    presence_rate, share_of_voice, aggregate_sentiment,
    avg_grounding, avg_placement, refusal_rate, hallucination_rate, geo_score
)
from .connectors.openai_conn import OpenAIConnector
from .connectors.anthropic_conn import AnthropicConnector
from .connectors.google_conn import GoogleConnector
from .connectors.perplexity_conn import PerplexityConnector
from .connectors.openmodel_conn import OpenModelConnector


def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_active_providers() -> List:
    """
    Build the provider list from the ACTIVE_PROVIDERS env var.
    Example: ACTIVE_PROVIDERS=openai,anthropic,google
    """
    provider_map = {
        "openai": OpenAIConnector,
        "anthropic": AnthropicConnector,
        "google": GoogleConnector,
        "perplexity": PerplexityConnector,
        "together": OpenModelConnector,
    }
    active_raw = os.getenv("ACTIVE_PROVIDERS", "")
    active = [p.strip().lower() for p in active_raw.split(",") if p.strip()]
    return [provider_map[p]() for p in active if p in provider_map]


def run_panel(panel_version: str, market: str, language: str, comments: str | None = None) -> None:
    init_db()

    registry = PromptRegistry(Path(__file__).parent / "prompts.yaml")
    prompts = [
        p for p in registry.panel(panel_version)
        if p.get("market") == market and p.get("language") == language
    ]
    if not prompts:
        print(f"No prompts found for panel='{panel_version}', market='{market}', language='{language}'.")
        return

    brand_dict = load_yaml(Path(__file__).parent / "competitors.yaml")["brands"]
    sunday_name = SETTINGS.brand_name

    providers = _build_active_providers()
    if not providers:
        print("No active providers specified. Set ACTIVE_PROVIDERS in .env, e.g. 'ACTIVE_PROVIDERS=openai,anthropic,google'.")
        return

    run_key = dt.datetime.utcnow().strftime("%Y%m%d") + f"-{panel_version}-{market}-{language}"
    with get_session() as s:
        # idempotent run
        if s.query(Run).filter_by(run_key=run_key).first():
            print(f"Run {run_key} already exists. Skipping.")
            return

        run = Run(run_key=run_key, panel_version=panel_version, comments=comments)
        s.add(run); s.flush()

        # store prompts for audit
        prompt_id_map: Dict[str, int] = {}
        for p in prompts:
            pr = Prompt(
                prompt_id=p["id"], text=p["text"], topic=p["topic"],
                intent=p["intent"], language=language, market=market
            )
            s.add(pr); s.flush()
            prompt_id_map[p["id"]] = pr.id

        # execute
        daily_rows: List[Dict[str, Any]] = []
        for prov in providers:
            for p in prompts:
                # Real call only — any failure raises, no fallback.
                resp = prov.generate(p["text"], language, market)

                # Reject anything that looks like a mock response
                is_mock = 1 if (resp.version and str(resp.version).startswith("mock")) else 0
                if is_mock:
                    raise RuntimeError(f"{prov.provider} returned mock output unexpectedly (prompt={p['id']}).")

                print(f"REAL CALL [{prov.provider}] prompt={p['id']} model={resp.model}")

                mc = ModelCall(
                    run_id=run.id,
                    prompt_ref=prompt_id_map[p["id"]],
                    provider=prov.provider,
                    model=resp.model,
                    model_version=resp.version,
                    temperature=0.2, top_p=None, max_tokens=None,
                    latency_ms=resp.latency_ms,
                    input_tokens=resp.usage.get("input_tokens", 0),
                    output_tokens=resp.usage.get("output_tokens", 0),
                    is_mock=0,
                    status="ok",
                    refusal_flag=int(resp.refusal),
                    raw_answer=resp.text,
                    citations_json={"citations": resp.citations},
                )
                s.add(mc); s.flush()

                # mentions
                mentions = extract_mentions(resp.text, brand_dict)
                for m in mentions:
                    s.add(ExtractedMention(
                        call_id=mc.id,
                        brand=m["brand"],
                        first_position=m["first_position"],
                        mention_count=m["mention_count"],
                        in_final_recommendation=m["in_final_recommendation"],
                    ))

                # sentiment
                sent, conf, rat = sentiment_score(resp.text)
                s.add(Sentiment(
                    call_id=mc.id,
                    sentiment_score=sent, confidence=conf,
                    rater_model="vader", rationale=rat
                ))

                # grounding
                total, cred, frac = grounding_quality(
                    resp.citations,
                    credible_whitelist=["example.com", "health.org", "nih.gov", "who.int"]
                )
                s.add(Grounding(
                    call_id=mc.id,
                    source_count=total,
                    credible_source_count=cred,
                    credible_fraction=frac,
                    credible_domains_json={"domains": resp.citations}
                ))

                # rollup row
                sunday_mentions = next(
                    (m for m in mentions if m["brand"].lower() == sunday_name.lower()), None
                )
                comp_mentions = sum(
                    m["mention_count"] for m in mentions if m["brand"].lower() != sunday_name.lower()
                )
                placement = placement_quality(resp.text, sunday_name) if sunday_mentions else 0.0

                daily_rows.append({
                    "provider": prov.provider,
                    "model": resp.model,
                    "market": market,
                    "topic": p["topic"],
                    "sunday_present": sunday_mentions is not None,
                    "sunday_mentions": sunday_mentions["mention_count"] if sunday_mentions else 0,
                    "competitor_mentions": comp_mentions,
                    "sentiment": sent,
                    "grounding_fraction": frac,
                    "placement": placement,
                    "refusal": resp.refusal,
                })

        # compute daily metrics and persist
        from .storage import MetricsDaily
        today = dt.date.today()
        from collections import defaultdict
        groups: Dict[tuple, list] = defaultdict(list)
        for row in daily_rows:
            key = (row["provider"], row["model"], row["market"], row["topic"])
            groups[key].append(row)

        for (prov, model, market, topic), rows in groups.items():
            pr = presence_rate(rows)
            sov = share_of_voice(rows)
            sent = aggregate_sentiment(rows)
            gq = avg_grounding(rows)
            pq = avg_placement(rows)
            rr = refusal_rate(rows)
            hr = hallucination_rate(rows)
            geo = geo_score(pr, sov, sent, gq, pq)
            s.add(MetricsDaily(
                date=today, provider=prov, model=model, market=market, topic=topic,
                presence_rate=pr, share_of_voice=sov, avg_sentiment=sent,
                grounding_quality=gq, placement_quality=pq, refusal_rate=rr,
                hallucination_rate=hr, geo_score=geo
            ))

        s.commit()
        print(f"Run {run_key} completed with {len(daily_rows)} call rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="default")
    parser.add_argument("--market", default=SETTINGS.market_default)
    parser.add_argument("--language", default=SETTINGS.language_default)
    parser.add_argument("--comments", default=None)
    args = parser.parse_args()
    run_panel(args.panel, args.market, args.language, args.comments)
