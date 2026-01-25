import argparse, re, json, time, sys, urllib.parse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from collections import Counter

from gsheets import read_prompts_dataframe
from llm_providers.openai_provider import OpenAIProvider
from llm_providers.gemini_provider import GeminiProvider
from metrics.presence import compute_presence_rate
from metrics.sentiment import compute_sentiment
from metrics.trust import compute_trustworthiness
from db import init_db, insert_run, insert_response, insert_metrics
from config import OPENAI_DEFAULT_MODEL, GEMINI_DEFAULT_MODEL

# Use the new brand detection module instead of the old heuristic code
from brand_detection import detect_competitor_brands

PROVIDERS = {"openai": OpenAIProvider, "gemini": GeminiProvider}

BRAND_NEEDLE = "Sunday Natural"

# --- Helpers -----------------------------------------------------------------

def _call_with_timeout(fn, timeout_s: int, retries: int, label: str):
    last_err = None
    for attempt in range(1, retries + 2):
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(fn)
                return fut.result(timeout=timeout_s)
        except TimeoutError:
            print(f"[timeout] {label} attempt {attempt} exceeded {timeout_s}s", file=sys.stderr)
            last_err = TimeoutError(f"{label} timed out")
        except Exception as e:
            print(f"[error] {label} attempt {attempt} failed: {e}", file=sys.stderr)
            last_err = e
        time.sleep(1.0 * attempt)
    return {
        "text": "",
        "latency_ms": None,
        "tokens_in": None,
        "tokens_out": None,
        "cost_usd": None,
        "sources": [],
        "error": str(last_err) if last_err else None
    }

def _preview(text: str, max_len: int) -> str:
    if max_len <= 0 or not text:
        return ""
    s = " ".join(str(text).split())
    return s if len(s) <= max_len else s[: max_len - 1] + "…"

_URL_RE = re.compile(r'\bhttps?://[^\s\)\]]+', re.IGNORECASE)
_MD_LINK_RE = re.compile(r'\[([^\]]{0,200})\]\((https?://[^\s\)]+)\)')

def _fallback_extract_sources(response_text: str):
    """
    Extract sources from response text when provider did not return any.
    Returns list of {"url":..., "title":...} de-duplicated by URL host+path.
    """
    if not response_text:
        return []

    found = []
    # Markdown links first: [title](url)
    for m in _MD_LINK_RE.finditer(response_text):
        title = m.group(1).strip() or None
        url = m.group(2).strip()
        found.append({"url": url, "title": title})

    # Bare URLs
    for m in _URL_RE.finditer(response_text):
        url = m.group(0).strip().rstrip(").,;")
        # Skip if already captured as md link
        if url not in [f["url"] for f in found]:
            found.append({"url": url, "title": None})

    # De-duplicate by normalized URL (host+path)
    dedup = {}
    for s in found:
        try:
            p = urllib.parse.urlparse(s["url"])
            key = (p.netloc.lower(), p.path)
            if key not in dedup:
                dedup[key] = {"url": s["url"], "title": s["title"] or None}
        except Exception:
            continue
    return list(dedup.values())

# --- Runner ------------------------------------------------------------------

def execute_all(provider_name: str, model: str, mode: str,
                limit: int | None = None, start: int = 0,
                market: str | None = None, lang: str | None = None,
                raw: bool = False,
                request_timeout: int = 60,
                max_retries: int = 1,
                sleep_ms: int = 0,
                log_question_len: int = 160):

    """
    Modes for both providers:
      internal      -> model knowledge only
      provider_web  -> provider-native web tools if available
    """
    init_db()
    df = read_prompts_dataframe()

    if start or limit:
        end = start + limit if limit else None
        df = df.iloc[start:end].reset_index(drop=True)
        print(f"[runner] Using slice start={start}, limit={limit or 'ALL'}, rows={len(df)}")

    ProviderCls = PROVIDERS[provider_name]
    provider = ProviderCls()

    if not model:
        model = OPENAI_DEFAULT_MODEL if provider_name == "openai" else GEMINI_DEFAULT_MODEL

    total = len(df)
    for idx, row in df.iterrows():
        prompt_id = row.get("prompt_id") or ""
        category  = row.get("category") or ""
        question  = row.get("question") or ""

        # --- Only change: skip empty questions --------------------------------
        if not str(question).strip():
            print(f"[{idx+1}/{total}] skipped empty question • prompt_id={prompt_id}")
            continue
        # ----------------------------------------------------------------------

        qprev = _preview(question, log_question_len)
        qpart = f" • q={qprev!r}" if qprev else ""
        print(f"[{idx+1}/{total}] run start • prompt_id={prompt_id}{qpart} • mode={mode} • raw={raw}")

        # Build final payload
        if raw:
            prompt_text = question
        else:
            header = ""
            if mode == "provider_web" and (market or lang):
                header = f"(Market: {market or '-'}; Language: {lang or '-'})\n\n"
            prompt_text = header + question

        # Store canonical mode only; raw flag goes into extra JSON
        run_id = insert_run(
            provider    = provider_name,
            model       = model,
            prompt_id   = prompt_id,
            category    = category,
            mode        = mode,
            question    = question,
            prompt_text = prompt_text,
            market      = market,
            lang        = lang,
            extra       = {"raw": bool(raw)}
        )
        print(f"[{idx+1}/{total}] run_id={run_id} inserted")

        # Provider call
        label = f"{provider_name}:{mode}"
        if mode == "provider_web" and hasattr(provider, "generate_provider_web"):
            result = _call_with_timeout(lambda: provider.generate_provider_web(prompt_text, model=model),
                                        request_timeout, max_retries, label)
        else:
            result = _call_with_timeout(lambda: provider.generate(prompt_text, model=model),
                                        request_timeout, max_retries, label)

        response_text    = result.get("text", "") or ""
        latency_ms       = result.get("latency_ms")
        tokens_in        = result.get("tokens_in")
        tokens_out       = result.get("tokens_out")
        cost_usd         = result.get("cost_usd")

        provider_sources = result.get("sources") or []
        if not provider_sources:
            # Fallback: scrape sources from the response text
            provider_sources = _fallback_extract_sources(response_text)
            if provider_sources:
                print(f"[{idx+1}/{total}] sources fallback extracted {len(provider_sources)} link(s)")

        insert_response(
            run_id=run_id,
            response_text=response_text,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            provider_sources=provider_sources
        )
        print(f"[{idx+1}/{total}] response saved • sources={len(provider_sources)} • chars={len(response_text)}")

        # ----------------- Metrics (your rule set) -----------------
        # Detect brands (other than our brand) from text+sources using LLM-based brand detection
        # The LLM understands context and won't return false positives like country/city names
        other_brands = detect_competitor_brands(
            response_text, 
            provider_sources, 
            BRAND_NEEDLE,
            industry="Supplements & Vitamins",  # Hardcoded for Sunday Natural
            market=market or "Germany"
        )

        # Presence of our brand in the answer (use your existing presence fn for robustness)
        presence_sn = compute_presence_rate(response_text, BRAND_NEEDLE)
        brand_present = bool(presence_sn and presence_sn > 0)

        # Apply your rules:
        # 1) If Sunday Natural is present -> keep computed presence (>0), compute sentiment.
        # 2) If others appear but SN does not -> presence = 0.0 (explicit zero).
        # 3) If no others appear and SN does not -> presence = None.
        if brand_present:
            presence = float(presence_sn)
            sentiment = compute_sentiment(response_text)
        else:
            if other_brands:
                presence = 0.0
                sentiment = None
            else:
                presence = None
                sentiment = None

        # Trustworthiness: always compute
        trust_authority, trust_sunday = compute_trustworthiness(response_text, provider_sources)

        details = {
            "brand_needle": BRAND_NEEDLE,
            "brand_present": bool(brand_present),
            "other_brands_detected": sorted(other_brands),
            "market": market,
            "lang": lang,
            "raw": bool(raw),
            # diagnostics
            "response_chars": len(response_text or ""),
            "n_sources": len(provider_sources or []),
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd,
        }

        try:
            insert_metrics(run_id, presence, sentiment, trust_authority, trust_sunday, details)
        except TypeError:
            insert_metrics(run_id, presence, sentiment, trust_authority, details)

        print(
            f"[{idx+1}/{total}] metrics saved • "
            f"presence={presence} • sentiment={sentiment} • "
            f"trustA={trust_authority} • trustS={trust_sunday} • "
            f"others={len(other_brands)}"
        )

        if sleep_ms:
            time.sleep(sleep_ms / 1000.0)

    print("[runner] all done")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "gemini"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", choices=["internal", "provider_web"], default="internal")
    ap.add_argument("--interval-minutes", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--market", type=str, default=None)
    ap.add_argument("--lang", type=str, default=None)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--request-timeout", type=int, default=60)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--sleep-ms", type=int, default=0)
    ap.add_argument("--log-question-len", type=int, default=160)
    args = ap.parse_args()

    model = args.model or (OPENAI_DEFAULT_MODEL if args.provider == "openai" else GEMINI_DEFAULT_MODEL)

    if args.interval_minutes > 0:
        print(f"Looping every {args.interval_minutes} minutes. Ctrl+C to stop.")
        while True:
            execute_all(args.provider, model, args.mode,
                        limit=args.limit, start=args.start,
                        market=args.market, lang=args.lang,
                        raw=args.raw,
                        request_timeout=args.request_timeout,
                        max_retries=args.max_retries,
                        sleep_ms=args.sleep_ms,
                        log_question_len=args.log_question_len)
            time.sleep(args.interval_minutes * 60)
    else:
        execute_all(args.provider, model, args.mode,
                    limit=args.limit, start=args.start,
                    market=args.market, lang=args.lang,
                    raw=args.raw,
                    request_timeout=args.request_timeout,
                    max_retries=args.max_retries,
                    sleep_ms=args.sleep_ms,
                    log_question_len=args.log_question_len)

if __name__ == "__main__":
    main()
