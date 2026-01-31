# api/services.py
"""
Service layer that wraps the existing GEO tracker for API use.
This is the bridge between the FastAPI endpoints and your existing code.

UPDATED:
- Correct LLM models as of January 2026
- Parallel provider execution
- AI-powered query generation
"""
import os
import sys
import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
import time
import re
import urllib.parse
import threading

# Add parent directory to path to import existing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import (
    init_db, insert_run, insert_response, insert_metrics, _connect,
    get_or_create_brand, record_brand_run
)
from config import (
    OPENAI_DEFAULT_MODEL, GEMINI_DEFAULT_MODEL,
    PERPLEXITY_DEFAULT_MODEL, ANTHROPIC_DEFAULT_MODEL
)
from llm_providers.openai_provider import OpenAIProvider
from llm_providers.gemini_provider import GeminiProvider
from llm_providers.perplexity_provider import PerplexityProvider
from llm_providers.anthropic_provider import AnthropicProvider
from metrics.presence import compute_presence_rate
from metrics.sentiment import compute_sentiment
from metrics.trust import compute_trustworthiness

# Import the new brand detection module
from brand_detection import detect_competitor_brands, normalize_brand

# Import query generator
from query_generator import (
    generate_queries, 
    get_fallback_queries,
    BusinessContext
)

from .jobs import Job, JobStatus
from .models import (
    RunConfigCreate, QueryCreate, QueryResult, RunSummary, 
    ProviderEnum, ModeEnum, SourceInfo
)

# Provider registry
PROVIDERS = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "perplexity": PerplexityProvider,
    "anthropic": AnthropicProvider,
}

# ============================================
# AVAILABLE MODELS - Updated January 2026
# ============================================
# Based on web search of OpenAI and Google Gemini documentation
# 
# OpenAI: https://platform.openai.com/docs/models
# - GPT-4.1 family: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano (1M context)
# - GPT-4o family: gpt-4o, gpt-4o-mini (still available)
# - Legacy: gpt-4-turbo (128K context), gpt-3.5-turbo (legacy)
#
# Google Gemini: https://ai.google.dev/gemini-api/docs/models
# - Gemini 2.5: gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite
# - Gemini 2.0: gemini-2.0-flash (deprecated March 2026, still works)
# - Preview: gemini-3-pro-preview, gemini-3-flash-preview
# ============================================

AVAILABLE_MODELS = {
    "openai": [
        {"id": "gpt-4.1", "name": "GPT-4.1", "description": "Latest flagship model (1M context)"},
        {"id": "gpt-4.1-mini", "name": "GPT-4.1 Mini", "description": "Fast & affordable (1M context)"},
        {"id": "gpt-4.1-nano", "name": "GPT-4.1 Nano", "description": "Fastest, cheapest (1M context)"},
        {"id": "gpt-4o", "name": "GPT-4o", "description": "Multimodal GPT-4"},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "description": "Smaller multimodal"},
        {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "description": "Legacy (128K context)"},
    ],
    "gemini": [
        {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "description": "Fast, recommended (1M context)"},
        {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "description": "Most capable (1M context)"},
        {"id": "gemini-2.5-flash-lite", "name": "Gemini 2.5 Flash-Lite", "description": "Cost-efficient"},
        {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "description": "Previous gen (until March 2026)"},
        {"id": "gemini-3-flash-preview", "name": "Gemini 3 Flash (Preview)", "description": "Preview: Advanced multimodal"},
    ],
    "perplexity": [
        {"id": "sonar", "name": "Sonar", "description": "Fast with native web search"},
        {"id": "sonar-pro", "name": "Sonar Pro", "description": "Advanced reasoning with web search"},
        {"id": "sonar-reasoning", "name": "Sonar Reasoning", "description": "Chain-of-thought with web search"},
    ],
    "anthropic": [
        {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku", "description": "Fast and affordable"},
        {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4", "description": "Balanced performance"},
        {"id": "claude-opus-4-20250514", "name": "Claude Opus 4", "description": "Most capable, extended thinking"},
    ],
}


# ============================================
# HELPER FUNCTIONS
# ============================================

def _call_with_timeout(fn, timeout_s: int, retries: int, label: str):
    """Execute function with timeout and retries."""
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


_URL_RE = re.compile(r'\bhttps?://[^\s\)\]]+', re.IGNORECASE)
_MD_LINK_RE = re.compile(r'\[([^\]]{0,200})\]\((https?://[^\s\)]+)\)')

def _fallback_extract_sources(response_text: str):
    """Extract sources from response text when provider did not return any."""
    if not response_text:
        return []
    found = []
    for m in _MD_LINK_RE.finditer(response_text):
        title = m.group(1).strip() or None
        url = m.group(2).strip()
        found.append({"url": url, "title": title})
    for m in _URL_RE.finditer(response_text):
        url = m.group(0).strip().rstrip(").,;")
        if url not in [f["url"] for f in found]:
            found.append({"url": url, "title": None})
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


# Thread-safe lock for database writes
_db_lock = threading.Lock()


# ============================================
# MAIN EXECUTION SERVICE
# ============================================

class GEOTrackerService:
    """
    Service class that wraps the GEO tracker execution.
    Supports configurable brand name, model selection, parallel execution, and progress tracking.
    """
    
    def __init__(self):
        init_db()
    
    def get_available_models(self, provider: str = None) -> Dict[str, List[Dict]]:
        """Get available models for providers."""
        if provider:
            return {provider: AVAILABLE_MODELS.get(provider, [])}
        return AVAILABLE_MODELS
    
    def generate_ai_queries(
        self,
        company_name: str,
        industry: str,
        description: Optional[str] = None,
        target_market: Optional[str] = None,
        language: str = "en",
        count: int = 25,
        focus_areas: Optional[List[str]] = None,
        competitor_names: Optional[List[str]] = None,
        provider: str = "auto",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate industry-specific queries using AI.
        
        Args:
            company_name: Name of the company/brand
            industry: Industry/sector
            description: Company description
            target_market: Target market description
            language: Language code (en, de, fr, es, it)
            count: Number of queries to generate
            focus_areas: Specific areas to focus on
            competitor_names: Known competitor names
            provider: LLM provider to use (openai, gemini, auto)
            model: Specific model to use
        
        Returns:
            Dict with generated queries and metadata
        """
        context = BusinessContext(
            company_name=company_name,
            industry=industry,
            description=description,
            target_market=target_market,
            language=language,
            focus_areas=focus_areas,
            competitor_names=competitor_names,
        )
        
        try:
            # Try to generate with AI
            queries = generate_queries(
                context=context,
                count=count,
                provider=provider,
                model=model,
            )
            generated_by = "ai"
        except Exception as e:
            print(f"[warning] AI query generation failed: {e}, using fallback")
            queries = get_fallback_queries(context, count)
            generated_by = "fallback"
        
        return {
            "queries": queries,
            "count": len(queries),
            "generated_by": generated_by,
            "context": {
                "company_name": company_name,
                "industry": industry,
                "language": language,
            }
        }
    
    def _process_single_query(
        self,
        provider_name_str: str,
        query: QueryCreate,
        config: RunConfigCreate,
        model: str,
        mode: str,
        brand_needle: str,
    ) -> Dict[str, Any]:
        """
        Process a single query for a single provider.
        This is designed to be run in parallel.
        Returns the result dictionary.
        """
        question = query.question
        
        # Build prompt with context
        if config.raw:
            prompt_text = question
        else:
            header = ""
            if mode == "provider_web" and (config.market or config.lang):
                header = f"(Market: {config.market or '-'}; Language: {config.lang or '-'})\n\n"
            prompt_text = header + question
        
        # Get provider instance (create new instance for thread safety)
        ProviderCls = PROVIDERS.get(provider_name_str)
        if not ProviderCls:
            return {"error": f"Unknown provider: {provider_name_str}"}
        
        provider = ProviderCls()
        
        # Insert run record (thread-safe)
        with _db_lock:
            run_id = insert_run(
                provider=provider_name_str,
                model=model,
                prompt_id=query.prompt_id or f"q_{hash(question) % 10000}",
                category=query.category or "custom",
                mode=mode,
                question=question,
                prompt_text=prompt_text,
                market=config.market,
                lang=config.lang,
                extra={
                    "raw": bool(config.raw),
                    "brand_name": brand_needle,
                    "company_id": config.company_id,
                }
            )
        
        # Make provider call
        try:
            label = f"{provider_name_str}:{mode}"
            if mode == "provider_web" and hasattr(provider, "generate_provider_web"):
                result = _call_with_timeout(
                    lambda: provider.generate_provider_web(prompt_text, model=model),
                    config.request_timeout,
                    config.max_retries,
                    label
                )
            else:
                result = _call_with_timeout(
                    lambda: provider.generate(prompt_text, model=model),
                    config.request_timeout,
                    config.max_retries,
                    label
                )
        except Exception as e:
            print(f"[error] Provider call failed: {e}", file=sys.stderr)
            result = {"text": "", "error": str(e), "sources": []}
        
        response_text = result.get("text", "") or ""
        latency_ms = result.get("latency_ms")
        tokens_in = result.get("tokens_in")
        tokens_out = result.get("tokens_out")
        cost_usd = result.get("cost_usd")
        
        provider_sources = result.get("sources") or []
        if not provider_sources:
            provider_sources = _fallback_extract_sources(response_text)
        
        # Save response to database (thread-safe)
        with _db_lock:
            insert_response(
                run_id=run_id,
                response_text=response_text,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                provider_sources=provider_sources
            )
        
        # Compute metrics with configurable brand
        # Use LLM-based brand detection with industry and market context
        other_brands = detect_competitor_brands(
            response_text, 
            provider_sources, 
            brand_needle,
            industry=config.industry or "",
            market=config.market or ""
        )
        
        presence_val = compute_presence_rate(response_text, brand_needle)
        brand_present = bool(presence_val and presence_val > 0)
        
        if brand_present:
            presence = float(presence_val)
            sentiment = compute_sentiment(response_text)
        else:
            if other_brands:
                presence = 0.0
                sentiment = None
            else:
                presence = None
                sentiment = None
        
        trust_authority, trust_sunday = compute_trustworthiness(response_text, provider_sources)
        
        # Store detailed metrics (thread-safe)
        details = {
            "brand_needle": brand_needle,
            "brand_present": brand_present,
            "other_brands_detected": sorted(other_brands),
            "company_id": config.company_id,
            "model": model,
            "openai_model": config.openai_model,
            "gemini_model": config.gemini_model,
        }
        
        with _db_lock:
            try:
                insert_metrics(run_id, presence, sentiment, trust_authority, trust_sunday, details)
            except TypeError:
                insert_metrics(run_id, presence, sentiment, trust_authority, details)
        
        # Build result object
        query_result = {
            "run_id": run_id,
            "prompt_id": query.prompt_id,
            "category": query.category,
            "question": question,
            "provider": provider_name_str,
            "model": model,
            "mode": mode,
            "response_text": response_text,
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "presence": presence,
            "sentiment": sentiment,
            "trust_authority": trust_authority,
            "trust_sunday": trust_sunday,
            "brand_mentioned": brand_present,
            "other_brands_detected": list(other_brands),
            "sources": [{"url": s.get("url"), "title": s.get("title")} for s in provider_sources],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        return query_result
    
    def execute_run(
        self,
        config: RunConfigCreate,
        queries: List[QueryCreate],
        job: Optional[Job] = None
    ) -> Dict[str, Any]:
        """
        Execute a GEO tracker run with the given configuration.
        
        PARALLEL EXECUTION: Providers run in parallel for better performance.
        SQLite with WAL mode handles concurrent writes safely.
        
        Args:
            config: Run configuration including brand name, providers, models
            queries: List of queries to execute
            job: Optional Job object for progress tracking
        
        Returns:
            Dictionary with run results
        """
        brand_needle = config.brand_name
        results = []
        
        # Calculate total tasks
        provider_count = len(config.providers)
        total_tasks = len(queries) * provider_count
        
        if job:
            job.total_tasks = total_tasks
        
        # Build list of all tasks to execute
        tasks = []
        for provider_name in config.providers:
            provider_name_str = provider_name.value if hasattr(provider_name, 'value') else str(provider_name)
            
            # Get model for this provider from config
            if provider_name_str == "openai":
                model = config.openai_model or OPENAI_DEFAULT_MODEL
            elif provider_name_str == "gemini":
                model = config.gemini_model or GEMINI_DEFAULT_MODEL
            elif provider_name_str == "perplexity":
                model = config.perplexity_model or PERPLEXITY_DEFAULT_MODEL
            elif provider_name_str == "anthropic":
                model = config.anthropic_model or ANTHROPIC_DEFAULT_MODEL
            else:
                model = OPENAI_DEFAULT_MODEL
            
            mode = config.mode.value if hasattr(config.mode, 'value') else str(config.mode)
            
            for query in queries:
                tasks.append({
                    "provider": provider_name_str,
                    "query": query,
                    "model": model,
                    "mode": mode,
                })
        
        # Execute tasks in parallel
        # Use max 4 workers to avoid overwhelming APIs
        max_workers = min(4, len(config.providers))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in tasks:
                if job and job.status == JobStatus.CANCELLED:
                    break
                
                future = executor.submit(
                    self._process_single_query,
                    task["provider"],
                    task["query"],
                    config,
                    task["model"],
                    task["mode"],
                    brand_needle,
                )
                future_to_task[future] = task
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                if job and job.status == JobStatus.CANCELLED:
                    break
                
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result and "error" not in result:
                        results.append(result)
                    
                    if job:
                        job.completed_tasks += 1
                        job.current_provider = task["provider"]
                        q = task["query"].question
                        job.current_query = q[:50] + "..." if len(q) > 50 else q
                        
                except Exception as e:
                    print(f"[error] Task failed: {e}", file=sys.stderr)
                    if job:
                        job.failed_tasks += 1
        
        # Calculate summary
        summary = self._calculate_summary(config, results)

        # Record brand run history
        if config.brand_name and results:
            try:
                # Get or create brand entry
                providers_list = [
                    p.value if hasattr(p, 'value') else str(p)
                    for p in config.providers
                ]
                brand_id = get_or_create_brand(
                    brand_name=config.brand_name,
                    industry=config.industry,
                    market=config.market,
                    company_id=config.company_id
                )

                # Record this run
                job_id = job.id if job else None
                record_brand_run(
                    brand_id=brand_id,
                    job_id=job_id,
                    providers=providers_list,
                    mode=config.mode.value if hasattr(config.mode, 'value') else str(config.mode),
                    total_queries=len(results),
                    visibility_pct=summary.get("overall_visibility", 0),
                    avg_sentiment=summary.get("avg_sentiment"),
                    avg_trust=summary.get("avg_trust_authority"),
                    competitor_summary=summary.get("competitor_visibility", {}),
                    extra={
                        "models_used": summary.get("models_used", {}),
                        "provider_visibility": summary.get("provider_visibility", {})
                    }
                )
                print(f"[brand_history] Recorded run for brand '{config.brand_name}' (id={brand_id})")
            except Exception as e:
                print(f"[brand_history] Failed to record brand run: {e}", file=sys.stderr)

        return {
            "summary": summary,
            "results": results
        }
    
    def _calculate_summary(self, config: RunConfigCreate, results: List[Dict]) -> Dict:
        """Calculate summary metrics from results."""
        total_queries = len(results)
        brand_mentioned_count = sum(1 for r in results if r.get("brand_mentioned"))
        
        # Overall visibility
        overall_visibility = (brand_mentioned_count / total_queries * 100) if total_queries > 0 else 0
        
        # Average sentiment (only where brand was mentioned)
        sentiments = [r["sentiment"] for r in results if r.get("sentiment") is not None]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else None
        
        # Average trust
        trusts = [r["trust_authority"] for r in results if r.get("trust_authority") is not None]
        avg_trust = sum(trusts) / len(trusts) if trusts else None
        
        # Per-provider visibility
        provider_visibility = {}
        for provider in config.providers:
            prov_str = provider.value if hasattr(provider, 'value') else str(provider)
            prov_results = [r for r in results if r["provider"] == prov_str]
            prov_mentioned = sum(1 for r in prov_results if r.get("brand_mentioned"))
            if prov_results:
                provider_visibility[prov_str] = round(prov_mentioned / len(prov_results) * 100, 2)
        
        # Competitor visibility - only include actual brands, not common words
        competitor_counts = {}
        for r in results:
            for comp in r.get("other_brands_detected", []):
                competitor_counts[comp] = competitor_counts.get(comp, 0) + 1
        
        # Sort by count and take top 15
        competitor_visibility = {}
        for comp, count in sorted(competitor_counts.items(), key=lambda x: -x[1])[:15]:
            competitor_visibility[comp] = round(count / total_queries * 100, 2)
        
        return {
            "run_id": results[0]["run_id"] if results else None,
            "company_id": config.company_id,
            "brand_name": config.brand_name,
            "status": "completed",
            "total_queries": total_queries,
            "total_responses": total_queries,
            "overall_visibility": round(overall_visibility, 2),
            "avg_sentiment": round(avg_sentiment, 3) if avg_sentiment is not None else None,
            "avg_trust_authority": round(avg_trust, 3) if avg_trust is not None else None,
            "provider_visibility": provider_visibility,
            "competitor_visibility": competitor_visibility,
            "models_used": {
                "openai": config.openai_model,
                "gemini": config.gemini_model,
                "perplexity": config.perplexity_model,
                "anthropic": config.anthropic_model,
            },
            "execution_mode": "parallel",
        }
    
    def get_results_from_db(
        self,
        company_id: Optional[str] = None,
        limit: int = 100,
        since_days: int = 7
    ) -> List[Dict]:
        """Fetch results from the database."""
        con = _connect()
        
        query = """
            SELECT 
                r.id as run_id,
                r.run_ts,
                r.provider,
                r.model,
                r.mode,
                r.prompt_id,
                r.category,
                r.question,
                r.prompt_text,
                r.market,
                r.lang,
                r.extra,
                resp.response_text,
                resp.latency_ms,
                resp.tokens_in,
                resp.tokens_out,
                resp.provider_sources,
                m.presence,
                m.sentiment,
                m.trust_authority,
                m.trust_sunday,
                m.details
            FROM runs r
            LEFT JOIN responses resp ON r.id = resp.run_id
            LEFT JOIN metrics m ON r.id = m.run_id
            WHERE r.run_ts >= datetime('now', ?)
            ORDER BY r.run_ts DESC
            LIMIT ?
        """
        
        cursor = con.cursor()
        cursor.execute(query, (f'-{since_days} days', limit))
        rows = cursor.fetchall()
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        for row in rows:
            result = dict(zip(columns, row))
            # Parse JSON fields
            if result.get("provider_sources"):
                try:
                    result["sources"] = json.loads(result["provider_sources"])
                except:
                    result["sources"] = []
            else:
                result["sources"] = []
            
            if result.get("details"):
                try:
                    details = json.loads(result["details"])
                    result["brand_mentioned"] = details.get("brand_present", False)
                    result["other_brands_detected"] = details.get("other_brands_detected", [])
                    result["brand_name"] = details.get("brand_needle", "")
                    # Filter by company_id if provided
                    if company_id and details.get("company_id") != company_id:
                        continue
                except:
                    result["brand_mentioned"] = False
                    result["other_brands_detected"] = []
            
            if result.get("extra"):
                try:
                    extra = json.loads(result["extra"])
                    result["brand_name"] = extra.get("brand_name", "")
                except:
                    pass
            
            results.append(result)
        
        return results


    def get_run_summaries(
        self,
        company_id: Optional[str] = None,
        limit: int = 50,
        since_days: int = 30
    ) -> List[Dict]:
        """
        Get summarized run data from brand_runs table.
        This provides a high-level view of all runs for the Previous Runs tab.
        Uses brand_runs which has proper grouping by job_id.
        """
        from db import get_all_brand_runs

        runs = get_all_brand_runs(
            company_id=company_id,
            limit=limit,
            since_days=since_days
        )

        # Transform to the format expected by frontend
        summaries = []
        for run in runs:
            # Calculate brand mentions from visibility
            total_queries = run.get("total_queries") or 0
            visibility_pct = run.get("visibility_pct") or 0
            brand_mentions = int(total_queries * visibility_pct / 100) if total_queries > 0 else 0

            summaries.append({
                "run_ts": run.get("run_at"),
                "job_id": run.get("job_id"),
                "brand_name": run.get("brand_name", ""),
                "providers": run.get("providers") or [],
                "mode": run.get("mode"),
                "market": run.get("market"),
                "lang": run.get("extra", {}).get("lang") if isinstance(run.get("extra"), dict) else None,
                "total_queries": total_queries,
                "brand_mentions": brand_mentions,
                "visibility_pct": round(visibility_pct, 2),
                "avg_sentiment": round(run["avg_sentiment"], 3) if run.get("avg_sentiment") else None,
                "avg_trust": round(run["avg_trust"], 3) if run.get("avg_trust") else None,
                "avg_latency_ms": run.get("extra", {}).get("avg_latency_ms") if isinstance(run.get("extra"), dict) else None,
                "company_id": run.get("company_id"),
            })

        return summaries


# Global service instance
geo_service = GEOTrackerService()
