"""
Service for generating AI-powered visibility reports and recommendations.

This service analyzes GEO tracker results and uses an LLM to generate
actionable recommendations for improving brand visibility in AI assistants.
"""
import json
import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import insert_recommendation, get_latest_recommendation

# Default analysis settings
DEFAULT_ANALYSIS_PROVIDER = "openai"
DEFAULT_ANALYSIS_MODEL = "gpt-4.1"


def _build_analysis_prompt(
    results_summary: Dict,
    detailed_results: List[Dict],
    brand_name: str
) -> str:
    """Build a detailed prompt for visibility analysis."""

    # Extract key metrics
    visibility = results_summary.get("overall_visibility", 0)
    sentiment = results_summary.get("avg_sentiment")
    trust = results_summary.get("avg_trust_authority")
    competitors = results_summary.get("competitor_visibility", {})
    provider_vis = results_summary.get("provider_visibility", {})
    total_queries = results_summary.get("total_queries", 0)

    # Get samples of responses where brand WAS mentioned vs NOT mentioned
    mentioned_samples = [r for r in detailed_results if r.get("brand_mentioned")][:3]
    not_mentioned_samples = [r for r in detailed_results if not r.get("brand_mentioned")][:3]

    prompt = f"""You are a GEO (Generative Engine Optimization) expert analyzing brand visibility in AI assistant responses.

## Brand Being Analyzed: {brand_name}

## Current Performance Metrics:
- Overall Visibility: {visibility:.1f}% (brand mentioned in {visibility:.1f}% of AI responses)
- Total Queries Analyzed: {total_queries}
- Average Sentiment when mentioned: {f"{sentiment:.2f}" if sentiment is not None else 'N/A'} (scale: -1 to +1)
- Average Trust Score: {f"{trust:.2f}" if trust is not None else 'N/A'} (scale: 0 to 1)

## Visibility by AI Provider:
"""

    for prov, vis in provider_vis.items():
        prompt += f"- {prov.title()}: {vis:.1f}%\n"

    prompt += "\n## Top Competitor Visibility (brands mentioned in AI responses):\n"

    if competitors:
        sorted_comps = sorted(competitors.items(), key=lambda x: -x[1])[:10]
        for comp, vis in sorted_comps:
            prompt += f"- {comp}: {vis:.1f}%\n"
    else:
        prompt += "- No competitor brands detected\n"

    prompt += "\n## Sample Responses WHERE BRAND WAS MENTIONED:\n"

    if mentioned_samples:
        for i, sample in enumerate(mentioned_samples, 1):
            sources = sample.get('sources', [])
            source_urls = [s.get('url', '') for s in sources[:3] if s.get('url')]
            prompt += f"""
--- Sample {i} ---
Question: {sample.get('question', '')}
Provider: {sample.get('provider', '').title()}
Response excerpt: {(sample.get('response_text', '') or '')[:500]}...
Sources cited: {source_urls if source_urls else 'None'}
"""
    else:
        prompt += "\n(No samples where brand was mentioned)\n"

    prompt += "\n## Sample Responses WHERE BRAND WAS NOT MENTIONED:\n"

    if not_mentioned_samples:
        for i, sample in enumerate(not_mentioned_samples, 1):
            competitors_in_response = sample.get('other_brands_detected', [])
            prompt += f"""
--- Sample {i} ---
Question: {sample.get('question', '')}
Provider: {sample.get('provider', '').title()}
Competitors mentioned instead: {competitors_in_response[:5] if competitors_in_response else 'None'}
Response excerpt: {(sample.get('response_text', '') or '')[:500]}...
"""
    else:
        prompt += "\n(Brand was mentioned in all responses)\n"

    prompt += """

## Your Task:
Analyze this data and provide a detailed visibility report with ACTIONABLE recommendations.

Structure your response EXACTLY as follows:

### EXECUTIVE SUMMARY
[2-3 sentences summarizing the brand's current AI visibility status and the most critical finding]

### KEY FINDINGS
1. [Finding 1 with supporting data]
2. [Finding 2 with supporting data]
3. [Finding 3 with supporting data]

### CONTENT OPTIMIZATION RECOMMENDATIONS
For each recommendation, explain WHY and HOW:

1. **[Recommendation Title]**
   - Why: [Explain the gap or opportunity based on the data]
   - Action: [Specific, actionable steps to take]
   - Expected Impact: [What improvement to expect]

2. **[Recommendation Title]**
   - Why: [...]
   - Action: [...]
   - Expected Impact: [...]

3. **[Recommendation Title]**
   - Why: [...]
   - Action: [...]
   - Expected Impact: [...]

### COMPETITIVE ANALYSIS
- Why competitors are being mentioned instead of your brand
- Specific gaps in your content/authority compared to competitors
- Strategies to close these gaps

### AUTHORITY BUILDING RECOMMENDATIONS
1. [Specific authority-building action with rationale]
2. [Specific authority-building action with rationale]
3. [Specific authority-building action with rationale]

### PRIORITY ACTION ITEMS
List the top 3 most impactful actions in order of priority:
1. **[Highest priority action]** - [Brief reason why this is #1]
2. **[Second priority action]** - [Brief reason]
3. **[Third priority action]** - [Brief reason]

Be specific, actionable, and base all recommendations on the actual data provided.
Focus on practical steps the brand can take to improve their AI visibility.
"""

    return prompt


async def generate_visibility_report(
    results_summary: Dict[str, Any],
    detailed_results: List[Dict[str, Any]],
    brand_name: str,
    job_id: Optional[str] = None,
    provider: str = DEFAULT_ANALYSIS_PROVIDER,
    model: str = DEFAULT_ANALYSIS_MODEL
) -> Dict[str, Any]:
    """
    Generate an AI-powered visibility report.

    Args:
        results_summary: Summary metrics from the run
        detailed_results: List of individual query results
        brand_name: Name of the brand being analyzed
        job_id: Optional job ID for caching
        provider: LLM provider to use (openai, gemini, anthropic, perplexity)
        model: Specific model to use

    Returns:
        {
            "report": "...",  # The full analysis text
            "generated_at": "...",
            "provider": "openai",
            "model": "gpt-4.1",
            "tokens_used": 1234,
            "saved": True/False
        }
    """
    prompt = _build_analysis_prompt(results_summary, detailed_results, brand_name)

    # Import provider dynamically to avoid circular imports
    from llm_providers.openai_provider import OpenAIProvider
    from llm_providers.gemini_provider import GeminiProvider

    # Get provider instance
    if provider == "openai":
        llm = OpenAIProvider()
    elif provider == "gemini":
        llm = GeminiProvider()
    else:
        # Default to OpenAI
        llm = OpenAIProvider()
        provider = "openai"

    # Generate the report
    try:
        result = llm.generate(prompt, model=model)
    except Exception as e:
        return {
            "report": f"Failed to generate report: {str(e)}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "tokens_used": 0,
            "saved": False,
            "error": str(e)
        }

    report_text = result.get("text", "")
    tokens_in = result.get("tokens_in") or 0
    tokens_out = result.get("tokens_out") or 0
    tokens_used = tokens_in + tokens_out

    # Save to database if job_id provided
    saved = False
    if job_id and report_text:
        try:
            insert_recommendation(
                job_id=job_id,
                analysis_type="visibility_report",
                content=report_text,
                brand_name=brand_name,
                provider=provider,
                model=model,
                tokens_used=tokens_used
            )
            saved = True
        except Exception as e:
            print(f"[report_service] Failed to save recommendation: {e}")

    return {
        "report": report_text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "tokens_used": tokens_used,
        "saved": saved,
        "brand_name": brand_name,
    }


def get_cached_report(job_id: str) -> Optional[Dict[str, Any]]:
    """Get a previously generated report for a job."""
    rec = get_latest_recommendation(job_id, analysis_type="visibility_report")
    if rec:
        return {
            "report": rec.get("content", ""),
            "generated_at": rec.get("created_at"),
            "provider": rec.get("provider"),
            "model": rec.get("model"),
            "tokens_used": rec.get("tokens_used"),
            "brand_name": rec.get("brand_name"),
            "from_cache": True
        }
    return None
