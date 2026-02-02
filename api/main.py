# api/main.py
"""
FastAPI application for GEO Tracker SaaS.

Run with:
    uvicorn api.main:app --reload --port 8000

Or:
    python -m uvicorn api.main:app --reload --port 8000
"""
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .models import (
    RunConfigCreate, QueryCreate, JobCreatedResponse, HealthResponse,
    ErrorResponse, RunProgress, ProviderEnum
)
from .jobs import job_manager, Job, JobStatus
from .services import geo_service
from .sheets_service import fetch_sheet_prompts, extract_sheet_id
from .report_service import generate_visibility_report, get_cached_report
from .email_service import send_lead_acknowledgment, send_lead_emails, is_email_service_configured
from .admin_service import (
    authenticate_admin, verify_token, initialize_default_admin,
    get_leads_for_role, can_update_lead, get_user_permissions
)
from .user_service import (
    register_user, authenticate_user, get_user_from_token,
    verify_user_token, update_user, initialize_demo_user,
    initialize_admin_dashboard_user, get_total_users
)
from db import insert_lead, get_leads_stats, update_lead_status, get_lead_by_id

# Security
security = HTTPBearer(auto_error=False)


# ============================================
# APP INITIALIZATION
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    print("🚀 GEO Tracker API starting...")
    # Initialize default admin users
    initialize_default_admin()
    # Initialize demo webapp user
    initialize_demo_user()
    yield
    # Shutdown
    print("👋 GEO Tracker API shutting down...")


app = FastAPI(
    title="GEO Tracker API",
    description="""
    API for running GEO (Generative Engine Optimization) analysis.
    
    Track your brand's visibility across AI assistants like ChatGPT, Claude, Gemini, and Perplexity.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# HEALTH CHECK
# ============================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and available providers."""
    available_providers = []

    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        available_providers.append("openai")

    # Check Gemini
    if os.getenv("GOOGLE_API_KEY"):
        available_providers.append("gemini")

    # Check Perplexity
    if os.getenv("PERPLEXITY_API_KEY"):
        available_providers.append("perplexity")

    # Check Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        available_providers.append("anthropic")

    return HealthResponse(
        status="healthy",
        version="2.0.0",
        providers_available=available_providers
    )


# ============================================
# MODELS ENDPOINT
# ============================================

@app.get(
    "/api/models",
    tags=["Models"],
    summary="Get available LLM models"
)
async def get_available_models(provider: Optional[str] = None):
    """
    Get available LLM models for each provider.
    
    Optionally filter by provider name (openai, gemini).
    """
    return geo_service.get_available_models(provider)


# ============================================
# RUN MANAGEMENT
# ============================================

@app.post(
    "/api/runs",
    response_model=JobCreatedResponse,
    tags=["Runs"],
    summary="Start a new GEO tracker run"
)
async def create_run(config: RunConfigCreate):
    """
    Start a new GEO tracker run.
    
    This endpoint queues a background job that will:
    1. Execute queries against the selected LLM providers
    2. Analyze responses for brand mentions, sentiment, etc.
    3. Store results in the database
    
    Use the returned `job_id` to check progress via `/api/runs/{job_id}/status`.
    """
    # Validate that we have queries
    if not config.queries or len(config.queries) == 0:
        raise HTTPException(
            status_code=400,
            detail="At least one query is required. Provide queries in the 'queries' field."
        )
    
    # Validate providers are available
    for provider in config.providers:
        prov_str = provider.value if hasattr(provider, 'value') else str(provider)
        if prov_str == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=400, detail="OpenAI API key not configured")
        if prov_str == "gemini" and not os.getenv("GOOGLE_API_KEY"):
            raise HTTPException(status_code=400, detail="Google API key not configured")
        if prov_str == "perplexity" and not os.getenv("PERPLEXITY_API_KEY"):
            raise HTTPException(status_code=400, detail="Perplexity API key not configured")
        if prov_str == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            raise HTTPException(status_code=400, detail="Anthropic API key not configured")
    
    # Calculate estimated tasks
    total_tasks = len(config.queries) * len(config.providers)
    
    # Create job
    job = job_manager.create_job(total_tasks=total_tasks)
    
    # Estimate duration (roughly 3-5 seconds per query per provider)
    estimated_duration = total_tasks * 4
    
    # Submit background job
    def run_geo_tracker(job: Job):
        """Background task to run GEO tracker."""
        result = geo_service.execute_run(
            config=config,
            queries=config.queries,
            job=job
        )
        return result
    
    job_manager.submit(run_geo_tracker, job=job)
    
    return JobCreatedResponse(
        job_id=job.id,
        run_id=job.id,  # Use job_id as run_id for now
        status=JobStatus.PENDING,
        message=f"Run queued with {total_tasks} tasks ({len(config.queries)} queries × {len(config.providers)} providers)",
        estimated_duration_seconds=estimated_duration
    )


@app.get(
    "/api/runs/{job_id}/status",
    response_model=RunProgress,
    tags=["Runs"],
    summary="Get run progress"
)
async def get_run_status(job_id: str):
    """
    Get the current status and progress of a run.
    
    Poll this endpoint to track progress of a running job.
    """
    status = job_manager.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return RunProgress(
        run_id=status["run_id"] or job_id,
        status=JobStatus(status["status"]),
        total_tasks=status["total_tasks"],
        completed_tasks=status["completed_tasks"],
        failed_tasks=status["failed_tasks"],
        progress_percent=status["progress_percent"],
        current_provider=status["current_provider"],
        current_query=status["current_query"],
        estimated_remaining_seconds=status["estimated_remaining_seconds"],
        started_at=datetime.fromisoformat(status["started_at"]) if status["started_at"] else None,
        updated_at=datetime.now(timezone.utc),
        error=status.get("error")
    )


@app.get(
    "/api/runs/{job_id}/results",
    tags=["Runs"],
    summary="Get run results"
)
async def get_run_results(job_id: str):
    """
    Get the results of a completed run.

    Returns the full results including:
    - Summary metrics (visibility, sentiment, etc.)
    - Individual query results
    - Competitor analysis

    First checks in-memory jobs (for active runs), then falls back
    to database (for historical runs after server restart).
    """
    # First try in-memory job manager (for active/recent runs)
    job = job_manager.get_job(job_id)
    if job:
        if job.status == JobStatus.PENDING:
            raise HTTPException(status_code=400, detail="Run has not started yet")

        if job.status == JobStatus.RUNNING:
            raise HTTPException(status_code=400, detail="Run is still in progress. Check /status for progress.")

        if job.status == JobStatus.FAILED:
            raise HTTPException(status_code=500, detail=f"Run failed: {job.error}")

        if job.status == JobStatus.CANCELLED:
            raise HTTPException(status_code=400, detail="Run was cancelled")

        if not job.result:
            raise HTTPException(status_code=500, detail="No results available")

        return job.result

    # Job not in memory - try to load from database (historical runs)
    db_results = geo_service.get_results_by_job_id(job_id)
    if db_results:
        return db_results

    # Not found anywhere
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@app.post(
    "/api/runs/{job_id}/cancel",
    tags=["Runs"],
    summary="Cancel a running job"
)
async def cancel_run(job_id: str):
    """Cancel a running job."""
    success = job_manager.cancel(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel job (not found or not running)")
    return {"message": "Cancellation requested", "job_id": job_id}


@app.get(
    "/api/runs",
    tags=["Runs"],
    summary="List recent jobs"
)
async def list_runs(limit: int = Query(default=20, ge=1, le=100)):
    """List recent run jobs."""
    return job_manager.list_jobs(limit=limit)


# ============================================
# RESULTS / HISTORY
# ============================================

@app.get(
    "/api/results",
    tags=["Results"],
    summary="Get historical results from database"
)
async def get_results(
    company_id: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    since_days: int = Query(default=7, ge=1, le=365)
):
    """
    Fetch historical results from the database.

    This returns persisted results from previous runs, not just the current session.
    """
    results = geo_service.get_results_from_db(
        company_id=company_id,
        limit=limit,
        since_days=since_days
    )
    return {"results": results, "count": len(results)}


@app.get(
    "/api/runs/history",
    tags=["Results"],
    summary="Get summarized run history"
)
async def get_run_history(
    company_id: Optional[str] = Query(None, description="Filter by company ID"),
    limit: int = Query(default=50, ge=1, le=200),
    since_days: int = Query(default=30, ge=1, le=365)
):
    """
    Get a summarized view of all previous runs.

    This returns aggregated data per run, including:
    - Brand name and timestamp
    - Providers and models used
    - Total queries and visibility percentage
    - Average sentiment and trust scores

    Ideal for displaying in a "Previous Runs" overview tab.
    """
    summaries = geo_service.get_run_summaries(
        company_id=company_id,
        limit=limit,
        since_days=since_days
    )
    return {"runs": summaries, "count": len(summaries)}


# ============================================
# QUERY GENERATION (AI-Powered)
# ============================================

from pydantic import BaseModel, Field

class QueryGenerationRequest(BaseModel):
    """Request model for AI query generation."""
    company_name: str = Field(..., description="Name of the company/brand")
    industry: str = Field(..., description="Industry/sector")
    description: Optional[str] = Field(None, description="Business context: brand positioning, products, target audience, unique selling points")
    target_market: Optional[str] = Field(None, description="Target market (e.g., 'Germany', 'Europe')")
    language: str = Field("en", description="Language code (en, de, fr, es, it)")
    count: int = Field(15, ge=5, le=25, description="Number of queries to generate (max 25)")
    focus_areas: Optional[List[str]] = Field(None, description="Specific areas to focus on")
    competitor_names: Optional[List[str]] = Field(None, description="Known competitor names")
    provider: str = Field("auto", description="LLM provider: openai, gemini, or auto")
    model: Optional[str] = Field(None, description="Specific model to use")


@app.post(
    "/api/queries/generate",
    tags=["Queries"],
    summary="Generate queries using AI"
)
async def generate_queries_endpoint(request: QueryGenerationRequest):
    """
    Generate industry-specific queries using AI (OpenAI or Gemini).
    
    This endpoint uses the configured LLM API to generate relevant, 
    industry-specific queries based on your business context.
    
    **API Key Required**: Set OPENAI_API_KEY or GOOGLE_API_KEY environment variable.
    
    The generated queries will include:
    - General product/service questions
    - Brand comparison queries
    - "Best of" queries
    - Problem-solving queries
    - Review/recommendation queries
    
    Returns a mix of queries, mostly NOT mentioning your brand name,
    to test organic visibility in AI responses.
    """
    # Check if we have API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_gemini = bool(os.getenv("GOOGLE_API_KEY"))
    
    if not has_openai and not has_gemini:
        raise HTTPException(
            status_code=400,
            detail="No LLM API key configured. Set OPENAI_API_KEY or GOOGLE_API_KEY environment variable."
        )
    
    # Validate provider choice
    if request.provider == "openai" and not has_openai:
        raise HTTPException(status_code=400, detail="OpenAI API key not configured")
    if request.provider == "gemini" and not has_gemini:
        raise HTTPException(status_code=400, detail="Google API key not configured")
    
    try:
        result = geo_service.generate_ai_queries(
            company_name=request.company_name,
            industry=request.industry,
            description=request.description,
            target_market=request.target_market,
            language=request.language,
            count=request.count,
            focus_areas=request.focus_areas,
            competitor_names=request.competitor_names,
            provider=request.provider,
            model=request.model,
        )
        
        return {
            "queries": result["queries"],
            "count": result["count"],
            "generated_by": result["generated_by"],
            "context": result["context"],
            "note": "Queries generated using AI. Review and customize as needed." if result["generated_by"] == "ai" else "Using fallback queries (AI generation failed). Check API key."
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query generation failed: {str(e)}"
        )


@app.get(
    "/api/queries/sample",
    tags=["Queries"],
    summary="Get sample queries (no API key required)"
)
async def get_sample_queries(
    industry: str = Query(..., description="Industry/sector"),
    company_name: str = Query(..., description="Company name"),
    language: str = Query("en", description="Language code"),
    count: int = Query(25, ge=5, le=50, description="Number of queries")
):
    """
    Get sample queries without using AI.
    
    This is useful for testing or when no API key is available.
    Returns template-based queries for the given industry.
    """
    from query_generator import get_fallback_queries, BusinessContext
    
    context = BusinessContext(
        company_name=company_name,
        industry=industry,
        language=language,
    )
    
    queries = get_fallback_queries(context, count)
    
    return {
        "queries": queries,
        "count": len(queries),
        "generated_by": "template",
        "note": "These are template-based queries. For AI-generated queries, use POST /api/queries/generate with an API key."
    }


# ============================================
# GOOGLE SHEETS INTEGRATION
# ============================================

class SheetFetchRequest(BaseModel):
    """Request to fetch prompts from a Google Sheet."""
    sheet_url: str = Field(..., description="Google Sheet URL or ID")
    worksheet_name: Optional[str] = Field(None, description="Worksheet name (defaults to first sheet)")
    force_refresh: bool = Field(False, description="Force refresh from API (bypass cache)")


@app.post(
    "/api/sheets/prompts",
    tags=["Sheets"],
    summary="Fetch prompts from a user's Google Sheet"
)
async def fetch_sheet_prompts_endpoint(request: SheetFetchRequest):
    """
    Fetch and parse prompts from a user-provided Google Sheet.

    The sheet must be shared with the application's service account.
    Auto-detects question/prompt and category columns (supports multiple languages).

    **How to use:**
    1. Get the service account email from your GOOGLE_APPLICATION_CREDENTIALS JSON
    2. Share your Google Sheet with that email (give Viewer access)
    3. Copy the sheet URL and paste it here

    Returns all prompts with metadata. Use the 'count' and 'start' params
    in the run config to select a subset.
    """
    try:
        result = fetch_sheet_prompts(
            sheet_url_or_id=request.sheet_url,
            worksheet_name=request.worksheet_name,
            force_refresh=request.force_refresh
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch sheet: {str(e)}")


@app.get(
    "/api/sheets/validate",
    tags=["Sheets"],
    summary="Validate a Google Sheet URL"
)
async def validate_sheet_url(url: str = Query(..., description="Google Sheet URL or ID")):
    """
    Quick validation that a sheet URL is accessible.
    Returns basic info about the sheet without fetching all data.
    """
    try:
        # Extract sheet ID to validate format
        sheet_id = extract_sheet_id(url)

        # Fetch with minimal processing
        result = fetch_sheet_prompts(url)
        return {
            "valid": True,
            "sheet_id": sheet_id,
            "sheet_title": result.get("sheet_title", ""),
            "total_prompts": result.get("total_count", 0),
            "columns": result.get("all_columns", []),
            "columns_detected": result.get("columns_detected", {}),
        }
    except ValueError as e:
        return {"valid": False, "error": str(e)}
    except Exception as e:
        return {"valid": False, "error": str(e)}


# ============================================
# VISIBILITY REPORTS
# ============================================

class ReportRequest(BaseModel):
    """Request for generating a visibility report."""
    job_id: Optional[str] = Field(None, description="Job ID to associate report with (for caching)")
    brand_name: str = Field(..., description="Brand name being analyzed")
    results_summary: Dict[str, Any] = Field(..., description="Summary metrics from the run")
    detailed_results: List[Dict[str, Any]] = Field(..., description="Detailed query results")
    provider: str = Field("openai", description="LLM provider for analysis (openai, gemini)")
    model: str = Field("gpt-4.1", description="Model to use for analysis")
    force_regenerate: bool = Field(False, description="Regenerate even if cached report exists")


@app.post(
    "/api/reports/visibility",
    tags=["Reports"],
    summary="Generate AI-powered visibility report"
)
async def generate_report_endpoint(request: ReportRequest):
    """
    Generate a detailed visibility report with recommendations.

    This uses an LLM to analyze the run results and provide:
    - Executive summary of current visibility status
    - Key findings from the data
    - Content optimization recommendations
    - Competitive analysis
    - Authority building recommendations
    - Priority action items

    Reports are cached per job_id. Use force_regenerate=True to generate a new one.

    **Note:** This endpoint may take 15-30 seconds as it calls the LLM for analysis.
    """
    # Check for cached report first
    if request.job_id and not request.force_regenerate:
        cached = get_cached_report(request.job_id)
        if cached:
            return cached

    try:
        # Generate report (this can take 10-30 seconds)
        report = await generate_visibility_report(
            results_summary=request.results_summary,
            detailed_results=request.detailed_results,
            brand_name=request.brand_name,
            job_id=request.job_id,
            provider=request.provider,
            model=request.model
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get(
    "/api/reports/{job_id}",
    tags=["Reports"],
    summary="Get cached report for a job"
)
async def get_report_endpoint(job_id: str):
    """
    Get a previously generated report for a job.

    Returns the cached report if one exists, or 404 if not found.
    """
    cached = get_cached_report(job_id)
    if not cached:
        raise HTTPException(status_code=404, detail="No report found for this job")
    return cached


# ============================================
# BRAND HISTORY
# ============================================

from db import (
    get_all_brands, get_brand_by_id, get_brand_by_name,
    get_brand_run_history, delete_brand
)


@app.get(
    "/api/brands",
    tags=["Brands"],
    summary="List all tracked brands"
)
async def list_brands(
    company_id: Optional[str] = Query(None, description="Filter by company ID"),
    limit: int = Query(50, ge=1, le=200, description="Max brands to return")
):
    """
    Get all brands that have been tracked.

    Each brand entry includes:
    - Brand name and metadata
    - Total runs and queries executed
    - Average visibility score across all runs
    - Last run timestamp
    """
    brands = get_all_brands(company_id=company_id, limit=limit)
    return {
        "brands": brands,
        "count": len(brands)
    }


@app.get(
    "/api/brands/{brand_id}",
    tags=["Brands"],
    summary="Get brand details with run history"
)
async def get_brand(brand_id: int):
    """
    Get details for a specific brand by ID, including run history.
    """
    brand = get_brand_by_id(brand_id)
    if not brand:
        raise HTTPException(status_code=404, detail=f"Brand {brand_id} not found")

    # Also fetch the run history for this brand
    history = get_brand_run_history(brand_id, limit=20)

    return {
        "brand": brand,
        "history": history
    }


@app.get(
    "/api/brands/search/{brand_name}",
    tags=["Brands"],
    summary="Search brand by name"
)
async def search_brand(
    brand_name: str,
    company_id: Optional[str] = Query(None, description="Filter by company ID")
):
    """
    Search for a brand by name (case-insensitive).
    """
    brand = get_brand_by_name(brand_name, company_id=company_id)
    if not brand:
        raise HTTPException(status_code=404, detail=f"Brand '{brand_name}' not found")
    return brand


@app.get(
    "/api/brands/{brand_id}/history",
    tags=["Brands"],
    summary="Get run history for a brand"
)
async def get_brand_history(
    brand_id: int,
    limit: int = Query(20, ge=1, le=100, description="Max runs to return")
):
    """
    Get the run history for a specific brand.

    Each run entry includes:
    - Run timestamp
    - Providers used
    - Query count
    - Visibility percentage
    - Sentiment and trust scores
    - Top competitors detected
    """
    # Verify brand exists
    brand = get_brand_by_id(brand_id)
    if not brand:
        raise HTTPException(status_code=404, detail=f"Brand {brand_id} not found")

    history = get_brand_run_history(brand_id, limit=limit)
    return {
        "brand": brand,
        "history": history,
        "count": len(history)
    }


@app.delete(
    "/api/brands/{brand_id}",
    tags=["Brands"],
    summary="Delete a brand and its history"
)
async def delete_brand_endpoint(brand_id: int):
    """
    Delete a brand and all its run history.

    **Warning:** This cannot be undone. The original run data (queries, responses)
    remains in the database but will no longer be associated with this brand.
    """
    success = delete_brand(brand_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Brand {brand_id} not found")
    return {"message": f"Brand {brand_id} deleted successfully"}


@app.delete(
    "/api/admin/clear-runs",
    tags=["Admin"],
    summary="Clear all run data (admin only)"
)
async def clear_all_runs():
    """
    Clear all run data from the database.

    **Warning:** This is destructive and cannot be undone.
    Deletes all runs, responses, metrics, and brand_runs.
    Brands themselves are preserved.
    """
    from db import clear_all_run_data
    result = clear_all_run_data()
    return {
        "message": "All run data cleared successfully",
        "deleted": result
    }


# ============================================
# LEAD CAPTURE
# ============================================

class LeadSubmission(BaseModel):
    """Lead submission from contact form."""
    company: str = Field(..., description="Company name")
    email: str = Field(..., description="Contact email")
    website: Optional[str] = Field(None, description="Company website")
    industry: Optional[str] = Field(None, description="Industry/sector")
    service: str = Field(..., description="Service interested in")
    contact_name: Optional[str] = Field(None, description="Contact person name")


@app.post(
    "/api/leads",
    tags=["Leads"],
    summary="Submit a lead and send emails"
)
async def submit_lead(lead: LeadSubmission):
    """
    Submit a lead from the contact form.

    This endpoint:
    1. Saves the lead to the database
    2. Sends an acknowledgment email to the lead via Resend
    3. Sends a notification email to admin(s) via Resend

    This is now the PRIMARY lead capture endpoint (Formspree removed).
    """
    # Send both lead acknowledgment AND admin notification emails
    email_result = send_lead_emails(
        company_name=lead.company,
        email=lead.email,
        service=lead.service,
        website=lead.website,
        industry=lead.industry,
        contact_name=lead.contact_name
    )

    # Extract results for database
    lead_email_success = email_result.get("lead_email", {}).get("success", False)
    lead_email_id = email_result.get("lead_email", {}).get("message_id")

    # Save lead to database
    try:
        lead_id = insert_lead(
            company=lead.company,
            email=lead.email,
            service=lead.service,
            website=lead.website,
            industry=lead.industry,
            contact_name=lead.contact_name,
            email_sent=lead_email_success,
            email_id=lead_email_id
        )
    except Exception as e:
        # Log error but don't fail the request
        print(f"[leads] Failed to save lead to database: {e}")
        lead_id = None

    # Build response
    admin_email_result = email_result.get("admin_email", {})

    if lead_email_success:
        return {
            "success": True,
            "message": "Lead received and emails sent",
            "lead_id": lead_id,
            "emails": {
                "lead_acknowledgment": {
                    "sent": True,
                    "message_id": lead_email_id
                },
                "admin_notification": {
                    "sent": admin_email_result.get("success", False),
                    "message_id": admin_email_result.get("message_id"),
                    "sent_to": admin_email_result.get("sent_to", []),
                    "error": admin_email_result.get("error") if not admin_email_result.get("success") else None
                }
            }
        }
    else:
        # Lead email failed - still save but report the issue
        return {
            "success": True,  # Lead was still captured
            "message": "Lead received but acknowledgment email could not be sent",
            "lead_id": lead_id,
            "emails": {
                "lead_acknowledgment": {
                    "sent": False,
                    "error": email_result.get("lead_email", {}).get("error")
                },
                "admin_notification": {
                    "sent": admin_email_result.get("success", False),
                    "message_id": admin_email_result.get("message_id"),
                    "error": admin_email_result.get("error") if not admin_email_result.get("success") else None
                }
            }
        }


@app.get(
    "/api/leads/status",
    tags=["Leads"],
    summary="Check email service status"
)
async def check_email_status():
    """Check if the email service is properly configured."""
    configured = is_email_service_configured()
    return {
        "email_service_configured": configured,
        "provider": "resend" if configured else None
    }


# ============================================
# ADMIN AUTHENTICATION
# ============================================

class AdminLoginRequest(BaseModel):
    """Admin login request."""
    username: str = Field(..., description="Admin username")
    password: str = Field(..., description="Admin password")


@app.post(
    "/api/admin/reset-users",
    tags=["Admin"],
    summary="Reset ALL users (admin + webapp users) - one-time fix"
)
async def reset_admin_users(reset_key: str = Query(..., description="Reset key for security")):
    """
    Reset BOTH admin users AND webapp users by deleting and recreating them.

    This is a one-time fix for when users were created with a different
    SECRET_KEY than currently configured.

    Call with: POST /api/admin/reset-users?reset_key=reset-admin-2024
    """
    # Simple security check - require a specific key
    expected_key = os.getenv("ADMIN_RESET_KEY", "reset-admin-2024")
    if reset_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid reset key")

    try:
        from db import _connect
        conn = _connect()
        cur = conn.cursor()

        # 1. Delete existing ADMIN users (admin panel)
        cur.execute("DELETE FROM admin_users WHERE username IN ('admin', 'demo')")
        admin_deleted = cur.rowcount

        # 2. Delete existing WEBAPP users (dashboard login) - both demo and admin
        cur.execute("DELETE FROM users WHERE email IN ('demo@geotracker.io', 'admin@geotracker.io')")
        user_deleted = cur.rowcount

        conn.commit()

        # 3. Reinitialize admin users with current secret key (for admin panel)
        initialize_default_admin()

        # 4. Reinitialize webapp users with current secret key (for dashboard)
        initialize_demo_user()
        initialize_admin_dashboard_user()

        admin_password = os.getenv("ADMIN_PASSWORD", "geotracker2024!")

        return {
            "success": True,
            "message": f"Reset complete. Deleted {admin_deleted} admin panel users and {user_deleted} dashboard users.",
            "credentials": {
                "admin_panel": {
                    "url": "/admin",
                    "description": "Lead management admin panel",
                    "admin": {
                        "username": "admin",
                        "password": admin_password
                    },
                    "demo": {
                        "username": "demo",
                        "password": "demo123"
                    }
                },
                "dashboard": {
                    "url": "/login",
                    "description": "GEO Tracker dashboard",
                    "admin": {
                        "email": "admin@geotracker.io",
                        "password": admin_password
                    },
                    "demo": {
                        "email": "demo@geotracker.io",
                        "password": "demo123"
                    }
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


class AdminLeadUpdate(BaseModel):
    """Update lead status."""
    status: str = Field(..., description="New status (new, contacted, qualified, converted, lost)")
    notes: Optional[str] = Field(None, description="Optional notes")


def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Dependency to get and verify current admin user from token."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return payload


@app.post(
    "/api/admin/login",
    tags=["Admin"],
    summary="Admin login"
)
async def admin_login(request: AdminLoginRequest):
    """
    Authenticate as an admin user.

    Returns a JWT-like token for subsequent API calls.

    **Users:**
    - `admin` - Full access to all features
    - `demo` - Limited access (masked emails, read-only)
    """
    result = authenticate_admin(request.username, request.password)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {
        "success": True,
        "token": result["token"],
        "username": result["username"],
        "role": result["role"],
        "permissions": get_user_permissions(result["role"]),
        "expires_in": result["expires_in"]
    }


@app.get(
    "/api/admin/me",
    tags=["Admin"],
    summary="Get current admin user info"
)
async def get_admin_info(admin: Dict = Depends(get_current_admin)):
    """Get information about the currently authenticated admin."""
    return {
        "username": admin["username"],
        "role": admin["role"],
        "permissions": get_user_permissions(admin["role"])
    }


@app.get(
    "/api/admin/leads",
    tags=["Admin"],
    summary="Get all leads (admin only)"
)
async def get_admin_leads(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    admin: Dict = Depends(get_current_admin)
):
    """
    Get all leads for admin panel.

    Demo users see masked email addresses.
    Admin users see full data.
    """
    leads = get_leads_for_role(
        role=admin["role"],
        status=status,
        limit=limit,
        offset=offset
    )
    return {
        "leads": leads,
        "count": len(leads),
        "role": admin["role"]
    }


@app.get(
    "/api/admin/leads/stats",
    tags=["Admin"],
    summary="Get lead statistics (admin only)"
)
async def get_admin_lead_stats(admin: Dict = Depends(get_current_admin)):
    """Get lead statistics for admin dashboard."""
    permissions = get_user_permissions(admin["role"])
    if not permissions.get("can_view_stats"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    stats = get_leads_stats()
    return stats


@app.get(
    "/api/admin/leads/{lead_id}",
    tags=["Admin"],
    summary="Get a specific lead (admin only)"
)
async def get_admin_lead(lead_id: int, admin: Dict = Depends(get_current_admin)):
    """Get details for a specific lead."""
    permissions = get_user_permissions(admin["role"])
    if not permissions.get("can_view_leads"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    lead = get_lead_by_id(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")

    # Mask email for demo users
    if not permissions.get("can_view_emails") and lead.get("email"):
        email = lead["email"]
        if "@" in email:
            local, domain = email.split("@", 1)
            masked_local = local[0] + "***" if len(local) > 1 else "***"
            lead["email"] = f"{masked_local}@{domain}"

    return lead


@app.patch(
    "/api/admin/leads/{lead_id}",
    tags=["Admin"],
    summary="Update lead status (admin only)"
)
async def update_admin_lead(
    lead_id: int,
    update: AdminLeadUpdate,
    admin: Dict = Depends(get_current_admin)
):
    """Update a lead's status. Admin only - demo users cannot update."""
    if not can_update_lead(admin["role"]):
        raise HTTPException(status_code=403, detail="Demo users cannot update leads")

    # Validate status
    valid_statuses = ["new", "contacted", "qualified", "converted", "lost"]
    if update.status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
        )

    success = update_lead_status(lead_id, update.status, update.notes)
    if not success:
        raise HTTPException(status_code=404, detail="Lead not found")

    return {"success": True, "message": f"Lead {lead_id} updated to status: {update.status}"}


# ============================================
# USER AUTHENTICATION (Webapp Users)
# ============================================

class UserSignupRequest(BaseModel):
    """User signup request."""
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="Password (min 6 characters)")
    name: str = Field(..., description="User's full name")
    company: Optional[str] = Field(None, description="Company name")


class UserLoginRequest(BaseModel):
    """User login request."""
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class UserProfileUpdate(BaseModel):
    """Update user profile."""
    name: Optional[str] = Field(None, description="Updated name")
    company: Optional[str] = Field(None, description="Updated company")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Dependency to get and verify current user from token."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = get_user_from_token(credentials.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return user


@app.post(
    "/api/auth/signup",
    tags=["Auth"],
    summary="Create a new user account"
)
async def user_signup(request: UserSignupRequest):
    """
    Create a new user account.

    Returns a JWT token for immediate login.

    **Requirements:**
    - Email must be unique
    - Password must be at least 6 characters
    """
    try:
        result = register_user(
            email=request.email,
            password=request.password,
            name=request.name,
            company=request.company
        )

        return {
            "success": True,
            "message": "Account created successfully",
            "token": result["token"],
            "user": result["user"],
            "expires_in": result["expires_in"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/api/auth/login",
    tags=["Auth"],
    summary="Login to user account"
)
async def user_login(request: UserLoginRequest):
    """
    Authenticate user and get access token.

    Returns a JWT token valid for 7 days.

    **Demo credentials:**
    - Email: demo@geotracker.io
    - Password: demo123
    """
    result = authenticate_user(request.email, request.password)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return {
        "success": True,
        "token": result["token"],
        "user": result["user"],
        "expires_in": result["expires_in"]
    }


@app.get(
    "/api/auth/me",
    tags=["Auth"],
    summary="Get current user info"
)
async def get_user_info(user: Dict = Depends(get_current_user)):
    """Get information about the currently authenticated user."""
    return {
        "success": True,
        "user": user
    }


@app.patch(
    "/api/auth/profile",
    tags=["Auth"],
    summary="Update user profile"
)
async def update_user_profile_endpoint(
    profile: UserProfileUpdate,
    user: Dict = Depends(get_current_user)
):
    """Update the current user's profile information."""
    success = update_user(
        user_id=user["id"],
        name=profile.name,
        company=profile.company
    )

    if not success:
        raise HTTPException(status_code=400, detail="No changes to update")

    # Get updated user info
    updated_user = get_user_from_token(user.get("token", "")) or user
    if profile.name:
        updated_user["name"] = profile.name
    if profile.company:
        updated_user["company"] = profile.company

    return {
        "success": True,
        "message": "Profile updated",
        "user": updated_user
    }


@app.post(
    "/api/auth/verify",
    tags=["Auth"],
    summary="Verify token validity"
)
async def verify_user_token_endpoint(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify if the provided token is still valid.

    Returns user info if valid, error if invalid/expired.
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="No token provided")

    payload = verify_user_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = get_user_from_token(credentials.credentials)
    return {
        "valid": True,
        "user": user,
        "expires_at": payload.get("exp")
    }


# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle unexpected errors."""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
