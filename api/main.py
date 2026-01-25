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
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .models import (
    RunConfigCreate, QueryCreate, JobCreatedResponse, HealthResponse,
    ErrorResponse, RunProgress, ProviderEnum
)
from .jobs import job_manager, Job, JobStatus
from .services import geo_service


# ============================================
# APP INITIALIZATION
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    print("🚀 GEO Tracker API starting...")
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
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
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
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
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
