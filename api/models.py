# api/models.py
"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


# ============================================
# ENUMS
# ============================================

class ProviderEnum(str, Enum):
    openai = "openai"
    gemini = "gemini"
    perplexity = "perplexity"
    anthropic = "anthropic"


class ModeEnum(str, Enum):
    internal = "internal"
    provider_web = "provider_web"


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


# ============================================
# COMPANY / CLIENT
# ============================================

class CompanyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    website: Optional[str] = None
    industry: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    brand_keywords: List[str] = Field(default_factory=list, description="Keywords that indicate brand mention")
    competitors: List[str] = Field(default_factory=list, description="Competitor brand names to track")


class Company(CompanyCreate):
    id: str
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============================================
# QUERIES
# ============================================

class QueryCreate(BaseModel):
    question: str = Field(..., min_length=1)
    category: Optional[str] = None
    prompt_id: Optional[str] = None


class Query(QueryCreate):
    id: int
    
    class Config:
        from_attributes = True


class QueryGenerateRequest(BaseModel):
    """Request to auto-generate queries using AI."""
    company_id: str
    count: int = Field(default=25, ge=5, le=100)
    focus_areas: Optional[List[str]] = None  # e.g., ["product recommendations", "brand comparisons"]


# ============================================
# RUN CONFIGURATION
# ============================================

class RunConfigCreate(BaseModel):
    """Configuration for starting a new GEO tracker run."""
    company_id: str
    brand_name: str = Field(..., min_length=1, description="Brand name to track (e.g., 'Sunday Natural')")
    industry: Optional[str] = Field(default="", description="Industry for competitor detection context")
    
    # Provider & Model selection
    providers: List[ProviderEnum] = Field(default=[ProviderEnum.openai])
    openai_model: Optional[str] = "gpt-4.1-mini"
    gemini_model: Optional[str] = "gemini-2.5-flash"
    perplexity_model: Optional[str] = "sonar"
    anthropic_model: Optional[str] = "claude-sonnet-4-20250514"
    
    # Mode
    mode: ModeEnum = ModeEnum.provider_web
    
    # Query selection
    queries: Optional[List[QueryCreate]] = None  # If provided, use these
    query_ids: Optional[List[int]] = None  # Or select by ID
    limit: Optional[int] = None  # Limit number of queries
    start: int = 0
    
    # Localization
    market: Optional[str] = "DE"
    lang: Optional[str] = "de"
    
    # Execution options
    raw: bool = False
    request_timeout: int = Field(default=60, ge=10, le=300)
    max_retries: int = Field(default=1, ge=0, le=5)
    sleep_ms: int = Field(default=0, ge=0, le=5000)


# ============================================
# RUN STATUS & RESULTS
# ============================================

class RunProgress(BaseModel):
    """Progress update for a running job."""
    run_id: str
    status: RunStatus
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    progress_percent: float
    current_provider: Optional[str] = None
    current_query: Optional[str] = None
    estimated_remaining_seconds: Optional[int] = None
    started_at: Optional[datetime] = None
    updated_at: datetime
    error: Optional[str] = None  # Error message if job failed


class SourceInfo(BaseModel):
    url: str
    title: Optional[str] = None


class QueryResult(BaseModel):
    """Result for a single query execution."""
    run_id: int
    prompt_id: Optional[str]
    category: Optional[str]
    question: str
    provider: str
    model: str
    mode: str
    
    # Response
    response_text: str
    latency_ms: Optional[int]
    tokens_in: Optional[int]
    tokens_out: Optional[int]
    
    # Metrics
    presence: Optional[float] = Field(None, description="0.0 = not mentioned, >0 = mentioned")
    sentiment: Optional[float] = Field(None, description="-1.0 to 1.0")
    trust_authority: Optional[float]
    trust_sunday: Optional[float]
    
    # Brand analysis
    brand_mentioned: bool = False
    other_brands_detected: List[str] = Field(default_factory=list)
    
    # Sources
    sources: List[SourceInfo] = Field(default_factory=list)
    
    timestamp: datetime


class RunSummary(BaseModel):
    """Summary metrics for a completed run."""
    run_id: str
    company_id: str
    brand_name: str
    status: RunStatus
    
    # Counts
    total_queries: int
    total_responses: int
    
    # Aggregated metrics
    overall_visibility: float = Field(description="% of queries where brand was mentioned")
    avg_sentiment: Optional[float]
    avg_trust_authority: Optional[float]
    
    # Per-provider breakdown
    provider_visibility: Dict[str, float] = Field(default_factory=dict)
    
    # Competitor visibility
    competitor_visibility: Dict[str, float] = Field(default_factory=dict)
    
    # Timing
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[int]


class RunResponse(BaseModel):
    """Full response for a run including summary and detailed results."""
    summary: RunSummary
    results: List[QueryResult]


# ============================================
# API RESPONSES
# ============================================

class JobCreatedResponse(BaseModel):
    """Response when a new run job is created."""
    job_id: str
    run_id: str
    status: RunStatus
    message: str
    estimated_duration_seconds: Optional[int] = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    providers_available: List[str]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
