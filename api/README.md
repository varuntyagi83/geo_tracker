# GEO Tracker API

FastAPI wrapper for the GEO Tracker engine, enabling SaaS-style usage.

## Updates (January 2026)

✅ **AI-Powered Query Generation**: Uses actual LLM API to generate industry-specific queries  
✅ **Parallel Execution**: Providers run in parallel for faster results  
✅ **Updated Models**: Latest OpenAI (GPT-4.1) and Gemini (2.5) models  
✅ **Proper Brand Detection**: 600+ stopwords to filter common words from competitor detection  
✅ **Business Context**: Queries are generated based on your company name and industry

## Quick Start

### 1. Install Dependencies

```bash
cd geo_tracker_full
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set Environment Variables (API Keys)

**⚠️ IMPORTANT: You need at least ONE API key for the tracker to work.**

```bash
# Copy the example file
cp .env .env

# Edit .env and add your API keys:
```

**Where to get API keys:**
- **OpenAI**: https://platform.openai.com/api-keys
- **Google Gemini**: https://aistudio.google.com/app/apikey

Your `.env` file should look like:
```bash
# At least ONE of these is required:
OPENAI_API_KEY=sk-your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
```

### 3. Run the API

```bash
# Development mode with auto-reload
python -m uvicorn api.main:app --reload --port 8000

# Or simply
cd geo_tracker_full
uvicorn api.main:app --reload --port 8000
```

### 4. Access the API

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## Available Models (January 2026)

### OpenAI
| Model ID | Name | Description |
|----------|------|-------------|
| `gpt-4.1` | GPT-4.1 | Latest flagship (1M context) |
| `gpt-4.1-mini` | GPT-4.1 Mini | Fast & affordable (recommended) |
| `gpt-4.1-nano` | GPT-4.1 Nano | Fastest, cheapest |
| `gpt-4o` | GPT-4o | Multimodal |
| `gpt-4o-mini` | GPT-4o Mini | Smaller multimodal |
| `gpt-4-turbo` | GPT-4 Turbo | Legacy (128K context) |

### Google Gemini
| Model ID | Name | Description |
|----------|------|-------------|
| `gemini-2.5-flash` | Gemini 2.5 Flash | Fast, recommended (1M context) |
| `gemini-2.5-pro` | Gemini 2.5 Pro | Most capable |
| `gemini-2.5-flash-lite` | Gemini 2.5 Flash-Lite | Cost-efficient |
| `gemini-2.0-flash` | Gemini 2.0 Flash | Previous gen (until March 2026) |
| `gemini-3-flash-preview` | Gemini 3 Flash | Preview |

---

## API Endpoints

### Health Check

```bash
GET /health
```

Returns API status and available providers.

### Start a Run

```bash
POST /api/runs
Content-Type: application/json

{
  "company_id": "company-123",
  "brand_name": "Sunday Natural",
  "providers": ["openai", "gemini"],
  "mode": "provider_web",
  "queries": [
    {"question": "What are the best vitamin D supplements?", "category": "product_recommendation"},
    {"question": "Compare Sunday Natural with other brands", "category": "brand_comparison"}
  ],
  "market": "DE",
  "lang": "de"
}
```

**Response:**
```json
{
  "job_id": "abc-123",
  "run_id": "abc-123",
  "status": "pending",
  "message": "Run queued with 4 tasks (2 queries × 2 providers)",
  "estimated_duration_seconds": 16
}
```

### Check Progress

```bash
GET /api/runs/{job_id}/status
```

**Response:**
```json
{
  "run_id": "abc-123",
  "status": "running",
  "total_tasks": 4,
  "completed_tasks": 2,
  "progress_percent": 50.0,
  "current_provider": "gemini",
  "current_query": "Compare Sunday Natural...",
  "estimated_remaining_seconds": 8
}
```

### Get Results

```bash
GET /api/runs/{job_id}/results
```

**Response:**
```json
{
  "summary": {
    "run_id": "abc-123",
    "brand_name": "Sunday Natural",
    "overall_visibility": 75.0,
    "avg_sentiment": 0.65,
    "provider_visibility": {
      "openai": 100.0,
      "gemini": 50.0
    },
    "competitor_visibility": {
      "Competitor A": 50.0,
      "Competitor B": 25.0
    }
  },
  "results": [
    {
      "question": "What are the best vitamin D supplements?",
      "provider": "openai",
      "brand_mentioned": true,
      "presence": 1.0,
      "sentiment": 0.8,
      "response_text": "...",
      "sources": [{"url": "...", "title": "..."}]
    }
  ]
}
```

### Generate Queries (AI-Powered)

```bash
POST /api/queries/generate
Content-Type: application/json

{
  "company_name": "Sunday Natural",
  "industry": "Supplements & Vitamins",
  "language": "de",
  "count": 25,
  "target_market": "Germany"
}
```

**Response:**
```json
{
  "queries": [
    {
      "question": "Was sind die besten Vitamin D Nahrungsergänzungsmittel?",
      "category": "product_recommendation",
      "intent": "User researching vitamin D products",
      "prompt_id": "ai_gen_1"
    },
    ...
  ],
  "count": 25,
  "generated_by": "ai",
  "note": "Queries generated using AI. Review and customize as needed."
}
```

---

### Get Sample Queries (No API Key Required)

```bash
GET /api/queries/sample?industry=Supplements&company_name=Sunday%20Natural&language=de&count=20
```

This returns template-based queries without calling the LLM API.

---

## Example: Full Workflow

```python
import requests
import time

BASE_URL = "http://localhost:8000"

# 1. Start a run
run_config = {
    "company_id": "demo-company",
    "brand_name": "Sunday Natural",
    "providers": ["openai"],
    "mode": "provider_web",
    "queries": [
        {"question": "Best vitamin brands Germany", "category": "product_recommendation"},
        {"question": "Sunday Natural review", "category": "brand_review"},
    ],
    "market": "DE",
    "lang": "de"
}

response = requests.post(f"{BASE_URL}/api/runs", json=run_config)
job = response.json()
job_id = job["job_id"]
print(f"Started job: {job_id}")

# 2. Poll for progress
while True:
    status = requests.get(f"{BASE_URL}/api/runs/{job_id}/status").json()
    print(f"Progress: {status['progress_percent']}% - {status['status']}")
    
    if status["status"] in ["completed", "failed", "cancelled"]:
        break
    
    time.sleep(2)

# 3. Get results
if status["status"] == "completed":
    results = requests.get(f"{BASE_URL}/api/runs/{job_id}/results").json()
    print(f"Visibility: {results['summary']['overall_visibility']}%")
    print(f"Sentiment: {results['summary']['avg_sentiment']}")
```

---

## Configuration Options

### Run Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `company_id` | string | required | Unique identifier for the company |
| `brand_name` | string | required | Brand name to track in responses |
| `providers` | list | `["openai"]` | LLM providers to use |
| `openai_model` | string | `"gpt-4.1-mini"` | OpenAI model to use |
| `gemini_model` | string | `"gemini-2.5-flash"` | Gemini model to use |
| `mode` | string | `"provider_web"` | `"internal"` or `"provider_web"` |
| `queries` | list | required | List of queries to execute |
| `market` | string | `"DE"` | Market/country code |
| `lang` | string | `"de"` | Language code |
| `raw` | boolean | `false` | Send raw question without headers |
| `request_timeout` | int | `60` | Timeout per request in seconds |
| `max_retries` | int | `1` | Max retries on failure |
| `sleep_ms` | int | `0` | Delay between requests in ms |

### Modes

- **`internal`**: Uses model's internal knowledge only (no web search)
- **`provider_web`**: Uses provider's built-in web search (recommended)

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   FastAPI                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │
│  │ /runs   │  │ /status │  │ /results        │ │
│  └────┬────┘  └────┬────┘  └────────┬────────┘ │
└───────┼────────────┼────────────────┼──────────┘
        │            │                │
        ▼            ▼                ▼
┌─────────────────────────────────────────────────┐
│              Job Manager (in-memory)             │
│  Background execution with progress tracking    │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│              GEO Tracker Service                 │
│  Wraps existing run.py functionality            │
└─────────────────────────────────────────────────┘
        │
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ OpenAI  │   │ Gemini  │   │ SQLite  │
   │Provider │   │Provider │   │   DB    │
   └─────────┘   └─────────┘   └─────────┘
```

---

## Next Steps

1. **Add Authentication**: Integrate Supabase Auth or similar
2. **Replace Job Manager**: Use Redis + Celery for production
3. **Add More Providers**: Claude, Perplexity, etc.
4. **Build Frontend**: Next.js dashboard
5. **Deploy with Docker**: See docker-compose setup

---

## Files

```
geo_tracker_full/
├── api/
│   ├── __init__.py      # Package init
│   ├── main.py          # FastAPI app & endpoints
│   ├── models.py        # Pydantic models
│   ├── services.py      # Business logic
│   └── jobs.py          # Background job manager
├── llm_providers/       # OpenAI, Gemini implementations
├── metrics/             # Presence, sentiment, trust
├── run.py               # Original execution engine
├── db.py                # SQLite database
├── config.py            # Environment config
└── requirements.txt     # Dependencies
```
