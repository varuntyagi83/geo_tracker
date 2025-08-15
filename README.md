# GEO Tracker: Generative Engine Optimization Monitoring

This repository contains a production-ready reference implementation to measure Sunday’s presence, frequency, placement, grounding quality, and sentiment across top foundation models.

## Quick start

1. Create and activate a virtual environment, then install dependencies:
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
2. Copy `.env.example` to `.env` and fill in API keys as available. If a key is missing, the connector auto-falls back to a deterministic mock so the pipeline still runs end-to-end.
3. Initialize the database and run a panel:
```
python scripts/run_panel.py --market DE --language en --panel default
```
4. Launch the Streamlit dashboard:
```
streamlit run app/dashboard.py
```

## Components

- **Prompt registry**: `prompts.yaml` with topics, intents, and phrasings
- **Connectors**: OpenAI, Anthropic, Google, Perplexity, and an open-model provider. All implement a common interface.
- **Extractor pipeline**: brand mention extraction, placement, grounding, sentiment
- **Storage**: SQLAlchemy ORM with SQLite by default. Switch via `DATABASE_URL`.
- **Metrics**: daily rollups and a composite GEO score
- **Dashboard**: Streamlit app to review trends and drill into raw answers

## Notes

- The code runs fully in mock mode without external keys, suitable for CI and demos.
- Replace or extend connectors to call real APIs in production.
