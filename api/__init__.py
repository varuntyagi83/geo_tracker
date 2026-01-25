# api/__init__.py
"""
GEO Tracker API Package

This package provides a FastAPI wrapper around the GEO tracker engine,
enabling it to be used as a SaaS service.
"""
from .main import app
from .services import geo_service
from .jobs import job_manager

__all__ = ["app", "geo_service", "job_manager"]
