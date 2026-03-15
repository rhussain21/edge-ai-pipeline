"""
ETL Package — Server-side pipeline that runs on Jetson.

Components:
    - pipeline.py  — Content ETL (extract, chunk, vectorize)
    - sources.py   — Content source downloaders (RSS/podcast)
    - signals.py   — Signal extraction via langextract
"""

from etl.pipeline import contentETL
from etl.sources import ContentSources
from etl.signals import SignalPipeline

__all__ = ["contentETL", "ContentSources", "SignalPipeline"]
