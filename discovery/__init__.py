"""
Source Discovery Module

Pipeline: JSON configs → Query Planner → tools/ adapters → LLM Classifier → Deduper → Approved Sources

Search adapters (RSS, Web, GitHub) live in tools/ — shared with agents.
"""

from discovery.models import SearchQuery, CandidateSource, ClassifiedCandidate, DiscoveryResult
from discovery.source_discovery_service import SourceDiscoveryService

__all__ = [
    "SourceDiscoveryService",
    "SearchQuery",
    "CandidateSource",
    "ClassifiedCandidate",
    "DiscoveryResult",
]
