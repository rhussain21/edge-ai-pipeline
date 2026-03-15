"""
Agents package — Multi-agent RAG system.

Agents:
    - RouterAgent  — query classification + dispatch
    - SQLAgent     — structured DB queries
    - VectorAgent  — semantic search + RAG synthesis
    - WebAgent     — external web search + synthesis

Base classes and utilities in agents.base.
Factory in agents.factory.
"""

from agents.base import BaseAgent, AgentResponse, AgentTimer
from agents.router_agent import RouterAgent
from agents.sql_agent import SQLAgent
from agents.vector_agent import VectorAgent
from agents.web_agent import WebAgent

# AgentFactory is not imported here to avoid pulling in heavy deps (faiss, torch).
# Import directly: from agents.factory import AgentFactory

__all__ = [
    'BaseAgent', 'AgentResponse', 'AgentTimer',
    'RouterAgent', 'SQLAgent', 'VectorAgent', 'WebAgent',
]
