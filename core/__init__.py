"""
Core framework for multi-agent system
Defines base classes and interfaces
"""

from .base_tool import BaseTool
from .base_agent import BaseAgent
from .agent_response import AgentResponse, AgentTimer

__all__ = ['BaseTool', 'BaseAgent', 'AgentResponse', 'AgentTimer']
