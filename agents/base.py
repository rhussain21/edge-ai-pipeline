"""Base Agent Interface and Agent Response models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    name: str = "base_agent"
    description: str = "Base agent description"
    
    def __init__(self, tools: Optional[List] = None, llm_client=None):
        self.tools = tools or []
        self.llm_client = llm_client
        logger.info(f"Initialized {self.name} with {len(self.tools)} tools")
    
    @abstractmethod
    def process(self, query: str, **kwargs) -> Any:
        pass
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__,
            "num_tools": len(self.tools),
            "tools": [tool.name if hasattr(tool, 'name') else str(tool) for tool in self.tools]
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', tools={len(self.tools)})"


@dataclass
class AgentResponse:

    agent_name: str
    status: str = "success"           # 'success' | 'error' | 'partial'
    data: Any = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return  {
            "agent_name": self.agent_name,  
            "status": self.status,
            "data": self.data,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat()
        }

    @classmethod
    def from_agent_output(cls, agent_name: str, output: Any, duration_ms: float = None) -> "AgentResponse":
        if isinstance(output, dict) and output.get('error') is None:
            return cls(agent_name=agent_name, status="success", data=output, duration_ms=duration_ms)
        else:
            return cls(agent_name=agent_name, status="error", data=output)
            
    

    @classmethod
    def error_response(cls, agent_name: str, error_msg: str) -> "AgentResponse":

        return cls(agent_name=agent_name, status="error", error=error_msg)


class AgentTimer:

    def __init__(self):
        self.start_time = None
        self.duration_ms = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            self.duration_ms = (time.perf_counter() - self.start_time) * 1000
