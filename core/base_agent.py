"""Base Agent Interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
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
