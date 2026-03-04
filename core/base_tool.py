"""Base Tool Interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    name: str = "base_tool"
    description: str = "Base tool description"
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass
    
    def get_info(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
