from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time


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


if __name__ == "__main__":
    resp1 = AgentResponse(agent_name="test_agent", data={"count": 5})
    print(resp1.to_dict())
    print(resp1)
    
    resp2 = AgentResponse.from_agent_output("test_agent", {"result": "success"}, 100.5)
    print(resp2)
    
    resp3 = AgentResponse.error_response("test_agent", "Something broke")
    print(resp3)
    
    print("\n--- Testing AgentTimer ---")
    with AgentTimer() as timer:
        for i in range(1000000):
            pass
    print(f"Timer duration: {timer.duration_ms:.2f}ms")

