"""Router Agent - Orchestrator with concurrent dispatch and standardized agent communication."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.agent_response import AgentResponse, AgentTimer

logger = logging.getLogger(__name__)


class RouterAgent:
    """Orchestrator with concurrent dispatch and standardized responses."""

    name = "router_agent"

    def __init__(self, max_workers: int = 3):
        self.agents = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"RouterAgent initialized (max_workers={max_workers})")

    def register_agent(self, agent_name: str, agent_instance):
        self.agents[agent_name] = agent_instance
        logger.info(f"Registered agent: {agent_name}")

    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """Main entry point. Classifies query and dispatches to appropriate agent."""
        decision = self._classify(query)
        agent_name = decision['agents'][0]
        return self._dispatch_single(agent_name, query, **kwargs)

    def _classify(self, query: str) -> Dict[str, Any]:
        """Classify query and determine which agent(s) should handle it."""
        q = query.lower().strip()

        concurrent_patterns = []

        sql_patterns = [
            "stats", "statistics", "how many", "count", "total",
            "list", "show all", "show me all", "what content", "documents", "records"
        ]

        vector_patterns = [
            "what is", "how does", "explain", "why", "analyze",
            "tell me about", "summarize", "compare", "trend", "time"
        ]

        web_patterns = [
            "search", "the web",
        ]

        if any(p in q for p in sql_patterns):
            return {"agents":['sql_agent'], "strategy":"single", "intent":"data_query"}

        if any(p in q for p in vector_patterns):
            return {"agents":['vector_agent'], "strategy":"single", "intent":"vector_query"}

        return {"agents": ["web_agent"], "strategy": "single", "intent": "default"}

    def _dispatch_single(self, agent_name: str, query: str, **kwargs):
        """Call a single agent and return its AgentResponse."""
        if agent_name not in self.agents.keys():
            return AgentResponse.error_response(agent_name, "Agent not found!")
        
        agent = self.agents[agent_name]
        with AgentTimer() as timer:
            return agent.process(query, **kwargs)
    def _dispatch_concurrent(self, agent_names: List[str], query: str, **kwargs) -> List:
        """Call multiple agents concurrently using ThreadPoolExecutor."""
        pass

    def _merge_responses(self, responses: List, classification: Dict) -> Dict[str, Any]:
        """Combine multiple AgentResponses into a single result."""

        merged = {
            'router':self.name,
            'strategy':classification['strategy'],
            'intent':classification['intent'],
            'agents':{},
            'agent_count': len(responses)
        }
        total_duration = 0
        for response in responses:
            merged["agents"][response.agent_name] = response.to_dict()
            total_duration += (response.duration_ms or 0)
        
        merged["total_duration_ms"] = round(total_duration, 2)
        return merged

    def list_agents(self) -> Dict[str, Any]:
        """List all registered agents and their info."""
        agents_info = {}
        for name, agent in self.agents.items():
            if hasattr(agent, 'get_info'): 
                agents_info[name] = agent.get_info()
            else:
                agents_info[name] = {"name": name, "type": type(agent).__name__}
        return agents_info

    def shutdown(self):
        """Clean up thread pool."""
        self.executor.shutdown(wait=True)
        logger.info("RouterAgent thread pool shut down")

    def __repr__(self):
        return f"RouterAgent(agents={list(self.agents.keys())})"


if __name__ == '__main__':

    router = RouterAgent()

    # Mock agents for testing without real DBs
    class MockAgent:
        def __init__(self, name):
            self.name = name
        def process(self, query, **kwargs):
            return AgentResponse.from_agent_output(self.name, {"answer": f"Mock response from {self.name}"})
        def get_info(self):
            return {"name": self.name, "type": "MockAgent"}

    router.register_agent("sql_agent", MockAgent("sql_agent"))
    router.register_agent("vector_agent", MockAgent("vector_agent"))
    router.register_agent("web_agent", MockAgent("web_agent"))

    test_queries = [
        "how many cats are there?",
        "what is AI manufacturing?",
        "search the web for trends",
        "show me all documents",
        "explain machine learning",
        "latest news about AI",
        "MEOW MEOW MOW",
    ]

    print("=== Router End-to-End Test (Single Strategy) ===")
    for query in test_queries:
        result = router.process(query)
        if isinstance(result, AgentResponse):
            print(f"Query: {query:<35} -> {result.agent_name} ({result.status})")
            print(f"Response: {result}")
        else:
            print(f"Query: {query:<35} -> {result}")

    