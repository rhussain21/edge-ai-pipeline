"""
AI Industry Signals - Multi-Agent System

Architecture:
    RouterAgent (orchestrator)
      ├── SQLAgent     -> relationalDB
      ├── VectorAgent  -> VectorDB + LLM (RAG)
      └── WebAgent     -> InternetSearchTool + LLM
"""

from dotenv import load_dotenv
load_dotenv()

from agents.factory import AgentFactory
from logging_config import setup_debug_logging
import os
import logging
import json

setup_debug_logging()
logger = logging.getLogger(__name__)

print("Initializing Multi-Agent System...")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")

factory = AgentFactory(
    corpus_vector_path="Vectors/corpus_vectors/",
    relational_db_path="Database/industry_signals.db",
    tavily_api_key=TAVILY_API_KEY,
    ollama_model=OLLAMA_MODEL
)

system = factory.create_system()

print("=" * 50)
print("Multi-Agent System Ready!")
print(f"  Agents: {list(system.agents.keys())}")
print(f"  Model: {OLLAMA_MODEL}")
print(f"  Web Search: {'Tavily' if TAVILY_API_KEY else 'DuckDuckGo'}")
print("=" * 50)


if __name__ == "__main__":
    test_queries = [
        "How many documents do we have? Show me stats",
        "List all content",
        "Find content titled manufacturing",
        "What is the current focus in manufacturing? Specifically around AI?",
        "How does smart manufacturing improve industrial efficiency?",
        "What is the Knicks' latest score?",
        "what is a dinosaur?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print(f"{'='*60}")

        result = system.process(query)

        print(f"Agent:     {result.agent_name}")
        print(f"Status:    {result.status}")
        print(f"Duration:  {result.duration_ms}ms" if hasattr(result, 'duration_ms') else "")
        print(f"Result:")
        print(json.dumps(result.data if hasattr(result, 'data') else result, indent=2, default=str)[:500])

    print(f"\nSystem Info:")
    print(json.dumps(factory.get_system_info(), indent=2, default=str))