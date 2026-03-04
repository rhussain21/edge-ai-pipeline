"""Agent Factory - Centralized creation and configuration of all agents."""

from typing import Dict, Any
import logging
import os
from dotenv import load_dotenv

load_dotenv()

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Clean imports from project root
from db_vector import VectorDB
from db_relational import relationalDB
from llm_client import OllamaClient
from tools.internet_search_tool import InternetSearchTool

from agents.sql_agent import SQLAgent
from agents.vector_agent import VectorAgent
from agents.web_agent import WebAgent
from agents.router_agent import RouterAgent

logger = logging.getLogger(__name__)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class AgentFactory:
    def __init__(self, vector_db_path: str, relational_db_path: str, 
                 tavily_api_key: str = None, ollama_model: str = "llama3:latest"):
        self.vector_db_path = vector_db_path
        self.relational_db_path = relational_db_path
        self.tavily_api_key = tavily_api_key
        self.ollama_model = ollama_model
        
        self._vector_db = None
        self._relational_db = None
        self._llm_client = None
        self._internet_tool = None
        self._agents = None
        self._router = None
    
    def _initialize_shared_resources(self):
        """Initialize shared resources (databases, LLM, tools)."""
        logger.info("Initializing shared resources...")
        
        self._vector_db = VectorDB(self.vector_db_path)
        self._relational_db = relationalDB(self.relational_db_path)
        self._llm_client = OllamaClient(model=self.ollama_model)
        
        if self.tavily_api_key:
            self._internet_tool = InternetSearchTool(
                provider="tavily", 
                api_key=self.tavily_api_key
            )
        else:
            self._internet_tool = InternetSearchTool(provider="duckduckgo")
            logger.warning("No Tavily API key, using DuckDuckGo fallback")
        
        logger.info("Shared resources initialized")
    
    def create_agents(self) -> Dict[str, Any]:
        """Create all agents with their dependencies."""
        if self._agents is None:
            self._initialize_shared_resources()
            logger.info("Creating agents...")
            agents = {}
            
            agents["sql_agent"] = SQLAgent(
                relational_db=self._relational_db,
                llm_client=self._llm_client
            )
            agents["vector_agent"] = VectorAgent(
                vector_db=self._vector_db,
                llm_client=self._llm_client
            )
            agents["web_agent"] = WebAgent(
                internet_tool=self._internet_tool,
                llm_client=self._llm_client
            )
            
            self._agents = agents
            logger.info(f"Created {len(agents)} agents")
        
        return self._agents
    
    def create_router(self, agents: Dict[str, Any] = None) -> RouterAgent:
        """Create router with all agents registered."""
        if self._router is None:
            if agents is None:
                agents = self.create_agents()
            
            logger.info("Creating router and registering agents...")
            router = RouterAgent()
            
            for name, agent in agents.items():
                router.register_agent(name, agent)
                logger.info(f"Registered {name}")
            
            self._router = router
            logger.info("Router created with all agents registered")
        
        return self._router
    
    def create_system(self) -> RouterAgent:
        """Create complete system (agents + router) in one call."""
        agents = self.create_agents()
        router = self.create_router(agents)
        return router
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the created system."""
        if not hasattr(self, '_agents') or not self._agents or not hasattr(self, '_router') or not self._router:
            return {"status": "System not created yet"}
        
        return {
            "status": "Ready",
            "agents": list(self._agents.keys()),
            "router_type": type(self._router).__name__,
            "vector_db_path": self.vector_db_path,
            "relational_db_path": self.relational_db_path,
            "internet_provider": self._internet_tool.provider if hasattr(self, '_internet_tool') and self._internet_tool else "Not initialized",
            "ollama_model": self.ollama_model
        }


def create_production_system():
    factory = AgentFactory(
        vector_db_path="Vectors/",
        relational_db_path="Database/industry_signals.db",
        tavily_api_key=TAVILY_API_KEY,
        ollama_model="llama3:latest"
    )
    
    return factory.create_system()


def create_development_system():
    factory = AgentFactory(
        vector_db_path="Vectors/",
        relational_db_path="Database/industry_signals.db"
    )
    
    return factory.create_system()


if __name__ == "__main__":
    factory = AgentFactory(
        vector_db_path="Vectors/",
        relational_db_path="Database/industry_signals.db",
        tavily_api_key=TAVILY_API_KEY
    )
    
    system = factory.create_system()
    info = factory.get_system_info()
    print("System info:", info)
