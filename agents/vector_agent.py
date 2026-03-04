"""Vector Agent - Semantic search and RAG synthesis."""

from typing import Any, Dict, Optional
import logging
from core.base_agent import BaseAgent
from core.agent_response import AgentResponse

logger = logging.getLogger(__name__)


class VectorAgent(BaseAgent):
    name = "vector_agent"
    description = "Semantic search and synthesis across the document library."

    def __init__(self, vector_db, llm_client):
        super().__init__(tools=[], llm_client=llm_client)
        self.vector_db = vector_db

    def process(self, query: str, top_k: int = 5, temperature: float = 0.7, **kwargs) -> AgentResponse:
        logger.info(f"VectorAgent processing: '{query[:60]}'")

        try:
            context_chunks = self.vector_db.search(query=query, top_k=top_k)

            if not context_chunks:
                return AgentResponse.from_agent_output(self.name, {
                    "answer": "No relevant context found in the document library.",
                    "sources": [],
                    "context_used": False
                })

            context = "\n\n".join([
                chunk if isinstance(chunk, str) else chunk.get("text", str(chunk))
                for chunk in context_chunks
            ])

            prompt = self._build_prompt(query, context)
            answer = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self._system_prompt(),
                temperature=temperature
            )

            return AgentResponse.from_agent_output(self.name, {
                "answer": answer,
                "sources_used": len(context_chunks),
                "context_used": True
            })

        except Exception as e:
            logger.error(f"VectorAgent error: {e}")
            return AgentResponse.error_response(self.name, str(e))

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""Context from internal document library:
{context}

Question: {query}

Instructions:
- Answer based ONLY on the provided context
- Identify themes and patterns across sources
- If information is not in context, say so clearly
- Cite specific excerpts where relevant
"""

    def _system_prompt(self) -> str:
        return """You are an internal analyst with access to a curated document library.
Your job is to synthesize themes, patterns, and insights across documents.
Be precise, cite your sources, and clearly distinguish what the library says vs what you infer."""
