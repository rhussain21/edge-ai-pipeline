"""Web Agent - External web search and synthesis."""

from typing import Any, Dict, Optional
import logging
from core.base_agent import BaseAgent
from core.agent_response import AgentResponse

logger = logging.getLogger(__name__)


class WebAgent(BaseAgent):
    name = "web_agent"
    description = "External web search for current events, market trends, and live information."

    def __init__(self, internet_tool, llm_client=None):
        super().__init__(tools=[internet_tool], llm_client=llm_client)
        self.internet_tool = internet_tool

    def process(self, query: str, max_results: int = 5, **kwargs) -> AgentResponse:
        logger.info(f"WebAgent processing: '{query[:60]}'")

        try:
            search_result = self.internet_tool.execute("search", query=query, max_results=max_results)

            if "error" in search_result:
                return AgentResponse.error_response(self.name, search_result["error"])

            results = search_result.get("results", [])

            if not results:
                return AgentResponse.from_agent_output(self.name, {
                    "answer": "No external results found for this query.",
                    "sources": [],
                    "query": query
                })

            if self.llm_client:
                answer = self._synthesize(query, results)
            else:
                answer = results[0].get("snippet", "") if results else ""

            return AgentResponse.from_agent_output(self.name, {
                "answer": answer,
                "sources": [
                    {"title": r.get("title"), "url": r.get("url"), "snippet": r.get("snippet")}
                    for r in results
                ],
                "result_count": len(results),
                "query": query
            })

        except Exception as e:
            logger.error(f"WebAgent error: {e}")
            return AgentResponse.error_response(self.name, str(e))

    def _synthesize(self, query: str, results: list) -> str:
        context = "\n\n".join([
            f"Source: {r.get('title', 'Unknown')}\nURL: {r.get('url', '')}\n{r.get('snippet', '')}"
            for r in results
        ])

        prompt = f"""Web search results for: "{query}"

{context}

Summarize the key findings from these external sources. 
Note any dates or recency indicators. Flag what might be worth ingesting into the internal library."""

        return self.llm_client.generate(
            prompt=prompt,
            system_prompt="""You are an external research analyst. Summarize web search results clearly,
cite sources, and highlight what is current vs potentially outdated."""
        )
