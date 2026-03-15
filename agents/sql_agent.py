"""SQL Agent - Structured data queries against relational database."""

from typing import Any, Dict, Optional
import logging
import re
from agents.base import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class SQLAgent(BaseAgent):
    name = "sql_agent"
    description = "Structured data lookups: counts, metadata filtering, listing, stats, and record retrieval."

    def __init__(self, relational_db, llm_client=None):
        super().__init__(tools=[], llm_client=llm_client)
        self.relational_db = relational_db

    def process(self, query: str, **kwargs) -> AgentResponse:
        intent = self._detect_intent(query)
        logger.info(f"SQLAgent intent: {intent} for query: '{query[:60]}'")

        try:
            if intent == "stats":
                result = self._handle_stats()
            elif intent == "list":
                result = self._handle_list(**kwargs)
            elif intent == "get_by_id":
                result = self._handle_get_by_id(query)
            elif intent == "search_db":
                result = self._handle_db_search(query, **kwargs)
            else:
                result = self._handle_db_search(query, **kwargs)

            return AgentResponse.from_agent_output(self.name, result)

        except Exception as e:
            logger.error(f"SQLAgent error: {e}")
            return AgentResponse.error_response(self.name, str(e))

    def _detect_intent(self, query: str) -> str:
        q = query.lower().strip()

        if any(w in q for w in ["stats", "statistics", "how many", "count", "total"]):
            return "stats"
        elif any(w in q for w in ["list", "show all", "show me all", "what content"]):
            return "list"
        elif q.startswith("id ") or q.startswith("get ") or q.startswith("#"):
            return "get_by_id"
        else:
            return "search_db"

    def _handle_stats(self) -> Dict[str, Any]:
        results = self.relational_db.query(
            "SELECT content_type, COUNT(*) as count FROM content GROUP BY content_type"
        )
        total = self.relational_db.query("SELECT COUNT(*) FROM content")

        return {
            "intent": "stats",
            "total_documents": total[0][0] if total else 0,
            "by_type": [{"type": row[0], "count": row[1]} for row in results] if results else []
        }

    def _handle_list(self, limit: int = 20, content_type: str = None, **kwargs) -> Dict[str, Any]:
        if content_type:
            query = f"SELECT id, title, content_type, created_at FROM content WHERE content_type = '{content_type}' ORDER BY created_at DESC LIMIT {limit}"
        else:
            query = f"SELECT id, title, content_type, created_at FROM content ORDER BY created_at DESC LIMIT {limit}"

        results = self.relational_db.query(query)

        return {
            "intent": "list",
            "count": len(results) if results else 0,
            "documents": [
                {"id": row[0], "title": row[1], "type": row[2], "created_at": str(row[3])}
                for row in results
            ] if results else []
        }

    def _handle_get_by_id(self, query: str) -> Dict[str, Any]:
        numbers = re.findall(r'\d+', query)
        if not numbers:
            return {"error": "No ID found in query", "intent": "get_by_id"}

        content_id = int(numbers[0])
        results = self.relational_db.query(
            f"SELECT id, title, content_type, created_at FROM content WHERE id = {content_id}"
        )

        return {
            "intent": "get_by_id",
            "id": content_id,
            "document": {"id": results[0][0], "title": results[0][1], "type": results[0][2]} if results else None,
            "found": bool(results)
        }

    def _handle_db_search(self, query: str, limit: int = 10, **kwargs) -> Dict[str, Any]:
        keyword = self._extract_keyword(query)
        results = self.relational_db.query(
            f"SELECT id, title, content_type FROM content WHERE title LIKE '%{keyword}%' LIMIT {limit}"
        )

        return {
            "intent": "search_db",
            "keyword": keyword,
            "count": len(results) if results else 0,
            "documents": [
                {"id": row[0], "title": row[1], "type": row[2]}
                for row in results
            ] if results else []
        }

    def _extract_keyword(self, query: str) -> str:
        stop_words = {
            "find", "search", "show", "get", "what", "which", "list",
            "the", "a", "an", "is", "are", "for", "about", "me",
            "all", "content", "titled", "named", "called", "with", "in",
            "did", "we", "cover", "where", "discussed"
        }
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords[0] if keywords else query.split()[-1]
