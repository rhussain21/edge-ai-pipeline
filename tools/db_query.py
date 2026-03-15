"""Database Query Tool - Structured queries against relational database."""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DBQueryTool:
    name = "db_query"
    description = """Query the relational database for structured content metadata.
    Supports listing content, filtering by type, getting stats, and looking up specific records."""

    def __init__(self, relational_db):
        self.db = relational_db

    def execute(self, action: str, **kwargs) -> Any:
        actions = {
            "list": self.list_content,
            "get": self.get_by_id,
            "stats": self.get_stats,
            "search": self.search_by_title,
            "query": self.raw_query,
        }

        if action not in actions:
            return {"error": f"Unknown action '{action}'. Available: {list(actions.keys())}"}

        try:
            return actions[action](**kwargs)
        except Exception as e:
            logger.error(f"DB query tool error ({action}): {e}")
            return {"error": str(e)}

    def list_content(self, limit: int = 20, offset: int = 0, content_type: Optional[str] = None) -> Dict[str, Any]:
        """List content with optional type filter."""
        if content_type:
            query = f"SELECT id, title, content_type, created_at FROM content WHERE content_type = '{content_type}' ORDER BY created_at DESC LIMIT {limit} OFFSET {offset}"
        else:
            query = f"SELECT id, title, content_type, created_at FROM content ORDER BY created_at DESC LIMIT {limit} OFFSET {offset}"

        results = self.db.query(query)
        columns = ["id", "title", "content_type", "created_at"]

        records = [dict(zip(columns, row)) for row in results]

        return {
            "count": len(records),
            "records": records
        }

    def get_by_id(self, content_id: int) -> Dict[str, Any]:
        """Get a single content record by ID."""
        results = self.db.query(f"SELECT * FROM content WHERE id = {content_id}")

        if not results:
            return {"error": f"No content found with id={content_id}"}

        col_results = self.db.query("DESCRIBE content")
        columns = [row[0] for row in col_results] if col_results else []

        if columns and len(columns) == len(results[0]):
            return dict(zip(columns, results[0]))
        else:
            return {"data": list(results[0])}

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        total = self.db.query("SELECT COUNT(*) FROM content")
        by_type = self.db.query(
            "SELECT content_type, COUNT(*) as cnt FROM content GROUP BY content_type ORDER BY cnt DESC"
        )
        recent = self.db.query(
            "SELECT title, content_type, created_at FROM content ORDER BY created_at DESC LIMIT 5"
        )

        return {
            "total_records": total[0][0] if total else 0,
            "by_type": [{"type": row[0], "count": row[1]} for row in by_type] if by_type else [],
            "recent": [
                {"title": row[0], "type": row[1], "created": str(row[2])}
                for row in recent
            ] if recent else []
        }

    def search_by_title(self, keyword: str, limit: int = 10) -> Dict[str, Any]:
        """Search content by title keyword (case-insensitive)."""
        results = self.db.query(
            f"SELECT id, title, content_type, created_at FROM content WHERE LOWER(title) LIKE '%{keyword.lower()}%' LIMIT {limit}"
        )
        columns = ["id", "title", "content_type", "created_at"]
        records = [dict(zip(columns, row)) for row in results]

        return {
            "query": keyword,
            "count": len(records),
            "records": records
        }

    def raw_query(self, sql: str) -> Dict[str, Any]:
        results = self.db.query(sql)
        return {
            "sql": sql,
            "count": len(results),
            "results": [list(row) for row in results]
        }

    def get_info(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__
        }
