"""
Content Screener — LLM quality gate between vectorization and signal extraction.

Evaluates whether ingested content is worth sending to the (expensive) signal
extraction step.  Approved content proceeds to signal extraction; rejected
content is flagged for future deletion (garbage-collection cron).

Pipeline position:
    download → extract → vectorize → **screen** → signal extraction

Usage:
    from etl.content_screener import ContentScreener

    screener = ContentScreener(db=db, llm_client=llm_client)
    results  = screener.screen_pending(limit=20)
"""

import json
import logging
import textwrap
import traceback
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)
from logging_config import syslog
from llm_client import _classify_error

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are a strict editorial gatekeeper for an industrial-manufacturing
intelligence dataset.  Your job is to decide whether a piece of content
contains enough substantive, factual information to justify extracting
structured signals (companies, technologies, events, metrics).

Approve content that contains ANY of:
  • Named companies, products, or technologies in manufacturing / industrial automation / AI
  • Concrete events: partnerships, funding rounds, product launches, acquisitions
  • Technical specifications, benchmarks, or performance metrics
  • Market data, forecasts, or regulatory developments

Reject content that is predominantly:
  • Generic marketing copy or press-release fluff with no specifics
  • Unrelated to manufacturing, industrial automation, or applied AI
  • Garbled / low-quality text (bad OCR, broken transcription, boilerplate)
  • Pure opinion with no factual claims or named entities

Respond with ONLY valid JSON — no markdown, no commentary:
{
  "decision": "approve" or "reject",
  "reason": "<one-sentence justification>",
  "confidence": <float 0.0-1.0>
}
""")

USER_PROMPT_TEMPLATE = textwrap.dedent("""\
Title: {title}
Source: {source_name}
Type: {source_type}
Length: {char_count} characters

--- Content excerpt (first ~2000 chars) ---
{excerpt}
--- End excerpt ---

Should this content be approved for signal extraction?
""")

MAX_EXCERPT_CHARS = 2000


class ContentScreener:
    """LLM-powered quality gate that screens content before signal extraction."""

    def __init__(self, db, llm_client):
        """
        Parameters
        ----------
        db : relationalDB
            Database handle (PostgreSQL or DuckDB).
        llm_client : OllamaClient | GeminiClient
            Any object that exposes `.generate(prompt, system_prompt, temperature)`.
        """
        self.db = db
        self.llm = llm_client

    # ── public API ────────────────────────────────────────────────────────

    def screen_pending(self, limit: int = 20) -> Dict[str, Any]:
        """Screen up to *limit* content items that are vectorized but not yet screened.

        Returns a summary dict: {approved: int, rejected: int, errors: int, details: [...]}
        """
        pending = self.db.query("""
            SELECT id, title, source_name, source_type, transcript
            FROM content
            WHERE (screening_status = 'pending' OR screening_status IS NULL)
              AND extraction_status IN ('completed', 'NA')
            ORDER BY id
            LIMIT ?
        """, [limit])

        if not pending:
            logger.info("No content pending screening")
            return {"approved": 0, "rejected": 0, "errors": 0, "details": []}

        logger.info(f"Screening {len(pending)} content items")
        summary = {"approved": 0, "rejected": 0, "errors": 0, "details": []}

        for i, row in enumerate(pending, 1):
            content_id = row["id"]
            title = row.get("title", "Untitled")
            print(f"  [{i}/{len(pending)}] Screening item {content_id}: {title[:50]}...")
            try:
                decision = self._screen_one(row)
                self._apply_decision(content_id, decision)

                status = decision["decision"]
                if status == "approve":
                    summary["approved"] += 1
                elif status == "reject":
                    summary["rejected"] += 1
                else:
                    summary["errors"] += 1
                summary["details"].append({
                    "id": content_id,
                    "title": title,
                    "decision": decision["decision"],
                    "reason": decision.get("reason", ""),
                    "confidence": decision.get("confidence", 0.0),
                })
                logger.info(
                    f"[{decision['decision'].upper()}] id={content_id} "
                    f"\"{title}\" — {decision.get('reason', '')}"
                )
            except Exception as e:
                error_cat = _classify_error(e)
                tb = traceback.format_exc()
                transcript_len = len(row.get('transcript') or '')

                logger.error(f"Screening failed for id={content_id}: {e}")
                syslog.error('pipeline', 'screening_error',
                             f'Screening failed [{error_cat}]: {title[:50]}',
                             content_id=content_id,
                             details={
                                 'error_category': error_cat,
                                 'error_type': type(e).__name__,
                                 'error_message': str(e)[:500],
                                 'traceback': tb[-500:],
                                 'content_id': content_id,
                                 'title': title,
                                 'source_type': row.get('source_type'),
                                 'source_name': row.get('source_name'),
                                 'transcript_length': transcript_len,
                             })
                # Mark as 'error' in DB with reason for later review
                self.db.update_record(content_id, {
                    'screening_status': 'error',
                    'screening_reason': f'[{error_cat}] {str(e)[:200]}',
                })
                summary["errors"] += 1
                summary["details"].append({
                    "id": content_id,
                    "title": title,
                    "decision": "error",
                    "reason": str(e),
                    "error_category": error_cat,
                })

        return summary

    # ── internals ─────────────────────────────────────────────────────────

    def _screen_one(self, row: dict) -> dict:
        """Ask the LLM to approve or reject a single content item."""
        transcript = row.get("transcript") or ""
        excerpt = transcript[:MAX_EXCERPT_CHARS]

        if not excerpt.strip():
            raise ValueError("Empty transcript — nothing to screen")

        prompt = USER_PROMPT_TEMPLATE.format(
            title=row.get("title", "Untitled"),
            source_name=row.get("source_name", "Unknown"),
            source_type=row.get("source_type", "unknown"),
            char_count=len(transcript),
            excerpt=excerpt,
        )

        raw = self.llm.generate(prompt, SYSTEM_PROMPT, temperature=0.1)
        return self._parse_response(raw, content_id=row.get("id"))

    @staticmethod
    def _parse_response(raw: str, content_id: int = None) -> dict:
        """Extract the JSON decision from the LLM response."""
        if raw is None:
            raise ValueError("LLM returned None response")

        text = raw.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        parsed = None
        # Try 1: Parse entire response as JSON
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try 2: Extract first valid JSON object using JSONDecoder
        if parsed is None:
            try:
                from json import JSONDecoder
                decoder = JSONDecoder()
                start = text.find("{")
                if start >= 0:
                    parsed, _ = decoder.raw_decode(text, start)
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Try 3: Find JSON boundaries manually
        if parsed is None:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
        
        if parsed is None:
            # Log the raw response for model training / debugging
            syslog.error('pipeline', 'screening_parse_fail',
                         f'Could not parse LLM response for content_id={content_id}',
                         content_id=content_id,
                         details={
                             'error_category': 'parse_error',
                             'raw_response': text[:500],
                             'response_length': len(text),
                         })
            raise ValueError(f"Could not parse LLM screening response: {text[:200]}")

        decision = parsed.get("decision", "").lower().strip()
        if decision not in ("approve", "reject"):
            syslog.error('pipeline', 'screening_invalid_decision',
                         f'LLM returned invalid decision "{decision}" for content_id={content_id}',
                         content_id=content_id,
                         details={
                             'error_category': 'validation_error',
                             'raw_decision': decision,
                             'parsed_json': parsed,
                             'raw_response': text[:500],
                         })
            raise ValueError(f"Invalid decision '{decision}', expected 'approve' or 'reject'")

        return {
            "decision": decision,
            "reason": parsed.get("reason", ""),
            "confidence": float(parsed.get("confidence", 0.5)),
        }

    def _apply_decision(self, content_id: int, decision: dict):
        """Persist the screening result to the content table."""
        status = "approved" if decision["decision"] == "approve" else "rejected"
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        update_data = {
            "screening_status": status,
            "screening_reason": decision.get("reason", ""),
            "screened_at": now,
        }

        if status == "rejected":
            update_data["marked_for_deletion"] = True

        self.db.update_record(content_id, update_data)

    # ── utility ───────────────────────────────────────────────────────────

    def get_approved_ids(self, limit: int = 50) -> List[int]:
        """Return content IDs that passed screening but haven't had signals extracted."""
        rows = self.db.query("""
            SELECT id FROM content
            WHERE screening_status = 'approved'
              AND (signal_processed = FALSE OR signal_processed IS NULL)
            ORDER BY id
            LIMIT ?
        """, [limit])
        return [r["id"] for r in rows]

    def get_rejection_report(self, limit: int = 100) -> List[dict]:
        """Return recently rejected items for review / garbage collection."""
        return self.db.query("""
            SELECT id, title, source_name, screening_reason, screened_at
            FROM content
            WHERE screening_status = 'rejected'
            ORDER BY screened_at DESC
            LIMIT ?
        """, [limit])

    def get_deletion_candidates(self) -> List[dict]:
        """Return all items marked for deletion (for cron / garbage collection)."""
        return self.db.query("""
            SELECT id, title, file_path, source_name, screening_reason, screened_at
            FROM content
            WHERE marked_for_deletion = TRUE
            ORDER BY screened_at
        """)
