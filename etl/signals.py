import os
import langextract as lx
import textwrap
import json
import logging
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
from logging_config import syslog
from llm_client import _classify_error
import pydantic
from pydantic import BaseModel, Field, validator


logger = logging.getLogger(__name__)

CONFIDENCE_MAP = {
    'match_exact': 0.95,
    'match_fuzzy': 0.75,
    'match_lesser': 0.60,
    None: 0.30
}


class ManufacturingNewsSignal(BaseModel):
    """Pydantic model for current events: partnerships, investments, launches, acquisitions."""
    signal_type: str
    entity: str = Field(..., min_length=2, max_length=500)
    description: str
    industry: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    impact_level: Optional[str] = None
    timeline: Optional[str] = None
    event_type: Optional[str] = None
    monetary_value: Optional[str] = None
    source_content_id: int
    metadata_json: Optional[str] = None
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    context_window: Optional[str] = None
    enriched_text: Optional[str] = None
    enrichment_version: Optional[str] = None

    @validator('signal_type')
    def validate_signal_type(cls, v):
        allowed = list(NEWS_TYPES) + ['metric']
        if v not in allowed:
            raise ValueError(f'News signal_type must be one of {allowed}')
        return v


class ManufacturingTechSignal(BaseModel):
    """Pydantic model for technical facts: specs, protocols, standards, performance data."""
    signal_type: str
    entity: str = Field(..., min_length=2, max_length=500)
    description: str
    industry: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    impact_level: Optional[str] = None
    technical_spec: Optional[str] = None
    performance_metric: Optional[str] = None
    source_content_id: int
    metadata_json: Optional[str] = None
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    context_window: Optional[str] = None
    enriched_text: Optional[str] = None
    enrichment_version: Optional[str] = None

    @validator('signal_type')
    def validate_signal_type(cls, v):
        allowed = list(TECH_TYPES) + ['metric']
        if v not in allowed:
            raise ValueError(f'Tech signal_type must be one of {allowed}')
        return v


NEWS_TYPES = {'company', 'person', 'people', 'event', 'market', 'location', 'goal', 'program', 'initiative'}
TECH_TYPES = {'technology', 'industry', 'standard', 'protocol', 'service', 'product', 'solution'}
METRIC_TYPE = 'metric'


SIGNAL_TYPES = [
    "trend",
    "product_launch",
    "partnership",
    "funding",
    "regulation",
    "disruption",
    "adoption",
]


PROMPT = textwrap.dedent("""\
    "Extract companies, people, technologies, and metrics from manufacturing content".

    Provide meaningful attributes for every entity to add context and depth.

    Important: Use exact text from the input for extraction_text. Do not paraphrase.
    Extract entities in order of appearance with no overlapping text spans.

    CRITICAL: Extract ONLY the entity name itself, not surrounding sentences.
    Examples:
    -"Rockwell Automation" (not "Rockwell Automation announced partnership")
    -"CompactLogix 5480" (not "The CompactLogix 5480 was novel")
    -"as a real" (not "we've always worked with, as a real, and we're trying")
""")
EXAMPLES = [
        lx.data.ExampleData(
            text="Rockwell Automation announced partnership with Microsoft",
            extractions=[
                lx.data.Extraction(
                    extraction_class="company",
                    extraction_text="Rockwell Automation",
                    attributes={"type": "industrial_automation"}
                )
            ]
        ),
        lx.data.ExampleData(
            text="The CompactLogix 5480 was novel in the Industrial IoT world as it combined the elements of a PLC and compute device by having an onboard operating system ",
            extractions=[
                lx.data.Extraction(
                    extraction_class="technology",
                    extraction_text="CompactLogix 5480",
                    attributes={"type": "PLC"}
                ),
                lx.data.Extraction(
                    extraction_class="industry",
                    extraction_text="industrial IoT",
                    attributes={"type": "manufacturing"}
                ),
                 lx.data.Extraction(
                    extraction_class="technology",
                    extraction_text="compute",
                    attributes={"type": "operating system"}
                )
            ]
        ),
        lx.data.ExampleData(
            text="GE announced $500M investment in predictive maintenance, aiming to reduce downtime by 35% by 2025",
            extractions=[
                lx.data.Extraction(
                    extraction_class="company",
                    extraction_text="GE",
                    attributes={"type": "conglomerate"}
                ),
                lx.data.Extraction(
                    extraction_class="event",
                    extraction_text="investment",
                    attributes={"type": "funding", "value": "$500M"}
                ),
                lx.data.Extraction(
                    extraction_class="technology",
                    extraction_text="predictive maintenance",
                    attributes={"type": "AI application"}
                ),
                lx.data.Extraction(
                    extraction_class="metric",
                    extraction_text="35%",
                    attributes={"type": "downtime_reduction"}
                ),
                lx.data.Extraction(
                    extraction_class="metric",
                    extraction_text="2025",
                    attributes={"type": "timeline"}
                )
            ]
        )
    ]


# ── Enrichment config ────────────────────────────────────────────────────
# USE_LLM_ENRICHMENT=true activates optional LLM rewrite of context windows.
# Default (false) uses deterministic "<entity> — <context_window>" which is
# cheaper, faster, and hallucination-free.
USE_LLM_ENRICHMENT = os.getenv('USE_LLM_ENRICHMENT', 'false').lower() == 'true'
DEFAULT_CONTEXT_CHARS = int(os.getenv('ENRICHMENT_CONTEXT_CHARS', '250'))
ENRICHMENT_VERSION = 'v1'

# Strict LLM prompt — only used when USE_LLM_ENRICHMENT=true
LLM_ENRICHMENT_PROMPT = textwrap.dedent("""\
    Rewrite the following into ONE factual sentence that describes the entity
    in the context of industrial manufacturing / automation.

    RULES:
    - Use ONLY information present in the provided context. Do NOT add facts.
    - Do NOT generalize, speculate, or hallucinate.
    - Keep the entity name exactly as given.
    - Output a single sentence, nothing else.

    Entity: {entity}
    Signal type: {signal_type}
    Context: {context}

    Sentence:""")


class SignalPipeline:
    def __init__(self, db, llm_client=None, llm_url=None):
        self.db = db
        self.llm = llm_client
        self.llm_url = llm_url
        
        # Log configuration for debugging
        logger.info(f"SignalPipeline initialized with model_id={llm_client}, model_url={llm_url}")
        
        # Check if Gemini API key is available
        if 'gemini' in str(llm_client).lower() or 'generativelanguage' in str(llm_url).lower():
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                logger.info("GEMINI_API_KEY found - will use paid tier")
            else:
                logger.warning("GEMINI_API_KEY not found - may use free tier or fail")


    def chunk_transcript(self, transcript, max_chars=1500):
        """Split long transcripts into smaller chunks to avoid timeout."""
        chunks = []
        for i in range(0, len(transcript), max_chars):
            chunk = transcript[i:i+max_chars]
            chunks.append(chunk)
        return chunks

    # ── Enrichment methods ───────────────────────────────────────────────

    @staticmethod
    def extract_context_window(text: str, char_start: int, char_end: int,
                               window: int = DEFAULT_CONTEXT_CHARS) -> str:
        """Extract a context window around a char-offset span.

        Uses the original source text and the char offsets stored in
        metadata_json during extraction. This grounds the context in the
        actual document — no LLM involved, no hallucination possible.

        Args:
            text:       The full source text (transcript / article body).
            char_start: Start character offset of the entity mention.
            char_end:   End character offset of the entity mention.
            window:     Number of characters to include before and after
                        the entity span (default ~250).

        Returns:
            A cleaned substring centered on the entity mention.
        """
        if not text:
            return ''
        start = max(0, char_start - window)
        end = min(len(text), char_end + window)
        snippet = text[start:end].strip()
        # Trim to nearest word boundary at edges
        if start > 0:
            first_space = snippet.find(' ')
            if first_space > 0 and first_space < 30:
                snippet = snippet[first_space + 1:]
        if end < len(text):
            last_space = snippet.rfind(' ')
            if last_space > 0 and (len(snippet) - last_space) < 30:
                snippet = snippet[:last_space]
        return snippet

    @staticmethod
    def build_enriched_text(entity: str, signal_type: str,
                            context_window: str, industry: str = '') -> str:
        """Build a deterministic enriched_text from entity + context.

        Option A (default): No LLM. Produces a structured string that is
        rich enough for embedding but guaranteed hallucination-free.

        Format: "<signal_type> | <entity> — <context_window>"
        The signal_type prefix helps the embedding model understand the
        semantic role of the entity (company vs technology vs metric).
        """
        prefix = f"{signal_type}"
        if industry:
            prefix += f" [{industry}]"
        return f"{prefix} | {entity} — {context_window}"

    def enrich_signals(self, content_ids: Optional[List[int]] = None,
                       batch_size: int = 100,
                       use_llm: bool = USE_LLM_ENRICHMENT) -> dict:
        """Enrich existing signals with context_window and enriched_text.

        Reads signals from the DB, looks up the source transcript, extracts
        a context window around each entity mention using char offsets stored
        in metadata_json, and writes enriched_text back.

        Can be run as a backfill on already-extracted signals.

        Args:
            content_ids: Optional list of content IDs to limit enrichment to.
                         If None, enriches all signals missing enriched_text.
            batch_size:  Number of signals to process per DB round-trip.
            use_llm:     If True, uses LLM to rewrite context into a single
                         grounded sentence. Default: False (deterministic).

        Returns:
            dict with counts: enriched, skipped, errors.
        """
        # Fetch signals that need enrichment
        if content_ids:
            placeholders = ', '.join(['?' for _ in content_ids])
            signals = self.db.query(
                f"""SELECT id, entity, signal_type, industry, metadata_json,
                           source_content_id
                    FROM signals
                    WHERE source_content_id IN ({placeholders})
                      AND (enriched_text IS NULL OR enriched_text = '')
                    ORDER BY id
                    LIMIT ?""",
                content_ids + [batch_size]
            )
        else:
            signals = self.db.query(
                """SELECT id, entity, signal_type, industry, metadata_json,
                          source_content_id
                   FROM signals
                   WHERE enriched_text IS NULL OR enriched_text = ''
                   ORDER BY id
                   LIMIT ?""",
                [batch_size]
            )

        if not signals:
            logger.info("No signals need enrichment")
            return {'enriched': 0, 'skipped': 0, 'errors': 0}

        # Group by source_content_id to batch-load transcripts
        content_ids_needed = list({s['source_content_id'] for s in signals})
        transcripts = {}
        for cid in content_ids_needed:
            rows = self.db.query(
                "SELECT id, transcript FROM content WHERE id = ?", [cid]
            )
            if rows and rows[0].get('transcript'):
                transcripts[cid] = rows[0]['transcript']

        enriched = 0
        skipped = 0
        errors = 0

        for sig in signals:
            try:
                meta = json.loads(sig.get('metadata_json') or '{}')
                char_start = meta.get('char_start')
                char_end = meta.get('char_end')
                transcript = transcripts.get(sig['source_content_id'], '')

                if char_start is None or char_end is None or not transcript:
                    skipped += 1
                    continue

                # Step 1: Extract grounded context window
                ctx = self.extract_context_window(
                    transcript, char_start, char_end
                )
                if not ctx:
                    skipped += 1
                    continue

                # Step 2: Build enriched_text
                if use_llm and self.llm:
                    # Option B: LLM rewrite (gated, strict prompt)
                    try:
                        prompt = LLM_ENRICHMENT_PROMPT.format(
                            entity=sig['entity'],
                            signal_type=sig['signal_type'],
                            context=ctx
                        )
                        from llm_client import GeminiClient, OllamaClient
                        if isinstance(self.llm, str) and 'gemini' in self.llm.lower():
                            client = GeminiClient(model=self.llm)
                        else:
                            client = OllamaClient(model=self.llm, base_url=self.llm_url)
                        enriched_str = client.generate(prompt).strip()
                        version = f"{ENRICHMENT_VERSION}-llm"
                    except Exception as e:
                        logger.warning(f"LLM enrichment failed for signal {sig['id']}, "
                                       f"falling back to deterministic: {e}")
                        enriched_str = self.build_enriched_text(
                            sig['entity'], sig['signal_type'], ctx, sig.get('industry', '')
                        )
                        version = f"{ENRICHMENT_VERSION}-fallback"
                else:
                    # Option A: Deterministic (default — no hallucination)
                    enriched_str = self.build_enriched_text(
                        sig['entity'], sig['signal_type'], ctx, sig.get('industry', '')
                    )
                    version = ENRICHMENT_VERSION

                # Step 3: Write back to DB
                self.db.execute(
                    """UPDATE signals
                       SET context_window = ?, enriched_text = ?, enrichment_version = ?
                       WHERE id = ?""",
                    [ctx, enriched_str, version, sig['id']]
                )
                enriched += 1

            except Exception as e:
                logger.error(f"Enrichment error for signal {sig.get('id')}: {e}")
                errors += 1

        logger.info(f"Enrichment complete: {enriched} enriched, {skipped} skipped, {errors} errors")
        return {'enriched': enriched, 'skipped': skipped, 'errors': errors}

    def extract_from_batch(self, content_ids: list) -> dict:
        placeholders = ', '.join(['?' for _ in content_ids])
        rows = self.db.query(
            f"SELECT id, title, transcript FROM content WHERE id IN ({placeholders})",
            content_ids
        )

        results = {}
        for row in rows:
            content_id = row['id']
            try:
                transcript = row['transcript']
                chunks = self.chunk_transcript(transcript) if len(transcript) > 1500 else [transcript]

                all_signals = []
                chunk_offset = 0
                for i, chunk in enumerate(chunks):
                    if len(chunks) > 1:
                        logger.info(f"Processing chunk {i+1}/{len(chunks)} for content_id={content_id}")
                    
                    try:
                        result = lx.extract(
                            text_or_documents=chunk,
                            prompt_description=PROMPT,
                            examples=EXAMPLES,
                            model_id=self.llm,
                            model_url=self.llm_url
                        )
                        all_signals.extend(self._parse_signals(
                            result, content_id,
                            source_text=transcript, chunk_offset=chunk_offset
                        ))
                    except Exception as e:
                        error_cat = _classify_error(e)
                        error_msg = str(e)
                        tb = traceback.format_exc()

                        # Classify recoverable vs fatal errors
                        recoverable = any(k in error_msg.lower() for k in [
                            'json', 'parse', 'format', 'decode',
                            'timeout', 'rate', 'quota', 'resource',
                        ])

                        syslog.error('pipeline', 'signal_chunk_error',
                                     f'LangExtract error [{error_cat}] chunk {i+1}/{len(chunks)}: {row["title"][:50]}',
                                     content_id=content_id,
                                     details={
                                         'error_category': error_cat,
                                         'error_type': type(e).__name__,
                                         'error_message': error_msg[:500],
                                         'traceback': tb[-500:],
                                         'chunk_index': i,
                                         'total_chunks': len(chunks),
                                         'chunk_length': len(chunk),
                                         'model_id': str(self.llm),
                                         'recoverable': recoverable,
                                     })

                        if recoverable:
                            logger.warning(f"LangExtract recoverable error for chunk {i+1}, "
                                           f"content_id={content_id} [{error_cat}]: {e}")
                            chunk_offset += len(chunk)
                            continue
                        else:
                            raise

                    chunk_offset += len(chunk)

                stored_count = self._store_signals(all_signals)
                logger.info(f"Stored {stored_count} signals")

                # Always mark as processed, even if some signals failed validation
                try:
                    self.db.update_record(content_id, {'signal_processed': True})
                    logger.info(f"Marked content_id={content_id} as processed (stored {stored_count} signals)")
                except Exception as e:
                    logger.error(f"Failed to update signal_processed flag for content_id={content_id}: {e}")
                    # Don't fail the entire extraction for this

                results[content_id] = {'status': 'success', 'signals_stored': stored_count}
                logger.info(f"Extracted and stored {stored_count} signals from content_id={content_id} ({row['title']})")
                syslog.info('pipeline', 'signals', f'Extracted {stored_count} signals from {row["title"][:50]}',
                            content_id=content_id, details={'chunks': len(chunks), 'processed': len(all_signals)})
            except Exception as e:
                error_cat = _classify_error(e)
                tb = traceback.format_exc()
                transcript_len = len(row.get('transcript') or '')

                results[content_id] = {
                    'status': 'failed',
                    'error': str(e),
                    'error_category': error_cat,
                }
                logger.error(f"Failed extraction for content_id={content_id}: {e}")
                logger.error(f"Full traceback for content_id={content_id}: {tb}")
                syslog.error('pipeline', 'signal_extraction_failed',
                             f'Signal extraction failed [{error_cat}]: {row["title"][:50]}',
                             content_id=content_id,
                             details={
                                 'error_category': error_cat,
                                 'error_type': type(e).__name__,
                                 'error_message': str(e)[:500],
                                 'traceback': tb[-500:],
                                 'content_id': content_id,
                                 'title': row.get('title'),
                                 'transcript_length': transcript_len,
                                 'chunks_attempted': len(chunks) if 'chunks' in dir() else 0,
                                 'signals_before_error': len(all_signals) if 'all_signals' in dir() else 0,
                                 'model_id': str(self.llm),
                             })
                continue

        return results

    def _parse_signals(self, extraction_result, content_id: int,
                       source_text: str = '', chunk_offset: int = 0) -> list:
        """Convert LangExtract Extraction objects into validated Pydantic signal models.

        When source_text is provided, enrichment happens inline:
        char offsets from the chunk are translated to full-document offsets
        (via chunk_offset), a context_window is extracted, and enriched_text
        is built deterministically.  This avoids a separate backfill pass
        for newly extracted signals.
        """
        validated = []
        for ext in extraction_result.extractions:
            if ext.char_interval is None:
                continue
            
            # Skip single-character entities (validation will fail)
            if len(ext.extraction_text.strip()) < 2:
                logger.debug(f"Skipping single-character entity: '{ext.extraction_text}'")
                syslog.info('pipeline', 'signal_validation_skip',
                            f'Skipped short entity: "{ext.extraction_text}"',
                            content_id=content_id,
                            details={
                                'reason': 'entity_too_short',
                                'entity': ext.extraction_text,
                                'extraction_class': ext.extraction_class,
                            })
                continue

            alignment_str = str(ext.alignment_status) if ext.alignment_status else None
            confidence = CONFIDENCE_MAP.get(ext.alignment_status, 0.30)
            attrs = ext.attributes or {}

            # Translate chunk-local char offsets to full-document offsets
            abs_start = ext.char_interval.start_pos + chunk_offset
            abs_end = ext.char_interval.end_pos + chunk_offset

            # Inline enrichment: extract context window from full source text
            ctx_window = None
            enriched = None
            e_version = None
            if source_text:
                ctx_window = self.extract_context_window(
                    source_text, abs_start, abs_end
                )
                if ctx_window:
                    enriched = self.build_enriched_text(
                        ext.extraction_text, ext.extraction_class,
                        ctx_window, attrs.get('type', '')
                    )
                    e_version = ENRICHMENT_VERSION

            signal_data = {
                'signal_type': ext.extraction_class,
                'entity': ext.extraction_text,
                'description': f"{ext.extraction_text} ({attrs.get('type', '')})",
                'industry': attrs.get('type', ''),
                'confidence': confidence,
                'source_content_id': content_id,
                'metadata_json': json.dumps({
                    'char_start': abs_start,
                    'char_end': abs_end,
                    'alignment': alignment_str,
                    'attributes': attrs
                }),
                'context_window': ctx_window,
                'enriched_text': enriched,
                'enrichment_version': e_version,
            }

            try:
                if ext.extraction_class in NEWS_TYPES:
                    signal = ManufacturingNewsSignal(**signal_data)
                elif ext.extraction_class in TECH_TYPES:
                    signal = ManufacturingTechSignal(**signal_data)
                elif ext.extraction_class == METRIC_TYPE:
                    signal = ManufacturingNewsSignal(**signal_data)
                else:
                    logger.warning(f"Unknown signal type '{ext.extraction_class}' for entity '{ext.extraction_text}', skipping")
                    syslog.warning('pipeline', 'signal_unknown_type',
                                   f'Unknown signal type: {ext.extraction_class}',
                                   content_id=content_id,
                                   details={
                                       'signal_type': ext.extraction_class,
                                       'entity': ext.extraction_text[:100],
                                       'known_news_types': list(NEWS_TYPES),
                                       'known_tech_types': list(TECH_TYPES),
                                   })
                    continue
                validated.append(signal)
            except Exception as e:
                logger.warning(f"Validation failed for '{ext.extraction_text}': {e}")
                syslog.warning('pipeline', 'signal_validation_failed',
                               f'Pydantic validation failed: {ext.extraction_text[:50]}',
                               content_id=content_id,
                               details={
                                   'error_type': type(e).__name__,
                                   'error_message': str(e)[:300],
                                   'entity': ext.extraction_text[:100],
                                   'signal_type': ext.extraction_class,
                                   'signal_data': {k: str(v)[:100] for k, v in signal_data.items()},
                               })
                continue

        return validated

    def _store_signals(self, signals: list) -> int:
        """Store validated Pydantic signal models into the signals table.

        Returns the number of signals successfully inserted.
        Uses db.insert_signal() which is backend-agnostic (DuckDB or PostgreSQL).
        """
        inserted = 0
        for signal in signals:
            try:
                self.db.insert_signal(signal)
                inserted += 1
            except Exception as e:
                logger.error(f"Failed to store signal '{signal.entity}': {e}")
                syslog.error('pipeline', 'signal_store_failed',
                             f'DB insert failed: {signal.entity[:50]}',
                             content_id=signal.source_content_id,
                             details={
                                 'error_type': type(e).__name__,
                                 'error_message': str(e)[:300],
                                 'entity': signal.entity[:100],
                                 'signal_type': signal.signal_type,
                             })
                continue
        logger.info(f"Stored {inserted}/{len(signals)} signals")
        return inserted

    # ── Vectorization prep ─────────────────────────────────────────────────

    def get_enriched_for_vectorization(self, batch_size: int = 500,
                                       min_confidence: float = 0.0
                                       ) -> List[Dict[str, Any]]:
        """Retrieve enriched signals ready for embedding on the client (M2).

        Returns a list of dicts with signal_id and enriched_text.
        Only returns signals that have been enriched but NOT yet vectorized,
        so this can be called repeatedly for incremental vectorization.

        Downstream usage:
            batches = pipeline.get_enriched_for_vectorization()
            texts = [b['enriched_text'] for b in batches]
            ids   = [b['signal_id']     for b in batches]
            embeddings = embedding_model.encode(texts, batch_size=64)
            # Then store embeddings in FAISS with ids as metadata

        Args:
            batch_size:     Max rows to return per call.
            min_confidence: Only include signals above this confidence.

        Returns:
            List of {'signal_id': int, 'enriched_text': str,
                     'entity': str, 'signal_type': str, 'industry': str}
        """
        rows = self.db.query(
            """SELECT id, entity, signal_type, industry, enriched_text
               FROM signals
               WHERE enriched_text IS NOT NULL
                 AND enriched_text != ''
                 AND (vectorized = FALSE OR vectorized IS NULL)
                 AND confidence >= ?
               ORDER BY id
               LIMIT ?""",
            [min_confidence, batch_size]
        )
        return [
            {
                'signal_id': r['id'],
                'enriched_text': r['enriched_text'],
                'entity': r['entity'],
                'signal_type': r['signal_type'],
                'industry': r.get('industry', ''),
            }
            for r in rows
        ]

    def mark_vectorized(self, signal_ids: List[int]) -> int:
        """Mark signals as vectorized after embedding.

        Call this after successfully storing embeddings in FAISS.

        Args:
            signal_ids: List of signal IDs that were vectorized.

        Returns:
            Number of rows updated.
        """
        if not signal_ids:
            return 0
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        updated = 0
        for sid in signal_ids:
            try:
                self.db.execute(
                    "UPDATE signals SET vectorized = ?, vectorized_at = ? WHERE id = ?",
                    [True, now, sid]
                )
                updated += 1
            except Exception as e:
                logger.error(f"Failed to mark signal {sid} as vectorized: {e}")
        logger.info(f"Marked {updated}/{len(signal_ids)} signals as vectorized")
        return updated


if __name__ == "__main__":

        from db_relational import relationalDB
        from device_config import config
        import os

        db_path = config.DB_PATH
        model_id = config.LLM_MODEL
        model_url = config.LLM_URL
        db = relationalDB(db_path)

        #query = "SELECT id, title, transcript FROM content WHERE id=5"
        #input_text = db.query(query)

        sp = SignalPipeline(db,model_id,model_url)
        content_ids=[5]
        signals = sp.extract_from_batch(content_ids)

        
        print("complete!")