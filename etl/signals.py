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


# ── Signal type taxonomy ──────────────────────────────────────────────
# extraction_class in LangExtract maps to signal_type in the DB.
# Each value answers: "What is happening?" — not "what kind of entity is this?"
SIGNAL_TYPES = {
    'partnership',        # Alliances, collaborations, joint ventures, integrations
    'product_launch',     # New products, versions, platforms, features announced
    'investment',         # Funding rounds, acquisitions, capital expenditure, M&A
    'adoption',           # Technology deployed, implemented, scaled in production
    'regulation',         # Standards published, compliance mandates, certifications
    'research',           # Studies, patents, technical breakthroughs, white papers
    'performance_claim',  # Quantitative capability claims, benchmarks, test results
    'market_trend',       # Industry growth, decline, directional shifts, forecasts
    'workforce',          # Hiring, layoffs, skills programs, org changes
}


class IndustrySignal(BaseModel):
    """Unified signal model — one record per actionable industry event/claim."""
    signal_type: str
    entity: str = Field(..., min_length=2, max_length=500)
    description: str
    industry: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    impact_level: Optional[str] = None
    source_content_id: int
    metadata_json: Optional[str] = None
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    context_window: Optional[str] = None
    enriched_text: Optional[str] = None
    enrichment_version: Optional[str] = None
    extraction_tool: Optional[str] = None

    @validator('signal_type')
    def validate_signal_type(cls, v):
        if v not in SIGNAL_TYPES:
            raise ValueError(
                f'signal_type must be one of {sorted(SIGNAL_TYPES)}, got "{v}"'
            )
        return v


PROMPT = textwrap.dedent("""\
    Extract industry signals from manufacturing, industrial automation, and
    technology content. A signal is an EVENT, ACTION, CLAIM, or RELATIONSHIP
    — not a bare noun or keyword.

    Classify each signal by what is HAPPENING:
    - partnership: alliances, collaborations, integrations between entities
    - product_launch: new products, versions, platforms, features announced
    - investment: funding, acquisitions, capital expenditure, M&A
    - adoption: technology being deployed or scaled in production
    - regulation: standards published, compliance mandates, certifications
    - research: studies, patents, technical breakthroughs
    - performance_claim: quantitative claims about capability or improvement
    - market_trend: industry growth, directional shifts, forecasts
    - workforce: hiring, layoffs, skills initiatives, org changes

    DO NOT extract:
    - Bare nouns (e.g., "processor", "data", "software")
    - Version numbers or dates without surrounding context
    - Generic industry terms (e.g., "manufacturing", "industry")

    For each signal provide these attributes:
    - entity_type: company, technology, standard, person, or organization
    - industry: the specific sector (e.g., industrial_automation, manufacturing,
      energy, cybersecurity, semiconductors)
    - detail: key qualifier (monetary value, percentage, date, etc.)

    Use exact text from the input for extraction_text. Do not paraphrase.
    Extract in order of appearance with no overlapping text spans.
    Extract ONLY the primary entity name, not full sentences.
""")

EXAMPLES = [
    lx.data.ExampleData(
        text="Rockwell Automation announced a strategic partnership with Microsoft to integrate Azure IoT services into its FactoryTalk platform for smart manufacturing.",
        extractions=[
            lx.data.Extraction(
                extraction_class="partnership",
                extraction_text="Rockwell Automation",
                attributes={
                    "entity_type": "company",
                    "industry": "industrial_automation",
                    "detail": "Azure IoT integration with FactoryTalk",
                }
            ),
        ]
    ),
    lx.data.ExampleData(
        text="Beckhoff released TwinCAT 3.1 Build 4148 with native OPC UA publish-subscribe support, targeting real-time industrial communication.",
        extractions=[
            lx.data.Extraction(
                extraction_class="product_launch",
                extraction_text="TwinCAT 3.1 Build 4148",
                attributes={
                    "entity_type": "technology",
                    "industry": "industrial_automation",
                    "detail": "OPC UA pub-sub support",
                }
            ),
        ]
    ),
    lx.data.ExampleData(
        text="GE invested $500M in predictive maintenance AI, aiming to reduce unplanned downtime by 35% across its aviation manufacturing facilities by 2025.",
        extractions=[
            lx.data.Extraction(
                extraction_class="investment",
                extraction_text="GE",
                attributes={
                    "entity_type": "company",
                    "industry": "manufacturing",
                    "detail": "$500M in predictive maintenance AI",
                }
            ),
            lx.data.Extraction(
                extraction_class="performance_claim",
                extraction_text="35%",
                attributes={
                    "entity_type": "metric",
                    "industry": "manufacturing",
                    "detail": "unplanned downtime reduction target by 2025",
                }
            ),
        ]
    ),
    lx.data.ExampleData(
        text="Siemens reported that over 200 factories now run its Industrial Edge platform, processing data locally for quality inspection using computer vision.",
        extractions=[
            lx.data.Extraction(
                extraction_class="adoption",
                extraction_text="Siemens",
                attributes={
                    "entity_type": "company",
                    "industry": "industrial_automation",
                    "detail": "200+ factories running Industrial Edge",
                }
            ),
        ]
    ),
    lx.data.ExampleData(
        text="The updated IEC 62443-4-2 standard now requires component-level cybersecurity certification for all programmable controllers sold in the EU.",
        extractions=[
            lx.data.Extraction(
                extraction_class="regulation",
                extraction_text="IEC 62443-4-2",
                attributes={
                    "entity_type": "standard",
                    "industry": "cybersecurity",
                    "detail": "component-level certification required for PLCs in EU",
                }
            ),
        ]
    ),
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
        
        # Check Gemini auth configuration
        if 'gemini' in str(llm_client).lower():
            gcp_project = os.getenv("GCP_PROJECT")
            if gcp_project:
                logger.info(f"Gemini will use Vertex AI (project={gcp_project})")
            elif os.getenv("GEMINI_API_KEY"):
                logger.info("Gemini will use AI Studio (api_key fallback)")
            else:
                logger.warning("No GCP_PROJECT or GEMINI_API_KEY found - Gemini calls may fail")


    def chunk_transcript(self, transcript, max_chars=8000):
        """Split long transcripts into smaller chunks to avoid timeout.

        Default 8000 chars (~2000 tokens) — Gemini handles this easily.
        Ollama/local models may need smaller chunks (pass max_chars=1500).
        """
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
                chunks = self.chunk_transcript(transcript) if len(transcript) > 8000 else [transcript]

                all_signals = []
                chunk_offset = 0

                # Build Vertex AI params for gemini models
                _lx_kwargs = {}
                if isinstance(self.llm, str) and 'gemini' in self.llm.lower():
                    gcp_project = os.getenv("GCP_PROJECT")
                    if gcp_project:
                        _lx_kwargs['language_model_params'] = {
                            'vertexai': True,
                            'project': gcp_project,
                            'location': os.getenv("GCP_LOCATION", "us-central1"),
                        }
                        logger.info(f"LangExtract will use Vertex AI (project={gcp_project})")

                for i, chunk in enumerate(chunks):
                    if len(chunks) > 1:
                        logger.info(f"Processing chunk {i+1}/{len(chunks)} for content_id={content_id}")

                    try:
                        result = lx.extract(
                            text_or_documents=chunk,
                            prompt_description=PROMPT,
                            examples=EXAMPLES,
                            model_id=self.llm,
                            model_url=self.llm_url,
                            **_lx_kwargs
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
        # Generic single-word tokens that add no signal value on their own.
        # Performance claims (numbers, percentages, bare metrics) are only
        # useful when the entity carries a real name — bare digits are noise.
        _GENERIC_TOKENS = {
            'companies', 'company', 'industry', 'industries', 'people',
            'professors', 'professor', 'colleges', 'college', 'universities',
            'university', 'customers', 'customer', 'users', 'user',
            'integrators', 'integrator', 'vendors', 'vendor', 'partners',
            'partner', 'clients', 'client', 'others', 'organizations',
            'organization', 'team', 'teams', 'staff', 'employees', 'workers',
            'students', 'student', 'schools', 'school', 'government', 'agencies',
            'things', 'stuff', 'data', 'information', 'content', 'software',
            'hardware', 'system', 'systems', 'solutions', 'technology',
            'technologies', 'product', 'products', 'service', 'services',
            'platform', 'platforms', 'tool', 'tools', 'project', 'projects',
        }

        validated = []
        for ext in extraction_result.extractions:
            if ext.char_interval is None:
                continue

            entity_clean = ext.extraction_text.strip()

            # --- Entity quality filters ---

            # 1. Too short to be meaningful
            if len(entity_clean) < 4:
                logger.debug(f"Skipping short entity: '{entity_clean}'")
                syslog.info('pipeline', 'signal_validation_skip',
                            f'Skipped short entity: "{entity_clean}"',
                            content_id=content_id,
                            details={'reason': 'entity_too_short', 'entity': entity_clean,
                                     'extraction_class': ext.extraction_class})
                continue

            # 2. Pure number or percentage (e.g. "300", "18s", "35%") with no
            #    alphabetic context — only valid for performance_claim where
            #    attrs.detail provides the meaning.
            import re as _re
            is_bare_number = bool(_re.fullmatch(r'[\d,\.\s%\$\+\-]+[a-zA-Z]{0,2}', entity_clean))
            if is_bare_number and ext.extraction_class != 'performance_claim':
                logger.debug(f"Skipping bare number entity: '{entity_clean}'")
                syslog.info('pipeline', 'signal_validation_skip',
                            f'Skipped bare number: "{entity_clean}"',
                            content_id=content_id,
                            details={'reason': 'bare_number', 'entity': entity_clean,
                                     'extraction_class': ext.extraction_class})
                continue

            # 3. Generic single-word token with no signal value
            if entity_clean.lower() in _GENERIC_TOKENS:
                logger.debug(f"Skipping generic token entity: '{entity_clean}'")
                syslog.info('pipeline', 'signal_validation_skip',
                            f'Skipped generic token: "{entity_clean}"',
                            content_id=content_id,
                            details={'reason': 'generic_token', 'entity': entity_clean,
                                     'extraction_class': ext.extraction_class})
                continue

            alignment_str = str(ext.alignment_status) if ext.alignment_status else None
            alignment_val = ext.alignment_status.value if ext.alignment_status else None
            confidence = CONFIDENCE_MAP.get(alignment_val, 0.30)
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
                        ctx_window, attrs.get('industry', '')
                    )
                    e_version = ENRICHMENT_VERSION

            # Build a meaningful description from attributes
            detail = attrs.get('detail', '')
            entity_type = attrs.get('entity_type', '')
            description = f"{ext.extraction_class}: {ext.extraction_text}"
            if detail:
                description += f" — {detail}"

            signal_data = {
                'signal_type': ext.extraction_class,
                'entity': ext.extraction_text,
                'description': description,
                'industry': attrs.get('industry', ''),
                'confidence': confidence,
                'source_content_id': content_id,
                'metadata_json': json.dumps({
                    'char_start': abs_start,
                    'char_end': abs_end,
                    'alignment': alignment_str,
                    'entity_type': entity_type,
                    'attributes': attrs
                }),
                'context_window': ctx_window,
                'enriched_text': enriched,
                'enrichment_version': e_version,
                'extraction_tool': 'langextract',
            }

            try:
                signal = IndustrySignal(**signal_data)
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

    @staticmethod
    def _dedup_signals(signals: list) -> list:
        """Remove duplicate signals within a batch before DB insertion.

        Two-pass deduplication:
        Pass 1 — exact key: (signal_type, normalised_entity). Catches the same
                  entity string appearing multiple times in one transcript.
        Pass 2 — fuzzy token overlap: within the same signal_type, two entities
                  whose token sets overlap ≥80% are treated as the same entity.
                  E.g. "Ignition" vs "Ignition software" vs "Ignition platform".
                  Highest-confidence copy wins in both passes.
        """
        import re as _re

        def _tokens(text: str) -> set:
            return set(_re.sub(r'[^a-z0-9 ]', '', text.lower()).split())

        # Pass 1: exact normalised key
        seen: dict = {}
        for sig in sorted(signals, key=lambda s: s.confidence, reverse=True):
            norm = _re.sub(r'[^a-z0-9]', '', sig.entity.lower())
            key = (sig.signal_type, norm)
            if key not in seen:
                seen[key] = sig
        pass1 = list(seen.values())

        # Pass 2: fuzzy token overlap (Jaccard ≥ 0.8 within same signal_type)
        groups: list = []
        for sig in sorted(pass1, key=lambda s: s.confidence, reverse=True):
            toks = _tokens(sig.entity)
            if not toks:
                groups.append(sig)
                continue
            merged = False
            for rep in groups:
                if rep.signal_type != sig.signal_type:
                    continue
                rep_toks = _tokens(rep.entity)
                if not rep_toks:
                    continue
                jaccard = len(toks & rep_toks) / len(toks | rep_toks)
                if jaccard >= 0.80:
                    merged = True
                    break
            if not merged:
                groups.append(sig)

        dropped = len(signals) - len(groups)
        if dropped:
            logger.info(f"Deduped {dropped} duplicate signals (kept {len(groups)})")
        return groups

    def _store_signals(self, signals: list) -> int:
        """Store validated Pydantic signal models into the signals table.

        Returns the number of signals successfully inserted.
        Uses db.insert_signal() which is backend-agnostic (DuckDB or PostgreSQL).
        """
        signals = self._dedup_signals(signals)
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
    import sys
    import argparse
    from db_relational import relationalDB
    from device_config import config

    parser = argparse.ArgumentParser(description="Extract signals from approved content")
    parser.add_argument('--ids', type=int, nargs='+', default=None,
                        help='Specific content IDs to extract')
    parser.add_argument('--limit', type=int, default=9999,
                        help='Max number of unprocessed items to pick (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be extracted but do not store signals')
    args = parser.parse_args()

    db = relationalDB(config.DB_PATH)
    sp = SignalPipeline(db, llm_client=config.LLM_MODEL, llm_url=config.LLM_URL)

    # Resolve content IDs
    if args.ids:
        content_ids = args.ids
    else:
        rows = db.query("""
            SELECT id, title, LENGTH(transcript) as char_len FROM content
            WHERE screening_status = 'approved'
              AND (signal_processed = FALSE OR signal_processed IS NULL)
              AND (do_not_vectorize = FALSE OR do_not_vectorize IS NULL)
            ORDER BY LENGTH(transcript) ASC
            LIMIT ?
        """, [args.limit])
        if not rows:
            print("No approved unprocessed content found.")
            sys.exit(0)
        content_ids = [r['id'] for r in rows]
        total_chars = sum(r.get('char_len', 0) or 0 for r in rows)
        est_tokens = total_chars // 4
        print(f"Selected {len(content_ids)} unprocessed items "
              f"(~{total_chars:,} chars / ~{est_tokens:,} tokens):")
        for r in rows:
            clen = r.get('char_len', 0) or 0
            print(f"  [{r['id']:>4}] {clen:>8,} chars  {r['title'][:70]}")

    print(f"\n{'='*60}")
    print(f"Extracting signals for content IDs: {content_ids}")
    print(f"Model: {config.LLM_MODEL}")
    print(f"{'='*60}\n")

    results = sp.extract_from_batch(content_ids)

    # ── Inspection: show every new signal ──
    print(f"\n{'='*60}")
    print("EXTRACTION RESULTS")
    print(f"{'='*60}")
    for cid, res in results.items():
        print(f"\n── content_id={cid}: {res['status']} ──")
        if res['status'] == 'failed':
            print(f"   ERROR: {res.get('error', '')[:200]}")
            continue
        print(f"   Signals stored: {res.get('signals_stored', 0)}")

    # Query back the newly inserted signals for inspection
    all_ids = [cid for cid, r in results.items() if r['status'] == 'success']
    if all_ids:
        placeholders = ', '.join(['?' for _ in all_ids])
        try:
            new_signals = db.query(f"""
                SELECT id, signal_type, entity, description, industry, confidence,
                       extraction_tool
                FROM signals
                WHERE source_content_id IN ({placeholders})
                ORDER BY source_content_id, id
            """, all_ids)
        except Exception:
            # extraction_tool column may not exist yet on this DB instance
            # (run: rsync db_relational.py to Jetson to trigger migration)
            new_signals = db.query(f"""
                SELECT id, signal_type, entity, description, industry, confidence
                FROM signals
                WHERE source_content_id IN ({placeholders})
                ORDER BY source_content_id, id
            """, all_ids)

        print(f"\n{'='*60}")
        print(f"SIGNAL DETAIL ({len(new_signals)} signals)")
        print(f"{'='*60}")
        for s in new_signals:
            print(f"  [{s['signal_type']:<18}] {s['entity'][:40]:<40}  "
                  f"industry={s.get('industry',''):<25}  conf={s['confidence']:.2f}")
            if s.get('description'):
                print(f"    desc: {s['description'][:100]}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")