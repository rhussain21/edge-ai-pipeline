import langextract as lx
import textwrap
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from logging_config import syslog
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
                        all_signals.extend(self._parse_signals(result, content_id))
                    except Exception as e:
                        # Handle langextract JSON parsing errors gracefully
                        error_msg = str(e)
                        if "JSONDecodeError" in error_msg or "FormatParseError" in error_msg:
                            logger.warning(f"LangExtract JSON parse error for chunk {i+1}, content_id={content_id}: {e}")
                            # Continue with other chunks instead of failing the entire content
                            continue
                        else:
                            # Re-raise non-JSON errors
                            raise

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
                results[content_id] = {'status': 'failed', 'error': str(e)}
                logger.error(f"Failed extraction for content_id={content_id}: {e}")
                import traceback
                logger.error(f"Full traceback for content_id={content_id}: {traceback.format_exc()}")
                syslog.error('pipeline', 'signals', f'Signal extraction failed: {row["title"][:50]}',
                             content_id=content_id, details={'error': str(e)})
                continue

        return results

    def _parse_signals(self, extraction_result, content_id: int) -> list:
        """Convert LangExtract Extraction objects into validated Pydantic signal models."""
        validated = []
        for ext in extraction_result.extractions:
            if ext.char_interval is None:
                continue
            
            # Skip single-character entities (validation will fail)
            if len(ext.extraction_text.strip()) < 2:
                logger.debug(f"Skipping single-character entity: '{ext.extraction_text}'")
                continue

            alignment_str = str(ext.alignment_status) if ext.alignment_status else None
            confidence = CONFIDENCE_MAP.get(ext.alignment_status, 0.30)
            attrs = ext.attributes or {}

            signal_data = {
                'signal_type': ext.extraction_class,
                'entity': ext.extraction_text,
                'description': f"{ext.extraction_text} ({attrs.get('type', '')})",
                'industry': attrs.get('type', ''),
                'confidence': confidence,
                'source_content_id': content_id,
                'metadata_json': json.dumps({
                    'char_start': ext.char_interval.start_pos,
                    'char_end': ext.char_interval.end_pos,
                    'alignment': alignment_str,
                    'attributes': attrs
                })
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
                    continue
                validated.append(signal)
            except Exception as e:
                logger.warning(f"Validation failed for '{ext.extraction_text}': {e}")
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
                continue
        logger.info(f"Stored {inserted}/{len(signals)} signals")
        return inserted

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