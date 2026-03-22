"""
LLM-powered candidate classifier for source discovery.

Evaluates candidate sources for relevance to an industrial automation corpus.
Uses batched LLM calls with structured JSON output.
Supports caching to avoid re-classifying identical candidates.

Designed to work with your OllamaClient.generate(prompt, system_prompt, temperature).
"""

import json
import logging
from typing import List, Optional, Callable

from discovery.models import CandidateSource, Classification, ClassifiedCandidate
from discovery.cache import TaskCache

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a technical librarian building an industrial automation knowledge corpus.

Your job: evaluate candidate documents and decide if they belong in the corpus.

RELEVANT topics include:
- PLC programming (IEC 61131-3, ladder logic, structured text, function blocks)
- Industrial networking (EtherNet/IP, PROFINET, OPC UA, Modbus, MQTT)
- Machine safety (ISO 13849, IEC 62443, safety PLCs, light curtains)
- Industrial cybersecurity (IEC 62443, network segmentation, OT security)
- Robotics (industrial robots, cobots, robot programming, ROS for industrial)
- SCADA / HMI (supervisory control, visualization, alarm management)
- Manufacturing systems (MES, digital twin, Industry 4.0, smart manufacturing)
- Motion control (servo drives, VFDs, CNC, stepper motors)
- Industrial AI/ML (predictive maintenance, quality inspection, edge AI)
- Vendor documentation (Siemens, Rockwell/Allen-Bradley, ABB, Fanuc, Beckhoff, etc.)

PREFERRED document types:
- manual, whitepaper, specification, technical guide, application note
- official documentation, tutorial, reference architecture, standard

REJECT these:
- Generic marketing pages with no technical depth
- Vague news articles with no actionable content
- Business fluff (earnings reports, press releases about partnerships)
- Low-signal landing pages or product overview brochures
- Content not related to industrial automation at all

Respond ONLY with valid JSON. No markdown, no explanation outside the JSON."""


BATCH_PROMPT_TEMPLATE = """Classify each candidate source below. For each, return a JSON object with these fields:
- "relevant": boolean — does this belong in an industrial automation corpus?
- "doc_type": string — one of: manual, whitepaper, specification, technical_guide, application_note, official_docs, tutorial, research_paper, blog_post, github_repo, podcast, video, other
- "topic_tags": list of strings — 1-5 topic tags from the relevant topics list
- "authority_guess": string — one of: vendor_official, standards_body, community, academic, media, unknown
- "confidence": float 0.0-1.0 — how confident are you in the relevance decision?
- "reason": string — one sentence explaining your decision

Return a JSON array with one object per candidate, in the same order.

Candidates:
{candidates_text}

Respond with ONLY a JSON array. Example:
[{{"relevant": true, "doc_type": "manual", "topic_tags": ["PLC", "IEC 61131-3"], "authority_guess": "vendor_official", "confidence": 0.9, "reason": "Official Siemens TIA Portal programming manual"}}]"""


class CandidateClassifier:
    """
    Classifies discovery candidates using batched LLM calls.

    Usage:
        classifier = CandidateClassifier(llm_generate_fn=ollama.generate)
        classified = classifier.classify_batch(candidates)

    The llm_generate_fn should match the signature:
        def generate(prompt: str, system_prompt: str, temperature: float) -> str
    """

    def __init__(
        self,
        llm_generate_fn: Callable,
        cache: Optional[TaskCache] = None,
        batch_size: int = 25,
        temperature: float = 0.1,
        model_name: str = "ollama",
    ):
        """
        Args:
            llm_generate_fn: Your LLM's generate function.
                             Signature: generate(prompt, system_prompt, temperature) -> str
            cache: Optional TaskCache for caching classifications.
            batch_size: Max candidates per LLM call (keep under context window limits).
            temperature: LLM temperature for classification (low = deterministic).
            model_name: Model name for cache metadata.
        """
        self.generate = llm_generate_fn
        self.cache = cache
        self.batch_size = batch_size
        self.temperature = temperature
        self.model_name = model_name

    def classify_batch(self, candidates: List[CandidateSource]) -> List[ClassifiedCandidate]:
        """
        Classify a list of candidates, using cache where possible.

        Steps:
            1. Check cache for each candidate
            2. Batch uncached candidates into LLM calls
            3. Parse structured JSON responses
            4. Cache new results
            5. Return all classified candidates

        Returns:
            List of ClassifiedCandidate with classification attached.
        """
        results: List[ClassifiedCandidate] = []
        uncached: List[CandidateSource] = []

        # Step 1: Check cache
        for candidate in candidates:
            cache_key = self._cache_key(candidate)
            if self.cache and self.cache.has("classify", cache_key):
                cached_data = self.cache.get("classify", cache_key)
                classification = self._parse_single_classification(cached_data)
                results.append(ClassifiedCandidate(
                    candidate=candidate,
                    classification=classification,
                    cached=True,
                ))
                logger.debug(f"Cache hit: {candidate.title[:50]}")
            else:
                uncached.append(candidate)

        logger.info(f"Classification: {len(results)} cached, {len(uncached)} need LLM")

        # Step 2: Batch classify uncached
        for i in range(0, len(uncached), self.batch_size):
            batch = uncached[i : i + self.batch_size]
            batch_results = self._classify_llm_batch(batch)
            results.extend(batch_results)

        return results

    def _classify_llm_batch(self, batch: List[CandidateSource]) -> List[ClassifiedCandidate]:
        """Send a batch of candidates to the LLM for classification."""
        # Build the candidates text block
        candidates_text = ""
        for idx, c in enumerate(batch):
            candidates_text += (
                f"\n--- Candidate {idx + 1} ---\n"
                f"Title: {c.title}\n"
                f"URL: {c.url}\n"
                f"Source Type: {c.source_type}\n"
                f"Publisher: {c.publisher or 'unknown'}\n"
                f"Snippet: {(c.snippet or 'N/A')[:300]}\n"
            )

        prompt = BATCH_PROMPT_TEMPLATE.format(candidates_text=candidates_text)

        try:
            raw_response = self.generate(prompt, SYSTEM_PROMPT, self.temperature)
            classifications = self._parse_batch_response(raw_response, len(batch))
        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            # Fall back to empty classifications
            classifications = [Classification() for _ in batch]

        results = []
        for candidate, classification in zip(batch, classifications):
            # Cache the result
            if self.cache:
                cache_key = self._cache_key(candidate)
                self.cache.put(
                    "classify",
                    cache_key,
                    self._classification_to_dict(classification),
                    model=self.model_name,
                )

            results.append(ClassifiedCandidate(
                candidate=candidate,
                classification=classification,
                cached=False,
            ))

        return results

    def _parse_batch_response(self, raw: str, expected_count: int) -> List[Classification]:
        """
        Parse the LLM JSON array response into Classification objects.

        Handles common LLM output quirks:
        - Markdown code fences around JSON
        - Trailing commas
        - Missing fields
        """
        # Strip markdown fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (``` markers)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Raw response: {raw[:500]}")
            return [Classification() for _ in range(expected_count)]

        if not isinstance(data, list):
            data = [data]

        classifications = []
        for i in range(expected_count):
            if i < len(data):
                classifications.append(self._parse_single_classification(data[i]))
            else:
                logger.warning(f"LLM returned fewer classifications than expected ({len(data)}/{expected_count})")
                classifications.append(Classification())

        return classifications

    @staticmethod
    def _parse_single_classification(d: dict) -> Classification:
        """Parse a single classification dict into a Classification dataclass."""
        return Classification(
            relevant=bool(d.get("relevant", False)),
            doc_type=str(d.get("doc_type", "unknown")),
            topic_tags=list(d.get("topic_tags", [])),
            authority_guess=str(d.get("authority_guess", "unknown")),
            confidence=float(d.get("confidence", 0.0)),
            reason=str(d.get("reason", "")),
        )

    @staticmethod
    def _classification_to_dict(c: Classification) -> dict:
        return {
            "relevant": c.relevant,
            "doc_type": c.doc_type,
            "topic_tags": c.topic_tags,
            "authority_guess": c.authority_guess,
            "confidence": c.confidence,
            "reason": c.reason,
        }

    @staticmethod
    def _cache_key(candidate: CandidateSource) -> str:
        """Build a deterministic cache key from candidate URL + title."""
        return f"{candidate.url}|{candidate.title}"

    def classify_single(self, candidate: CandidateSource) -> ClassifiedCandidate:
        """Convenience method: classify a single candidate."""
        return self.classify_batch([candidate])[0]
