import os
import requests
import json
import time
import logging
import traceback
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_BACKOFF = 2  # seconds


def _classify_error(e: Exception) -> str:
    """Classify an exception into a category for structured logging."""
    msg = str(e).lower()
    etype = type(e).__name__

    if isinstance(e, requests.exceptions.Timeout):
        return 'timeout'
    if isinstance(e, requests.exceptions.ConnectionError):
        return 'connection_error'
    if isinstance(e, requests.exceptions.HTTPError):
        code = getattr(e.response, 'status_code', 0) if hasattr(e, 'response') else 0
        if code == 429:
            return 'rate_limit'
        if code >= 500:
            return 'server_error'
        return f'http_{code}'
    if 'rate' in msg and 'limit' in msg:
        return 'rate_limit'
    if 'quota' in msg:
        return 'quota_exceeded'
    if 'timeout' in msg:
        return 'timeout'
    if 'resource' in msg and 'exhausted' in msg:
        return 'quota_exceeded'
    if 'json' in msg or 'parse' in msg or 'decode' in msg:
        return 'parse_error'
    if 'safety' in msg or 'blocked' in msg:
        return 'safety_filter'
    return etype


class OllamaClient:
    def __init__(self, model="llama3:latest", base_url="http://localhost:11434"):
        self.model = model
        self.model_name = model
        self.base_url = base_url
    
    def generate(self, prompt, system_prompt, temperature=0.7):
        full_prompt = f"{system_prompt}\n\n{prompt}"
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "temperature": temperature,
                        "stream": False
                    },
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()

                if "response" in result:
                    return result["response"]
                elif "error" in result:
                    raise RuntimeError(f"Ollama error: {result['error']}")
                else:
                    raise KeyError(f"Unexpected Ollama response format. Got: {json.dumps(result, indent=2)}")

            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as e:
                last_error = e
                wait = BASE_BACKOFF ** attempt
                logger.warning(f"Ollama attempt {attempt}/{MAX_RETRIES} failed ({_classify_error(e)}), retrying in {wait}s")
                time.sleep(wait)
            except requests.exceptions.HTTPError as e:
                error_cat = _classify_error(e)
                if error_cat == 'rate_limit' and attempt < MAX_RETRIES:
                    wait = BASE_BACKOFF ** attempt * 2
                    logger.warning(f"Ollama rate limited, waiting {wait}s")
                    time.sleep(wait)
                    last_error = e
                else:
                    raise
            except Exception:
                raise

        raise last_error or RuntimeError(f"Ollama failed after {MAX_RETRIES} attempts")


class GeminiClient:
    def __init__(self, model="gemini-2.5-flash"):
        from google import genai
        from google.genai import types
        self.model_name = model
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self._client = genai.Client(api_key=api_key)
        self._types = types

    def generate(self, prompt, system_prompt, temperature=0.7):
        full_prompt = f"{system_prompt}\n\n{prompt}"
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=self._types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=8192,
                    )
                )
                if response.text is None:
                    # Safety filter or empty response
                    candidates = getattr(response, 'candidates', [])
                    reason = 'unknown'
                    if candidates:
                        reason = getattr(candidates[0], 'finish_reason', 'unknown')
                    raise RuntimeError(f"Gemini returned empty response (finish_reason={reason})")
                return response.text

            except Exception as e:
                error_cat = _classify_error(e)
                last_error = e

                if error_cat in ('rate_limit', 'quota_exceeded', 'timeout', 'server_error', 'connection_error'):
                    wait = BASE_BACKOFF ** attempt * (3 if error_cat == 'rate_limit' else 1)
                    logger.warning(f"Gemini attempt {attempt}/{MAX_RETRIES} failed ({error_cat}), retrying in {wait}s")
                    time.sleep(wait)
                elif error_cat == 'safety_filter':
                    # Don't retry safety blocks — they'll keep failing
                    raise
                elif attempt < MAX_RETRIES:
                    wait = BASE_BACKOFF ** attempt
                    logger.warning(f"Gemini attempt {attempt}/{MAX_RETRIES} failed ({error_cat}: {e}), retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise

        raise last_error or RuntimeError(f"Gemini failed after {MAX_RETRIES} attempts")

class LLMClient:
    def __init__(self, vector_db, relational_db, llm_client):
        self.vdb = vector_db
        self.db = relational_db
        self.llm_client = llm_client
    
    def query(self, question, top_k=5, temperature=0.7):
        context = self.retrieve(query=question, top_k=top_k)
        prompt = self.generate_prompt(question=question, context=context)
        response = self.generate_response(prompt, temperature=temperature)
        
        return response
        
    def retrieve(self, query, top_k=5, filters=None):
        results = self.vdb.search(query=query, top_k=top_k)

        if not results:
            return "No relevant context found."
        

        context_chunks = []
        for r in results:
            if isinstance(r, (list, tuple)):
                text = r[0]
            elif isinstance(r, dict):
                text = r.get("document") or r.get("text") or str(r)
            else:
                text = str(r)
            context_chunks.append(text.strip())

        return "\n\n".join(context_chunks)
 
    def generate_prompt(self, question, context):
        system_prompt = """You are an expert industry analyst specializing in smart manufacturing, 
        industrial automation, and technology trends. Provide accurate, well-cited answers 
        based on the provided context documents."""
        
        user_prompt = f"""Context from industry documents:
        {context}

        Question: {question}

        Instructions:
        - Answer based ONLY on the provided context
        - Be specific and technical when appropriate
        - If information is not in context, say so clearly
        - Provide concise, factual responses
        """
                
        return user_prompt
            
    def generate_response(self, prompt, temperature=0.7):
        system_prompt = """You are an expert industry analyst specializing in smart manufacturing, 
        industrial automation, and technology trends. Provide accurate, well-cited answers 
        based on the provided context documents."""
        
        return self.llm_client.generate(prompt, system_prompt, temperature)

if __name__ == "__main__":
    from db_vector import VectorDB
    from db_relational import relationalDB

    rel_path = os.path.expanduser('~/Documents/ai-projects/ai_industry_signals/')
    content_dir = os.path.join(rel_path, 'content_files')

    db_path = os.path.join(rel_path, 'Database/industry_signals.db')
    vdb_path = os.path.join(rel_path, 'Vectors/corpus_vectors')

    print("Initializing Relational Database...")
    db = relationalDB(db_path)
    db.init_db()
    
    print("Intializing Vector Database...")
    vdb = VectorDB("Vectors/corpus_vectors/")
    vdb.load("corpus_vectors")

    llm_client = OllamaClient(model="llama3:latest")
    llm_client = LLMClient(vdb, db, llm_client)
    
    query = "What is the current focus in manufacturing? Specifically around AI?"

    results = llm_client.query(query)
    print(f"Results at temp: 0.7: {results}")

    results = llm_client.query(query, temperature=0.2)
    print(f"Results at temp: 0.2: {results}")