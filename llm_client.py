import os
import requests
import json
from typing import List, Dict, Any, Optional

class OllamaClient:
    def __init__(self, model="llama3:latest", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def generate(self, prompt, system_prompt, temperature=0.7):
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "temperature": temperature,
                "stream": False
            }
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Handle different response formats
        if "response" in result:
            return result["response"]
        elif "error" in result:
            raise RuntimeError(f"Ollama error: {result['error']}")
        else:
            # Unexpected format - show what we got
            import json
            raise KeyError(f"Unexpected Ollama response format. Got: {json.dumps(result, indent=2)}")

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
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=self._types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=8192,
            )
        )
        return response.text

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