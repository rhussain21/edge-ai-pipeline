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
        
        return response.json()["response"]

class RAGEngine:
    def __init__(self, vector_db, relational_db, llm_client):
        self.vdb = vector_db
        self.db = relational_db
        self.llm_client = llm_client
    
    def query(self, question, top_k=5, temperature=0.7):
        # 1. Retrieve relevant segments
        context = self.retrieve(query=question, top_k=top_k)
        
        # 2. Generate prompt
        prompt = self.generate_prompt(question=question, context=context)
        
        # 3. Generate response
        response = self.generate_response(prompt, temperature=temperature)
        
        return response
        
    def retrieve(self, query, top_k=5, filters=None):
        # Semantic search in vector DB
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
        # Create structured prompt
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
        # Call LLM API
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
    vdb_path = os.path.join(rel_path, 'Vectors/industry_signals_vectors')

    print("Initializing Relational Database...")
    db = relationalDB(db_path)
    db.init_db()
    
    print("Intializing Vector Database...")
    vdb = VectorDB("Vectors/")
    vdb.load("industry_signals_vectors")

    llm_client = OllamaClient(model="llama3:latest")
    rag_engine = RAGEngine(vdb, db, llm_client)
    
    query = "What is the current focus in manufacturing? Specifically around AI?"

    results = rag_engine.query(query)
    print(f"Results at temp: 0.7: {results}")

    results = rag_engine.query(query, temperature=0.2)
    print(f"Results at temp: 0.2: {results}")