
import os
import requests
import json
from src.config import settings
from src.vector_store import VectorStoreManager

class ResponseGenerator:
    def __init__(self):
        self.store_manager = VectorStoreManager()
        # Load the local FAISS index built in Rung 3
        self.vector_store = self.store_manager.load_index()
        self.ollama_url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        self.model_name = settings.LLM_MODEL

    def generate_answer(self, question: str) -> dict:
        """Retrieves matching context and uses local Llama 3 to generate an accurate answer."""
        if not self.vector_store:
            # Fallback check to try and reload the index if it was missing
            self.vector_store = self.store_manager.load_index()
            if not self.vector_store:
                return {"error": "Vector database index not initialized. Run vector_store module first."}

        # 1. Retrieve the top matching text chunks from FAISS
        print(f"\n🔍 Retrieving relevant context for query: '{question}'...")
        matched_docs = self.vector_store.similarity_search(question, k=2)
        
        # Combine the content of matched chunks into a single text block
        context_text = "\n---\n".join([doc.page_content for doc in matched_docs])
        
        # Track the sources to verify the AI's integrity later
        sources = [os.path.basename(doc.metadata.get('source', 'unknown')) for doc in matched_docs]

        # 2. Construct the strict Enterprise Guardrail Prompt
        prompt_template = f"""You are a secure corporate AI assistant. Answer the user's question strictly using the provided trusted context. 
If the context does not contain the answer, state clearly that you do not possess that information. Do not use outside knowledge or hallucinate facts.

[TRUSTED CONTEXT]:
{context_text}

[USER QUESTION]: 
{question}

[EXPERT ANSWER]:"""

        # 3. Ship the payload over to the local Ollama system service
        print(f"🤖 Sending augmented payload to local LLM runtime ({self.model_name})...")
        payload = {
            "model": self.model_name,
            "prompt": prompt_template,
            "stream": False  # Set to False to get the full answer back in one clean string
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            response.raise_for_status()
            response_json = response.json()
            
            return {
                "answer": response_json.get("response", "").strip(),
                "sources": list(set(sources))
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Failed to communicate with local Ollama engine. Ensure 'ollama run {self.model_name}' is active. Details: {str(e)}"
            }

if __name__ == "__main__":
    # Local verification run
    generator = ResponseGenerator()
    
    # Let's ask a question that requires tracing consequences from your policy file
    test_query = "What happens if I forget to use the corporate VPN when working from home?"
    print(f"❓ User Question: {test_query}")
    
    result = generator.generate_answer(test_query)
    
    print("\n✨ AI Engine Response Output:")
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(result["answer"])
        print(f"\n📊 Verified Source Documents: {result['sources']}")