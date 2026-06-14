import os
from typing import Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import settings
from src.ingestion import DataIngestionPipeline

class VectorStoreManager:
    def __init__(self):
        self.index_dir = settings.FAISS_INDEX_DIR
        print("📥 Loading local Embedding Model (all-MiniLM-L6-v2)...")
        # Downloads model once to your local machine cache and executes on CPU
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

    def build_and_save_index(self) -> Optional[FAISS]:
        """Runs document ingestion, embeds chunks, and saves FAISS index to disk."""
        # 1. Fetch text chunks from Rung 2 pipeline
        ingestion_pipeline = DataIngestionPipeline()
        chunks = ingestion_pipeline.run()

        if not chunks:
            print("🛑 Ingestion pipeline returned zero chunks. Aborting index building.")
            return None

        print(f"🧬 Converting {len(chunks)} chunks into embeddings and indexing via FAISS...")
        
        # 2. Convert text chunks to vectors and build the index database
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # 3. Save database index locally to avoid re-embedding files every time
        vector_store.save_local(self.index_dir)
        print(f"💾 Vector Database index successfully stored at: {self.index_dir}")
        return vector_store

    def load_index(self) -> Optional[FAISS]:
        """Loads an existing local FAISS database index."""
        if not os.path.exists(self.index_dir):
            print(f"⚠️ No index found at {self.index_dir}. You need to build it first.")
            return None
        
        print(f"📖 Loading existing FAISS index from: {self.index_dir}")
        # allow_dangerous_deserialization is required for loading local pickle-based FAISS files safely
        return FAISS.load_local(self.index_dir, self.embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    # Local verification run
    manager = VectorStoreManager()
    
    print("\n--- Testing Vector Database Creation ---")
    db = manager.build_and_save_index()
    
    if db:
        print("\n--- Testing Similarity Search Capability ---")
        query = "How do engineers connect from home?"
        print(f"🔍 Question: '{query}'")
        
        # Search for the top matching document chunk based on pure semantic meaning
        results = db.similarity_search(query, k=1)
        for doc in results:
            print(f"\n🎯 Nearest Vector Match found:\n{doc.page_content}")