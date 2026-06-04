# Local Enterprise Knowledge Engine: Cost-Optimized & Secure RAG Microservice

🚀 **An enterprise-grade, fully containerized Retrieval-Augmented Generation (RAG) microservice designed for zero-cloud-cost execution and 100% data privacy.**

Traditional enterprise RAG implementations rely on expensive public cloud APIs, introducing severe data privacy risks and unpredictable per-token billing overhead. This project delivers a high-performance, production-ready alternative engineered entirely on local infrastructure. 

---

## 🏗️ System Architecture

The microservice is decoupled into two primary operational pipelines—**Data Ingestion** and **Retrieval & Generation**—orchestrated through an asynchronous API layer.


![alt text](Gemini_Generated_Image_f11ptyf11ptyf11p.png)

🛠️ Tech Stack & Infrastructure
Framework: LangChain (Orchestration Engine)

Web Server: FastAPI (Asynchronous REST API)

Vector Database: ChromaDB (Isolated Local Vector Storage)

Local Inference: Ollama (Llama 3 / Phi-3)

Embeddings Model: sentence-transformers/all-MiniLM-L6-v2

Containerization: Docker & Docker Compose
