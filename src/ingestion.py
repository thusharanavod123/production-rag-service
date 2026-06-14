import os
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import settings

class DataIngestionPipeline:
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        # Slices text into smart, overlapping chunks so no context is cut in half
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_documents(self) -> List[Document]:
        """Scans the data directory and loads PDFs and TXT files."""
        documents = []
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"📁 Created missing data directory at: {self.data_dir}")
            return documents

        # 1. Load PDFs if any exist
        try:
            pdf_loader = PyPDFDirectoryLoader(self.data_dir)
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            if pdf_docs:
                print(f"📄 Loaded {len(pdf_docs)} pages from PDFs.")
        except Exception as e:
            print(f"⚠️ Error loading PDFs: {str(e)}")

        # 2. Load plain text files
        for file in os.listdir(self.data_dir):
            if file.endswith(".txt"):
                file_path = os.path.join(self.data_dir, file)
                try:
                    txt_loader = TextLoader(file_path, encoding="utf-8")
                    documents.extend(txt_loader.load())
                    print(f"📝 Loaded text file: {file}")
                except Exception as e:
                    print(f"⚠️ Error loading text file {file}: {str(e)}")

        return documents

    def run(self) -> List[Document]:
        """Executes the extraction and recursive chunking logic."""
        print("\n⚙️ Executing Data Ingestion Pipeline...")
        raw_docs = self.load_documents()
        
        if not raw_docs:
            print("🛑 No source documents found to process. Drop a file in /data!")
            return []

        chunks = self.text_splitter.split_documents(raw_docs)
        print(f"✂️ Sliced {len(raw_docs)} source documents into {len(chunks)} optimized chunks.")
        return chunks

if __name__ == "__main__":
    # Local verification run
    pipeline = DataIngestionPipeline()
    processed_chunks = pipeline.run()
    
    # Print out what the machine chunked to inspect its work
    for i, chunk in enumerate(processed_chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk.page_content)