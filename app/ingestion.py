# app/ingestion.py

import os
from langchain.storage import InMemoryStore
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores.utils import filter_complex_metadata

# --- CORRECTED IMPORTS ---
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DB_BASE_PATH = "./db"

def process_and_get_retriever(file_path: str, document_id: str):
    db_path = os.path.join(DB_BASE_PATH, document_id)
    
    try:
        print(f"Loading document with Unstructured (incl. OCR): {file_path}")
        loader = UnstructuredLoader(file_path)
        docs = loader.load()

        filtered_docs = filter_complex_metadata(docs)
        
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        
        # Use the corrected class names
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectorstore = Chroma(
            collection_name=document_id,
            embedding_function=embeddings,
            persist_directory=db_path
        )
        
        store = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )
        
        print("Adding documents to ParentDocumentRetriever...")
        retriever.add_documents(filtered_docs, ids=None, add_to_docstore=True)
        
        full_text = "\n\n".join(doc.page_content for doc in filtered_docs)
        
        print("✅ Ingestion and retriever setup complete.")
        return retriever, full_text

    except Exception as e:
        print(f"❌ An error occurred during ingestion and retriever setup: {e}")
        return None, None
