from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import os

def setup_qdrant_collection():
    """Set up Qdrant collection with PDF documents"""
    
    # Configuration
    PDF_FILES = [
        "attention-is-all-you-need.pdf",
        "denoising-diffusion-probabilistic-models.pdf",
        "rag.pdf",
    ]
    
    # Set up Google API key
    os.environ["GOOGLE_API_KEY"] = "AIzaSyA9151bCImNiIOi2sY0pSejOaejw-duhqw"
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    all_chunks = []
    
    # Debug: Check current directory and file paths
    print("Current script location:", Path(__file__).parent)
    print("Looking for PDFs in:", Path(__file__).parent / "PDFs")
    
    # Load and process PDFs
    for file_name in PDF_FILES:
        pdf_path = Path(__file__).parent / "PDFs" / file_name
        print(f"Checking file: {pdf_path}")
        print(f"File exists: {pdf_path.exists()}")
        
        if pdf_path.exists():
            try:
                loader = PyPDFLoader(file_path=str(pdf_path))
                docs = loader.load()
                split_docs = text_splitter.split_documents(docs)
                all_chunks.extend(split_docs)
                print(f"Successfully loaded {file_name} - {len(split_docs)} chunks")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
        else:
            print(f"File not found: {pdf_path}")
    
    print(f"Total chunks loaded: {len(all_chunks)}")
    
    if not all_chunks:
        print("No chunks loaded. Please check your PDF files and paths.")
        return
    
    # Initialize embedder
    print("Initializing embedder...")
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    
    # Create Qdrant vector store
    print("Creating Qdrant vector store...")
    try:
        vector_store = QdrantVectorStore.from_documents(
            documents=all_chunks,
            embedding=embedder,
            url="http://localhost:6333",
            collection_name="learning_langchain",
            force_recreate=True  # This will recreate the collection if it exists
        )
        print("Successfully created Qdrant collection!")
        
        # Test the retrieval
        print("\nTesting retrieval...")
        query = "What is RAG?"
        search_results = vector_store.similarity_search(query=query, k=3)
        
        print(f"Found {len(search_results)} similar documents for query: '{query}'")
        for i, chunk in enumerate(search_results, 1):
            print(f"\n--- Chunk {i} ---")
            print(f"Content preview: {chunk.page_content[:200]}...")
            print(f"Metadata: {chunk.metadata}")
            
    except Exception as e:
        print(f"Error creating Qdrant collection: {e}")
        print("Make sure Qdrant is running on localhost:6333")

if __name__ == "__main__":
    setup_qdrant_collection()