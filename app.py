from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

PDF_FILES = [
    "attention-is-all-you-need.pdf",
    "denoising-diffusion-probabilistic-models.pdf", 
    "rag.pdf",
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
)

all_chunks = []

# Debug: Check current directory and file paths
print("Current script location:", Path(__file__).parent)
print("Looking for PDFs in:", Path(__file__).parent / "PDFs")

for file_name in PDF_FILES:
    pdf_path = Path(__file__).parent / "PDFs" / file_name
    print(f"Checking file: {pdf_path}")
    print(f"File exists: {pdf_path.exists()}")
    
    if pdf_path.exists():
        loader = PyPDFLoader(file_path=str(pdf_path))
        docs = loader.load()
        split_docs = text_splitter.split_documents(docs)
        all_chunks.extend(split_docs)
        print(f"Successfully loaded {file_name}")
    else:
        print(f"File not found: {pdf_path}")

print(f"Total chunks loaded: {len(all_chunks)}")

# Only create embedder if we have chunks
if all_chunks:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyA9151bCImNiIOi2sY0pSejOaejw-duhqw"
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    print("Embedder created successfully")
