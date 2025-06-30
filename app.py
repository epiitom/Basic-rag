from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
import os
from typing import List

app = FastAPI(title="RAG Chat API", description="A RAG-based chat system")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

class SearchResult(BaseModel):
    content: str
    metadata: dict

# Global variables for embedder and retriever
embedder = None
retriever = None
qa_chain = None

def initialize_rag_system():
    """Initialize the RAG system with embeddings and vector store"""
    global embedder, retriever, qa_chain
    
    try:
        # Set up Google API key
        os.environ["GOOGLE_API_KEY"] = "AIzaSyA9151bCImNiIOi2sY0pSejOaejw-duhqw"
        
        # Initialize embedder
        embedder = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        
        # Initialize retriever from existing Qdrant collection
        retriever = QdrantVectorStore.from_existing_collection(
            url="http://localhost:6333",
            collection_name="learning_langchain",
            embedding=embedder
        )
        
        # Initialize LLM with correct model name
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Updated model name
            temperature=0.3
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        print("RAG system initialized successfully")
        
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        raise

def load_and_process_pdfs():
    """Load and process PDF files into the vector store"""
    PDF_FILES = [
        "attention-is-all-you-need.pdf",
        "denoising-diffusion-probabilistic-models.pdf",
        "rag.pdf",
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    all_chunks = []
    
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
    return all_chunks

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    initialize_rag_system()

@app.get("/")
async def read_root():
    """Serve the chat interface"""
    return {"message": "RAG Chat API is running. Visit /chat for the interface."}

@app.get("/chat", response_class=HTMLResponse)
async def chat_interface():
    """Serve the HTML chat interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Chat Interface</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .chat-container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.1);
                width: 100%;
                max-width: 800px;
                height: 600px;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            
            .chat-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                text-align: center;
                font-size: 24px;
                font-weight: bold;
            }
            
            .chat-messages {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                background: #f8f9fa;
            }
            
            .message {
                margin-bottom: 15px;
                padding: 12px 16px;
                border-radius: 18px;
                max-width: 80%;
                word-wrap: break-word;
            }
            
            .user-message {
                background: #007bff;
                color: white;
                margin-left: auto;
                text-align: right;
            }
            
            .bot-message {
                background: white;
                color: #333;
                border: 1px solid #e0e0e0;
                margin-right: auto;
            }
            
            .sources {
                font-size: 12px;
                color: #666;
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid #eee;
            }
            
            .chat-input-container {
                padding: 20px;
                background: white;
                border-top: 1px solid #e0e0e0;
                display: flex;
                gap: 10px;
            }
            
            .chat-input {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #e0e0e0;
                border-radius: 25px;
                font-size: 16px;
                outline: none;
                transition: border-color 0.3s;
            }
            
            .chat-input:focus {
                border-color: #667eea;
            }
            
            .send-button {
                padding: 12px 24px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                transition: transform 0.2s;
            }
            
            .send-button:hover {
                transform: translateY(-2px);
            }
            
            .send-button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .typing-indicator {
                display: none;
                padding: 12px 16px;
                background: white;
                border: 1px solid #e0e0e0;
                border-radius: 18px;
                max-width: 80px;
                margin-right: auto;
                margin-bottom: 15px;
            }
            
            .typing-indicator span {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background-color: #999;
                margin: 0 2px;
                animation: typing 1.4s infinite ease-in-out;
            }
            
            .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
            .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
            
            @keyframes typing {
                0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
                40% { transform: scale(1); opacity: 1; }
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                RAG Chat Assistant
            </div>
            <div class="chat-messages" id="chatMessages">
                <div class="message bot-message">
                    Hello! I'm your RAG assistant. I can help you find information from the uploaded documents. Ask me anything!
                </div>
            </div>
            <div class="typing-indicator" id="typingIndicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
            <div class="chat-input-container">
                <input type="text" class="chat-input" id="chatInput" placeholder="Type your question here..." onkeypress="handleKeyPress(event)">
                <button class="send-button" id="sendButton" onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            const chatMessages = document.getElementById('chatMessages');
            const chatInput = document.getElementById('chatInput');
            const sendButton = document.getElementById('sendButton');
            const typingIndicator = document.getElementById('typingIndicator');

            function addMessage(content, isUser = false, sources = []) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
                
                let messageContent = content;
                if (!isUser && sources.length > 0) {
                    messageContent += `<div class="sources"><strong>Sources:</strong> ${sources.join(', ')}</div>`;
                }
                
                messageDiv.innerHTML = messageContent;
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function showTyping() {
                typingIndicator.style.display = 'block';
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function hideTyping() {
                typingIndicator.style.display = 'none';
            }

            async function sendMessage() {
                const query = chatInput.value.trim();
                if (!query) return;

                // Add user message
                addMessage(query, true);
                chatInput.value = '';
                
                // Disable input and button
                chatInput.disabled = true;
                sendButton.disabled = true;
                showTyping();

                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ query: query })
                    });

                    const data = await response.json();
                    
                    hideTyping();
                    
                    if (response.ok) {
                        addMessage(data.answer, false, data.sources);
                    } else {
                        addMessage('Sorry, I encountered an error processing your request.', false);
                    }
                } catch (error) {
                    hideTyping();
                    addMessage('Sorry, I encountered an error connecting to the server.', false);
                }

                // Re-enable input and button
                chatInput.disabled = false;
                sendButton.disabled = false;
                chatInput.focus();
            }

            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }

            // Focus on input when page loads
            window.onload = function() {
                chatInput.focus();
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests and return responses with sources"""
    if not qa_chain:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        # Get response from QA chain using invoke method (not deprecated)
        result = qa_chain.invoke({"query": request.query})
        
        # Extract source information
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    source_file = Path(doc.metadata['source']).name
                    page = doc.metadata.get('page', 'Unknown')
                    sources.append(f"{source_file} (page {page})")
        
        return ChatResponse(
            answer=result["result"],
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/api/search", response_model=List[SearchResult])
async def search(query: str, k: int = 3):
    """Search for similar documents"""
    if not retriever:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        search_results = retriever.similarity_search(query=query, k=k)
        
        results = []
        for doc in search_results:
            results.append(SearchResult(
                content=doc.page_content,
                metadata=doc.metadata
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "embedder_initialized": embedder is not None,
        "retriever_initialized": retriever is not None,
        "qa_chain_initialized": qa_chain is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)