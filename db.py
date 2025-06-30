from langchain_qdrant import QdrantVectorStore
from app import all_chunks, embedder  
# This part should be run only once
# It stores all the chunks and their embeddings into Qdrant
 #You can comment this after the first run

retriever = QdrantVectorStore.from_existing_collection(
      url="http://localhost:6333",
      collection_name="learning_langchain",
      embedding=embedder
  )
query = "What is Rag?"
search_result = retriever.similarity_search(query=query)
for i, chunk in enumerate(search_result, 1):
    print(f"\n--- Chunk {i} ---\n{chunk.page_content}")