import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_cohere import CohereRerank
from pinecone import Pinecone

load_dotenv()

def get_retriever(namespace: str = "default"):
    # 1. Initialize Embeddings (Must match ingest.py)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # 2. Connect to Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("rag-app-index")
    
    vectorstore = PineconeVectorStore(
        index=index, 
        embedding=embeddings, 
        text_key="text", # Tells LangChain where the chunk text is in metadata
        namespace=namespace # CRUCIAL: Search only in this namespace
    )

    # 3. Setup Base Retriever with MMR
    # Requirement 3a: Top-k retrieval (MMR)
    # fetch_k=20 is the pool size, k=10 is what's passed to the reranker
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 10, 'fetch_k': 20, 'lambda_mult': 0.5}
    )

    # 4. Setup Reranker
    # Requirement 3b: Apply a reranker (Cohere)
    compressor = CohereRerank(
        cohere_api_key=os.getenv("COHERE_API_KEY"), 
        model="rerank-english-v3.0", 
        top_n=5 # Final chunks sent to LLM
    )

    # 5. Create the Compression Retriever with dynamic import
    try:
        from langchain.retrievers.document_compressors import DocumentCompressorPipeline  # type: ignore
        from langchain.retrievers import ContextualCompressionRetriever  # type: ignore
        
        compression_retriever_inner = DocumentCompressorPipeline(
            transformers=[compressor]
        )
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compression_retriever_inner, 
            base_retriever=base_retriever
        )
        return compression_retriever
    except ImportError:
        # Fallback: return base retriever with top 5 results
        print("Warning: Using base retriever without reranking (compression module not found)")
        return base_retriever

if __name__ == "__main__":
    # Test the retriever
    query = "What are the main requirements for this app?"
    retriever = get_retriever()
    results = retriever.invoke(query)
    
    print(f"\nRetrieved {len(results)} results:\n")
    for i, doc in enumerate(results[:5]):  # Show top 5
        print(f"--- Chunk {i+1} (Source: {doc.metadata.get('source', 'N/A')}) ---")
        print(doc.page_content[:200] + "...\n")
