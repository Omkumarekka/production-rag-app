import os
import uuid
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()

# 1. Setup Embeddings
# We use Google's latest text-embedding-004 (768 dimensions)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 2. Setup Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-app-index")

def ingest_text(text: str, source_name: str, title: str, namespace: str = "default"):
    """
    Chunks text and upserts to Pinecone with mandatory metadata for citations.
    namespace: Allows organizing multiple documents in the same index.
    """
    # Requirement 2b: 800-1,200 tokens with 10-15% overlap
    # We use a character proxy (1 token ≈ 4 chars) for the splitter
    # 1000 tokens ≈ 4000 chars; 15% overlap ≈ 600 chars
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, 
        chunk_overlap=600,
        separators=["\n\n", "\n", ".", " ", ""],
        add_start_index=True # Helps with the 'position' metadata requirement
    )

    # Requirement 2c: Store metadata (source, title, section, position)
    chunks = text_splitter.split_text(text)
    
    vectors = []
    for i, chunk in enumerate(chunks):
        # Generate embedding
        vector_values = embeddings.embed_query(chunk)
        
        # Unique ID for each chunk
        chunk_id = str(uuid.uuid4())
        
        # Metadata construction for citations
        metadata = {
            "text": chunk,
            "source": source_name,
            "title": title,
            "section": f"Chunk {i+1}",
            "position": i
        }
        
        vectors.append({
            "id": chunk_id,
            "values": vector_values,
            "metadata": metadata
        })

    # Requirement 1b: Upsert Strategy with Namespace
    # Batching upserts is a production-minded approach to avoid API timeouts
    index.upsert(vectors=vectors, namespace=namespace)
    print(f"Successfully upserted {len(vectors)} chunks to namespace: {namespace}")

if __name__ == "__main__":
    # Test Data - Actual document content
    sample_text = """
    Retrieval-Augmented Generation (RAG) is a powerful architecture that combines language models with vector databases to provide accurate, grounded responses. 
    
    The RAG system works by first embedding user queries into a vector space, then retrieving the most relevant documents from a vector index based on semantic similarity. 
    These retrieved documents are passed as context to a language model, which generates responses based on the provided context rather than relying solely on its training data.
    
    This approach has several key advantages: First, it reduces hallucinations by grounding responses in actual documents. Second, it enables knowledge updates without retraining the model - new documents can be added to the vector database instantly. 
    Third, it provides source attribution and citations, allowing users to verify where information comes from.
    
    The vector database stores embeddings of document chunks. Each chunk typically contains 800 to 1200 tokens to balance context size with retrieval precision. 
    Embeddings capture semantic meaning, allowing the system to find relevant documents even when they don't share exact keywords with the query.
    
    Pinecone is a managed vector database service that makes it easy to store, index, and search embeddings at scale. 
    It provides fast similarity search, filtering by metadata, and automatic scaling. The service handles indexing and optimization transparently.
    
    Google's text-embedding models convert text into high-dimensional vectors that capture semantic information. 
    These embeddings enable semantic search and similarity matching. The models are trained on diverse data and can handle various domains and languages.
    
    Chunking strategies are critical for RAG performance. Recursive text splitting breaks documents into coherent chunks at sentence, paragraph, and word boundaries. 
    Overlapping chunks help preserve context between boundaries. Metadata attached to chunks enables filtering and source attribution.
    
    The metadata stored with each chunk includes the document source, title, section information, and position within the document. 
    This allows the system to provide accurate citations and helps users understand the context of retrieved information.
    
    LangChain provides convenient abstractions for building RAG systems. It offers text splitters, embedding integrations, and vector store connectors. 
    These tools simplify the implementation of RAG pipelines significantly.
    
    Production RAG systems need careful attention to retrieval quality, chunk sizing, and metadata management. 
    Monitoring and evaluation are important for ensuring the system provides accurate, helpful responses. Continuous updates to the vector database keep information current.
    """ * 5  # Repeat to ensure multiple chunks
    
    ingest_text(sample_text, source_name="rag_guide_v1.pdf", title="RAG Systems Guide")
