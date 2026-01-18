import streamlit as st
import time
from ingest import ingest_text
from generator import generate_answer
from pypdf import PdfReader
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize session state for multi-document support
if "available_docs" not in st.session_state:
    st.session_state.available_docs = {}  # Maps filename -> namespace
if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None

# App Configuration
st.set_page_config(page_title="RAG Thought Partner", layout="wide")
st.title("🚀 Production-Minded RAG App")

# Sidebar: Data Ingestion & Document Management
with st.sidebar:
    st.header("1. Document Management")
    
    # Document Selector - choose which doc to chat with
    st.markdown("**📄 Select a Document to Chat With**")
    if st.session_state.available_docs:
        doc_options = list(st.session_state.available_docs.keys())
        selected_doc = st.selectbox(
            "Choose a document:",
            options=doc_options,
            key="doc_selector",
            help="Select which document to query about"
        )
        st.session_state.selected_doc = selected_doc
        active_namespace = st.session_state.available_docs[selected_doc]
        st.success(f"✅ Chatting with: **{selected_doc}**")
    else:
        st.info("📂 No documents indexed yet. Upload a document below.")
        st.session_state.selected_doc = None
    
    st.divider()
    
    # Document-specific Clear Button
    if st.session_state.selected_doc and st.button("🗑️ Delete Selected Document"):
        try:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index("rag-app-index")
            namespace = st.session_state.available_docs[st.session_state.selected_doc]
            index.delete(delete_all=True, namespace=namespace)
            # Remove from tracking
            del st.session_state.available_docs[st.session_state.selected_doc]
            st.session_state.selected_doc = None
            st.success(f"✅ Document deleted successfully!")
            st.rerun()
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                st.info("ℹ️ Document is already empty.")
                del st.session_state.available_docs[st.session_state.selected_doc]
                st.session_state.selected_doc = None
                st.rerun()
            else:
                st.error(f"Error deleting document: {str(e)}")
    
    # Clear All Documents Button
    if st.button("🗑️ Clear All Documents", help="Delete all documents from the index"):
        try:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index("rag-app-index")
            index.delete(delete_all=True)
            st.session_state.available_docs.clear()
            st.session_state.selected_doc = None
            st.success("✅ All documents cleared successfully!")
            st.rerun()
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                st.info("ℹ️ Index is already empty.")
            else:
                st.error(f"Error clearing index: {str(e)}")
    
    st.divider()
    
    # AUTOMATIC FILE INDEXING with Namespace Support
    st.markdown("**⬆️ Upload & Auto-Index**")
    st.markdown("Files are automatically indexed when uploaded.")
    uploaded_file = st.file_uploader("Upload a file", type=['txt', 'pdf'])
    
    # AUTOMATIC TRIGGER: If a file is uploaded and NOT yet in our 'available_docs'
    if uploaded_file is not None and uploaded_file.name not in st.session_state.available_docs:
        with st.spinner(f"Automatically indexing {uploaded_file.name}..."):
            try:
                # Extraction logic
                if uploaded_file.type == "application/pdf":
                    reader = PdfReader(uploaded_file)
                    text = "".join([page.extract_text() for page in reader.pages])
                    print(f"DEBUG: Extracted PDF text: {text[:500]}")
                else:
                    text = uploaded_file.read().decode("utf-8")
                    print(f"DEBUG: Extracted TXT text: {text[:500]}")
                
                # Generate namespace from filename (sanitized)
                namespace = uploaded_file.name.replace(" ", "_").replace(".", "_").lower()
                
                # Run the ingestion script WITH namespace
                ingest_text(text, source_name=uploaded_file.name, title=uploaded_file.name, namespace=namespace)
                
                # Track this document with its namespace
                st.session_state.available_docs[uploaded_file.name] = namespace
                st.session_state.selected_doc = uploaded_file.name
                st.success(f"✅ {uploaded_file.name} is ready!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to index: {str(e)}")
    elif uploaded_file is not None:
        st.info(f"✅ {uploaded_file.name} is already indexed.")

# Main Area: Querying
st.header("2. Ask a Question")

# Show which document is being queried
if st.session_state.selected_doc:
    st.info(f"📄 Currently chatting with: **{st.session_state.selected_doc}**")
    query = st.text_input("What would you like to know?")
    
    if query:
        start_time = time.time()
        active_namespace = st.session_state.available_docs[st.session_state.selected_doc]
        
        with st.spinner("Retrieving, Reranking, and Generating..."):
            answer, sources = generate_answer(query, namespace=active_namespace)
        
        end_time = time.time()
        duration = end_time - start_time

        # Display Answer
        st.markdown("### Answer")
        st.write(answer)

        # Requirement 5b: Timing and Cost Estimates
        # Rough estimate: 1 token ≈ 4 characters
        input_tokens = len(query) / 4 + (sum(len(s['content']) for s in sources) / 4)
        output_tokens = len(answer) / 4
        
        # 2026 pricing estimate (Groq Llama 3.3): Free tier available, ~$0.03-0.10 per 1M input/output
        est_cost = ((input_tokens / 1_000_000) * 0.05) + ((output_tokens / 1_000_000) * 0.15)

        st.info(f"⏱️ **Time:** {duration:.2f}s  |  💰 **Estimated Cost:** ${est_cost:.6f}")

        # Requirement 4b: Show source snippets
        with st.expander("View Source Snippets & Citations"):
            for s in sources:
                st.markdown(f"**[{s['id']}] {s['title']}** (Source: {s['source']})")
                st.caption(s['content'])
                st.divider()
else:
    st.warning("⚠️ Please upload and select a document to start chatting.")

# Footer for Requirements
st.markdown("---")
st.caption("Built with LangChain + Pinecone + Groq Llama 3.3 + Cohere Rerank")
