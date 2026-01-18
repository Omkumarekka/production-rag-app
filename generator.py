import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from retriever import get_retriever

load_dotenv()

# 1. Setup the LLM (Groq is fast and has generous free tier)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0, # Set to 0 for grounded, factual responses
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 2. Define the System Prompt
# Requirement 4b & 4c: Citations and No-answer handling
RAG_PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions strictly based on the provided context.

INSTRUCTIONS:
1. Use the provided context to answer the user's question.
2. If the answer is not contained within the context, gracefully state that you do not have enough information to answer. Do not make up an answer.
3. Every claim you make must be followed by an inline citation in brackets, e.g., [1], [2].
4. Match these numbers to the order of the source documents provided below.

CONTEXT:
{context}

QUESTION: 
{question}

HELPFUL ANSWER:
"""

def generate_answer(query: str, namespace: str = "default"):
    # Get the retriever from Step 4
    retriever = get_retriever(namespace=namespace)
    
    # 3. Retrieve and Rerank documents
    docs = retriever.invoke(query)
    
    # Check if we have any documents (Requirement 4c)
    if not docs:
        return "I'm sorry, I couldn't find any relevant information in the documents to answer your question.", []

    # 4. Format the context for the prompt with numbers for citations
    context_text = ""
    sources = []
    for i, doc in enumerate(docs):
        citation_num = i + 1
        context_text += f"[{citation_num}] {doc.page_content}\n\n"
        sources.append({
            "id": citation_num,
            "source": doc.metadata.get("source", "Unknown"),
            "title": doc.metadata.get("title", "Untitled"),
            "content": doc.page_content
        })

    # 5. Generate Answer
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    chain = prompt | llm
    
    response = chain.invoke({
        "context": context_text,
        "question": query
    })

    return response.content, sources

if __name__ == "__main__":
    # Test the full pipeline
    user_query = "What are the core requirements for the RAG app?"
    answer, source_list = generate_answer(user_query)
    
    print("\n--- ANSWER ---")
    print(answer)
    print("\n--- SOURCES ---")
    for s in source_list:
        print(f"[{s['id']}] {s['title']} ({s['source']})")
