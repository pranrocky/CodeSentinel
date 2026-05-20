from langchain_core.tools import tool
from ingestion.indexer import load_persisted_index
import config

@tool
def rag_query(query: str) -> str:
    """
    Search the codebase and bug pattern library using semantic similarity.
    Use this to find relevant code chunks, understand how features are implemented,
    or locate where specific bug patterns might exist.
    
    Args:
        query: A natural language description of what to find (e.g., 'authentication logic').
    """
    print(f"[Tool: RAG] Searching vector DB for: '{query}'...")
    try:
        # FIX: We load the index fresh from the disk every single time.
        # This guarantees we are always searching the most recently ingested repository,
        # never a stale one left over in memory.
        index = load_persisted_index(config.INDEX_DIR)
        retriever = index.as_retriever(similarity_top_k=5)
        
        nodes = retriever.retrieve(query)
        
        if not nodes:
            return "No relevant code found for your query."
            
        chunks = []
        for n in nodes:
            file_path = n.metadata.get('file_path', 'Unknown File')
            header = f"### File: {file_path} ###\n"
            chunks.append(header + n.text)
            
        return "\n\n---\n\n".join(chunks)
        
    except Exception as e:
        return f"Error executing RAG search: {str(e)}"

# --- Execution Test Block ---
if __name__ == "__main__":
    print("\nExecuting RAG Tool on 'routing initialization'...")
    rag_output = rag_query.invoke({"query": "routing initialization"})
    
    print("\n" + "="*50)
    print(rag_output[:1000] + "\n\n... [TRUNCATED] ...")
    print("="*50)