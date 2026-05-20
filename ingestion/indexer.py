import os
import subprocess
import faiss
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Import our global paths
import config

# graphify runs inside the codesen conda env
_GRAPHIFY_CMD = ["conda", "run", "-n", "codesen", "graphify"]


def build_knowledge_graph(repo_path: str) -> str:
    """
    Updates the knowledge dependency graph for the ingested repo using graphify.
    Requires the graph to have been initialised first via `/graphify .` in Claude Code.
    Uses `graphify update` (AST-only, no LLM needed) to keep the graph current.
    Returns the path to GRAPH_REPORT.md, or an empty string on failure.
    """
    graph_json = os.path.join(repo_path, "graphify-out", "graph.json")
    if not os.path.exists(graph_json):
        print(
            "[!] No existing knowledge graph found. "
            "Run `/graphify .` once inside Claude Code to build the initial graph, "
            "then re-ingest to enable auto-updates."
        )
        return ""

    print(f"[*] Updating knowledge graph for {repo_path}...")
    try:
        subprocess.run(
            _GRAPHIFY_CMD + ["update", repo_path],
            check=True,
            capture_output=True,
            timeout=120,
        )
        report = os.path.join(repo_path, "graphify-out", "GRAPH_REPORT.md")
        print(f"[*] Knowledge graph updated: {report}")
        return report
    except subprocess.CalledProcessError as e:
        print(f"[!] graphify update failed: {e.stderr.decode()[:300]}")
        return ""
    except FileNotFoundError:
        print("[!] graphify not found — run: conda activate codesen && pip install graphifyy")
        return ""

def get_embedding_model():
    """Loads the BGE-M3 model on CPU."""
    print(f"[*] Loading embedding model: {config.EMBEDDING_MODEL}...")
    return HuggingFaceEmbedding(
        model_name=config.EMBEDDING_MODEL, 
        device='cuda' 
    )

def build_and_persist_index(nodes, persist_dir=config.INDEX_DIR):
    """Embeds nodes and saves the FAISS index to disk."""
    if not nodes:
        raise ValueError("No nodes provided to index.")

    print(f"[*] Building FAISS index for {len(nodes)} nodes...")
    
    embed_model = get_embedding_model()

    # BGE-M3 outputs 1024 dimensions
    faiss_idx = faiss.IndexFlatIP(1024) 
    vector_store = FaissVectorStore(faiss_index=faiss_idx)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes, 
        storage_context=storage_context, 
        embed_model=embed_model
    )
    
    index.storage_context.persist(persist_dir)
    print(f"[*] Index successfully persisted to {persist_dir}")
    
    return index

def load_persisted_index(persist_dir=config.INDEX_DIR):
    """Loads a previously built index from disk."""
    print(f"[*] Loading existing index from {persist_dir}...")
    
    embed_model = get_embedding_model()
    
    vector_store = FaissVectorStore.from_persist_dir(persist_dir)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, 
        persist_dir=persist_dir
    )
    
    index = load_index_from_storage(
        storage_context=storage_context, 
        embed_model=embed_model
    )
    
    return index

# --- Execution Test Block ---
if __name__ == "__main__":
    from ingestion.loader import load_repo
    from ingestion.chunker import chunk_code
    
    test_url = "https://github.com/pallets/flask"
    
    try:
        # 1. Load & Chunk
        docs, _ = load_repo(test_url)
        nodes = chunk_code(docs)
        
        # 2. Build the Index 
        index = build_and_persist_index(nodes)
        
        # 3. Test Retrieval
        print("\n[*] Testing Retrieval Engine...")
        retriever = index.as_retriever(similarity_top_k=3)
        
        results = retriever.retrieve("How does the app route HTTP requests?")
        
        print("\n=== TOP RETRIEVAL RESULT ===")
        if results:
            print(f"File: {results[0].metadata.get('file_path')}")
            print(f"Score: {results[0].score:.3f}") 
            print("-" * 40)
            print(results[0].text[:300] + "...")
        else:
            print("No results found.")
            
    except Exception as e:
        print(f"Pipeline failed: {e}")