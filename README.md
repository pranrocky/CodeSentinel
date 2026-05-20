# CodeSentinel

> Agentic AI code review and security analysis — fully local, no API keys required.

CodeSentinel lets you point an AI agent at any GitHub repository or local codebase and have a conversation with it: find bugs, trace logic, audit for security issues, explain architecture, and get fix suggestions — all running on your own hardware via Ollama.

---

## How It Works

```
GitHub URL / Local Path
        │
        ▼
  [ Ingestion Pipeline ]
   load_repo → chunk_code (AST) → FAISS index + Knowledge Graph
        │
        ▼
  [ LangGraph Agent ]
   LLM (qwen2.5-coder:14b) ↔ 11 Specialist Tools
        │
        ▼
  [ Gradio Chat UI ]
   Streaming responses at localhost:7860
```

1. **Ingest** — paste a GitHub URL or local path. The repo is cloned, parsed at AST boundaries, embedded with `BAAI/bge-m3`, and stored in a FAISS vector index. A knowledge graph is built in parallel.
2. **Ask** — type any question in natural language. The LangGraph agent decides which tools to call, reasons over the results, and streams its response back.
3. **Iterate** — the full conversation history is preserved per session so you can drill down, follow up, or pivot to a new file.

---

## Features

- **RAG over code** — semantic search across the entire indexed codebase using 1024-dim BGE-M3 embeddings
- **Knowledge dependency graph** — pre-built graph for fast, cheap relationship lookups before hitting the vector index
- **AST-aware chunking** — splits at function/class boundaries (LlamaIndex `CodeSplitter`) with `SentenceSplitter` fallback
- **11 specialist tools** bound to the LLM:

| Tool | What it does |
|---|---|
| `query_kdg` | Query the pre-built knowledge dependency graph |
| `rag_query` | Semantic search over the FAISS vector index |
| `search_symbol` | AST-based symbol definition finder |
| `get_call_slice` | Forward program slice from a Python function |
| `run_linter` | Run `ruff` on any file and return diagnostics |
| `suggest_fix` | LLM-generated fix suggestions for linter issues |
| `explain_trace` | Interpret Python stack traces with RAG context |
| `universal_grep` | Regex search across the entire repo |
| `list_repo_files` | Enumerate all files in the repository |
| `read_file` | Read any specific file |
| `read_entire_codebase` | Full-repo read for global summaries |

- **Pre-generated bug patterns** — JSON pattern library (mutable defaults, bare excepts, SQL injection via f-strings, etc.) loaded at startup for RAG enrichment
- **Ollama fallback parser** — intercepts raw JSON tool calls from models that don't emit structured tool-call tokens, so the agent degrades gracefully
- **Multi-language ingestion** — Python, JavaScript, TypeScript, Java, C/C++, Go, Rust, HTML, CSS, and Markdown READMEs

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with `qwen2.5-coder:14b` pulled
- CUDA GPU recommended for embedding speed (CPU works but is slow)
- `git` available on PATH (for cloning remote repos)
- `ruff` installed (for the linter tool)

---

## Installation

```bash
# 1. Clone this repo
git clone https://github.com/YOUR_USERNAME/CodeSentinel.git
cd CodeSentinel

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the LLM model into Ollama
ollama pull qwen2.5-coder:14b

# 5. Pre-generate bug pattern files (one-time setup)
python generate_patterns.py
```

---

## Usage

### Start the UI

```bash
python ui/app.py
```

Open `http://localhost:7860` in your browser.

### In the interface

1. **Ingest Codebase** (left panel) — enter a GitHub URL or absolute local path, then click **Build Knowledge Base**. Watch the status box for progress.
2. **Analysis & Review** (right panel) — once ingestion completes, type your question and press **Send**.

### Example questions

```
Find any SQL injection vulnerabilities in this codebase.
Where is the Flask class defined and what does __init__ do?
Explain the call chain starting from app.run().
Run the linter on src/models/user.py and suggest fixes.
Are there any bare except clauses that swallow errors silently?
Summarize the overall architecture of this project.
```

---

## Configuration

All configuration lives in `config.py`:

```python
LLM_MODEL       = 'qwen2.5-coder:14b'   # any Ollama model
EMBEDDING_MODEL  = 'BAAI/bge-m3'         # HuggingFace embedding model
INDEX_DIR        = 'data/index'           # FAISS index persistence path
BUG_PATTERNS_DIR = 'data/bug_patterns'   # pre-generated JSON patterns
```

To swap the LLM, pull any Ollama model and update `LLM_MODEL`. The agent and tools are model-agnostic.

---

## Project Structure

```
CodeSentinel/
├── config.py               # Central config (models, paths)
├── generate_patterns.py    # One-time bug pattern pre-generation
├── ui/
│   └── app.py              # Gradio web interface
├── agent/
│   ├── graph.py            # LangGraph StateGraph (agent ↔ tools loop)
│   ├── state.py            # AgentState TypedDict
│   └── prompts.py          # System prompt builder
├── ingestion/
│   ├── loader.py           # Git clone / local dir loader (multi-language)
│   ├── chunker.py          # AST + sentence chunking
│   └── indexer.py          # FAISS index builder + knowledge graph
├── tools/
│   ├── rag_tool.py         # Semantic search
│   ├── kdg_tool.py         # Knowledge dependency graph query
│   ├── slicer_tool.py      # Forward program slicer
│   ├── linter_tool.py      # Ruff linter wrapper
│   ├── fix_tool.py         # LLM fix suggester
│   ├── symbol_tool.py      # AST symbol finder
│   ├── trace_tool.py       # Stack trace explainer
│   ├── grep_tool.py        # Universal grep
│   ├── explorer_tool.py    # File lister + reader
│   └── repo_map_tool.py    # Full codebase reader
└── data/
    ├── index/              # Persisted FAISS index (auto-created)
    └── bug_patterns/       # Pre-generated JSON bug patterns
```

---

## Stack

| Layer | Technology |
|---|---|
| LLM | `qwen2.5-coder:14b` via Ollama (OpenAI-compatible API) |
| Embeddings | `BAAI/bge-m3` (HuggingFace, 1024-dim) |
| Vector DB | FAISS (via LlamaIndex), persisted to disk |
| Agent framework | LangGraph (`StateGraph` with conditional edges) |
| LLM client | LangChain `ChatOpenAI` (pointed at Ollama) |
| Chunking | LlamaIndex `CodeSplitter` (AST) + `SentenceSplitter` |
| Linter | Ruff |
| UI | Gradio 5 |

---

## Privacy

CodeSentinel is entirely local. No code, queries, or embeddings leave your machine. Ollama runs the LLM on localhost; HuggingFace model weights are downloaded once and cached locally.

---

## License

MIT
