from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.core.schema import Document

def chunk_code(documents: list[Document]):
    print(f"[*] Starting dynamic chunking for {len(documents)} documents...")
    all_nodes = []
    
    # Group documents by their language metadata
    docs_by_lang = {}
    for doc in documents:
        lang = doc.metadata.get('language', 'unknown')
        if lang not in docs_by_lang:
            docs_by_lang[lang] = []
        docs_by_lang[lang].append(doc)
        
    for lang, docs in docs_by_lang.items():
        print(f"    -> Chunking {len(docs)} files for language: '{lang}'")
        
        # Text/Markdown gets standard sentence splitting
        if lang == 'markdown' or lang == 'unknown':
            splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
            all_nodes.extend(splitter.get_nodes_from_documents(docs))
        else:
            try:
                # Code gets AST splitting
                splitter = CodeSplitter(
                    language=lang,
                    chunk_lines=40,
                    chunk_lines_overlap=5,
                    max_chars=1500
                )
                all_nodes.extend(splitter.get_nodes_from_documents(docs))
            except Exception as e:
                # If tree-sitter fails to load a specific C-binary, safely fall back to text
                print(f"    [!] Failed to load AST parser for {lang}. Falling back to text splitting.")
                fallback = SentenceSplitter(chunk_size=512, chunk_overlap=50)
                all_nodes.extend(fallback.get_nodes_from_documents(docs))
                
    print(f"[*] Chunking complete. Generated {len(all_nodes)} total nodes.")
    return all_nodes