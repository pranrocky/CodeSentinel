import os
import shutil
import tempfile
from git import Repo
from llama_index.core import Document

SUPPORTED_EXTS = {
    '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
    '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.go': 'go', '.rs': 'rust',
    '.html': 'html', '.css': 'css'
}

def scrape_directory(dir_path: str) -> list[Document]:
    print(f"[*] Scraping directory for code and documentation...")
    documents = []
    
    for root, _, files in os.walk(dir_path):
        if '.git' in root or 'node_modules' in root or '__pycache__' in root:
            continue
            
        for file in files:
            filepath = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            is_code = ext in SUPPORTED_EXTS
            is_readme = file.lower() == 'readme.md'
            
            if is_code or is_readme:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lang = SUPPORTED_EXTS.get(ext, 'markdown') if is_code else 'markdown'
                        
                        doc = Document(
                            text=content,
                            metadata={
                                'file_path': filepath,
                                'file_name': file,
                                'language': lang,
                                'is_readme': is_readme
                            }
                        )
                        documents.append(doc)
                except Exception:
                    pass # Silently skip unreadable binary/corrupted files
                    
    print(f"[*] Successfully loaded {len(documents)} files.")
    return documents

def load_repo(source_url: str) -> tuple[list[Document], str]:
    """Loads a repository from a local directory or clones a GitHub URL."""
    
    # FIX: Native support for local directories
    if os.path.isdir(source_url):
        print(f"[*] Local directory detected. Using: {source_url}")
        docs = scrape_directory(source_url)
        return docs, source_url

    # FIX: Prevent disk leaks by using a single, reusable workspace
    workspace_dir = os.path.join(tempfile.gettempdir(), "codesentinel_workspace")
    
    if os.path.exists(workspace_dir):
        print(f"[*] Cleaning up previous workspace at {workspace_dir}...")
        shutil.rmtree(workspace_dir)
        
    os.makedirs(workspace_dir)
    print(f"[*] Created fresh workspace at: {workspace_dir}")
    
    print(f"[*] Cloning repository from {source_url}...")
    Repo.clone_from(source_url, workspace_dir)
    
    docs = scrape_directory(workspace_dir)
    return docs, workspace_dir