import os

# 1. Base Project Directory
# __file__ is a special Python variable containing the path to THIS file.
# os.path.dirname gets the folder containing this file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Data Directories
# We use os.path.join so this works on Windows or Linux seamlessly.
DATA_DIR = os.path.join(BASE_DIR, 'data')
INDEX_DIR = os.path.join(DATA_DIR, 'index')
BUG_PATTERNS_DIR = os.path.join(DATA_DIR, 'bug_patterns')

# 3. Model Configuration
# The exact string Ollama expects.
LLM_MODEL = 'qwen2.5-coder:14b'
EMBEDDING_MODEL = 'BAAI/bge-m3'