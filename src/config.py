import os
from dotenv import load_dotenv

# Load environment variables from .env file
# It's good practice to call this early, 
# so that environment variables are available when other modules are imported.
load_dotenv()

# --- LLM API Configuration ---
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
LLM_CHAT_MODEL = os.getenv("LLM_CHAT_MODEL", "glm-z1-flash") # Confirm actual model name

# --- Embedding Model Configuration ---
# Defaulting to a common Chinese text embedding model from HuggingFace
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# For GPU, you might set device in HuggingFaceEmbeddings, e.g. model_kwargs={'device': 'cuda'}
# For CPU, model_kwargs={'device': 'cpu'} is usually default or can be explicit.

# --- Vector Store Configuration ---
# Path where the FAISS vector store index will be saved/loaded from
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store/my_kb_faiss")
# Directory containing the source documents for the knowledge base
DATA_PATH = "data/"

# --- RAG Parameters --- 
# These can be tuned for performance and relevance
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
# Number of top relevant documents to retrieve from the vector store
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 3))

# --- Sanity Checks & Warnings ---
if not LLM_API_KEY:
    print("Warning: LLM_API_KEY is not set. LLM functionality will be affected.")

print(f"Configuration Loaded:")
print(f"  LLM Model: {LLM_CHAT_MODEL}")
print(f"  Embedding Model: {EMBEDDING_MODEL_NAME}")
print(f"  Vector Store Path: {VECTOR_STORE_PATH}")
print(f"  Knowledge Base Data Path: {DATA_PATH}")
print(f"  Chunk Size: {CHUNK_SIZE}, Chunk Overlap: {CHUNK_OVERLAP}, Top K Results: {TOP_K_RESULTS}") 