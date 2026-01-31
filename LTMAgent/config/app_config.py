"""
Configuration settings for the LTM application.

Simple configuration with environment variables used only for sensitive data.
"""
import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "model_provider": "groq",  # Options: 'ollama', 'groq'
    "model_name": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "searx_host": os.getenv("LTM_SEARX_HOST", "http://127.0.0.1:8080"),
    "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    "ollama_model": "mistral:latest",
    "vector_k_results": 3,
    "allow_external_requests": os.getenv("LTM_ALLOW_EXTERNAL", "").lower() == "true",
    "use_https": os.getenv("LTM_USE_HTTPS", "").lower() == "true",
    "request_timeout": 10,
    # Mongo configuration for persistent memory (used by MemoryManager)
    # Prefer a standard MONGODB_URI environment variable (e.g. from Atlas). Fall back to LTM_MONGODB_URI.
    "mongodb_uri": os.getenv("MONGODB_URI", os.getenv("LTM_MONGODB_URI", "mongodb://localhost:27017")),
    "mongodb_db": os.getenv("LTM_MONGODB_DB", "agent-memory"),
    # Default collection name used by MemoryManager when creating stores
    "mongodb_collection": os.getenv("LTM_MONGODB_COLLECTION", "memories"),
    # Embedding/index configuration for the store (example defaults)
    "memory_index": {
        "dims": os.getenv("LTM_EMBED_DIMS", "384"),
        # Use a provider string compatible with langmem/langgraph store index
        "embed": os.getenv("LTM_EMBED_MODEL", "hf:sentence-transformers/all-MiniLM-L6-v2"),
    },
}

def get_config(key, default=None):
    """Get a configuration value with fallback to default."""
    return CONFIG.get(key, default)


def validate_config():
    """Perform minimal validation of required configuration keys.

    This function is intentionally small and non-fatal for MVP — it logs
    warnings if configuration values look suspicious but does not raise
    unless a required key is missing.
    """
    provider = CONFIG.get("model_provider")
    if provider not in ("groq", "ollama"):
        print(f"[WARNING] Unexpected model_provider '{provider}' in config. Expected 'groq' or 'ollama'.")

    if provider == "groq" and not CONFIG.get("model_name"):
        print("[WARNING] model_provider is 'groq' but 'model_name' is not set.")

    if provider == "ollama" and not CONFIG.get("ollama_model"):
        print("[WARNING] model_provider is 'ollama' but 'ollama_model' is not set.")

    # Validate vector_k_results is an int >= 1
    k = CONFIG.get("vector_k_results")
    try:
        if int(k) < 1:
            print(f"[WARNING] vector_k_results should be >= 1; got {k}")
    except Exception:
        print(f"[WARNING] vector_k_results is not an integer: {k}")


def get_mongodb_store_config():
    """Return a dict containing MongoDB connection info and index config.

    This helper centralizes how MemoryManager should read Mongo-related config
    values so the rest of the codebase uses a single source of truth.
    """
    return {
        "uri": CONFIG.get("mongodb_uri"),
        "db": CONFIG.get("mongodb_db"),
        "collection": CONFIG.get("mongodb_collection"),
        "index": CONFIG.get("memory_index"),
    }