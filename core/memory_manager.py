"""
Fallback Memory Manager for LTM application.

This module provides a lightweight in-memory store when a persistent database
(e.g., MongoDB) is not available. WARNING: InMemoryStore does NOT persist data
across process restarts and should only be used for development or as a safe
fallback.
"""

import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.store.memory import InMemoryStore

# Configure logger
logger = logging.getLogger(__name__)

def create_memory_store():
    """
    Initializes the memory store.

    CRITICAL: Currently using InMemoryStore as a fallback.
    Data will NOT be persisted across restarts.
    TODO: Implement MongoDBStore or langchain_mongodb VectorStore for production.
    """
    try:
        # Placeholder for future MongoDB implementation
        # from langgraph.store.mongodb import MongoDBStore
        # return MongoDBStore(...)
        pass
    except ImportError:
        pass

    logger.warning("⚠️  Using InMemoryStore for Semantic Memory. ALL MEMORY WILL BE LOST ON RESTART.")
    return InMemoryStore()

# Initialize the store
memory_store = create_memory_store()

# Keep embeddings available for the graph builder if needed
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
