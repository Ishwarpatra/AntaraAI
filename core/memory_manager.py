"""
MongoDB-backed Memory Manager for LTM application.

This module creates a real connection to a MongoDB instance using `pymongo` and
initializes `langmem` memory tools with a Mongo-backed vector store.
This implementation will fail fast if MongoDB is not reachable so the application
operator is forced to configure a persistent database for production use.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from unittest.mock import MagicMock # Import MagicMock

# External deps
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Langgraph / langmem store
from langgraph.store.mongodb import MongoDBStore
from langmem import create_manage_memory_tool, create_search_memory_tool

# Load environment
load_dotenv()

logger = logging.getLogger(__name__)

# --- MONGODB CONNECTION (LAZY INITIALIZATION) ---
_mongo_client_instance = None
_db_instance = None
_collection_instance = None

def _initialize_mongodb_connection():
    global _mongo_client_instance, _db_instance, _collection_instance
    if _mongo_client_instance is not None:
        return _mongo_client_instance, _db_instance, _collection_instance

    MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    DATABASE_NAME = os.getenv("MONGODB_DATABASE", "ltm_database")
    COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "memory_store")

    try:
        _mongo_client_instance = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Force server selection to validate connection; will raise if unreachable
        _mongo_client_instance.server_info()
        _db_instance = _mongo_client_instance[DATABASE_NAME]
        _collection_instance = _db_instance[COLLECTION_NAME]
    except Exception as e:
        logger.error("Failed to connect to MongoDB at %s: %s", MONGO_URI, e)
        raise RuntimeError(f"MongoDB connection failed: {e}") from e
    
    return _mongo_client_instance, _db_instance, _collection_instance

# Expose db/collection for other modules, initialized lazily
mongo_client = None
db = None
collection = None

if os.getenv("TESTING_ENV", "false").lower() == "true":
    # In testing environment, provide mock objects for MongoDB
    mongo_client = MagicMock()
    db = MagicMock()
    collection = MagicMock()

    # In testing environment, provide a mock _initialize_embeddings
    def _initialize_embeddings():
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 384
        mock_embed.embed_documents.return_value = [[0.1] * 384]
        return mock_embed
else:
    # In production/development, initialize real connection
    mongo_client, db, collection = _initialize_mongodb_connection()

    _embeddings_instance = None
    def _initialize_embeddings():
        """Initializes and returns the preferred embeddings model."""
        global _embeddings_instance
        if _embeddings_instance is not None:
            return _embeddings_instance

        # Attempt to use available, preferred embeddings in order
        try:
            _embeddings_instance = OpenAIEmbeddings(model="text-embedding-ada-002")
            logger.info("Using OpenAI embeddings")
        except Exception as e:
            logger.warning(f"OpenAI embeddings failed: {e}. Trying Google Generative.")
            try:
                _embeddings_instance = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                logger.info("Using Google Generative embeddings")
            except Exception as e:
                logger.warning(f"Google Generative embeddings failed: {e}. Falling back to HuggingFace.")
                try:
                    _embeddings_instance = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    logger.info("Falling back to HuggingFace embeddings")
                except Exception as e:
                    logger.error(f"All embedding providers failed to initialize: {e}")
                    raise RuntimeError("No embedding provider could be initialized.")
        return _embeddings_instance

_memory_store_instance = None
def get_memory_store():
    global _memory_store_instance
    if _memory_store_instance is None:
        _memory_store_instance = MongoDBStore(
            collection=collection,
            index={
                "dims": 384, # Default dims, will be updated if embed has specific dims
                "embed": _initialize_embeddings(), # This will now be called when get_memory_store is invoked
                "fields": ["content"],
            },
        )
    return _memory_store_instance

# ==========================================================================
# Memory Schemas
# ==========================================================================
class Episode(BaseModel):
    """Episodic memory captures specific experiences and learning moments."""
    observation: str = Field(..., description="The context and setup - what happened")
    thoughts: str = Field(..., description="Internal reasoning process")
    action: str = Field(..., description="What was done")
    result: str = Field(..., description="Outcome and retrospective analysis")
    significance_score: int = Field(..., description="A 1-10 rating of significance")

class Triple(BaseModel):
    """Semantic memory stores factual information as triples."""
    subject: str = Field(..., description="The entity being described")
    predicate: str = Field(..., description="The relationship or property")
    object: str = Field(..., description="The target of the relationship")
    context: str | None = Field(None, description="Optional additional context")

class Procedural(BaseModel):
    """Procedural memory stores instructions, rules, and procedures."""
    task: str = Field(..., description="The task or process")
    steps: List[str] = Field(..., description="Step-by-step instructions")
    conditions: str | None = Field(None, description="When to apply this procedure")
    outcome: str | None = Field(None, description="Expected result")

class Associative(BaseModel):
    """Associative memory stores connections between concepts and ideas."""
    concept_a: str = Field(..., description="The first concept in the association")
    concept_b: str = Field(..., description="The second concept in the association")
    strength: float = Field(..., description="Strength of the association (0.0-1.0)")
    context: str | None = Field(None, description="Context in which the association was formed")

# ==========================================================================
# Memory Tools (LangMem)
# ==========================================================================
manage_episodic_memory_tool = create_manage_memory_tool(
    namespace=("memories", "{user_id}", "episodes"),
    store=get_memory_store()
)
search_episodic_memory_tool = create_search_memory_tool(
    namespace=("memories", "{user_id}", "episodes"),
    store=get_memory_store()
)

manage_semantic_memory_tool = create_manage_memory_tool(
    namespace=("memories", "{user_id}", "triples"),
    store=get_memory_store()
)
search_semantic_memory_tool = create_search_memory_tool(
    namespace=("memories", "{user_id}", "triples"),
    store=get_memory_store()
)

manage_procedural_memory_tool = create_manage_memory_tool(
    namespace=("memories", "{user_id}", "procedures"),
    store=get_memory_store()
)
search_procedural_memory_tool = create_search_memory_tool(
    namespace=("memories", "{user_id}", "procedures"),
    store=get_memory_store()
)

manage_associative_memory_tool = create_manage_memory_tool(
    namespace=("memories", "{user_id}", "associations"),
    store=get_memory_store()
)
search_associative_memory_tool = create_search_memory_tool(
    namespace=("memories", "{user_id}", "associations"),
    store=get_memory_store()
)

manage_general_memory_tool = create_manage_memory_tool(
    namespace=("memories", "{user_id}"),
    store=get_memory_store()
)
search_general_memory_tool = create_search_memory_tool(
    namespace=("memories", "{user_id}"),
    store=get_memory_store()
)

logger.info("✅ MongoDB-backed memory manager initialized and memory tools created")