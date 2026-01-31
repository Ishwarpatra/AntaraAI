"""
MongoDB-backed Memory Manager for LTM application.

This module provides:
1. MongoDB connection for general data persistence (mood logs, sessions, etc.)
2. Memory tools using langmem with InMemoryStore fallback
3. Graceful degradation if advanced features aren't available

The design prioritizes reliability - the app will run even if some
advanced memory features aren't available.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment
load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# MONGODB CONNECTION
# =============================================================================

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DATABASE_NAME = os.getenv("MONGODB_DATABASE", "ltm_database")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "memory_store")

# MongoDB client and database
mongo_client = None
db = None
collection = None

try:
    from pymongo import MongoClient
    
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Force server selection to validate connection
    mongo_client.server_info()
    
    db = mongo_client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    logger.info("✅ Connected to MongoDB at %s", MONGO_URI)
except ImportError:
    logger.warning("pymongo not installed. Using mock database.")
except Exception as e:
    logger.warning("MongoDB connection failed: %s. Using mock database.", e)

# Create mock database if MongoDB failed
if db is None:
    class MockCollection:
        """Mock collection that stores data in memory."""
        def __init__(self):
            self._data = []
        
        def insert_one(self, doc):
            self._data.append(doc)
            return type('InsertOneResult', (), {'inserted_id': len(self._data)})()
        
        def find(self, query=None, projection=None):
            return MockCursor(self._data)
        
        def find_one(self, query=None):
            return self._data[-1] if self._data else None
        
        def update_one(self, filter, update, upsert=False):
            return type('UpdateResult', (), {'modified_count': 0})()
        
        def delete_one(self, filter):
            return type('DeleteResult', (), {'deleted_count': 0})()
    
    class MockCursor:
        def __init__(self, data):
            self._data = data
            self._cursor = iter(data)
        
        def sort(self, field, direction=-1):
            return self
        
        def limit(self, n):
            self._data = self._data[:n]
            return self
        
        def __iter__(self):
            return iter(self._data)
        
        def __next__(self):
            return next(self._cursor)
    
    class MockDatabase:
        """Mock database that creates collections on demand."""
        def __init__(self):
            self._collections = {}
        
        def __getitem__(self, name):
            if name not in self._collections:
                self._collections[name] = MockCollection()
            return self._collections[name]
    
    db = MockDatabase()
    collection = db[COLLECTION_NAME]
    logger.warning("⚠️ Using in-memory mock database. Data will NOT persist across restarts!")

# =============================================================================
# EMBEDDINGS CONFIGURATION
# =============================================================================

embeddings = None

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        logger.info("Using Google Generative AI embeddings")
except Exception as e:
    logger.debug("Google embeddings not available: %s", e)

if embeddings is None:
    try:
        from langchain_openai import OpenAIEmbeddings
        if os.getenv("OPENAI_API_KEY"):
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            logger.info("Using OpenAI embeddings")
    except Exception as e:
        logger.debug("OpenAI embeddings not available: %s", e)

if embeddings is None:
    try:
        from langchain_community.embeddings import FakeEmbeddings
        embeddings = FakeEmbeddings(size=384)
        logger.warning("Using FakeEmbeddings - vector search will not work properly")
    except Exception:
        embeddings = None
        logger.warning("No embeddings available")

# =============================================================================
# MEMORY STORE
# =============================================================================

memory_store = None

# Try to use langgraph's InMemoryStore (more widely available)
try:
    from langgraph.store.memory import InMemoryStore
    memory_store = InMemoryStore()
    logger.info("Using InMemoryStore for memory management")
except ImportError:
    logger.warning("langgraph.store.memory not available")

# =============================================================================
# Memory Schemas
# =============================================================================

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
    context: Optional[str] = Field(None, description="Optional additional context")

class Procedural(BaseModel):
    """Procedural memory stores instructions, rules, and procedures."""
    task: str = Field(..., description="The task or process")
    steps: List[str] = Field(..., description="Step-by-step instructions")
    conditions: Optional[str] = Field(None, description="When to apply this procedure")
    outcome: Optional[str] = Field(None, description="Expected result")

class Associative(BaseModel):
    """Associative memory stores connections between concepts and ideas."""
    concept_a: str = Field(..., description="The first concept in the association")
    concept_b: str = Field(..., description="The second concept in the association")
    strength: float = Field(..., description="Strength of the association (0.0-1.0)")
    context: Optional[str] = Field(None, description="Context in which the association was formed")

# =============================================================================
# Memory Tools (with graceful fallback)
# =============================================================================

memory_tools = []

if memory_store is not None:
    try:
        from langmem import create_manage_memory_tool, create_search_memory_tool
        
        manage_episodic_memory_tool = create_manage_memory_tool(
            namespace=("memories", "{user_id}", "episodes"),
            store=memory_store
        )
        search_episodic_memory_tool = create_search_memory_tool(
            namespace=("memories", "{user_id}", "episodes"),
            store=memory_store
        )

        manage_semantic_memory_tool = create_manage_memory_tool(
            namespace=("memories", "{user_id}", "triples"),
            store=memory_store
        )
        search_semantic_memory_tool = create_search_memory_tool(
            namespace=("memories", "{user_id}", "triples"),
            store=memory_store
        )

        manage_procedural_memory_tool = create_manage_memory_tool(
            namespace=("memories", "{user_id}", "procedures"),
            store=memory_store
        )
        search_procedural_memory_tool = create_search_memory_tool(
            namespace=("memories", "{user_id}", "procedures"),
            store=memory_store
        )

        manage_associative_memory_tool = create_manage_memory_tool(
            namespace=("memories", "{user_id}", "associations"),
            store=memory_store
        )
        search_associative_memory_tool = create_search_memory_tool(
            namespace=("memories", "{user_id}", "associations"),
            store=memory_store
        )

        manage_general_memory_tool = create_manage_memory_tool(
            namespace=("memories", "{user_id}"),
            store=memory_store
        )
        search_general_memory_tool = create_search_memory_tool(
            namespace=("memories", "{user_id}"),
            store=memory_store
        )

        memory_tools = [
            manage_episodic_memory_tool,
            search_episodic_memory_tool,
            manage_semantic_memory_tool,
            search_semantic_memory_tool,
            manage_procedural_memory_tool,
            search_procedural_memory_tool,
            manage_associative_memory_tool,
            search_associative_memory_tool,
            manage_general_memory_tool,
            search_general_memory_tool,
        ]
        
        logger.info("✅ Memory tools created successfully (%d tools)", len(memory_tools))
        
    except ImportError as e:
        logger.warning("langmem not available: %s. Memory tools disabled.", e)
    except Exception as e:
        logger.warning("Failed to create memory tools: %s", e)
else:
    logger.warning("Memory store not available. Memory tools disabled.")

# Export memory tools for use in tools.py
__all__ = [
    'db', 'collection', 'memory_store', 'embeddings', 'memory_tools',
    'Episode', 'Triple', 'Procedural', 'Associative'
]

logger.info("Memory manager initialization complete")
