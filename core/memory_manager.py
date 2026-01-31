"""
Memory management for the LTM application.
Implements the 4 LangMem layers (Episodic, Semantic, Procedural, Associative) with MongoDB-backed persistence.
"""

from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.store.mongodb import MongoStore
from pydantic import BaseModel, Field
from langmem import create_manage_memory_tool, create_search_memory_tool
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("✅ Initializing Memory Manager with MongoDB persistence")

# --- MONGODB CONNECTION ---
# Connect to MongoDB
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client["ltm_database"]
collection = db["memories"]

# --- EMBEDDINGS CONFIGURATION ---
# Use appropriate embeddings based on available services
try:
    # Try OpenAI embeddings first
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
except:
    try:
        # Fallback to Google embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except:
        # Fallback to Hugging Face embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- MONGODB STORE ---
memory_store = MongoStore(
    client=client,
    database_name="ltm_database",
    collection_name="memory_store",
    index={
        "dims": embeddings.model_kwargs.get("dimensions", 384) if hasattr(embeddings, 'model_kwargs') else 384,
        "embed": embeddings,
        "fields": ["content"],
    }
)
# ---------------------

# ============================================================================
# MEMORY SCHEMAS
# ============================================================================

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


# ============================================================================
# MEMORY TOOLS
# ============================================================================

# Episodic Memory Tools
manage_episodic_memory_tool = create_manage_memory_tool(
    namespace=("memories", "{user_id}", "episodes"),
    store=memory_store
)
search_episodic_memory_tool = create_search_memory_tool(
    namespace=("memories", "{user_id}", "episodes"),
    store=memory_store
)

# Semantic Memory Tools
manage_semantic_memory_tool = create_manage_memory_tool(
    namespace=("memories", "{user_id}", "triples"),
    store=memory_store
)
search_semantic_memory_tool = create_search_memory_tool(
    namespace=("memories", "{user_id}", "triples"),
    store=memory_store
)

# Procedural Memory Tools
manage_procedural_memory_tool = create_manage_memory_tool(
    namespace=("memories", "{user_id}", "procedures"),
    store=memory_store
)
search_procedural_memory_tool = create_search_memory_tool(
    namespace=("memories", "{user_id}", "procedures"),
    store=memory_store
)

# Associative Memory Tools (for associative/relationship connections)
manage_associative_memory_tool = create_manage_memory_tool(
    namespace=("memories", "{user_id}", "associations"),
    store=memory_store
)
search_associative_memory_tool = create_search_memory_tool(
    namespace=("memories", "{user_id}", "associations"),
    store=memory_store
)

# General Memory Tools (for mixed usage)
manage_general_memory_tool = create_manage_memory_tool(
    namespace=("memories", "{user_id}"),
    store=memory_store
)
search_general_memory_tool = create_search_memory_tool(
    namespace=("memories", "{user_id}"),
    store=memory_store
)
