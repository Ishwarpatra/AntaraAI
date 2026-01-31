"""
Memory management for the LTM application using LangMem with MongoDB Store.
Supports Episodic, Semantic, Procedural, and Associative memory types.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.store.mongodb.base import MongoDBStore, create_vector_index_config
from pymongo import MongoClient
import os
from pydantic import BaseModel, Field
from langmem import create_manage_memory_tool, create_search_memory_tool
from typing import List

print("Initializing MongoDB Memory Store")

# MongoDB connection
mongo_client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
db = mongo_client["ltm_agent"]
collection = db["memories"]

# LangGraph MongoDB store for LangMem tools
memory_store = MongoDBStore(
    collection=collection,
    index_config=create_vector_index_config(
        embed=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        dims=384,  # all-MiniLM-L6-v2 has 384 dimensions
        fields=["content"],
    ),
    auto_index_timeout=60  # Wait up to 60 seconds for index creation
)

# ============================================================================
# MEMORY SCHEMAS
# ============================================================================

class Episode(BaseModel):
    """Episodic memory captures specific experiences and learning moments.
    Each episode records what happened, the agent's reasoning, actions taken, and outcomes."""

    observation: str = Field(..., description="The context and setup - what happened in the situation")
    thoughts: str = Field(
        ...,
        description="Internal reasoning process and observations of the agent that led to the action and result"
    )
    action: str = Field(
        ...,
        description="What was done, how, and in what format. Include salient details for success"
    )
    result: str = Field(
        ...,
        description="Outcome and retrospective analysis. What worked well? What could be improved?"
    )

class Triple(BaseModel):
    """Semantic memory stores factual information as triples.
    Captures relationships, preferences, and structured knowledge."""

    subject: str = Field(..., description="The entity being described")
    predicate: str = Field(..., description="The relationship or property")
    object: str = Field(..., description="The target of the relationship")
    context: str | None = Field(None, description="Optional additional context or clarification")

class Procedural(BaseModel):
    """Procedural memory stores instructions, rules, and procedures for recurring tasks.
    Captures how to perform specific actions or follow established processes."""

    task: str = Field(..., description="The task or process this procedure applies to")
    steps: List[str] = Field(..., description="Step-by-step instructions for completing the task")
    conditions: str | None = Field(None, description="When or under what circumstances to apply this procedure")
    outcome: str | None = Field(None, description="Expected result or success criteria")


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

# General Memory Tools (for mixed usage)
manage_general_memory_tool = create_manage_memory_tool(
    namespace=("memories", "{user_id}"),
    store=memory_store
)
search_general_memory_tool = create_search_memory_tool(
    namespace=("memories", "{user_id}"),
    store=memory_store
)
