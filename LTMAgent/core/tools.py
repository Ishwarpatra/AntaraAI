"""
Tools configuration for the LTM application.

This module defines and configures the tools available to the agent.
"""
from langchain_community.tools import SearxSearchResults
from langchain_community.utilities import SearxSearchWrapper
from langchain_core.tools import StructuredTool
from typing import List

from config.app_config import get_config
from core.memory_manager import (
    memory_store,
    manage_episodic_memory_tool,
    search_episodic_memory_tool,
    manage_semantic_memory_tool,
    search_semantic_memory_tool,
    manage_procedural_memory_tool,
    search_procedural_memory_tool,
    manage_general_memory_tool,
    search_general_memory_tool
)

# Try to import LangMem tools (for fallback if needed)
try:
    from langmem import create_manage_memory_tool, create_search_memory_tool
    LANGMEM_AVAILABLE = True
except ImportError:
    LANGMEM_AVAILABLE = False

search_internet_tool = SearxSearchResults(
    num_results=5,
    wrapper=SearxSearchWrapper(searx_host=get_config("searx_host"))
)

# Memory management tools - all types available
memory_tools = [
    manage_episodic_memory_tool,    # Create/update/delete episodic memories
    search_episodic_memory_tool,    # Search episodic memories (experiences/learning)
    manage_semantic_memory_tool,    # Create/update/delete semantic memories (facts/triples)
    search_semantic_memory_tool,    # Search semantic memories (facts/relationships)
    manage_procedural_memory_tool,  # Create/update/delete procedural memories (instructions/rules)
    search_procedural_memory_tool,  # Search procedural memories (how-to knowledge)
    manage_general_memory_tool,     # General memory management (associative relationships)
    search_general_memory_tool,     # General memory search (associative retrieval)
]

all_tools = [
    search_internet_tool
]

# Add all memory tools
all_tools.extend(memory_tools)