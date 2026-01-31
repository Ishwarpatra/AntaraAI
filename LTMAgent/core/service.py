"""
Service layer for the LTM application.

This module serves as an interface between the UI and the core functionality.
It provides clean, high-level methods that can be used by any UI implementation
(CLI, web, desktop) without needing to understand the internal implementation.
"""

import uuid
from typing import Dict, List, Any, Generator, Optional, Tuple

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from config.app_config import get_config
from core.tools import all_tools
from core.graph_builder import build_graph

class LTMService:
    """Service class to manage LTM agent interactions."""
    
    def __init__(self):
        """Initialize the LTM service."""
        self.model = None
        self.model_with_tools = None
        self.graph = None
        
        # Store model configuration to avoid repeated config calls
        self.model_provider = get_config("model_provider")
        self.model_name = get_config("model_name")
        self.ollama_host = get_config("ollama_host")
        self.ollama_model = get_config("ollama_model")
        
        try:
            self._initialize_model()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LTM service: {e}") from e
        
    def _initialize_model(self):
        """Initialize the language model based on configuration."""
        # Determine which model provider to use
        if self.model_provider == "groq":
            # Use Groq for Llama models
            self.model = ChatGroq(model=self.model_name)
        else:
            # Use Ollama for other models
            self.model = ChatOllama(base_url=self.ollama_host, model=self.ollama_model)
        
        # Bind tools to the model
        self.model_with_tools = self.model.bind_tools(all_tools)
        
        # Build the conversation graph
        self.graph = build_graph(self.model_with_tools)
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the currently loaded model.
        
        Returns:
            Dict[str, str]: Dictionary with model information
        """
        if self.model_provider == "groq":
            provider = "Groq"
            model_name = self.model_name
        else:
            provider = "Ollama"
            model_name = self.ollama_model
        
        return {
            "provider": provider,
            "model_name": model_name
        }
    
    def get_available_users(self) -> List[str]:
        """Get a list of existing user IDs from memory.
        
        This is a placeholder - in a real implementation, you would
        retrieve this information from a database or file.
        
        Returns:
            List[str]: List of user IDs
        """
        # Placeholder implementation
        return ["user1", "user2", "user3"]
    
    def create_user_id(self) -> str:
        """Generate a new user ID.
        
        Returns:
            str: A new user ID
        """
        return str(uuid.uuid4())[:8]
    
    def get_threads_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get conversation threads for a given user.
        
        Args:
            user_id: The user ID to get threads for
            
        Returns:
            List[Dict[str, Any]]: List of thread information
        """
        # Placeholder implementation
        return [
            {"id": "thread1", "created": "2023-05-01", "title": "First conversation"},
            {"id": "thread2", "created": "2023-05-02", "title": "Second conversation"}
        ]
    
    def create_thread_id(self) -> str:
        """Generate a new thread ID.
        
        Returns:
            str: A new thread ID
        """
        return str(uuid.uuid4())[:8]
    
    def process_message(self, user_prompt: str, user_id: str, 
                        thread_id: str) -> Generator[Dict[str, Any], None, None]:
        """Process a user message and generate a response.
        
        Args:
            user_prompt: The user's message
            user_id: The user ID
            thread_id: The conversation thread ID
            
        Yields:
            Dict[str, Any]: Output chunks from the graph execution
        """
        # Configure runtime settings
        config = {"configurable": {
            "user_id": user_id, 
            "thread_id": thread_id
        }}
        
        # Process the user input and yield results
        for chunk in self.graph.stream({"messages": [("user", user_prompt)]}, config=config):
            yield chunk