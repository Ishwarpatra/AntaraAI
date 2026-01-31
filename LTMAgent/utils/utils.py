from langchain_core.runnables import RunnableConfig
import requests
from typing_extensions import TypedDict


class KnowledgeTriple(TypedDict):
    subject: str
    predicate: str
    object_: str


def get_user_id(config: RunnableConfig) -> str:
    """Get the user ID from the runtime configuration.
    
    Args:
        config: Runtime configuration object
        
    Returns:
        str: User ID
        
    Raises:
        ValueError: If user ID is not provided in the configuration
    """
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")
    return user_id


def get_thread_id(config: RunnableConfig) -> str:
    """Get the thread ID from the runtime configuration.
    
    Args:
        config: Runtime configuration object
        
    Returns:
        str: Thread ID or None if not provided
    """
    return config["configurable"].get("thread_id", "default")