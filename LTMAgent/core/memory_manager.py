from langchain_core.runnables import RunnableConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
#from langchain_core.vectorstores import InMemoryVectorStore
import uuid
from typing import List, Optional

from utils.utils import get_user_id, get_thread_id, KnowledgeTriple
from config.app_config import get_config
from langgraph.store.mongodb import MongoDBStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize the VectorStore where memories will be stored
print("Initializing Vector Store")
recall_vector_store = MongoDBStore(
    connection_string="your_mongo_atlas_connection_string",
    db_name="your_db",
    collection_name="recall_memory",
    embedding=GoogleGenerativeAIEmbeddings()
)



# Cache configuration values to avoid repeated lookups
VECTOR_K_RESULTS = get_config("vector_k_results", 3)

def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    user_id = get_user_id(config)
    thread_id = get_thread_id(config)

    doc_id = str(uuid.uuid4())
    document = Document(
        page_content=memory.strip(),
        id=doc_id,
        metadata={
            "user_id": user_id,
            "thread_id": thread_id,
            "timestamp": str(uuid.uuid1())
        }
    )
    # Add the document to the vector store
    recall_vector_store.put(("recall", user_id, thread_id), doc_id, document)
    return memory

def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    # Get the user ID and thread ID from the configuration
    user_id = get_user_id(config)
    thread_id = get_thread_id(config)
    # Define a filter to get only memories for this user and thread
    results = recall_vector_store.search(("recall", user_id, thread_id), query=query.strip(), limit=VECTOR_K_RESULTS)
    return [res.value.page_content for res in results]
    
    


def save_recall_memory_knowledge_triple(memories: List[KnowledgeTriple], config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    # Get the user ID and thread ID from the configuration
    user_id = get_user_id(config)
    thread_id = get_thread_id(config)
    
    # Process each memory in the list
    for memory in memories:
        # Create a serialized representation of the knowledge triple
        serialized = " ".join(memory.values())
        
        # Create a document with the serialized content and metadata
        document = Document(
            page_content=serialized,
            id=str(uuid.uuid4()),
            metadata={
                "user_id": user_id,
                "thread_id": thread_id,
                "timestamp": str(uuid.uuid1()),
                **memory,
            },
        )
        
        # Add the document to the vector store
        recall_vector_store.add_documents([document])
    
    return memories