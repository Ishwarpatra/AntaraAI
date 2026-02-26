import pytest
from unittest.mock import MagicMock, patch
import pymongo.errors
import sys

# Create a mock embedding instance to be returned by patched embedding classes
mock_embedding_instance = MagicMock()
mock_embedding_instance.embed_query.return_value = [0.1] * 384 # Dummy embedding value
mock_embedding_instance.embed_documents.return_value = [[0.1] * 384] # Dummy embedding for documents

@pytest.fixture(autouse=True, scope='session')
def mock_module_level_patches():
    # --- Start Patchers ---
    
    # Patch _initialize_embeddings before core.memory_manager is imported
    _initialize_embeddings_patcher = patch('core.memory_manager._initialize_embeddings', return_value=mock_embedding_instance)
    _mock_initialize_embeddings = _initialize_embeddings_patcher.start()

    # Patch _initialize_mongodb_connection
    _initialize_mongodb_connection_patcher = patch('core.memory_manager._initialize_mongodb_connection')
    _mock_initialize_mongodb_connection = _initialize_mongodb_connection_patcher.start()
    
    # Configure _mock_initialize_mongodb_connection to return mock client, db, collection
    mock_mongo_client = MagicMock()
    mock_db = MagicMock()
    mock_collection = MagicMock()
    _mock_initialize_mongodb_connection.return_value = (mock_mongo_client, mock_db, mock_collection)

    # Patch pymongo.MongoClient (the actual class) and configure its return value
    _mongo_client_patcher = patch('pymongo.MongoClient')
    MockMongoClient = _mongo_client_patcher.start()
    mock_mongo_client_instance = MockMongoClient.return_value
    mock_mongo_client_instance.server_info.return_value = {} # Prevent connection error
    mock_db_instance = MagicMock()
    mock_mongo_client_instance.__getitem__.return_value = mock_db_instance # db property (for core.memory_manager.db)
    
    # Patch pymongo.errors.ServerSelectionTimeoutError
    _server_selection_timeout_error_patcher = patch('pymongo.errors.ServerSelectionTimeoutError')
    MockServerSelectionTimeoutError = _server_selection_timeout_error_patcher.start()

    # Patch rag_embeddings_model in core.tools
    _rag_embeddings_model_patcher = patch('core.tools.rag_embeddings_model')
    MockRagEmbeddingsModel = _rag_embeddings_model_patcher.start()
    MockRagEmbeddingsModel.embed_query.return_value = [0.1] * 384 # Dummy embedding

    # Patch core.memory_manager.db to use our mocked db instance
    # This ensures that when core.memory_manager.db is accessed, it's our mock
    _memory_manager_db_patcher = patch('core.memory_manager.db', new=mock_db_instance)
    _mock_memory_manager_db = _memory_manager_db_patcher.start()

    yield # Yield control to the tests

    # --- Stop Patchers ---
    _memory_manager_db_patcher.stop()
    _rag_embeddings_model_patcher.stop()
    _server_selection_timeout_error_patcher.stop()
    _mongo_client_patcher.stop()
    _initialize_mongodb_connection_patcher.stop()
    _initialize_embeddings_patcher.stop()

# This fixture is no longer needed as the db is globally mocked by mock_module_level_patches
# @pytest.fixture
# def mock_db_client_db():
#     return MagicMock()