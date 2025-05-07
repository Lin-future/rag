import os
from typing import List, Optional

from langchain_core.documents import Document # Updated import path
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS # Updated import path

from . import config
from .embedding_utils import get_embedding_model # Ensure this import is correct

# Global variable to hold the loaded vector store instance to avoid reloading
_vector_store_instance: Optional[FAISS] = None

def get_or_create_vector_store(
    chunks: Optional[List[Document]] = None,
    force_recreate: bool = False
) -> Optional[FAISS]:
    """Loads an existing FAISS vector store or creates a new one if it doesn't exist or if forced.

    If `chunks` are provided and `force_recreate` is True, or if the store doesn't exist,
    a new vector store is created from the chunks and saved.
    Otherwise, it attempts to load an existing store from the path specified in `config.VECTOR_STORE_PATH`.

    Args:
        chunks: A list of Document objects (chunks) to build the vector store from. 
                Required if creating a new store.
        force_recreate: If True, always creates a new vector store even if one exists.

    Returns:
        A FAISS vector store instance, or None if loading/creation fails.
    """
    global _vector_store_instance
    embedding_model = get_embedding_model() # Ensure embedding model is ready
    store_path = config.VECTOR_STORE_PATH

    if not force_recreate and _vector_store_instance is not None:
        print("Returning already loaded vector store instance.")
        return _vector_store_instance

    if not force_recreate and os.path.exists(store_path):
        print(f"Attempting to load existing vector store from: {store_path}")
        try:
            # allow_dangerous_deserialization=True is often needed for FAISS local loading
            _vector_store_instance = FAISS.load_local(
                store_path, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
            print("Vector store loaded successfully from local path.")
            return _vector_store_instance
        except Exception as e:
            print(f"Error loading vector store from '{store_path}': {e}. \
                  Will attempt to recreate if chunks are provided.")
            _vector_store_instance = None # Reset instance if loading failed
            # Fall through to recreation logic if chunks are available

    if chunks:
        print("Creating new vector store...")
        if not embedding_model:
            print("Error: Embedding model is not available. Cannot create vector store.")
            return None
        try:
            _vector_store_instance = FAISS.from_documents(chunks, embedding_model)
            # Create directory if it doesn't exist before saving
            os.makedirs(os.path.dirname(store_path), exist_ok=True)
            _vector_store_instance.save_local(store_path)
            print(f"New vector store created and saved to: {store_path}")
            return _vector_store_instance
        except Exception as e:
            print(f"Error creating or saving new vector store: {e}")
            _vector_store_instance = None
            return None
    else:
        if force_recreate:
            print("Error: `force_recreate` is True, but no `chunks` were provided to build a new vector store.")
        else:
            print(f"Vector store not found at '{store_path}' and no chunks provided to create a new one.")
        return None

def get_retriever_from_store(vector_store: FAISS) -> Optional[VectorStoreRetriever]:
    """Creates a retriever from the given FAISS vector store.

    Args:
        vector_store: The FAISS vector store instance.

    Returns:
        A VectorStoreRetriever instance, or None if the input vector_store is invalid.
    """
    if not vector_store or not isinstance(vector_store, FAISS):
        print("Error: Invalid vector store provided for creating a retriever.")
        return None
        
    retriever = vector_store.as_retriever(
        search_type="similarity", # Can also be "mmr", "similarity_score_threshold"
        search_kwargs={'k': config.TOP_K_RESULTS} # Retrieve top K results
    )
    print(f"Retriever created from vector store. Will retrieve top {config.TOP_K_RESULTS} results.")
    return retriever

# Example usage (can be run directly for testing this module)
if __name__ == '__main__':
    print("--- Testing Vector Store Manager ---")

    # 1. Setup: Create dummy chunks (simulate output from data_processor)
    dummy_docs = [
        Document(page_content="第一块文本，关于苹果。", metadata={'source': 'doc1.txt'}),
        Document(page_content="第二块文本，关于香蕉。", metadata={'source': 'doc1.txt'}),
        Document(page_content="第三块文本，关于苹果和橙子。", metadata={'source': 'doc2.txt'}),
    ]
    # Ensure a clean state for testing by removing any existing store if necessary
    # This part is tricky because VECTOR_STORE_PATH is from config, which might be shared.
    # For robust testing, use a dedicated test path.
    test_store_path = "./temp_test_vector_store"
    original_store_path = config.VECTOR_STORE_PATH 
    config.VECTOR_STORE_PATH = test_store_path

    if os.path.exists(test_store_path):
        import shutil
        print(f"Cleaning up existing test store at: {test_store_path}")
        shutil.rmtree(test_store_path)
    
    # 2. Test 1: Create a new vector store
    print("\nTest 1: Creating new vector store...")
    vs = get_or_create_vector_store(chunks=dummy_docs, force_recreate=True)
    assert vs is not None, "Test 1 Failed: Vector store creation returned None."
    print("Test 1 Passed: Vector store created.")

    # 3. Test 2: Load the existing vector store (not forcing recreate)
    print("\nTest 2: Loading existing vector store...")
    _vector_store_instance = None # Reset global to force reload from disk
    vs_loaded = get_or_create_vector_store()
    assert vs_loaded is not None, "Test 2 Failed: Loading existing vector store returned None."
    print("Test 2 Passed: Vector store loaded from disk.")

    # 4. Test 3: Get retriever
    print("\nTest 3: Getting retriever...")
    retriever = get_retriever_from_store(vs_loaded)
    assert retriever is not None, "Test 3 Failed: Could not get retriever."
    print("Test 3 Passed: Retriever obtained.")

    # 5. Test 4: Use retriever (simple query)
    print("\nTest 4: Testing retriever with a query...")
    if retriever:
        query = "关于苹果的信息"
        try:
            results = retriever.invoke(query)
            print(f"Retrieved {len(results)} documents for query '{query}':")
            for i, doc_res in enumerate(results):
                print(f"  Result {i+1}: {doc_res.page_content[:50]}... (Source: {doc_res.metadata.get('source')})")
            assert len(results) > 0, "Test 4 Assertion: Retriever did not return any results for a relevant query."
        except Exception as e:
            print(f"Error during retriever test: {e}")
            assert False, f"Test 4 Failed due to retriever error: {e}"
    print("Test 4 Passed (or skipped if retriever was None).")
    
    # 6. Test 5: Attempt to load non-existent store without chunks
    print("\nTest 5: Attempting to load non-existent store without chunks...")
    _vector_store_instance = None # Reset global
    if os.path.exists(test_store_path):
        import shutil
        shutil.rmtree(test_store_path) # Delete the store first
    vs_non_existent = get_or_create_vector_store()
    assert vs_non_existent is None, "Test 5 Failed: Should return None for non-existent store and no chunks."
    print("Test 5 Passed: Correctly handled non-existent store without chunks.")
    
    # Clean up and reset config
    if os.path.exists(test_store_path):
        import shutil
        shutil.rmtree(test_store_path)
    config.VECTOR_STORE_PATH = original_store_path
    _vector_store_instance = None # Clear global instance

    print("\n--- Vector Store Manager Testing Completed ---") 