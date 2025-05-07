from langchain_community.embeddings import HuggingFaceEmbeddings # Updated import path
from langchain_core.embeddings import Embeddings # For type hinting
from . import config

_embedding_model_instance: Embeddings = None

def get_embedding_model() -> Embeddings:
    """Initializes and returns a singleton instance of the embedding model.

    This function ensures that the embedding model is loaded only once.
    The model name and device are configured in config.py.

    Returns:
        An instance of the LangChain Embeddings class.
    """
    global _embedding_model_instance
    if _embedding_model_instance is None:
        print(f"Initializing embedding model: {config.EMBEDDING_MODEL_NAME}")
        # Determine device, default to CPU if not specified or invalid
        # You can add more sophisticated device checking (e.g., torch.cuda.is_available())
        # For simplicity, we assume 'cpu' or 'cuda' might be set in model_kwargs if needed.
        model_kwargs = {'device': 'cpu'} # Default to CPU
        # If you have a GPU and want to use it:
        # import torch
        # if torch.cuda.is_available():
        #    model_kwargs = {'device': 'cuda'}
        # else:
        #    print("CUDA not available, using CPU for embeddings.")

        encode_kwargs = {
            'normalize_embeddings': True # Normalizing embeddings can improve similarity search results
        }
        
        try:
            _embedding_model_instance = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL_NAME,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            print(f"Embedding model '{config.EMBEDDING_MODEL_NAME}' loaded successfully on device '{model_kwargs['device']}'.")
        except Exception as e:
            print(f"Error loading embedding model '{config.EMBEDDING_MODEL_NAME}': {e}")
            print("Please ensure the model name is correct and you have an internet connection if downloading.")
            print("Falling back to a default CPU configuration attempt if possible or raise error.")
            # Potentially try a more fail-safe default or raise the error
            # For now, we let it propagate if the primary attempt fails with specified model
            raise # Re-raise the exception if loading fails
            
    return _embedding_model_instance

# Example usage (can be run directly for testing this module)
if __name__ == '__main__':
    print("--- Testing Embedding Model Loading ---")
    try:
        embedding_fn = get_embedding_model()
        if embedding_fn:
            print("Successfully retrieved embedding model instance.")
            # Test embedding a sample text
            sample_text = "这是一个用于测试嵌入模型的句子。"
            print(f"Attempting to embed sample text: '{sample_text}'")
            try:
                embedding_vector = embedding_fn.embed_query(sample_text)
                print(f"Successfully embedded text. Vector dimension: {len(embedding_vector)}")
                # print(f"Sample vector (first 5 dims): {embedding_vector[:5]}")
            except Exception as e:
                print(f"Error embedding sample text: {e}")
        else:
            print("Failed to retrieve embedding model instance.")
    except Exception as e:
        print(f"An error occurred during the embedding model test: {e}") 