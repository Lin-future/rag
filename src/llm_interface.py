from langchain_openai import ChatOpenAI # For OpenAI-compatible APIs
# from langchain_community.chat_models import ... # Check for specific LLM integration if available
from langchain_core.language_models.chat_models import BaseChatModel # For type hinting
from langchain_core.utils.function_calling import convert_to_openai_function
from . import config

_llm_instance: BaseChatModel = None

def get_llm() -> BaseChatModel:
    """Initializes and returns a singleton instance of the LLM client.

    This function assumes the LLM API is compatible with the OpenAI API format
    and uses ChatOpenAI with a custom base URL and API key.
    If llm has a dedicated LangChain integration, that should be preferred.

    Returns:
        An instance of a LangChain BaseChatModel for LLM.
        Returns None if API key is not set or initialization fails.
    """
    global _llm_instance
    if _llm_instance is not None:
        # print("Returning already initialized LLM instance.")
        return _llm_instance

    if not config.LLM_API_KEY:
        print("Error: LLM_API_KEY is not set in environment variables or .env file.")
        print("LLM functionality will not be available.")
        return None

    print(f"Initializing LLM: Model - {config.LLM_CHAT_MODEL}, API Base - {config.LLM_API_BASE_URL}")
    
    try:
        # Using ChatOpenAI with custom base URL for llm if it's OpenAI API compatible.
        # You MUST verify llms's API documentation for the correct base URL, model names, and auth.
        _llm_instance = ChatOpenAI(
            model_name=config.LLM_CHAT_MODEL, 
            openai_api_base=config.LLM_API_BASE_URL,
            openai_api_key=config.LLM_API_KEY,
            temperature=0.7,  # Default temperature, can be adjusted
            # max_tokens=1024, # Optional: configure max tokens if needed
            # request_timeout=60, # Optional: set a timeout for API requests
            # streaming=False, # Set to True if you want to handle streaming responses
        )
        print("LLM (via ChatOpenAI wrapper) initialized successfully.")
        # Perform a simple test call if desired (be mindful of API costs)
        # try:
        #     print("Performing a simple test call to the LLM...")
        #     response = _llm_instance.invoke("你好")
        #     print(f"LLM Test call successful. Response: {response.content[:50]}...")
        # except Exception as test_e:
        #     print(f"LLM Test call failed: {test_e}. Check API key, base URL, and model name.")
        #     _llm_instance = None # Nullify if test fails

    except Exception as e:
        print(f"Error initializing LLM via ChatOpenAI wrapper: {e}")
        print("Please ensure:")
        print("  1. Your LLM_API_KEY is correct.")
        print("  2. LLM_API_BASE_URL points to the correct API endpoint (OpenAI-compatible if using ChatOpenAI).")
        print("  3. LLM_CHAT_MODEL is a valid model name for your API key.")
        print("  4. You have an active internet connection and can reach the API endpoint.")
        _llm_instance = None # Ensure instance is None if initialization fails

    return _llm_instance

# Example usage (can be run directly for testing this module)
if __name__ == '__main__':
    print("--- Testing LLM Interface ---")
    # Ensure API key is loaded from .env for this direct test if you have one
    # from dotenv import load_dotenv
    # load_dotenv() # Already called in config.py if this module imports it
    
    # print(f"API Key from config: {'SET' if config.LLM_API_KEY else 'NOT SET'}")
    # print(f"API Base from config: {config.LLM_API_BASE_URL}")
    # print(f"Model from config: {config.LLM_CHAT_MODEL}")

    llm_client = get_llm()

    if llm_client:
        print("Successfully retrieved LLM client instance.")
        test_query = "简单介绍一下什么是大型语言模型（LLM）？请用中文回答。"
        print(f"Attempting to invoke LLM with query: '{test_query}'")
        try:
            response = llm_client.invoke(test_query)
            # Assuming response is AIMessage or similar with a 'content' attribute
            if hasattr(response, 'content'):
                print(f"LLM Response (first 100 chars): {response.content[:100]}...")
            else:
                print(f"LLM Response (raw): {response}")
            print("LLM invocation test completed.")
        except Exception as e:
            print(f"Error invoking LLM: {e}")
            print("This might be due to an invalid API key, network issues, or incorrect API endpoint/model configuration.")
    else:
        print("Failed to retrieve LLM client instance. Check API key and configurations.")

    print("--- LLM Interface Testing Completed ---") 