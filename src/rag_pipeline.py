from typing import Dict, List, Tuple, Optional

from langchain_core.documents import Document # Updated import path
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever
import os

from . import config

# Chinese prompt template for RAG
RAG_PROMPT_TEMPLATE_ZH = """
你是一个专注于回答用户提问的AI助手。
请根据下面提供的 "上下文信息" 来回答 "用户问题"。
你的回答应该完全基于这些上下文信息，确保答案准确、简洁、专业且不包含任何在上下文之外的推测或信息。
如果上下文中没有足够的信息来回答问题，请明确告知："根据提供的上下文信息，我无法回答该问题。"
请务必使用中文进行回答。

上下文信息:
{context}

用户问题:
{question}

回答:
"""

# Chinese prompt template for LLM-only (no RAG)
LLM_ONLY_PROMPT_TEMPLATE_ZH = """
你是一个乐于助人的AI助手。
请直接、准确地回答下面的用户问题。
请使用中文进行回答。

用户问题:
{question}

回答:
"""

def format_docs_for_context(docs: List[Document]) -> str:
    """Formats a list of retrieved documents into a single string for the context.

    Args:
        docs: A list of LangChain Document objects.

    Returns:
        A string concatenating the page content of the documents, separated by newlines.
    """
    if not docs:
        return "没有在知识库中找到相关信息。"
    # Joining content with a clear separator. Adding source and page number can be useful.
    # For simplicity here, just joining page_content.
    # Consider adding metadata like "Source: {doc.metadata.get('source', 'N/A')}" if useful
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

class RAGSystem:
    """Encapsulates the Retrieval-Augmented Generation (RAG) system logic."""

    def __init__(self, retriever: VectorStoreRetriever, llm: BaseChatModel):
        """Initializes the RAGSystem.

        Args:
            retriever: A LangChain VectorStoreRetriever instance for document retrieval.
            llm: A LangChain BaseChatModel instance (e.g., configured LLM).
        """
        if not retriever or not llm:
            raise ValueError("Retriever and LLM must be provided and initialized.")
        self.retriever = retriever
        self.llm = llm

        # --- RAG Chain Setup ---
        rag_prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE_ZH,
            input_variables=["context", "question"],
        )

        # LCEL (LangChain Expression Language) chain for RAG
        # 1. Retrieve documents based on the question.
        # 2. Format the retrieved documents into a context string.
        # 3. Pass the context and original question to the prompt.
        # 4. Pass the formatted prompt to the LLM.
        # 5. Parse the LLM output.
        self.rag_chain = (
            {
                "context": RunnableLambda(lambda x: format_docs_for_context(self.retriever.invoke(x["question"]))),
                "question": RunnablePassthrough() # Passes the original question through
            }
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )

        # --- LLM-Only Chain Setup (for comparison) ---
        llm_only_prompt = PromptTemplate(
            template=LLM_ONLY_PROMPT_TEMPLATE_ZH,
            input_variables=["question"],
        )
        self.llm_only_chain = (
            {"question": RunnablePassthrough()} 
            | llm_only_prompt 
            | self.llm 
            | StrOutputParser()
        )
        print("RAGSystem initialized with RAG chain and LLM-only chain.")

    def ask_with_rag(self, question: str) -> Dict[str, any]:
        """Answers a question using the RAG pipeline, including retrieved documents.

        Args:
            question: The user's question.

        Returns:
            A dictionary containing the question, the RAG answer, 
            and the list of retrieved documents.
        """
        print(f"Processing RAG query: '{question[:50]}...'")
        # Invoke the RAG chain. The chain internally handles retrieval.
        # For returning documents, we might need to run retriever separately or adjust chain.
        
        # Option 1: Run retriever separately to get docs for the return dict
        retrieved_docs = self.retriever.invoke(question)
        formatted_context = format_docs_for_context(retrieved_docs)
        
        # Now invoke the chain using a slightly modified setup if we pass context directly
        # Or, ensure the main rag_chain uses the input question to fetch context as defined.
        # The current self.rag_chain is set up to take `question` and derive `context`.
        answer = self.rag_chain.invoke({"question": question})
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "formatted_context": formatted_context # The actual context string passed to LLM
        }

    def ask_llm_only(self, question: str) -> str:
        """Answers a question using only the LLM (no RAG).

        Args:
            question: The user's question.

        Returns:
            The LLM's direct answer as a string.
        """
        print(f"Processing LLM-only query: '{question[:50]}...'")
        return self.llm_only_chain.invoke({"question": question})

# Example usage (can be run directly for testing this module)
if __name__ == '__main__':
    print("--- Testing RAG Pipeline ---")

    # Mocking components for standalone testing
    # In a real scenario, these would be initialized by main.py or a similar orchestrator
    from .embedding_utils import get_embedding_model
    from .vector_store_manager import get_or_create_vector_store, get_retriever_from_store
    from .llm_interface import get_llm

    print("Setting up mock components for RAG pipeline test...")
    # 1. Setup: Create dummy knowledge base for retriever
    dummy_kb_docs = [
        Document(page_content="DeepSeek是一家专注于人工智能研究的公司，致力于开发先进的大语言模型。", metadata={'source': 'deepseek_intro.txt'}),
        Document(page_content="大型语言模型（LLM）是基于大量文本数据训练的深度学习模型，能够理解和生成人类语言。", metadata={'source': 'llm_basics.txt'}),
        Document(page_content="中国的首都是北京，它是一座历史悠久的文化名城。", metadata={'source': 'geo_facts.txt'}),
        Document(page_content="太阳系中最大的行星是木星。", metadata={'source': 'space_facts.txt'})
    ]
    
    # Use a temporary path for testing vector store to avoid conflicts
    original_store_path = config.VECTOR_STORE_PATH
    test_store_path_pipeline = "./temp_test_vector_store_pipeline"
    config.VECTOR_STORE_PATH = test_store_path_pipeline
    if os.path.exists(test_store_path_pipeline):
        import shutil
        shutil.rmtree(test_store_path_pipeline)
    os.makedirs(os.path.dirname(test_store_path_pipeline), exist_ok=True)

    mock_retriever = None
    try:
        embedding_function = get_embedding_model() # Needs a valid model or it will fail
        if embedding_function:
            print("Mock embedding function loaded.")
            mock_vector_store = get_or_create_vector_store(chunks=dummy_kb_docs, force_recreate=True)
            if mock_vector_store:
                print("Mock vector store created.")
                mock_retriever = get_retriever_from_store(mock_vector_store)
                if mock_retriever:
                    print("Mock retriever created.")
    except Exception as e:
        print(f"Could not set up mock retriever due to: {e}. RAG test might be limited.")

    mock_llm = get_llm() # Needs API Key in .env for this test to actually call LLM
    if not mock_llm:
        print("Warning: Mock LLM (DeepSeek) could not be initialized. API Key might be missing or invalid.")
        print("LLM-dependent tests will likely fail or be skipped.")

    if mock_retriever and mock_llm:
        print("Initializing RAGSystem with mock components...")
        rag_system = RAGSystem(retriever=mock_retriever, llm=mock_llm)
        
        test_question_rag = "DeepSeek公司是做什么的？"
        print(f"\n--- Testing RAG Query: '{test_question_rag}' ---")
        rag_result = rag_system.ask_with_rag(test_question_rag)
        print(f"  Question: {rag_result['question']}")
        print(f"  RAG Answer: {rag_result['answer']}")
        print(f"  Retrieved Documents ({len(rag_result['retrieved_documents'])}):")
        for i, doc in enumerate(rag_result['retrieved_documents']):
            print(f"    Doc {i+1} (Source: {doc.metadata.get('source')}): '{doc.page_content[:50]}...'")

        test_question_no_context = "天空为什么是蓝色的？"
        print(f"\n--- Testing RAG Query (No Context in KB): '{test_question_no_context}' ---")
        rag_result_no_ctx = rag_system.ask_with_rag(test_question_no_context)
        print(f"  Question: {rag_result_no_ctx['question']}")
        print(f"  RAG Answer: {rag_result_no_ctx['answer']}") # Expected: model says it cannot answer based on context

        print(f"\n--- Testing LLM-Only Query: '{test_question_rag}' (same as RAG for comparison) ---")
        llm_only_answer = rag_system.ask_llm_only(test_question_rag)
        print(f"  LLM-Only Answer: {llm_only_answer}")
        
        # Example for a question likely not in small KB but LLM might know
        test_question_general_llm = "中国的首都是哪里？"
        print(f"\n--- Testing RAG Query (Potentially in KB): '{test_question_general_llm}' ---")
        rag_result_geo = rag_system.ask_with_rag(test_question_general_llm)
        print(f"  Question: {rag_result_geo['question']}")
        print(f"  RAG Answer: {rag_result_geo['answer']}")

        print(f"\n--- Testing LLM-Only Query: '{test_question_general_llm}' ---")
        llm_only_answer_geo = rag_system.ask_llm_only(test_question_general_llm)
        print(f"  LLM-Only Answer: {llm_only_answer_geo}")

    else:
        print("Skipping RAGSystem tests as mock retriever or LLM could not be initialized.")

    # Clean up test vector store path and reset config
    if os.path.exists(test_store_path_pipeline):
        import shutil
        shutil.rmtree(test_store_path_pipeline)
    config.VECTOR_STORE_PATH = original_store_path
    # Reset global instances from other modules if they were affected by test setup
    from . import vector_store_manager
    vector_store_manager._vector_store_instance = None 

    print("\n--- RAG Pipeline Testing Completed ---") 