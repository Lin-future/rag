import argparse
import os
import sys

# Ensure the src directory is in the Python path for direct execution of main.py
# This is more relevant if you run `python src/main.py` from the project root.
# If running as a module `python -m src.main`, this might not be strictly necessary
# but doesn't hurt.
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir) # This assumes main.py is in src/
# sys.path.insert(0, project_root)

from . import config # Loads .env, sets up paths
from . import data_processor
from . import vector_store_manager
from . import llm_interface
from .rag_pipeline import RAGSystem

# --- Global RAG System Instance ---
# This will be initialized once when needed to avoid re-initializing components repeatedly.
_rag_system_instance: RAGSystem = None

def get_rag_system_instance() -> RAGSystem:
    """Initializes and returns a singleton RAGSystem instance.
    
    This function handles the lazy initialization of the RAG system components
    (LLM, vector store, retriever) and the RAGSystem class itself.
    """
    global _rag_system_instance
    if _rag_system_instance is None:
        print("Initializing RAG system components...")
        llm = llm_interface.get_llm()
        if not llm:
            print("Error: Failed to initialize LLM. RAG system cannot operate.")
            return None # Or raise an exception

        # Attempt to load or get the vector store (does not create if not exists and no chunks)
        vector_store = vector_store_manager.get_or_create_vector_store()
        if not vector_store:
            # This is acceptable if we are only doing LLM-only queries and KB init is separate.
            # However, for RAG queries, it is an issue.
            print("Warning: Vector store not found or could not be loaded. RAG queries will fail.")
            # We might still proceed if only LLM-only queries are made, 
            # but RAGSystem constructor expects a retriever.
            # For robust RAG, ensure KB is initialized first.
            # Let's assume for now that if vector_store is None, retriever will also be None,
            # and RAGSystem might handle this or we check before creating it for RAG queries.
            # To simplify, RAGSystem strictly needs a retriever.
            print(f"Please initialize the knowledge base first using: python -m src.main --init-kb")
            return None
            
        retriever = vector_store_manager.get_retriever_from_store(vector_store)
        if not retriever:
            print("Error: Failed to create retriever from vector store. RAG system cannot operate.")
            return None

        try:
            _rag_system_instance = RAGSystem(retriever=retriever, llm=llm)
            print("RAG system initialized successfully.")
        except ValueError as ve:
            print(f"Error initializing RAGSystem: {ve}")
            return None
            
    return _rag_system_instance

def initialize_knowledge_base_cli(force_recreate: bool = False):
    """Command-line interface function to initialize the knowledge base."""
    print("Starting knowledge base initialization via CLI...")
    
    # 1. Check if data directory exists and has files
    if not os.path.exists(config.DATA_PATH) or not os.listdir(config.DATA_PATH):
        print(f"Error: Knowledge base data directory '{config.DATA_PATH}' is empty or does not exist.")
        print("Please add your source documents (e.g., .txt, .md, .pdf) to this directory.")
        return False

    # 2. Load documents
    print("Loading documents...")
    documents = data_processor.load_documents()
    if not documents:
        print("No documents were loaded. Aborting knowledge base initialization.")
        return False
    
    # 3. Split documents into chunks
    print("Splitting documents into chunks...")
    chunks = data_processor.split_documents_into_chunks(documents)
    if not chunks:
        print("Failed to split documents into chunks. Aborting knowledge base initialization.")
        return False
    
    # 4. Create/update vector store
    # This will also load the embedding model via get_embedding_model() call inside
    print("Creating or updating vector store...")
    vector_store = vector_store_manager.get_or_create_vector_store(chunks=chunks, force_recreate=force_recreate)
    
    if vector_store:
        print("Knowledge base initialization completed successfully.")
        # Clear any cached RAG system instance as its retriever might be outdated
        global _rag_system_instance
        _rag_system_instance = None 
        return True
    else:
        print("Failed to create or load vector store. Knowledge base initialization failed.")
        return False

def handle_rag_query_cli(query: str):
    """Handles a RAG query from the CLI."""
    print(f"\nProcessing RAG query via CLI: '{query[:100]}...'")
    rag_system = get_rag_system_instance()
    if not rag_system:
        print("RAG system is not available. Cannot process RAG query.")
        return

    result = rag_system.ask_with_rag(query)
    
    print("\n--- RAG System Answer ---")
    print(result["answer"])
    print("\n--- Retrieved Documents (Snippets) ---")
    if result["retrieved_documents"]:
        for i, doc in enumerate(result["retrieved_documents"]):
            source = doc.metadata.get('source', 'N/A')
            page_content_snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"\n[Document {i+1}] Source: {source}")
            print(f"  Snippet: {page_content_snippet}")
    else:
        print("No documents were retrieved for this query.")
    print("\n" + "-"*30 + "\n")

def handle_llm_only_query_cli(query: str):
    """Handles an LLM-only query from the CLI."""
    print(f"\nProcessing LLM-only query via CLI: '{query[:100]}...'")
    
    # We still use get_rag_system_instance to ensure LLM is initialized,
    # but we will call its llm_only_chain method.
    # Alternatively, get LLM directly if RAG system setup is too heavy or fails.
    llm = llm_interface.get_llm()
    if not llm:
        print("LLM is not available. Cannot process LLM-only query.")
        return
    
    # Re-create RAGSystem for ask_llm_only if not already available
    # This is a bit of a workaround if get_rag_system_instance failed due to retriever issues
    # but LLM itself is fine.
    rag_system_for_llm_only = _rag_system_instance
    if not rag_system_for_llm_only:
        # Create a dummy RAGSystem if retriever part failed but we need the llm_only_chain logic
        # This needs careful handling: RAGSystem expects a retriever.
        # Simpler: just use the llm_only_chain directly from rag_pipeline.py if possible
        # For now, let's rely on the llm_interface if RAGSystem cannot be formed.
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from .rag_pipeline import LLM_ONLY_PROMPT_TEMPLATE_ZH # Import the template
        
        llm_only_prompt = PromptTemplate(
            template=LLM_ONLY_PROMPT_TEMPLATE_ZH,
            input_variables=["question"],
        )
        direct_llm_chain = ({ "question": lambda x: x["question"] } | llm_only_prompt | llm | StrOutputParser())
        answer = direct_llm_chain.invoke({"question": query})
    else:
        answer = rag_system_for_llm_only.ask_llm_only(query)

    print("\n--- LLM-Only Answer ---")
    print(answer)
    print("\n" + "-"*30 + "\n")

def main_cli():
    """Main command-line interface function."""
    parser = argparse.ArgumentParser(
        description="A RAG (Retrieval-Augmented Generation) System using LLM.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--init-kb", 
        action="store_true", 
        help="Initialize or re-initialize the knowledge base from documents in the data directory."
    )
    parser.add_argument(
        "--force-recreate-kb",
        action="store_true",
        help="Force recreation of the knowledge base vector store, even if one already exists. Use with --init-kb."
    )
    parser.add_argument(
        "--query", 
        type=str, 
        metavar='"Your Question"' , 
        help="Ask a question using the RAG system (retrieval + LLM)."
    )
    parser.add_argument(
        "--query-llm-only", 
        type=str, 
        metavar='"Your Question"', 
        help="Ask a question directly to the LLM (without RAG retrieval), for comparison."
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        print("\nExamples:")
        print("  python -m src.main --init-kb")
        print("  python -m src.main --query \"DeepSeek公司是做什么的？\"")
        print("  python -m src.main --query-llm-only \"介绍一下RAG技术。\"")
        sys.exit(1)

    args = parser.parse_args()

    if args.init_kb:
        initialize_knowledge_base_cli(force_recreate=args.force_recreate_kb)
    
    if args.query:
        handle_rag_query_cli(args.query)
    
    if args.query_llm_only:
        handle_llm_only_query_cli(args.query_llm_only)
    
    # If no specific action was taken (e.g., only --force-recreate-kb without --init-kb)
    # or if only help was implicitly called by no args, we might have already exited.
    # This is just a fallback if somehow no action was triggered but args were parsed.
    # if not (args.init_kb or args.query or args.query_llm_only):
    #     print("No action specified. Use --help for options.")

if __name__ == "__main__":
    # This allows the script to be run directly, e.g., `python src/main.py --init-kb`
    # For module execution `python -m src.main --init-kb`, the imports work as is.
    main_cli() 