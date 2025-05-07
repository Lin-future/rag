from typing import List
import os
from langchain_community.document_loaders import ( # Updated import path
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    # Add other loaders as needed, e.g., CSVLoader, Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Updated import path
from . import config

def load_documents(directory_path: str = config.DATA_PATH) -> List[Document]:
    """Loads documents from the specified directory using various loaders.

    Args:
        directory_path: The path to the directory containing source documents.

    Returns:
        A list of loaded Document objects.
    """
    print(f"Loading documents from: {directory_path}")
    if not os.path.exists(directory_path) or not os.listdir(directory_path):
        print(f"Warning: Document directory '{directory_path}' is empty or does not exist.")
        return []

    # Using a dictionary to map glob patterns to loader classes and their kwargs
    # This makes it easier to extend with more document types
    loader_configs = {
        "**/*.txt": {"loader_cls": TextLoader, "loader_kwargs": {'encoding': 'utf-8'}},
        "**/*.md": {"loader_cls": UnstructuredMarkdownLoader, "loader_kwargs": {}},
        "**/*.pdf": {"loader_cls": PyPDFLoader, "loader_kwargs": {}},
        # Example for CSV (ensure CSVLoader is imported and relevant library installed)
        # "**/*.csv": {"loader_cls": CSVLoader, "loader_kwargs": {'encoding': 'utf-8', 'source_column': 'text_content_column_name'}},
    }

    loaded_documents: List[Document] = []
    for glob_pattern, conf in loader_configs.items():
        try:
            loader = DirectoryLoader(
                directory_path,
                glob=glob_pattern,
                loader_cls=conf["loader_cls"],
                loader_kwargs=conf["loader_kwargs"],
                show_progress=True,
                use_multithreading=True,
                silent_errors=True  # Silently ignore files that fail to load with a specific loader
            )
            docs = loader.load()
            if docs:
                loaded_documents.extend(docs)
                print(f"  Successfully loaded {len(docs)} file(s) matching '{glob_pattern}'.")
        except Exception as e:
            print(f"  Warning: Error loading files with pattern '{glob_pattern}': {e}")
            # Optionally, decide if you want to halt or continue on specific errors
    
    if not loaded_documents:
        print("No documents were successfully loaded.")
    else:
        print(f"Total documents loaded: {len(loaded_documents)}")
    return loaded_documents

def split_documents_into_chunks(documents: List[Document]) -> List[Document]:
    """Splits the loaded documents into smaller chunks for processing.

    Args:
        documents: A list of Document objects to be split.

    Returns:
        A list of chunked Document objects.
    """
    if not documents:
        print("No documents to split.")
        return []

    print(f"Splitting {len(documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,  # Adds metadata about the chunk's start index in the original doc
        separators=["\n\n", "\n", " ", "", "\uff0c", "\u3002"] # Common separators, including Chinese punctuation
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Successfully split documents into {len(chunks)} chunks.")
    return chunks

# Example usage (can be run directly for testing this module)
if __name__ == '__main__':
    # Create dummy data directory and files for testing
    test_data_path = "./temp_test_data"
    os.makedirs(test_data_path, exist_ok=True)
    with open(os.path.join(test_data_path, "test_doc1.txt"), "w", encoding="utf-8") as f:
        f.write("这是第一个测试文件。它包含一些简单的中文文本。\n这是第二行。")
    with open(os.path.join(test_data_path, "test_doc2.md"), "w", encoding="utf-8") as f:
        f.write("# 这是一个Markdown测试\n\n包含一些*Markdown*格式的文本。")
    # Simulate a PDF - this won't be readable by PyPDFLoader as a text file, 
    # but shows the loader attempting to load based on glob.
    # For actual PDF testing, place a real PDF file.
    # with open(os.path.join(test_data_path, "test_doc3.pdf"), "w", encoding="utf-8") as f:
    #     f.write("%PDF-1.4 fake content") 

    print("--- Testing Document Loading ---")
    # Override config.DATA_PATH for this test
    original_data_path = config.DATA_PATH
    config.DATA_PATH = test_data_path
    docs = load_documents()
    config.DATA_PATH = original_data_path # Reset for other modules

    if docs:
        print(f"\n--- Loaded {len(docs)} documents ---")
        for i, doc in enumerate(docs):
            print(f"Doc {i+1} (Source: {doc.metadata.get('source')}, Snippet: {doc.page_content[:50]}...)")
        
        print("\n--- Testing Document Splitting ---")
        chunks = split_documents_into_chunks(docs)
        if chunks:
            print(f"\n--- Split into {len(chunks)} chunks ---")
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i+1} (Source: {chunk.metadata.get('source')}, Start Index: {chunk.metadata.get('start_index')}, Snippet: {chunk.page_content[:50]}...)")
    else:
        print("Document loading test did not produce any documents.")

    # Clean up dummy files and directory
    # import shutil
    # shutil.rmtree(test_data_path)
    print(f"\n(Note: Test files created in {test_data_path}. Manual cleanup might be needed if not using shutil)") 