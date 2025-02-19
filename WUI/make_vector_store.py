import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


# Function to process PDFs and store embeddings
def process_pdfs_to_vector_store(folder_path, vector_store_path):
    """
    Process all PDFs in a folder, generate embeddings, and store them in a local FAISS vector store.

    Args:
        folder_path (str): Path to the folder containing PDF files.
        vector_store_path (str): Path to save the FAISS vector store.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key="your API key here")
    vector_store = None

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_splits = text_splitter.split_documents(docs)
            if vector_store is None:
                vector_store = FAISS.from_documents(all_splits, embeddings)
            else:
                vector_store.add_documents(all_splits)

    if vector_store:
        vector_store.save_local(vector_store_path)
    else:
        print("No PDF files found or processed.")


if __name__ == "__main__":
    folder_location = "Research"  # Replace with your folder path containing PDF documents
    vector_store_location = "Vectorstore"  # Replace with your desired vector store path
    process_pdfs_to_vector_store(folder_location, vector_store_location)
