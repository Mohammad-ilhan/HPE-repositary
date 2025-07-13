from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Step 1: Load a single PDF
def load_pdf(file_path):
    if not os.path.isfile(file_path):  # Check if the file exists
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

# Step 2: Create Chunks
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Step 3: Setup embeddings model
def get_embedding_model(Ollama_model_name):
    embeddings = OllamaEmbeddings(model=Ollama_model_name)
    return embeddings

# Step 4: Index Documents and Store in FAISS
def index_documents(text_chunks, Ollama_model_name, faiss_db_path):
    embeddings = get_embedding_model(Ollama_model_name)
    faiss_db = FAISS.from_documents(text_chunks, embeddings)
    faiss_db.save_local(faiss_db_path)
    return faiss_db

# Initialize FAISS DB globally
file_path = 'udhr.pdf'  # Ensure the file exists in the 'pdfs/' directory
Faiss_DB_PATH = "vectorstore/db_faiss"
Ollama_model_name = "deepseek-r1:7b"

# Load the PDF
documents = load_pdf(file_path)
print(f"Loaded {len(documents)} pages from the PDF.")

# Create text chunks
text_chunks = create_chunks(documents)
print(f"Created {len(text_chunks)} text chunks.")

# Index documents and save to FAISS
faiss_db = index_documents(text_chunks, Ollama_model_name, Faiss_DB_PATH)
print("FAISS vector store saved successfully.")