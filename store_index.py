from src.helper import load_pdf_files, split_text_into_chunks, download_huggingface_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Load and process the PDF data
print("Loading PDF files...")
extracted_data = load_pdf_files("data/")
print(f"Loaded {len(extracted_data)} pages.")

print("Splitting into chunks...")
text_chunks = split_text_into_chunks(extracted_data)
print(f"Created {len(text_chunks)} chunks.")

print("Loading embeddings model...")
embeddings = download_huggingface_embeddings()
print("Embeddings model loaded.")

# Initialize Pinecone
print("Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "textbot"
existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    print(f"Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Index '{index_name}' created.")
else:
    print(f"Index '{index_name}' already exists.")

# Upload documents to Pinecone
print("Uploading documents to Pinecone vector store...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)
print("✓ Documents successfully stored in Pinecone!")