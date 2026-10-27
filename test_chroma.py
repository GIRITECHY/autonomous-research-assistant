import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY was not found.")
    exit()

# Create Gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=api_key
)

# Create sample research documents
documents = [
    "Artificial Intelligence can assist doctors in medical diagnosis.",
    "Machine learning can analyze large amounts of healthcare data.",
    "AI systems can help detect diseases from medical images.",
    "Artificial Intelligence can support personalized treatment planning.",
]

# Create Chroma vector database
vector_store = Chroma(
    collection_name="research_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Add documents
vector_store.add_texts(documents)

print("\nDocuments successfully stored in Chroma!")

# Test semantic search
query = "How can AI help doctors?"

results = vector_store.similarity_search(query, k=2)

print("\nSearch Results:\n")

for i, result in enumerate(results, start=1):
    print(f"Result {i}:")
    print(result.page_content)
    print("-" * 60)