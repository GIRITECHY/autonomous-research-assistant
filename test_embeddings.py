import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY was not found.")
    exit()


# Create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=api_key
)


# Convert text into an embedding
text = "Artificial Intelligence in Healthcare"

vector = embeddings.embed_query(text)


print("\nEmbedding test successful!")
print("Vector type:", type(vector))
print("Vector dimensions:", len(vector))
print("First 10 values:")
print(vector[:10])