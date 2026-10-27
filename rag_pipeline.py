import os
from dotenv import load_dotenv

from tavily import TavilyClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# ============================================================
# 1. LOAD API KEYS
# ============================================================

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not gemini_api_key:
    print("ERROR: GEMINI_API_KEY was not found.")
    exit()

if not tavily_api_key:
    print("ERROR: TAVILY_API_KEY was not found.")
    exit()


# ============================================================
# 2. CREATE TAVILY CLIENT
# ============================================================

tavily_client = TavilyClient(
    api_key=tavily_api_key
)


# ============================================================
# 3. CREATE GEMINI EMBEDDING MODEL
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=gemini_api_key
)


# ============================================================
# 4. CREATE CHROMA VECTOR DATABASE
# ============================================================

vector_store = Chroma(
    collection_name="research_documents_v2",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# ============================================================
# 5. GET RESEARCH TOPIC
# ============================================================

topic = input(
    "\nEnter your research topic: "
).strip()

if not topic:
    print("ERROR: Research topic cannot be empty.")
    exit()


print("\nSearching the web...")
print("-" * 60)


# ============================================================
# 6. SEARCH USING TAVILY
# ============================================================

response = tavily_client.search(
    query=topic,
    max_results=3
)

results = response.get("results", [])

if not results:
    print("No search results were found.")
    exit()

print(f"Found {len(results)} web sources.")


# ============================================================
# 7. PREPARE SMALL RESEARCH CHUNKS
# ============================================================

documents = []
metadatas = []

for result in results:

    title = result.get("title", "")
    url = result.get("url", "")
    content = result.get("content", "")

    if not content:
        continue

    # Limit content size
    content = content[:3000]

    # Split content into smaller chunks
    chunk_size = 1000

    for start in range(0, len(content), chunk_size):

        chunk = content[start:start + chunk_size]

        if not chunk.strip():
            continue

        document_text = (
            f"Title: {title}\n"
            f"URL: {url}\n\n"
            f"Content:\n{chunk}"
        )

        documents.append(document_text)

        metadatas.append({
            "title": title,
            "url": url,
            "topic": topic
        })


print(
    f"Created {len(documents)} smaller research chunks."
)


# ============================================================
# 8. STORE CHUNKS IN CHROMA
# ============================================================

print("\nCreating embeddings and storing research...")
print("-" * 60)

vector_store.add_texts(
    texts=documents,
    metadatas=metadatas
)

print(
    f"Stored {len(documents)} research chunks in ChromaDB."
)


# ============================================================
# 9. SEMANTIC RETRIEVAL
# ============================================================

query = input(
    "\nEnter a question about this research: "
).strip()

if not query:
    print("ERROR: Question cannot be empty.")
    exit()


print("\nSearching the research database...")
print("-" * 60)


retrieved_documents = vector_store.similarity_search(
    query,
    k=3
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

if not retrieved_documents:
    print("No relevant research was found.")
    exit()


print("\nRelevant Research:\n")

for i, document in enumerate(
    retrieved_documents,
    start=1
):

    print(f"Result {i}")
    print("=" * 60)

    print(document.page_content)

    print("\nSource:")
    print(document.metadata.get("url", "Unknown"))

    print("-" * 60)


print("\nRAG pipeline test completed successfully!")