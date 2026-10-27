import os
from dotenv import load_dotenv

from tavily import TavilyClient
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
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
# 3. CREATE GEMINI EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=gemini_api_key
)


# ============================================================
# 4. CREATE GEMINI ANALYST MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=gemini_api_key
)


# ============================================================
# 5. CREATE CHROMA VECTOR DATABASE
# ============================================================

vector_store = Chroma(
    collection_name="research_documents_v2",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# ============================================================
# 6. GET RESEARCH TOPIC
# ============================================================

topic = input(
    "\nEnter your research topic: "
).strip()

if not topic:
    print("ERROR: Research topic cannot be empty.")
    exit()


# ============================================================
# 7. SEARCH WEB
# ============================================================

print("\nSearching the web...")
print("-" * 60)

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
# 8. PREPARE RESEARCH CHUNKS
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

    # Split content into chunks
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
    f"Created {len(documents)} research chunks."
)


# ============================================================
# 9. STORE RESEARCH IN CHROMA
# ============================================================

print("\nStoring research in ChromaDB...")
print("-" * 60)

vector_store.add_texts(
    texts=documents,
    metadatas=metadatas
)

print(
    f"Stored {len(documents)} research chunks."
)


# ============================================================
# 10. CREATE USER RESEARCH QUESTION
# ============================================================

query = input(
    "\nWhat would you like the Analyst Agent to analyze? "
).strip()

if not query:
    print("ERROR: Question cannot be empty.")
    exit()


# ============================================================
# 11. RETRIEVE RELEVANT RESEARCH
# ============================================================

print("\nRetrieving relevant research...")
print("-" * 60)

retrieved_documents = vector_store.similarity_search(
    query,
    k=3
)

if not retrieved_documents:
    print("No relevant research was found.")
    exit()


# ============================================================
# 12. COMBINE RETRIEVED RESEARCH
# ============================================================

retrieved_text = "\n\n".join(
    document.page_content
    for document in retrieved_documents
)


# ============================================================
# 13. CREATE ANALYST PROMPT
# ============================================================

prompt = f"""
You are the Analyst Agent in an Autonomous Research Assistant.

Research Topic:
{topic}

User's Research Question:
{query}

Retrieved Research:
{retrieved_text}

Analyze ONLY the retrieved research provided above.

Do not invent facts.
Do not use information that is not present in the retrieved research.

Create a structured analysis using:

1. Main Findings
2. Important Facts
3. Key Patterns and Insights
4. Benefits and Applications
5. Risks and Challenges
6. Evidence from Sources
7. Overall Analysis

Keep the analysis clear and concise.

If the retrieved research does not contain enough information
for a section, write:

"Insufficient information in the retrieved research."
"""


# ============================================================
# 14. RUN ANALYST AGENT
# ============================================================

print("\nAnalyst Agent is analyzing...")
print("-" * 60)

analysis_response = llm.invoke(prompt)


# ============================================================
# 15. DISPLAY FINAL ANALYSIS
# ============================================================

print("\nANALYST AGENT REPORT")
print("=" * 60)

if isinstance(analysis_response.content, list):

    for item in analysis_response.content:

        if item.get("type") == "text":
            print(item.get("text", ""))

else:
    print(analysis_response.content)


print("\n" + "=" * 60)
print("RAG + Analyst pipeline completed successfully!")