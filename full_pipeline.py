import os
from dotenv import load_dotenv

from tavily import TavilyClient
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
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
# 2. CREATE CLIENTS
# ============================================================

tavily_client = TavilyClient(
    api_key=tavily_api_key
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=gemini_api_key
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=gemini_api_key
)


# ============================================================
# 3. CREATE CHROMA VECTOR STORE
# ============================================================

vector_store = Chroma(
    collection_name="full_research_pipeline",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# ============================================================
# 4. GET RESEARCH TOPIC
# ============================================================

topic = input(
    "\nEnter your research topic: "
).strip()

if not topic:
    print("ERROR: Research topic cannot be empty.")
    exit()


# ============================================================
# 5. RESEARCHER AGENT
# ============================================================

print("\n[1/4] RESEARCHER AGENT")
print("-" * 60)

print("Searching the web...")

search_response = tavily_client.search(
    query=topic,
    max_results=3
)

results = search_response.get("results", [])

if not results:
    print("No research sources found.")
    exit()

print(f"Found {len(results)} web sources.")


# ============================================================
# 6. PREPARE RESEARCH CHUNKS
# ============================================================

documents = []
metadatas = []

for result in results:

    title = result.get("title", "")
    url = result.get("url", "")
    content = result.get("content", "")

    if not content:
        continue

    # Limit source content
    content = content[:3000]

    # Split into smaller chunks
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


print(f"Created {len(documents)} research chunks.")


# ============================================================
# 7. STORE RESEARCH IN CHROMA
# ============================================================

print("\nStoring research in ChromaDB...")

vector_store.add_texts(
    texts=documents,
    metadatas=metadatas
)

print(
    f"Stored {len(documents)} research chunks."
)


# ============================================================
# 8. RAG RETRIEVAL
# ============================================================

print("\n[2/4] RAG RETRIEVAL")
print("-" * 60)

research_question = input(
    "What should the system analyze? "
).strip()

if not research_question:
    print("ERROR: Research question cannot be empty.")
    exit()

print("Retrieving relevant research...")

retrieved_documents = vector_store.similarity_search(
    research_question,
    k=3
)

if not retrieved_documents:
    print("No relevant research was retrieved.")
    exit()

print(
    f"Retrieved {len(retrieved_documents)} relevant research chunks."
)


# ============================================================
# 9. PREPARE RETRIEVED RESEARCH
# ============================================================

retrieved_text = "\n\n".join(
    document.page_content
    for document in retrieved_documents
)


# ============================================================
# 10. ANALYST AGENT
# ============================================================

print("\n[3/4] ANALYST AGENT")
print("-" * 60)

analyst_prompt = f"""
You are the Analyst Agent in an Autonomous Research Assistant.

Research Topic:
{topic}

Research Question:
{research_question}

Retrieved Research:
{retrieved_text}

Analyze ONLY the retrieved research.

Do not invent facts.
Do not introduce unsupported information.

Create a structured analysis using:

1. Main Findings
2. Important Facts
3. Key Patterns and Insights
4. Benefits and Applications
5. Risks and Challenges
6. Evidence from Sources
7. Overall Analysis

If the retrieved research does not contain enough
information for a section, state:

"Insufficient information in the retrieved research."
"""

analyst_response = llm.invoke(analyst_prompt)

if isinstance(analyst_response.content, list):

    analyst_parts = []

    for item in analyst_response.content:

        if item.get("type") == "text":
            analyst_parts.append(
                item.get("text", "")
            )

    analysis = "\n".join(analyst_parts)

else:
    analysis = analyst_response.content


print("Analysis completed.")


# ============================================================
# 11. WRITER AGENT
# ============================================================

print("\n[4/4] WRITER AGENT")
print("-" * 60)

writer_prompt = f"""
You are the Writer Agent in an Autonomous Research Assistant.

Research Topic:
{topic}

Research Question:
{research_question}

Analyst Agent Analysis:
{analysis}

Write a professional research report using ONLY
the information provided by the Analyst Agent.

Do not invent facts.
Do not create fake sources.
Do not introduce unsupported information.

Use this structure:

# {topic}

## 1. Executive Summary

## 2. Introduction

## 3. Key Findings

## 4. Important Facts and Insights

## 5. Benefits and Applications

## 6. Risks and Challenges

## 7. Overall Analysis

## 8. Conclusion

## 9. Source Evidence

Keep the report clear, professional,
and suitable for an academic project.

Preserve limitations from the Analyst Agent.
If information is insufficient, do not invent it.
"""

writer_response = llm.invoke(writer_prompt)

if isinstance(writer_response.content, list):

    writer_parts = []

    for item in writer_response.content:

        if item.get("type") == "text":
            writer_parts.append(
                item.get("text", "")
            )

    final_report = "\n".join(writer_parts)

else:
    final_report = writer_response.content


# ============================================================
# 12. DISPLAY FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL AUTONOMOUS RESEARCH REPORT")
print("=" * 70)

print(final_report)

print("\n")
print("=" * 70)
print("FULL RESEARCH PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)