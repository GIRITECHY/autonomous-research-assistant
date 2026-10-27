import os
import re
import uuid
from typing import TypedDict
from urllib.parse import urlparse

from dotenv import load_dotenv
from tavily import TavilyClient

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY is missing from .env")


# ============================================================
# 2. INITIALIZE GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=gemini_api_key,
)


# ============================================================
# 3. INITIALIZE TAVILY
# ============================================================

tavily_client = TavilyClient(api_key=tavily_api_key)


# ============================================================
# 4. INITIALIZE GEMINI EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=gemini_api_key,
)


# ============================================================
# 5. CHROMA DATABASE
# ============================================================

vectorstore = Chroma(
    collection_name="langgraph_research",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)


# ============================================================
# 6. LANGGRAPH STATE
# ============================================================

class ResearchState(TypedDict):
    topic: str
    run_id: str
    research_results: list
    verified_sources: list
    retrieved_documents: list
    analysis: str
    final_report: str


# ============================================================
# 7. HELPER FUNCTION
# ============================================================

def get_response_text(response):
    """
    Converts Gemini response content into normal text.
    Handles both string and structured content.
    """

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))

        return "\n".join(parts)

    return str(content)


# ============================================================
# 8. RESEARCHER AGENT
# ============================================================

def researcher_node(state: ResearchState):

    topic = state["topic"]

    print("\n" + "=" * 70)
    print("RESEARCHER AGENT")
    print("=" * 70)

    print(f"Research topic: {topic}")

    # --------------------------------------------------------
    # Generate multiple search queries using Gemini
    # --------------------------------------------------------

    query_prompt = f"""
You are a research assistant.

Research topic:
{topic}

Generate exactly 5 different web search queries that will help
research this topic from different perspectives.

Consider:
- Definition / background
- Important facts
- Applications
- Benefits
- Risks / challenges

Return ONLY the 5 queries, one per line.
Do not number them.
"""

    response = llm.invoke(query_prompt)

    query_text = get_response_text(response)

    queries = [
        line.strip()
        for line in query_text.splitlines()
        if line.strip()
    ]

    # Keep only 5 queries
    queries = queries[:5]

    print("\nGenerated search queries:")

    for i, query in enumerate(queries, 1):
        print(f"{i}. {query}")

    # --------------------------------------------------------
    # Search Tavily
    # --------------------------------------------------------

    all_results = []

    for query in queries:

        print(f"\nSearching: {query}")

        try:
            search_results = tavily_client.search(
                query=query,
                max_results=3,
            )

            for result in search_results.get("results", []):

                title = result.get("title", "").strip()
                url = result.get("url", "").strip()
                content = result.get("content", "").strip()

                all_results.append(
                    {
                        "title": title,
                        "url": url,
                        "content": content[:3000],
                    }
                )

        except Exception as e:
            print(f"Search error: {e}")

    # --------------------------------------------------------
    # Basic deduplication
    # --------------------------------------------------------

    unique_results = []
    seen_urls = set()

    for result in all_results:

        url = result["url"].rstrip("/").lower()

        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)

    # Keep a manageable number of sources
    unique_results = unique_results[:10]

    print("\n" + "-" * 70)
    print(f"Researcher found {len(unique_results)} unique sources.")
    print("-" * 70)

    for i, result in enumerate(unique_results, 1):
        print(f"{i}. {result['title']}")
        print(f"   {result['url']}")

    return {
        **state,
        "research_results": unique_results,
    }


# ============================================================
# 9. SOURCE VERIFICATION
# ============================================================

def is_valid_url(url):
    """
    Checks whether the URL has HTTP/HTTPS scheme and a domain.
    """

    try:
        parsed = urlparse(url)

        return (
            parsed.scheme in ["http", "https"]
            and bool(parsed.netloc)
        )

    except Exception:
        return False


def normalize_url(url):
    """
    Normalizes URLs for duplicate detection.
    """

    url = url.strip().lower()

    if url.endswith("/"):
        url = url[:-1]

    return url


def verify_source(source):
    """
    Performs technical source verification.

    This does NOT prove that the information is factually true.
    """

    title = source.get("title", "").strip()
    url = source.get("url", "").strip()
    content = source.get("content", "").strip()

    reasons = []

    # --------------------------------------------------------
    # Check title
    # --------------------------------------------------------

    if not title:
        reasons.append("Missing title")

    # --------------------------------------------------------
    # Check URL
    # --------------------------------------------------------

    if not url:
        reasons.append("Missing URL")

    elif not is_valid_url(url):
        reasons.append("Invalid URL")

    # --------------------------------------------------------
    # Check HTTPS
    # --------------------------------------------------------

    if url.startswith("http://"):
        reasons.append("URL does not use HTTPS")

    # --------------------------------------------------------
    # Check content
    # --------------------------------------------------------

    if not content:
        reasons.append("Missing source content")

    elif len(content) < 100:
        reasons.append("Source content is too short")

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        **source,
        "verified": len(reasons) == 0,
        "verification_reasons": reasons,
    }


def source_verification_node(state: ResearchState):

    print("\n" + "=" * 70)
    print("SOURCE VERIFICATION AGENT")
    print("=" * 70)

    research_results = state["research_results"]

    verified_sources = []

    seen_urls = set()

    for source in research_results:

        checked_source = verify_source(source)

        url = normalize_url(
            checked_source.get("url", "")
        )

        # ----------------------------------------------------
        # Duplicate URL check
        # ----------------------------------------------------

        if url and url in seen_urls:

            checked_source["verified"] = False
            checked_source["verification_reasons"].append(
                "Duplicate URL"
            )

        # ----------------------------------------------------
        # Keep valid source
        # ----------------------------------------------------

        if checked_source["verified"]:

            seen_urls.add(url)
            verified_sources.append(checked_source)

            print("\n✓ VERIFIED")
            print(f"  Title : {checked_source['title']}")
            print(f"  URL   : {checked_source['url']}")

        else:

            print("\n✗ REJECTED")
            print(f"  Title : {checked_source.get('title', 'Unknown')}")
            print(f"  URL   : {checked_source.get('url', 'Unknown')}")
            print(
                "  Reason:",
                ", ".join(
                    checked_source["verification_reasons"]
                ),
            )

    print("\n" + "-" * 70)
    print(
        f"Verified sources: {len(verified_sources)} "
        f"/ {len(research_results)}"
    )
    print("-" * 70)

    if not verified_sources:
        raise ValueError(
            "No valid sources remained after verification."
        )

    return {
        **state,
        "verified_sources": verified_sources,
    }


# ============================================================
# 10. RAG NODE
# ============================================================

def rag_node(state: ResearchState):

    print("\n" + "=" * 70)
    print("RAG SYSTEM")
    print("=" * 70)

    topic = state["topic"]
    run_id = state["run_id"]
    verified_sources = state["verified_sources"]

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    documents = []
    metadatas = []
    ids = []

    for source_index, source in enumerate(verified_sources):

        content = source["content"]

        chunks = text_splitter.split_text(content)

        for chunk_index, chunk in enumerate(chunks):

            documents.append(chunk)

            metadatas.append(
                {
                    "title": source["title"],
                    "url": source["url"],
                    "topic": topic,
                    "run_id": run_id,
                    "source_index": source_index,
                }
            )

            ids.append(
                f"{run_id}_{source_index}_{chunk_index}"
            )

    print(f"Created {len(documents)} research chunks.")

    # --------------------------------------------------------
    # Store chunks
    # --------------------------------------------------------

    if documents:

        vectorstore.add_texts(
            texts=documents,
            metadatas=metadatas,
            ids=ids,
        )

    print(f"Stored {len(documents)} chunks.")

    # --------------------------------------------------------
    # Retrieve ONLY current run
    # --------------------------------------------------------

    retrieved = vectorstore.similarity_search(
        topic,
        k=15,
        filter={
            "run_id": run_id
        },
    )

    print(
        f"Retrieved {len(retrieved)} "
        f"candidate chunks from current run."
    )

    # --------------------------------------------------------
    # Remove duplicate sources
    # --------------------------------------------------------

    unique_sources = []
    seen_urls = set()

    for doc in retrieved:

        url = doc.metadata.get("url", "").strip()

        normalized = normalize_url(url)

        if normalized in seen_urls:
            continue

        seen_urls.add(normalized)

        unique_sources.append(
            {
                "title": doc.metadata.get(
                    "title",
                    "Unknown"
                ),
                "url": url,
                "content": doc.page_content,
            }
        )

        if len(unique_sources) >= 3:
            break

    print(
        f"Selected {len(unique_sources)} "
        f"unique sources for analysis."
    )

    # --------------------------------------------------------
    # Display retrieved sources
    # --------------------------------------------------------

    print("\nRetrieved sources:")

    for i, source in enumerate(unique_sources, 1):

        print(f"\nSource {i}")
        print(f"Title: {source['title']}")
        print(f"URL  : {source['url']}")

    return {
        **state,
        "retrieved_documents": unique_sources,
    }


# ============================================================
# 11. ANALYST AGENT
# ============================================================

def analyst_node(state: ResearchState):

    print("\n" + "=" * 70)
    print("ANALYST AGENT")
    print("=" * 70)

    topic = state["topic"]
    documents = state["retrieved_documents"]

    research_text = ""

    for i, document in enumerate(documents, 1):

        research_text += f"""
SOURCE {i}

Title:
{document['title']}

URL:
{document['url']}

Content:
{document['content']}

----------------------------------------
"""

    prompt = f"""
You are the Analyst Agent in an Autonomous Research Assistant.

Research topic:
{topic}

Below is retrieved research from verified sources.

{research_text}

Analyze ONLY the information provided above.

Do not invent facts.
Do not invent statistics.
Do not invent sources.
Do not use information that is not present in the retrieved research.

Prepare the analysis using exactly these sections:

1. Main Findings
2. Important Facts
3. Key Patterns and Insights
4. Benefits and Applications
5. Risks and Challenges
6. Evidence from Sources
7. Overall Analysis

In "Evidence from Sources", include the actual source
title and URL wherever relevant.

If there is not enough information for a section,
clearly say:

"Insufficient information in the retrieved research."
"""

    response = llm.invoke(prompt)

    analysis = get_response_text(response)

    print("\nAnalyst completed successfully.")

    return {
        **state,
        "analysis": analysis,
    }


# ============================================================
# 12. WRITER AGENT
# ============================================================

def writer_node(state: ResearchState):

    print("\n" + "=" * 70)
    print("WRITER AGENT")
    print("=" * 70)

    topic = state["topic"]
    analysis = state["analysis"]
    documents = state["retrieved_documents"]

    source_text = ""

    for i, document in enumerate(documents, 1):

        source_text += f"""
SOURCE {i}

Title:
{document['title']}

URL:
{document['url']}

Evidence:
{document['content']}

----------------------------------------
"""

    prompt = f"""
You are the Writer Agent in an Autonomous Research Assistant.

Research Topic:
{topic}

ANALYST OUTPUT:
{analysis}

SOURCE INFORMATION:
{source_text}

Create a professional structured research report.

Use ONLY the information provided.

Do not:
- invent facts
- invent statistics
- invent citations
- invent URLs
- invent sources

Preserve the real source URLs.

Use these sections:

1. Executive Summary
2. Introduction
3. Key Findings
4. Important Facts and Insights
5. Benefits and Applications
6. Risks and Challenges
7. Overall Analysis
8. Conclusion
9. Source Evidence

For Source Evidence, list the actual source title
and URL.

Make the report clear, professional and readable.
"""

    response = llm.invoke(prompt)

    final_report = get_response_text(response)

    print("\nWriter completed successfully.")

    return {
        **state,
        "final_report": final_report,
    }


# ============================================================
# 13. BUILD LANGGRAPH
# ============================================================

builder = StateGraph(ResearchState)

builder.add_node(
    "researcher",
    researcher_node
)

builder.add_node(
    "source_verification",
    source_verification_node
)

builder.add_node(
    "rag",
    rag_node
)

builder.add_node(
    "analyst",
    analyst_node
)

builder.add_node(
    "writer",
    writer_node
)


# ============================================================
# 14. GRAPH FLOW
# ============================================================

builder.add_edge(
    START,
    "researcher"
)

builder.add_edge(
    "researcher",
    "source_verification"
)

builder.add_edge(
    "source_verification",
    "rag"
)

builder.add_edge(
    "rag",
    "analyst"
)

builder.add_edge(
    "analyst",
    "writer"
)

builder.add_edge(
    "writer",
    END
)


graph = builder.compile()


# ============================================================
# 15. USER INPUT
# ============================================================

print("\n")
print("=" * 70)
print("AUTONOMOUS RESEARCH ASSISTANT")
print("=" * 70)

topic = input(
    "\nEnter your research topic: "
).strip()

if not topic:
    raise ValueError(
        "Research topic cannot be empty."
    )


# ============================================================
# 16. CREATE UNIQUE RUN ID
# ============================================================

run_id = uuid.uuid4().hex[:8]

print(f"\nResearch Run ID: {run_id}")


# ============================================================
# 17. RUN LANGGRAPH
# ============================================================

initial_state = {
    "topic": topic,
    "run_id": run_id,
    "research_results": [],
    "verified_sources": [],
    "retrieved_documents": [],
    "analysis": "",
    "final_report": "",
}


final_state = graph.invoke(initial_state)


# ============================================================
# 18. FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL RESEARCH REPORT")
print("=" * 70)

print("\n")
print(final_state["final_report"])

print("\n")
print("=" * 70)
print("RESEARCH COMPLETED")
print("=" * 70)