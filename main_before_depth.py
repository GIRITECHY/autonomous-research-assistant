import os
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

tavily_client = TavilyClient(
    api_key=tavily_api_key
)


# ============================================================
# 4. INITIALIZE EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=gemini_api_key,
)


# ============================================================
# 5. CHROMA
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
# 7. RESPONSE TEXT HELPER
# ============================================================

def get_response_text(response):

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
                    parts.append(
                        str(item["text"])
                    )

        return "\n".join(parts)

    return str(content)


# ============================================================
# 8. RESEARCHER AGENT
# ============================================================

def researcher_node(state: ResearchState):

    topic = state["topic"]

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

    response = llm.invoke(
        query_prompt
    )

    query_text = get_response_text(
        response
    )

    queries = [
        line.strip()
        for line in query_text.splitlines()
        if line.strip()
    ]

    queries = queries[:5]

    all_results = []

    for query in queries:

        try:

            search_results = tavily_client.search(
                query=query,
                max_results=3,
            )

            for result in search_results.get(
                "results",
                []
            ):

                title = result.get(
                    "title",
                    ""
                ).strip()

                url = result.get(
                    "url",
                    ""
                ).strip()

                content = result.get(
                    "content",
                    ""
                ).strip()

                all_results.append(
                    {
                        "title": title,
                        "url": url,
                        "content": content[:3000],
                    }
                )

        except Exception as e:

            print(
                f"Search error: {e}"
            )

    # Remove duplicate URLs

    unique_results = []

    seen_urls = set()

    for result in all_results:

        url = result["url"].rstrip(
            "/"
        ).lower()

        if url and url not in seen_urls:

            seen_urls.add(url)

            unique_results.append(
                result
            )

    unique_results = unique_results[:10]

    return {
        **state,
        "research_results": unique_results,
    }


# ============================================================
# 9. SOURCE VERIFICATION
# ============================================================

def is_valid_url(url):

    try:

        parsed = urlparse(url)

        return (
            parsed.scheme
            in ["http", "https"]
            and bool(parsed.netloc)
        )

    except Exception:

        return False


def normalize_url(url):

    url = url.strip().lower()

    if url.endswith("/"):

        url = url[:-1]

    return url


def verify_source(source):

    title = source.get(
        "title",
        ""
    ).strip()

    url = source.get(
        "url",
        ""
    ).strip()

    content = source.get(
        "content",
        ""
    ).strip()

    reasons = []

    if not title:

        reasons.append(
            "Missing title"
        )

    if not url:

        reasons.append(
            "Missing URL"
        )

    elif not is_valid_url(url):

        reasons.append(
            "Invalid URL"
        )

    if url.startswith(
        "http://"
    ):

        reasons.append(
            "URL does not use HTTPS"
        )

    if not content:

        reasons.append(
            "Missing source content"
        )

    elif len(content) < 100:

        reasons.append(
            "Source content is too short"
        )

    return {
        **source,
        "verified": len(reasons) == 0,
        "verification_reasons": reasons,
    }


def source_verification_node(
    state: ResearchState
):

    research_results = state[
        "research_results"
    ]

    verified_sources = []

    seen_urls = set()

    for source in research_results:

        checked_source = verify_source(
            source
        )

        url = normalize_url(
            checked_source.get(
                "url",
                ""
            )
        )

        if url and url in seen_urls:

            checked_source[
                "verified"
            ] = False

            checked_source[
                "verification_reasons"
            ].append(
                "Duplicate URL"
            )

        if checked_source[
            "verified"
        ]:

            seen_urls.add(url)

            verified_sources.append(
                checked_source
            )

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

    topic = state["topic"]

    run_id = state["run_id"]

    verified_sources = state[
        "verified_sources"
    ]

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
    )

    documents = []

    metadatas = []

    ids = []

    for source_index, source in enumerate(
        verified_sources
    ):

        content = source[
            "content"
        ]

        chunks = text_splitter.split_text(
            content
        )

        for chunk_index, chunk in enumerate(
            chunks
        ):

            documents.append(
                chunk
            )

            metadatas.append(
                {
                    "title": source[
                        "title"
                    ],
                    "url": source[
                        "url"
                    ],
                    "topic": topic,
                    "run_id": run_id,
                    "source_index": source_index,
                }
            )

            ids.append(
                f"{run_id}_{source_index}_{chunk_index}"
            )

    if documents:

        vectorstore.add_texts(
            texts=documents,
            metadatas=metadatas,
            ids=ids,
        )

    retrieved = (
        vectorstore.similarity_search(
            topic,
            k=15,
            filter={
                "run_id": run_id
            },
        )
    )

    unique_sources = []

    seen_urls = set()

    for doc in retrieved:

        url = doc.metadata.get(
            "url",
            ""
        ).strip()

        normalized = normalize_url(
            url
        )

        if normalized in seen_urls:

            continue

        seen_urls.add(
            normalized
        )

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

    return {
        **state,
        "retrieved_documents": unique_sources,
    }


# ============================================================
# 11. ANALYST AGENT
# ============================================================

def analyst_node(
    state: ResearchState
):

    topic = state["topic"]

    documents = state[
        "retrieved_documents"
    ]

    research_text = ""

    for i, document in enumerate(
        documents,
        1
    ):

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

Use these sections:

1. Main Findings
2. Important Facts
3. Key Patterns and Insights
4. Benefits and Applications
5. Risks and Challenges
6. Evidence from Sources
7. Overall Analysis

In Evidence from Sources, include the
actual source title and URL.

If there is not enough information,
say:

"Insufficient information in the retrieved research."
"""

    response = llm.invoke(
        prompt
    )

    analysis = get_response_text(
        response
    )

    return {
        **state,
        "analysis": analysis,
    }


# ============================================================
# 12. WRITER AGENT
# ============================================================

def writer_node(
    state: ResearchState
):

    topic = state["topic"]

    analysis = state[
        "analysis"
    ]

    documents = state[
        "retrieved_documents"
    ]

    source_text = ""

    for i, document in enumerate(
        documents,
        1
    ):

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

For Source Evidence, list the
actual source title and URL.

Make the report clear,
professional and readable.
"""

    response = llm.invoke(
        prompt
    )

    final_report = get_response_text(
        response
    )

    return {
        **state,
        "final_report": final_report,
    }


# ============================================================
# 13. BUILD LANGGRAPH
# ============================================================

builder = StateGraph(
    ResearchState
)

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
# 14. GRAPH CONNECTIONS
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
# 15. NORMAL RESEARCH FUNCTION
# ============================================================

def run_research(topic: str):

    topic = topic.strip()

    if not topic:

        raise ValueError(
            "Research topic cannot be empty."
        )

    run_id = uuid.uuid4().hex[:8]

    initial_state = {
        "topic": topic,
        "run_id": run_id,
        "research_results": [],
        "verified_sources": [],
        "retrieved_documents": [],
        "analysis": "",
        "final_report": "",
    }

    return graph.invoke(
        initial_state
    )


# ============================================================
# 16. STREAMING RESEARCH FUNCTION
# ============================================================

def stream_research(topic: str):

    topic = topic.strip()

    if not topic:

        raise ValueError(
            "Research topic cannot be empty."
        )

    run_id = uuid.uuid4().hex[:8]

    initial_state = {
        "topic": topic,
        "run_id": run_id,
        "research_results": [],
        "verified_sources": [],
        "retrieved_documents": [],
        "analysis": "",
        "final_report": "",
    }

    final_state = None

    for event in graph.stream(
        initial_state,
        stream_mode="updates"
    ):

        for node_name, node_output in event.items():

            final_state = node_output

            yield node_name, node_output


# ============================================================
# 17. TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "AUTONOMOUS RESEARCH ASSISTANT"
    )

    print("=" * 70)

    topic = input(
        "\nEnter your research topic: "
    ).strip()

    result = run_research(
        topic
    )

    print("\n")
    print("=" * 70)

    print(
        "FINAL RESEARCH REPORT"
    )

    print("=" * 70)

    print(
        result["final_report"]
    )