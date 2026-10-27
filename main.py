
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
# 5. INITIALIZE CHROMA
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
    depth: str
    run_id: str

    search_queries: list
    query_count: int
    total_search_results: int

    research_results: list
    verified_sources: list
    retrieved_documents: list

    analysis: str
    final_report: str


# ============================================================
# 7. RESEARCH DEPTH CONFIGURATION
# ============================================================

DEPTH_CONFIG = {

    "quick": {
        "query_count": 3,
        "results_per_query": 2,
        "max_sources": 6,
    },

    "standard": {
        "query_count": 5,
        "results_per_query": 3,
        "max_sources": 10,
    },

    "deep": {
        "query_count": 8,
        "results_per_query": 3,
        "max_sources": 16,
    },
}


# ============================================================
# 8. RESPONSE TEXT HELPER
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
# 9. RESEARCHER AGENT
# ============================================================

def researcher_node(state: ResearchState):

    topic = state["topic"]

    depth = state.get(
        "depth",
        "standard"
    ).lower()

    config = DEPTH_CONFIG.get(
        depth,
        DEPTH_CONFIG["standard"]
    )

    query_count = config[
        "query_count"
    ]

    results_per_query = config[
        "results_per_query"
    ]

    max_sources = config[
        "max_sources"
    ]

    query_prompt = f"""
You are the Researcher Agent in an Autonomous Research Assistant.

Research topic:
{topic}

Research depth:
{depth}

Generate exactly {query_count} different web search queries.

The queries should investigate different aspects of the topic.

Consider areas such as:

- Definition and background
- Important facts
- Current developments
- Applications
- Benefits
- Risks and challenges
- Real-world examples
- Future developments

For quick research, keep the queries focused.

For standard research, provide balanced coverage.

For deep research, provide broader and more detailed coverage.

Return ONLY the search queries,
one query per line.

Do not number the queries.
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

    queries = queries[
        :query_count
    ]

    all_results = []

    executed_queries = []

    for query in queries:

        try:

            search_results = tavily_client.search(
                query=query,
                max_results=results_per_query,
            )

            executed_queries.append(
                query
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

    # --------------------------------------------------------
    # REMOVE DUPLICATE URLS
    # --------------------------------------------------------

    unique_results = []

    seen_urls = set()

    for result in all_results:

        url = result[
            "url"
        ].rstrip(
            "/"
        ).lower()

        if url and url not in seen_urls:

            seen_urls.add(
                url
            )

            unique_results.append(
                result
            )

    unique_results = unique_results[
        :max_sources
    ]

    return {
        **state,

        "search_queries": executed_queries,

        "query_count": len(
            executed_queries
        ),

        "total_search_results": len(
            all_results
        ),

        "research_results": unique_results,
    }


# ============================================================
# 10. SOURCE VALIDATION
# ============================================================

def is_valid_url(url):

    try:

        parsed = urlparse(
            url
        )

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


# ============================================================
# 11. SOURCE VERIFICATION NODE
# ============================================================

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

            seen_urls.add(
                url
            )

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
# 12. RAG NODE
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
# 13. ANALYST AGENT
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

Below is retrieved research from technically verified sources.

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

In Evidence from Sources,
include the actual source title and URL.

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
# 14. WRITER AGENT
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

For Source Evidence,
list the actual source title and URL.

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
# 15. BUILD LANGGRAPH
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
# 16. LANGGRAPH CONNECTIONS
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
# 17. CREATE INITIAL STATE
# ============================================================

def create_initial_state(
    topic: str,
    depth: str = "standard"
):

    topic = topic.strip()

    if not topic:

        raise ValueError(
            "Research topic cannot be empty."
        )

    depth = depth.lower().strip()

    if depth not in DEPTH_CONFIG:

        depth = "standard"

    run_id = uuid.uuid4().hex[:8]

    return {
        "topic": topic,
        "depth": depth,
        "run_id": run_id,

        "search_queries": [],
        "query_count": 0,
        "total_search_results": 0,

        "research_results": [],
        "verified_sources": [],
        "retrieved_documents": [],

        "analysis": "",
        "final_report": "",
    }


# ============================================================
# 18. NORMAL RESEARCH FUNCTION
# ============================================================

def run_research(
    topic: str,
    depth: str = "standard"
):

    initial_state = create_initial_state(
        topic,
        depth
    )

    return graph.invoke(
        initial_state
    )


# ============================================================
# 19. STREAMING RESEARCH FUNCTION
# ============================================================

def stream_research(
    topic: str,
    depth: str = "standard"
):

    initial_state = create_initial_state(
        topic,
        depth
    )

    for event in graph.stream(
        initial_state,
        stream_mode="updates"
    ):

        for node_name, node_output in event.items():

            yield node_name, node_output


# ============================================================
# 20. TERMINAL TEST
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

    print(
        "\nChoose research depth:"
    )

    print(
        "1. Quick"
    )

    print(
        "2. Standard"
    )

    print(
        "3. Deep"
    )

    choice = input(
        "\nEnter choice (1/2/3): "
    ).strip()

    depth_map = {
        "1": "quick",
        "2": "standard",
        "3": "deep",
    }

    depth = depth_map.get(
        choice,
        "standard"
    )

    print(
        f"\nResearch depth: {depth.upper()}"
    )

    result = run_research(
        topic,
        depth
    )

    print("\n")

    print("=" * 70)

    print(
        "RESEARCH ACTIVITY STATISTICS"
    )

    print("=" * 70)

    print(
        f"Search Queries Executed: {result['query_count']}"
    )

    print(
        f"Total Search Results: {result['total_search_results']}"
    )

    print(
        f"Unique Sources Selected: {len(result['research_results'])}"
    )

    print(
        f"Technically Verified Sources: {len(result['verified_sources'])}"
    )

    print(
        f"RAG Retrieved Sources: {len(result['retrieved_documents'])}"
    )

    print(
        f"Research Mode: {result['depth'].upper()}"
    )

    print("\n")

    print("=" * 70)

    print(
        "SEARCH QUERIES"
    )

    print("=" * 70)

    for index, query in enumerate(
        result["search_queries"],
        1
    ):

        print(
            f"{index}. {query}"
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