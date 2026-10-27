import os
from typing import TypedDict

from dotenv import load_dotenv
from tavily import TavilyClient

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_chroma import Chroma

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")


if not tavily_api_key:
    print("ERROR: TAVILY_API_KEY was not found.")
    exit()


if not gemini_api_key:
    print("ERROR: GEMINI_API_KEY was not found.")
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
# 4. CREATE GEMINI LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=gemini_api_key
)


# ============================================================
# 5. CREATE CHROMA VECTOR STORE
# ============================================================

vector_store = Chroma(
    collection_name="langgraph_research",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# ============================================================
# 6. LANGGRAPH SHARED STATE
# ============================================================

class ResearchState(TypedDict):

    topic: str

    research_results: list

    retrieved_documents: list

    analysis: str

    final_report: str


# ============================================================
# 7. RESEARCHER NODE
# ============================================================

def researcher_node(state: ResearchState):

    topic = state["topic"]

    print("\n[Researcher Node]")
    print("-" * 60)

    print("Research topic:", topic)
    print("Searching the web...")


    response = tavily_client.search(
        query=topic,
        max_results=3
    )


    results = response.get(
        "results",
        []
    )


    if not results:

        print("No research sources found.")

        return {
            "research_results": []
        }


    print(
        f"Found {len(results)} web sources."
    )


    research_results = []


    for result in results:

        title = result.get(
            "title",
            ""
        )

        url = result.get(
            "url",
            ""
        )

        content = result.get(
            "content",
            ""
        )


        if not content:
            continue


        content = content[:3000]


        research_results.append({

            "title": title,

            "url": url,

            "content": content

        })


    print(
        f"Prepared {len(research_results)} research sources."
    )


    for index, source in enumerate(
        research_results,
        start=1
    ):

        print(
            f"\nSource {index}:"
        )

        print(
            "Title:",
            source["title"]
        )

        print(
            "URL:",
            source["url"]
        )


    return {

        "research_results":
            research_results

    }


# ============================================================
# 8. RAG NODE
# ============================================================

def rag_node(state: ResearchState):

    print("\n[RAG Node]")
    print("-" * 60)


    research = state[
        "research_results"
    ]


    print(
        f"Received {len(research)} research sources."
    )


    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    documents = []

    metadatas = []


    for source in research:

        title = source.get(
            "title",
            ""
        )

        url = source.get(
            "url",
            ""
        )

        content = source.get(
            "content",
            ""
        )


        if not content:
            continue


        content = content[:3000]


        chunk_size = 1000


        for start in range(
            0,
            len(content),
            chunk_size
        ):

            chunk = content[
                start:start + chunk_size
            ]


            if not chunk.strip():
                continue


            document_text = (

                f"Title: {title}\n"

                f"URL: {url}\n\n"

                f"Content:\n{chunk}"

            )


            documents.append(
                document_text
            )


            metadatas.append({

                "title": title,

                "url": url,

                "topic": state["topic"]

            })


    print(
        f"Created {len(documents)} research chunks."
    )


    # --------------------------------------------------------
    # Store in ChromaDB
    # --------------------------------------------------------

    if documents:

        print(
            "Creating embeddings and storing in ChromaDB..."
        )


        vector_store.add_texts(

            texts=documents,

            metadatas=metadatas

        )


        print(
            f"Stored {len(documents)} chunks."
        )


    # --------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------

    query = state[
        "topic"
    ]


    print(
        "Performing semantic retrieval..."
    )


    # Retrieve more candidates.
    # Multiple chunks can belong to the same URL.

    candidate_documents = (
        vector_store.similarity_search(
            query,
            k=15
        )
    )


    print(
        f"Retrieved {len(candidate_documents)} candidate chunks."
    )


    # --------------------------------------------------------
    # Select unique sources
    # --------------------------------------------------------

    unique_sources = {}

    duplicate_count = 0


    for document in candidate_documents:

        title = document.metadata.get(
            "title",
            "Unknown Source"
        )

        url = document.metadata.get(
            "url",
            ""
        )

        content = document.page_content


        if url.strip():

            source_key = (
                url.strip()
                .lower()
                .rstrip("/")
            )

        else:

            source_key = (
                title.strip()
                .lower()
            )


        if source_key in unique_sources:

            duplicate_count += 1

            continue


        unique_sources[source_key] = {

            "title": title,

            "url": url,

            "content": content

        }


        # We only need up to 3 unique sources.

        if len(unique_sources) >= 3:

            break


    print(
        f"Removed {duplicate_count} duplicate source chunks."
    )

    print(
        f"Selected {len(unique_sources)} unique sources."
    )


    # --------------------------------------------------------
    # Convert sources into text
    # --------------------------------------------------------

    retrieved_texts = []


    for index, source in enumerate(
        unique_sources.values(),
        start=1
    ):

        retrieved_text = (

            f"Source {index}\n"

            f"Title: {source['title']}\n"

            f"URL: {source['url']}\n\n"

            f"Content:\n{source['content']}"

        )


        retrieved_texts.append(
            retrieved_text
        )


    # --------------------------------------------------------
    # Display selected sources
    # --------------------------------------------------------

    for index, document in enumerate(
        retrieved_texts,
        start=1
    ):

        print(
            f"\nRetrieved Source {index}:"
        )

        print(
            "-" * 60
        )


        lines = document.split(
            "\n"
        )


        for line in lines[:4]:

            print(line)


    # --------------------------------------------------------
    # Update state
    # --------------------------------------------------------

    return {

        "retrieved_documents":
            retrieved_texts

    }


# ============================================================
# 9. ANALYST NODE
# ============================================================

def analyst_node(state: ResearchState):

    print("\n[Analyst Node]")
    print("-" * 60)


    documents = state[
        "retrieved_documents"
    ]

    topic = state[
        "topic"
    ]


    print(
        f"Received {len(documents)} documents."
    )


    retrieved_text = "\n\n".join(
        documents
    )


    prompt = f"""
You are the Analyst Agent in an Autonomous Research Assistant.

Research Topic:
{topic}

Retrieved Research:
{retrieved_text}

Analyze ONLY the retrieved research provided above.

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

For Evidence from Sources:

- Mention the actual source title.
- Mention the actual source URL.
- Connect findings to the relevant source.
- Do not invent sources or URLs.

If the retrieved research does not contain enough
information for a section, write:

"Insufficient information in the retrieved research."

Keep the analysis clear and concise.
"""


    print(
        "Sending retrieved research to Gemini..."
    )


    response = llm.invoke(
        prompt
    )


    if isinstance(
        response.content,
        list
    ):

        analysis_parts = []


        for item in response.content:

            if item.get("type") == "text":

                analysis_parts.append(
                    item.get(
                        "text",
                        ""
                    )
                )


        analysis = "\n".join(
            analysis_parts
        )


    else:

        analysis = response.content


    print(
        "Analysis completed successfully."
    )


    return {

        "analysis":
            analysis

    }


# ============================================================
# 10. WRITER NODE
# ============================================================

def writer_node(state: ResearchState):

    print("\n[Writer Node]")
    print("-" * 60)


    topic = state[
        "topic"
    ]

    analysis = state[
        "analysis"
    ]

    retrieved_documents = state[
        "retrieved_documents"
    ]


    print(
        "Received completed analysis."
    )

    print(
        "Generating final research report..."
    )


    source_information = "\n\n".join(
        retrieved_documents
    )


    prompt = f"""
You are the Writer Agent in an Autonomous Research Assistant.

Research Topic:
{topic}

Analyst's Research Analysis:
{analysis}

Retrieved Source Information:
{source_information}

Write a clear, structured research report.

Use ONLY the information provided above.

Do not invent facts.
Do not invent statistics.
Do not invent sources.
Do not create URLs that were not provided.

Use this structure:

# Research Report

## 1. Executive Summary

## 2. Introduction

## 3. Key Findings

## 4. Important Facts and Insights

## 5. Benefits and Applications

## 6. Risks and Challenges

If insufficient information exists, write:

"Insufficient information in the retrieved research."

## 7. Overall Analysis

## 8. Conclusion

## 9. Source Evidence

For every source include:

- Source Title
- Source URL
- Important evidence or finding

Important rules:

- Preserve the actual source URLs.
- Do not invent source information.
- Do not make unsupported claims.
- Keep the report professional and well organized.
"""


    response = llm.invoke(
        prompt
    )


    if isinstance(
        response.content,
        list
    ):

        report_parts = []


        for item in response.content:

            if item.get("type") == "text":

                report_parts.append(
                    item.get(
                        "text",
                        ""
                    )
                )


        final_report = "\n".join(
            report_parts
        )


    else:

        final_report = response.content


    print(
        "Final report generated successfully."
    )


    return {

        "final_report":
            final_report

    }


# ============================================================
# 11. CREATE LANGGRAPH
# ============================================================

graph_builder = StateGraph(
    ResearchState
)


# ============================================================
# 12. ADD NODES
# ============================================================

graph_builder.add_node(
    "researcher",
    researcher_node
)

graph_builder.add_node(
    "rag",
    rag_node
)

graph_builder.add_node(
    "analyst",
    analyst_node
)

graph_builder.add_node(
    "writer",
    writer_node
)


# ============================================================
# 13. CONNECT NODES
# ============================================================

graph_builder.add_edge(
    START,
    "researcher"
)

graph_builder.add_edge(
    "researcher",
    "rag"
)

graph_builder.add_edge(
    "rag",
    "analyst"
)

graph_builder.add_edge(
    "analyst",
    "writer"
)

graph_builder.add_edge(
    "writer",
    END
)


# ============================================================
# 14. COMPILE GRAPH
# ============================================================

graph = graph_builder.compile()


# ============================================================
# 15. GET RESEARCH TOPIC
# ============================================================

topic = input(
    "\nEnter your research topic: "
).strip()


if not topic:

    print(
        "ERROR: Research topic cannot be empty."
    )

    exit()


# ============================================================
# 16. INITIAL STATE
# ============================================================

initial_state: ResearchState = {

    "topic":
        topic,

    "research_results":
        [],

    "retrieved_documents":
        [],

    "analysis":
        "",

    "final_report":
        ""

}


# ============================================================
# 17. RUN LANGGRAPH
# ============================================================

print("\n")

print(
    "=" * 60
)

print(
    "LANGGRAPH RESEARCH + RAG TEST"
)

print(
    "=" * 60
)


final_state = graph.invoke(
    initial_state
)


# ============================================================
# 18. DISPLAY FINAL STATE
# ============================================================

print("\n")

print(
    "=" * 60
)

print(
    "FINAL LANGGRAPH STATE"
)

print(
    "=" * 60
)


print(
    "\nTopic:"
)

print(
    final_state["topic"]
)


print(
    "\nResearch Sources:"
)

print(
    len(
        final_state[
            "research_results"
        ]
    )
)


print(
    "\nRetrieved Documents:"
)

print(
    len(
        final_state[
            "retrieved_documents"
        ]
    )
)


print(
    "\nAnalysis:"
)

print(
    final_state["analysis"]
)


print(
    "\nFinal Report:"
)

print(
    final_state["final_report"]
)


print("\n")

print(
    "=" * 60
)

print(
    "LANGGRAPH RESEARCH + RAG TEST COMPLETED!"
)

print(
    "=" * 60
)