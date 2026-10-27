import os

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------------------------
# 1. Load API keys
# --------------------------------------------------

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not gemini_api_key:
    print("ERROR: GEMINI_API_KEY was not found.")
    exit()

if not tavily_api_key:
    print("ERROR: TAVILY_API_KEY was not found.")
    exit()


# --------------------------------------------------
# 2. Create Gemini
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=gemini_api_key
)


# --------------------------------------------------
# 3. Create Tavily
# --------------------------------------------------

tavily_client = TavilyClient(
    api_key=tavily_api_key
)


# --------------------------------------------------
# 4. Generate Research Queries
# --------------------------------------------------

def generate_queries(topic):

    prompt = f"""
You are a research planning assistant.

Research topic:
{topic}

Create exactly 5 different web-search queries
that investigate different important aspects of this topic.

The queries should cover areas such as:
- basic concepts
- applications or uses
- benefits
- risks or challenges
- future developments

Return ONLY the 5 queries, one query per line.

Do not number them.
Do not add explanations.
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, list):

        text_parts = []

        for item in response.content:

            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))

        query_text = "\n".join(text_parts)

    else:

        query_text = response.content


    queries = []

    for line in query_text.splitlines():

        line = line.strip()

        if line:
            queries.append(line)


    return queries[:5]


# --------------------------------------------------
# 5. Search Multiple Queries
# --------------------------------------------------

def search_web(queries):

    all_results = []

    for i, query in enumerate(queries, start=1):

        print(f"\nSearching {i}/5:")
        print(query)

        search_response = tavily_client.search(
            query=query,
            max_results=3
        )

        results = search_response["results"]

        all_results.extend(results)


    return all_results


# --------------------------------------------------
# 6. Create Final Research Summary
# --------------------------------------------------

def create_summary(topic, results):

    research_text = ""

    for i, result in enumerate(results, start=1):

        research_text += (
            f"\nSOURCE {i}\n"
            f"Title: {result['title']}\n"
            f"URL: {result['url']}\n"
            f"Content: {result['content']}\n"
            f"\n"
        )


    prompt = f"""
You are an autonomous research assistant.

Research topic:
{topic}

Below are research sources collected from multiple web searches:

{research_text}

Create a structured research summary.

Include:

1. Overview
2. Main Findings
3. Important Facts
4. Benefits / Applications
5. Risks / Challenges
6. Future Developments
7. Sources Used

Use only the information provided in the sources.

Do not invent facts.

Clearly distinguish information supported by the sources.
"""


    response = llm.invoke(prompt)


    if isinstance(response.content, list):

        text_parts = []

        for item in response.content:

            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))

        return "\n".join(text_parts)

    else:

        return response.content


# --------------------------------------------------
# 7. Main Researcher Agent
# --------------------------------------------------

def researcher_agent_v2(topic):

    print("\n" + "=" * 70)
    print("AUTONOMOUS RESEARCHER AGENT v2")
    print("=" * 70)

    print(f"\nResearch Topic: {topic}")


    # Generate queries
    print("\nGenerating research queries...")

    queries = generate_queries(topic)

    print("\nGenerated Queries:")
    print("-" * 70)

    for i, query in enumerate(queries, start=1):
        print(f"{i}. {query}")


    # Search the web
    print("\n" + "=" * 70)
    print("WEB RESEARCH")
    print("=" * 70)

    results = search_web(queries)


    # Display source count
    print("\n" + "=" * 70)
    print(f"TOTAL SOURCES COLLECTED: {len(results)}")
    print("=" * 70)


    # Generate summary
    print("\nGenerating research summary...")

    summary = create_summary(topic, results)


    # Display summary
    print("\n" + "=" * 70)
    print("FINAL RESEARCH SUMMARY")
    print("=" * 70)

    print(summary)


# --------------------------------------------------
# 8. Run Program
# --------------------------------------------------

if __name__ == "__main__":

    topic = input("\nEnter your research topic: ")

    researcher_agent_v2(topic)