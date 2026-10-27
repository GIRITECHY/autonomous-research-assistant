import os
from urllib.parse import urlparse

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

Cover:
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
# 5. Search Web
# --------------------------------------------------

def search_web(queries):

    all_results = []

    for i, query in enumerate(queries, start=1):

        print(f"\nSearching {i}/{len(queries)}:")
        print(query)

        search_response = tavily_client.search(
            query=query,
            max_results=3
        )

        results = search_response.get("results", [])

        all_results.extend(results)

    return all_results


# --------------------------------------------------
# 6. Verify URL
# --------------------------------------------------

def is_valid_url(url):

    if not url:
        return False

    try:

        parsed = urlparse(url)

        return (
            parsed.scheme in ("http", "https")
            and bool(parsed.netloc)
        )

    except Exception:

        return False


# --------------------------------------------------
# 7. Verify Individual Source
# --------------------------------------------------

def verify_source(result):

    title = result.get("title", "").strip()
    url = result.get("url", "").strip()
    content = result.get("content", "").strip()

    if not title:
        return False, "Missing title"

    if not is_valid_url(url):
        return False, "Invalid URL"

    if not content:
        return False, "Missing content"

    return True, "Valid"


# --------------------------------------------------
# 8. Deduplicate and Verify Sources
# --------------------------------------------------

def verify_and_deduplicate(results):

    verified_results = []

    seen_urls = set()

    duplicate_count = 0
    invalid_count = 0

    for result in results:

        url = result.get("url", "").strip()

        # Normalize URL for duplicate detection
        normalized_url = url.rstrip("/").lower()

        # Check duplicate
        if normalized_url in seen_urls:

            duplicate_count += 1
            continue

        # Verify source
        valid, reason = verify_source(result)

        if not valid:

            print(f"\nRejected source: {reason}")
            invalid_count += 1
            continue

        seen_urls.add(normalized_url)
        verified_results.append(result)


    print("\n" + "=" * 70)
    print("SOURCE VERIFICATION")
    print("=" * 70)

    print(f"Raw sources collected: {len(results)}")
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Invalid sources removed: {invalid_count}")
    print(f"Verified sources: {len(verified_results)}")

    return verified_results


# --------------------------------------------------
# 9. Create Research Summary
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

The following sources passed basic source verification:

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

Use only information supported by the provided sources.

Do not invent facts.

Do not claim that a source proves something unless
the provided source content supports that statement.
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
# 10. Main Researcher Agent v3
# --------------------------------------------------

def researcher_agent_v3(topic):

    print("\n" + "=" * 70)
    print("AUTONOMOUS RESEARCHER AGENT v3")
    print("=" * 70)

    print(f"\nResearch Topic: {topic}")


    # Generate queries
    print("\nGenerating research queries...")

    queries = generate_queries(topic)

    print("\nGenerated Queries:")
    print("-" * 70)

    for i, query in enumerate(queries, start=1):

        print(f"{i}. {query}")


    # Search
    print("\n" + "=" * 70)
    print("WEB RESEARCH")
    print("=" * 70)

    raw_results = search_web(queries)


    # Verify + deduplicate
    verified_results = verify_and_deduplicate(raw_results)


    # Stop if nothing survived verification
    if not verified_results:

        print("\nNo verified sources were available.")
        return


    # Generate summary
    print("\nGenerating research summary...")

    summary = create_summary(
        topic,
        verified_results
    )


    # Display final summary
    print("\n" + "=" * 70)
    print("FINAL VERIFIED RESEARCH SUMMARY")
    print("=" * 70)

    print(summary)


# --------------------------------------------------
# 11. Run Program
# --------------------------------------------------

if __name__ == "__main__":

    topic = input("\nEnter your research topic: ")

    researcher_agent_v3(topic)