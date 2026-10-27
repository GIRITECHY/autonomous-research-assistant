import os

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------------------------
# 1. Load API keys from .env
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
# 2. Create Gemini LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=gemini_api_key
)


# --------------------------------------------------
# 3. Create Tavily Search Client
# --------------------------------------------------

tavily_client = TavilyClient(
    api_key=tavily_api_key
)


# --------------------------------------------------
# 4. Researcher Agent Function
# --------------------------------------------------

def researcher_agent(topic):

    print("\n" + "=" * 70)
    print("RESEARCHER AGENT")
    print("=" * 70)

    print(f"\nResearch Topic: {topic}")

    # --------------------------------------------------
    # 4A. Search the web
    # --------------------------------------------------

    search_response = tavily_client.search(
        query=topic,
        max_results=5
    )

    results = search_response["results"]

    print("\nWeb Sources Found:")
    print("-" * 70)

    for i, result in enumerate(results, start=1):

        print(f"\n{i}. {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Content: {result['content'][:500]}")


    # --------------------------------------------------
    # 4B. Prepare research information for Gemini
    # --------------------------------------------------

    research_text = ""

    for result in results:

        research_text += (
            f"\nTitle: {result['title']}"
            f"\nURL: {result['url']}"
            f"\nContent: {result['content']}"
            f"\n"
        )


    # --------------------------------------------------
    # 4C. Create Gemini prompt
    # --------------------------------------------------

    prompt = f"""
You are a research assistant.

Research topic:
{topic}

Below are web search results:

{research_text}

Based only on the information provided above:

1. Summarize the main findings.
2. Identify important facts.
3. Mention the sources used.
4. Do not invent information.

Give the answer in a clear structured format.
"""


    # --------------------------------------------------
    # 4D. Send research to Gemini
    # --------------------------------------------------

    response = llm.invoke(prompt)


    # --------------------------------------------------
    # 4E. Display final research summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("RESEARCH SUMMARY")
    print("=" * 70)

    if isinstance(response.content, list):

        for item in response.content:

            if item.get("type") == "text":
                print(item.get("text", ""))

    else:

        print(response.content)


# --------------------------------------------------
# 5. Run the Researcher Agent
# --------------------------------------------------

if __name__ == "__main__":

    topic = input("\nEnter your research topic: ")

    researcher_agent(topic)