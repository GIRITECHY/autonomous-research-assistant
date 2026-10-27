import os
from dotenv import load_dotenv
from tavily import TavilyClient

# Load API keys from .env
load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    print("ERROR: TAVILY_API_KEY was not found.")
    exit()

# Create Tavily client
tavily_client = TavilyClient(api_key=api_key)

# Perform a web search
response = tavily_client.search(
    query="What is artificial intelligence?",
    max_results=3
)

print("\nTavily Search Results:\n")

for result in response["results"]:
    print("Title:", result["title"])
    print("URL:", result["url"])
    print("Content:", result["content"][:300])
    print("-" * 80)