import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load variables from .env
load_dotenv()

# Create Gemini LLM through LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Send a message
response = llm.invoke("Explain Artificial Intelligence in one simple sentence.")

# Print the response
print("\nLangChain + Gemini Response:")
print(response.content)