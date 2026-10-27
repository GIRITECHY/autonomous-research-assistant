import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# 1. LOAD API KEY
# ============================================================

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    print("ERROR: GEMINI_API_KEY was not found.")
    exit()


# ============================================================
# 2. CREATE GEMINI LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=gemini_api_key
)


# ============================================================
# 3. GET RESEARCH TOPIC
# ============================================================

topic = input(
    "\nEnter the research topic: "
).strip()

if not topic:
    print("ERROR: Research topic cannot be empty.")
    exit()


# ============================================================
# 4. GET RESEARCH INFORMATION
# ============================================================

print("\nEnter the research information.")
print("Paste the retrieved RAG content below.")
print("When finished, type END on a new line.")
print("-" * 60)

research_lines = []

while True:

    line = input()

    if line.strip() == "END":
        break

    research_lines.append(line)


research_text = "\n".join(research_lines)

if not research_text.strip():
    print("ERROR: Research information cannot be empty.")
    exit()


# ============================================================
# 5. CREATE ANALYST PROMPT
# ============================================================

prompt = f"""
You are the Analyst Agent in an Autonomous Research Assistant.

Research Topic:
{topic}

Research Information:
{research_text}

Analyze ONLY the research information provided above.

Do not invent facts.
Do not add information that is not supported by the provided research.

Produce the analysis using the following structure:

1. Main Findings
2. Important Facts
3. Key Patterns and Insights
4. Benefits and Applications
5. Risks and Challenges
6. Evidence from Sources
7. Overall Analysis

For each section, provide clear and concise points.

If the provided research does not contain enough information for a section,
write:
"Insufficient information in the retrieved research."
"""


# ============================================================
# 6. RUN ANALYST AGENT
# ============================================================

print("\nAnalyzing research...")
print("-" * 60)

response = llm.invoke(prompt)


# ============================================================
# 7. DISPLAY ANALYSIS
# ============================================================

print("\nANALYST AGENT REPORT")
print("=" * 60)

if isinstance(response.content, list):

    for item in response.content:

        if item.get("type") == "text":
            print(item.get("text", ""))

else:
    print(response.content)


print("\n" + "=" * 60)
print("Analyst Agent completed successfully!")