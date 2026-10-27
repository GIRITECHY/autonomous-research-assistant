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
# 2. CREATE GEMINI WRITER MODEL
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
# 4. GET ANALYST OUTPUT
# ============================================================

print("\nPaste the Analyst Agent's analysis below.")
print("When finished, type END on a new line.")
print("-" * 60)

analysis_lines = []

while True:

    line = input()

    if line.strip() == "END":
        break

    analysis_lines.append(line)


analysis = "\n".join(analysis_lines)

if not analysis.strip():
    print("ERROR: Analyst information cannot be empty.")
    exit()


# ============================================================
# 5. CREATE WRITER PROMPT
# ============================================================

prompt = f"""
You are the Writer Agent in an Autonomous Research Assistant.

Research Topic:
{topic}

Analyst Agent's Analysis:
{analysis}

Create a professional and well-structured research report
using ONLY the information provided by the Analyst Agent.

Do not invent facts.
Do not introduce unsupported information.
Do not create fake references or sources.

Use the following structure:

# {topic}

## 1. Executive Summary

Write a concise summary of the research.

## 2. Introduction

Explain the topic based on the provided analysis.

## 3. Key Findings

Present the major findings clearly.

## 4. Important Facts and Insights

Present important facts and patterns identified by the Analyst Agent.

## 5. Benefits and Applications

Describe the benefits and applications supported by the analysis.

## 6. Risks and Challenges

Describe the risks and challenges mentioned by the Analyst Agent.

If the Analyst Agent states that information is insufficient,
preserve that limitation instead of inventing details.

## 7. Overall Analysis

Present a concise synthesis of the research findings.

## 8. Conclusion

Summarize the main points without introducing new information.

## 9. Source Evidence

List the sources or source evidence mentioned by the Analyst Agent.

Make the report clear, professional, readable, and suitable
for an academic research project.
"""


# ============================================================
# 6. RUN WRITER AGENT
# ============================================================

print("\nWriter Agent is generating the research report...")
print("-" * 60)

response = llm.invoke(prompt)


# ============================================================
# 7. DISPLAY FINAL REPORT
# ============================================================

print("\nFINAL RESEARCH REPORT")
print("=" * 70)

if isinstance(response.content, list):

    for item in response.content:

        if item.get("type") == "text":
            print(item.get("text", ""))

else:
    print(response.content)


print("\n" + "=" * 70)
print("Writer Agent completed successfully!")