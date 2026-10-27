# Autonomous Research Assistant

An educational AI research application that collects web search results, checks basic source metadata, retrieves relevant source text with RAG, and produces a structured research report.

## How the application works

The Streamlit interface in `app.py` calls the LangGraph workflow in `main.py`:

1. Gemini generates search queries for the topic and selected research depth.
2. Tavily searches the web and returns source titles, URLs, and text snippets.
3. The workflow checks that each source has a title, an HTTP(S) URL, and enough text. This is a technical check; it is not independent fact-checking.
4. Source text is chunked, embedded with Gemini, stored in ChromaDB, and retrieved for the current research run.
5. Gemini analyzes the retrieved text and drafts a structured report.

Research depths use 3 queries (Quick), 5 (Standard), or 8 (Deep). Results can be downloaded from the app as Markdown or text.

## Project files

- `app.py` — Streamlit interface.
- `main.py` — Current LangGraph workflow and terminal entry point.
- `researcher_agent*.py`, `rag_pipeline.py`, `rag_analyst.py`, `full_pipeline.py`, `langgraph_pipeline.py`, `writer_agent.py` — Standalone learning and pipeline experiments.
- `main_backup.py`, `main_before_depth.py`, `main_before_statistics.py` — Earlier workflow snapshots kept for reference.
- `test_*.py` — Small integration examples for Gemini, LangChain, Tavily, embeddings, and ChromaDB. Running these can make external API calls.
- `chroma_db/` — Created locally by ChromaDB when the app runs; it is not required in the submission archive.

## Setup

Use Python 3.10 or newer, then from the project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and add your own API keys:

```text
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Keep `.env` private. Do not upload it or share its API keys.

## Run

```powershell
streamlit run app.py
```

The terminal version is also available with `python main.py`.

## Limitations

- Search results and report content vary between runs.
- Source checks validate basic fields and URL format; they do not confirm the truth of every claim or that a website is authoritative.
- Gemini and Tavily API access and internet connectivity are required for a research run.
- This is an educational project; review generated research and source material before relying on it.

## Technologies

Python, Streamlit, LangGraph, LangChain, Google Gemini, Tavily Search, ChromaDB, and Gemini embeddings.
