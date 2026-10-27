# Autonomous Research Assistant — Project Documentation

**Project type:** Educational AI research assistant  
**Interface:** Python + Streamlit  
**Workflow:** LangGraph sequential graph  
**AI/search/retrieval:** Google Gemini, Tavily Search, ChromaDB, Gemini embeddings

## 1. Overview

The application accepts a research topic and a depth setting, gathers web search results, applies technical source-screening rules, retrieves relevant text from a local vector store, and asks Gemini to prepare analysis and a structured report. It displays workflow progress and source information and offers Markdown and plain-text downloads.

Source screening checks metadata and URL/content conditions. It does **not** independently prove factual correctness or that a website is authoritative. Review important claims against the original sources.

## 2. Workflow

`START → researcher → source_verification → rag → analyst → writer → END`

1. The user enters a non-empty topic and selects Quick, Standard, or Deep.
2. Gemini generates search queries.
3. Tavily returns candidate web results.
4. The source-screening stage checks required title/URL/content fields, HTTPS, minimum content length, and duplicate URLs.
5. Accepted text is split into chunks, embedded, and stored in ChromaDB with run metadata.
6. Retrieval is filtered to the current run; the Analyst and Writer stages prepare analysis and the report.
7. Streamlit displays workflow activity, source evidence, report output, and download controls.

The implemented graph is a fixed sequential workflow. A retry path described in the original idea/proposal is not represented as an implemented conditional retry in this workflow.

## 3. Research-depth settings

| Mode | Query count | Results per query | Source cap |
|---|---:|---:|---:|
| Quick | 3 | 2 | 6 |
| Standard | 5 | 3 | 10 |
| Deep | 8 | 3 | 16 |

Counts are configuration limits, not guaranteed result counts. Actual results depend on external search responses and source screening.

## 4. Project files

- `app.py` — Streamlit user interface.
- `main.py` — current LangGraph workflow and terminal entry point.
- `requirements.txt` — Python package dependencies.
- `researcher_agent*.py`, `rag_pipeline.py`, `rag_analyst.py`, `full_pipeline.py`, `langgraph_pipeline.py`, `writer_agent.py`, and `analyst_agent.py` — supporting learning experiments or earlier pipeline implementations.
- `main_backup.py`, `main_before_depth.py`, `main_before_statistics.py` — earlier snapshots retained for reference.
- `test_*.py` — small integration examples; some may call external APIs and incur usage.
- `.env.example` — example environment-variable names, if present in the project copy. Never include a real `.env` file or API keys in a submission.

## 5. Requirements

- Windows or another environment that supports Python and Streamlit.
- Python 3.10 or newer, as stated in the project README.
- Internet connectivity.
- A valid Gemini API key and Tavily API key.

## 6. Setup on Windows (PowerShell)

Open PowerShell in the project directory and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a local `.env` file from `.env.example` (if that example file is present):

```powershell
Copy-Item .env.example .env
```

Open `.env` and enter your own credentials, using the variable names expected by the project:

```text
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Keep `.env` private. Do not paste keys into the report, source repository, screenshots, or shared documentation. If `.env.example` is absent in your copy, create `.env` manually with the variable names above, then confirm they match the code before running.

## 7. Run the application

From the activated virtual environment and project directory:

```powershell
streamlit run app.py
```

Streamlit normally prints a local URL in the terminal. Open that URL in a browser, enter a topic, choose a research mode, and start the research. The terminal workflow entry point is also available with:

```powershell
python main.py
```

## 8. Source screening and retrieval notes

The implementation checks title and URL fields, HTTPS, content length (at least 100 characters), and duplicate URLs. The RAG stage chunks accepted content using a 1,000-character chunk size and 150-character overlap, stores metadata including the run identifier, and retrieves within the current run. These are technical controls; they are not a fact-checking service or an independent source-authority rating.

## 9. Recorded Quick-mode evidence

The recorded full-display run documented 2 generated queries, 4 search results, 4 technically verified source records, 3 RAG source records, completion of the five workflow stages, and visible report/download controls. This is evidence for that run only; it is not a performance benchmark or proof that the report's claims are all correct.

## 10. Limitations and safe use

- Search results and model-generated text can vary from run to run.
- External Gemini and Tavily services, valid API credentials, and internet access are required.
- Technical source screening does not prove source authority or factual truth.
- No performance benchmark, production deployment, user-adoption measure, or organizational-impact metric is documented in the supplied project materials.
- Review sources and important claims manually before relying on the generated report.

## 11. Related official documentation

- Streamlit API: https://docs.streamlit.io/
- LangGraph documentation: https://docs.langchain.com/oss/python/langgraph/overview
- Chroma documentation: https://docs.trychroma.com/
- Tavily documentation: https://docs.tavily.com/
- Google AI for Developers: https://ai.google.dev/gemini-api/docs

---

**Link note:** This is a local accompanying documentation file, not a publicly hosted project website. Keep this Markdown file in the same folder as the internship report so the Appendix C file link can open it.
