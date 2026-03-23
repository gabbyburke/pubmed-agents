# 🧬 PubMed Multi-Agent Research Assistant

An autonomous, production-grade **Evidence-Based Medicine (EBM) Literature Search and Synthesis Pipeline** built using the Google **Agent Development Kit (ADK)** and **Vertex AI Agent Engine**.

This repository converts messy, natural-language patient case notes into a clean, deduplicated, and scored table of medical literature—complete with actionable summaries and zero LLM hallucination!

---

## 🎯 Use Case: automated clinical review

### The Problem
Medical research literature is expanding exponentially. Clinicians, researchers, and specialists waste hours manually querying databases, filtering out low-quality studies, and summarizing dense PDFs for complex patient cases. Traditional manual searches are slow, non-standardized, and prone to selection bias.

### The Solution: The Autonomous Medical Research Assistant
This agent acts as an automated EBM review engine. It takes conversational clinical findings (e.g., "4-year male with high-risk neuroblastoma") and autonomously:
1.  **Translates Terms Semantically**: Auto-expands into vector-search keywords.
2.  **Appraises Study Quality using Gemini Flash**: Scores articles using standard medical rubrics (Trial Type, Journal Impact, Relevancy).
3.  **Synthesizes Actionable Summaries**: Formats a beautiful **11-section markdown report** with valid clickable links and zero duplicates.

---

## 🏗️ Architecture: Sequential Graph Pattern

Unlike standard prompt-routers which can get confused about hand-offs, this system uses a **hardwired Sequential graph** ensuring deterministic data flow through specialized sub-agents:

### 1. 🔍 Clinical Librarian Agent (`LlmAgent`)
*   **Goal**: Translate natural language queries (patient case notes) into clean vector-search keywords.
*   **Action**: Queries BigQuery `public-data.pmc_open_access_commercial` using on-the-fly embeddings!

### 2. ⚖️ Evidence Analyst Agent (`LlmAgent`)
*   **Goal**: Appraise literature objectivity without human bias.
*   **Action**: Iterates through articles and applies a standard scoring rubric using Gemini Flash for sub-second execution speeds!

### 3. 📝 Synthesizer/Reporter Agent (`LlmAgent`)
*   **Goal**: Clean tabular output + human-readable analysis.
*   **Action**: Formats a beautiful markdown table with valid clickable links and zero duplicates.

---

## 🚀 Getting Started

### 📦 Prerequisites
Install standard dependencies from the repo root in your Python environment:

```bash
pip install -r requirements.txt
```

### 🔐 Environment Variables
Create a `.env` inside your `pubmed_multi_agent` folder with your service accounts:

```ini
PROJECT_ID="your-google-cloud-project"
LOCATION="us-central1"
# Dataset paths for BigQuery embeddings and journal impact values
USER_DATASET="pubmed_demo"
PUBMED_TABLE="bigquery-public-data.pmc_open_access_commercial.articles"
```

---

## 🏃 Testing Locally (ADK Web)

You can spin up local web browser views to test your agent before cloud deployments:

```bash
# Uses ADK built-in web frontend server
adk web pubmed_multi_agent
```

---

## 🌌 Deploying to Vertex AI Agent Engine

Because multi-agent projects rely heavily on folder module references, zipping them without flattening errors requires standard Python packaging. 

Use the custom **`deploy_agent.py`** script to push directly to Vertex Reasoning Engines without `cloudpickle` unpickling errors:

```bash
# From the project root
python deploy_agent.py
```

### 🔍 Key Features of `deploy_agent.py`:
*   Uses **`PatchedAdkApp`** to fix standard parse bugs in user interfaces.
*   Explicitly sets **`extra_packages=["pubmed_multi_agent"]`** during unpickling to prevent cloud NameErrors!
*   Sets standard airflow deployment paths without folder-shifting conflicts!

---

## 💡 Technical Accomplishments
*   **Sub-Second Caching**: We utilize standard DataFrame in-memory shuttling, bypassing prompt-bloat!
*   **Chunk-Aware Deduplication via SQL `QUALIFY`**: We increase `top_k` and group selections by `title`, ensuring we only keep one high-scoring chunk per paper (cutting out duplicates!).
*   **True DB Link Preservation**: Avoids classic RAG hallucinations by pulling original primary database IDs (`pmid`) rather than asking LLMs to guess them.
