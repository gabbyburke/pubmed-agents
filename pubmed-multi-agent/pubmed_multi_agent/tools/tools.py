# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
import math
import pandas as pd
from typing import Dict, List, Any
from google.cloud import bigquery
from google.adk.tools import FunctionTool
from vertexai.generative_models import GenerativeModel, GenerationConfig

# -------------------------------------------------------------------------
# Configuration & State
# -------------------------------------------------------------------------
PROJECT_ID = os.environ.get("PROJECT_ID", "gb-demos")
USER_DATASET = "pubmed_demo"
PUBMED_TABLE = "bigquery-public-data.pmc_open_access_commercial.articles"
MODEL_ID = os.environ.get("MODEL_ID", "gemini-2.5-flash")

# The Cache is used to pass DataFrames between the Librarian and Analyst
# without bloating the LLM's prompt window with thousands of tokens.
_SEARCH_RESULTS_CACHE = [None]

DEFAULT_CRITERIA = [
    {"name": "journal_impact", "description": "High-impact journal (automatic SJR lookup)", "type": "special", "weight": 25},
    {"name": "year_penalty", "description": "Penalty per year old", "type": "special", "weight": -5},
    {"name": "event_match", "description": "Points per matching event", "type": "special", "weight": 15},
    {"name": "novelty", "description": "Presents novel/innovative findings or approaches", "type": "boolean", "weight": 10},
    {"name": "disease_match", "description": "Discusses the specific disease from the case", "type": "boolean", "weight": 70},
    {"name": "pediatric_focus", "description": "Focuses on pediatric patients", "type": "boolean", "weight": 50},
    {"name": "treatment_shown", "description": "Shows treatment efficacy or outcomes", "type": "boolean", "weight": 80},
    {"name": "drugs_tested", "description": "Tests or discusses specific drugs/therapies", "type": "boolean", "weight": 5},
    {"name": "clinical_trial", "description": "Is a clinical trial", "type": "boolean", "weight": 50},
    {"name": "review_article", "description": "Is a review article", "type": "boolean", "weight": -5},
    {"name": "case_report", "description": "Is a case report", "type": "boolean", "weight": 5},
    {"name": "case_series", "description": "Is a case series or series of case reports", "type": "boolean", "weight": 10},
    {"name": "cell_studies", "description": "Includes cell/in-vitro studies", "type": "boolean", "weight": 5},
    {"name": "animal_studies", "description": "Includes animal/mouse model studies", "type": "boolean", "weight": 10},
    {"name": "clinical_study", "description": "Is a clinical study (observational or interventional)", "type": "boolean", "weight": 15},
    {"name": "clinical_study_on_children", "description": "Is a clinical study specifically on children", "type": "boolean", "weight": 20},
]

ANALYSIS_TEMPLATE = """You are a research analyst synthesizing findings from a comprehensive literature review. Your goal is to provide insights that are valuable for research purposes.

RESEARCH CONTEXT:
Original Query/Case: {case_description}

Primary Focus: {primary_focus}
Key Concepts Searched: {key_concepts}

ANALYZED ARTICLES:
{articles_content}

Based on the research context and analyzed articles above, please provide a comprehensive synthesis in markdown format with the following sections:

## Literature Analysis: {primary_focus}

### 1. Executive Summary
Provide a concise overview of the key findings from the literature review, highlighting:
- Main themes identified across the literature
- Most significant insights relevant to the research query
- Overall quality and quantity of available evidence
- Key takeaways for researchers in this field

### 2. Key Findings by Concept
| Concept | Articles Discussing | Key Findings | Evidence Quality |
|---------|-------------------|--------------|------------------|
[For each key concept searched, summarize what the literature reveals about it. In "Articles Discussing", list articles using their PMCID as clickable links, e.g., [PMC7654321](https://pmc.ncbi.nlm.nih.gov/articles/PMC7654321/)]

### 3. Methodological Landscape
| Research Method | Frequency | Notable Studies | Insights Generated |
|-----------------|-----------|-----------------|-------------------|
[Map the research methodologies used across the analyzed articles. Reference studies by PMCID]

### 4. Temporal Trends
| Time Period | Research Focus | Key Developments | Paradigm Shifts |
|-------------|----------------|------------------|-----------------|
[Analyze how research in this area has evolved over time. Cite articles using PMCID]

### 5. Cross-Study Patterns
| Pattern | Supporting Evidence | Implications | Confidence Level |
|---------|-------------------|--------------|------------------|
[Identify patterns that appear across multiple studies. List supporting evidence with PMCID references]

### 6. Controversies & Unresolved Questions
| Issue | Different Perspectives | Evidence For/Against | Current Consensus |
|-------|----------------------|---------------------|-------------------|
[Highlight areas of disagreement or ongoing debate in the literature. Cite specific articles by PMCID]

### 7. Knowledge Gaps & Future Research
| Gap Identified | Why It Matters | Potential Approaches | Expected Impact |
|----------------|----------------|---------------------|-----------------|
[Map areas where further research is needed based on the analyzed articles]

### 8. Practical Applications
Based on the synthesized literature, identify:
- How these findings can be applied in practice
- Recommendations for researchers entering this field
- Tools, methods, or frameworks that emerge from the literature
- Potential interdisciplinary connections

### 9. Quality & Reliability Assessment
Evaluate the overall body of literature:
- **Study Types**: Distribution of research designs (experimental, observational, reviews, etc.)
- **Sample Characteristics**: Common sample sizes, populations studied
- **Geographic Distribution**: Where research is being conducted
- **Publication Patterns**: Journal quality, publication years, citation patterns
- **Methodological Rigor**: Strengths and limitations observed

### 10. Synthesis & Conclusions
Provide an integrated narrative that:
- Connects findings across all analyzed articles
- Identifies the strongest evidence and most reliable findings
- Suggests how this research area is likely to develop
- Offers guidance for stakeholders interested in this topic

### 11. Bibliography
**Most Relevant Articles** (in order of relevance to the research query):
[For each article, format as follows:
- Title, Journal (Year). [PMCID: PMCxxxxxx](https://pmc.ncbi.nlm.nih.gov/articles/PMCxxxxxx/) | [PMID: xxxxxxxx](https://pubmed.ncbi.nlm.nih.gov/xxxxxxxx/)]

---

IMPORTANT NOTES:
- When referencing articles throughout the analysis, ALWAYS use their PMCID or PMID identifiers, not generic labels like "Article 1"
- Format all article references as clickable links: [PMCxxxxxx](https://pmc.ncbi.nlm.nih.gov/articles/PMCxxxxxx/)
- Maintain objectivity and clearly distinguish between strong evidence and preliminary findings
- Use accessible language while preserving scientific accuracy
- All claims must be traceable to specific articles in the analysis
- When evidence is conflicting, present all viewpoints fairly
- Focus on research insights and knowledge synthesis rather than prescriptive recommendations
- Highlight both the strengths and limitations of the current literature
"""


_JOURNAL_DICT_CACHE = [None]

def get_journal_dict() -> dict:
    if _JOURNAL_DICT_CACHE[0] is not None:
        return _JOURNAL_DICT_CACHE[0]
    
    client = bigquery.Client(project=PROJECT_ID)
    try:
        query = f"SELECT journal_title, sjr FROM `{PROJECT_ID}.{USER_DATASET}.journal_impact` WHERE sjr IS NOT NULL"
        results = client.query(query).to_dataframe()
        _JOURNAL_DICT_CACHE[0] = dict(zip(results['journal_title'], results['sjr']))
    except Exception as e:
        print(f"Warning: Could not load journal data: {e}")
        _JOURNAL_DICT_CACHE[0] = {}
    return _JOURNAL_DICT_CACHE[0]

def normalize_journal_score(sjr, max_points):
    if not sjr or sjr <= 0:
        return 0
    normalized = math.log(sjr + 1) * (max_points / 12)
    return min(normalized, max_points)

def calculate_article_score(metadata, events_list):
    from datetime import datetime
    score = 0
    breakdown = {}
    journal_dict = get_journal_dict()

    for criterion in DEFAULT_CRITERIA:
        name = criterion['name']
        weight = criterion['weight']
        ctype = criterion.get('type', 'boolean')

        if name == 'journal_impact':
            journal_title = metadata.get('journal_title', '')
            sjr = journal_dict.get(journal_title, 0)
            if not sjr:
                for jt, s in journal_dict.items():
                    if jt.lower() == journal_title.lower():
                        sjr = s
                        break
            if sjr and sjr > 0:
                pts = normalize_journal_score(sjr, weight)
                score += pts
                breakdown['journal_impact'] = round(pts, 2)

        elif name == 'year_penalty':
            try:
                year = int(metadata.get('year', 0))
                if year > 0:
                    diff = datetime.now().year - year
                    pts = weight * diff
                    score += pts
                    breakdown['year_penalty'] = pts
            except (ValueError, TypeError):
                pass

        elif name == 'event_match':
            events = metadata.get('actionable_events', [])
            if isinstance(events, str):
                events = [e.strip() for e in events.split(',') if e.strip()]
            matched = 0
            if isinstance(events, list):
                for ev in events:
                    if isinstance(ev, dict):
                        if ev.get('matches_query', False):
                            matched += 1
                    elif isinstance(ev, str):
                        if any(qe.lower() in ev.lower() for qe in events_list):
                            matched += 1
            if matched > 0:
                pts = matched * weight
                score += pts
                breakdown['event_match'] = pts

        elif ctype == 'boolean':
            val = metadata.get(name, False)
            if val is True or str(val).lower() == 'true':
                score += weight
                breakdown[name] = weight

        elif ctype == 'numeric':
            val = metadata.get(name, 0)
            if val:
                pts = weight * (float(val) / 100)
                score += pts
                breakdown[name] = round(pts, 2)

    return round(score, 2), breakdown

def analyze_single_article(article_row, disease, events_list):
    """Analyze a single article using Gemini to extract metadata metrics."""
    # Use Flash explicitly for scoring to speed up sequential loop
    model = GenerativeModel("gemini-2.5-flash")
    content = article_row.get('content', '')[:3000]

    criteria = [c for c in DEFAULT_CRITERIA if c['type'] != 'special']
    criteria_prompts = [
        "1. disease_match: Does the article discuss " + disease + "? (true/false)",
        "2. title: Article title",
        "3. journal_title: Extract the journal name from the article",
        "4. year: Publication year (integer)",
        "5. actionable_events: List which of these events are mentioned: " + str(events_list),
        "6. paper_type: Type of study (Clinical Trial, Review, Case Report, etc.)",
        "7. key_findings: Brief summary of main findings (1-2 sentences)",
    ]
    field_num = 8
    for c in criteria:
        if c['type'] == 'boolean':
            criteria_prompts.append(f"{field_num}. {c['name']}: {c['description']} (true/false)")
            field_num += 1

    criteria_text = "\n    ".join(criteria_prompts)
    prompt = f"""Analyze this medical research article for relevance to:
Disease: {disease}
Actionable Events: {', '.join(events_list)}

For this article, extract:
{criteria_text}

Return as a single JSON object.

Article:
PMID: {article_row.get('PMID', 'N/A')}
Content: {content}
"""
    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error analyzing article PMID {article_row.get('PMID')}: {e}")
        return {}

# -------------------------------------------------------------------------
# Tool 2: search_literature_tool
# -------------------------------------------------------------------------
def search_literature(disease: str, concepts: list) -> str:
    """
    Queries BigQuery Vector Search and caches the full article results.
    Returns a Markdown summary table of titles and links.
    """
    client = bigquery.Client(project=PROJECT_ID)
    embedding_model = f"{PROJECT_ID}.{USER_DATASET}.textembed"
    query_text = f"{disease} {' '.join(concepts)}"
    
    sql = f"""
    WITH search_results AS (
        SELECT base.*, distance
        FROM VECTOR_SEARCH(
            TABLE `{PUBMED_TABLE}`, 'ml_generate_embedding_result',
            (SELECT ml_generate_embedding_result FROM ML.GENERATE_EMBEDDING(
                MODEL `{embedding_model}`, (SELECT @query AS content))),
            top_k => 20
        )
    )
    SELECT pmid, pmc_id, title, article_text AS content, retracted
    FROM search_results
    WHERE (retracted IS NULL OR LOWER(retracted) != 'yes')
    QUALIFY ROW_NUMBER() OVER(PARTITION BY COALESCE(pmid, title) ORDER BY distance) = 1
    LIMIT 10
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("query", "STRING", query_text)]
    )
    df = client.query(sql, job_config=job_config).to_dataframe()
    
    # Store in memory for the next sub-agent
    _SEARCH_RESULTS_CACHE[0] = {"disease": disease, "concepts": concepts, "articles": df}
    
    if df.empty:
        return "No articles found."

    # Return a nicely formatted Markdown table
    table = "| Title | PMID | PMCID |\n| :--- | :--- | :--- |\n"
    for _, row in df.iterrows():
        title = row.get('title', 'Unknown')
        pmid = row.get('pmid')
        pmcid = row.get('pmc_id')
        pmid_str = str(pmid).strip() if pmid else ""
        pmcid_str = str(pmcid).strip() if pmcid else ""
        
        # Check if it is a valid numeric string and not '0' or 'nan'
        is_valid_pmid = pmid_str.isdigit() and pmid_str != "0" and pmid_str.lower() != "nan"
        is_valid_pmcid = pmcid_str.lower().startswith("pmc") or (pmcid_str.isdigit() and pmcid_str != "0")
        
        pmid_link = f"[{pmid_str}](https://pubmed.ncbi.nlm.nih.gov/{pmid_str}/)" if is_valid_pmid else "N/A"
        pmcid_link = f"[{pmcid_str}](https://pmc.ncbi.nlm.nih.gov/articles/{pmcid_str}/)" if is_valid_pmcid else "N/A"
        
        table += f"| {title} | {pmid_link} | {pmcid_link} |\n"
        
    return f"### 📚 Found {len(df)} Articles\n\n{table}"

search_literature_tool = FunctionTool(func=search_literature)

# -------------------------------------------------------------------------
# Tool 3: score_articles_tool
# -------------------------------------------------------------------------
def score_articles() -> str:
    """
    Applies the 16-point scoring logic to the cached articles.
    Returns a Markdown summary table of top scores and links.
    """
    if _SEARCH_RESULTS_CACHE[0] is None:
        return "No articles found in cache. Please run search_literature first."
    
    cache = _SEARCH_RESULTS_CACHE[0]
    df = cache["articles"]
    disease = cache["disease"]
    concepts = cache["concepts"]
    
    analyzed_list = []
    for _, row in df.iterrows():
        # Step 1: Extract metrics using Gemini
        metadata = analyze_single_article(row, disease, concepts)
        
        # Merge metadata with row
        row_dict = row.to_dict()
        # Preserve BigQuery pmid (lowercase) as per dataset schema
        bq_pmid = row_dict.get('pmid')
        row_dict.update(metadata)
        if bq_pmid:
            row_dict['pmid'] = bq_pmid
        analyzed_list.append(row_dict)
        
    df = pd.DataFrame(analyzed_list)
    
    # Step 2: Apply weights
    scored_list = []
    for _, row in df.iterrows():
        score, breakdown = calculate_article_score(row.to_dict(), concepts)
        row_dict = row.to_dict()
        row_dict['score'] = score
        row_dict['score_breakdown'] = breakdown
        scored_list.append(row_dict)
        
    scored_df = pd.DataFrame(scored_list).sort_values('score', ascending=False)
    _SEARCH_RESULTS_CACHE[0]["articles"] = scored_df
    
    if scored_df.empty:
        return "Scoring complete, but no articles to score."

    # Return Top 5 in a nicely formatted Markdown table
    table = "| Title | Score | PMID |\n| :--- | :--- | :--- |\n"
    for idx, (_, art) in enumerate(scored_df.head(5).iterrows()):
        title = art.get('title', 'Unknown')
        score = art.get('score', 0)
        pmid = art.get('pmid')
        pmid_link = f"[{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)" if pmid else "N/A"
        table += f"| {title} | {score:.1f} | {pmid_link} |\n"
        
    return f"### ✅ Scoring Complete (Top 5)\n\n{table}"

score_articles_tool = FunctionTool(func=score_articles)

# -------------------------------------------------------------------------
# Tool 4: synthesize_report_tool
# -------------------------------------------------------------------------
def synthesize_report(case_notes: str) -> str:
    """
    Generates the final 11-section markdown synthesis report.
    """
    cache = _SEARCH_RESULTS_CACHE[0]
    if cache is None or "score" not in cache["articles"].columns:
        return "Articles must be searched and scored before synthesis."
    
    model = GenerativeModel(MODEL_ID)
    from vertexai.generative_models import GenerationConfig
    
    # Prepend a nice sorted table of articles to the final report
    top_articles = cache["articles"].head(10)
    table = "| Title | Score | PMID |\n| :--- | :--- | :--- |\n"
    for _, art in top_articles.iterrows():
        title = art.get('title', 'Unknown')
        score = art.get('score', 0)
        pmid = art.get('pmid') # Use BigQuery lowercase pmid
        pmid_link = f"[{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)" if pmid else "N/A"
        table += f"| {title} | {score:.1f} | {pmid_link} |\n"
    
    table_section = f"### 📚 Scored Search Results (Top 10)\n\n{table}\n\n---\n\n"
    
    # Format articles for the synth prompt
    articles_content_parts = []
    for idx, (_, art) in enumerate(top_articles.iterrows(), 1):
        content = art.get('content', 'No content available')[:5000]
        journal = art.get('journal_title', 'Unknown')
        pmid = art.get('pmid', 'N/A')
        pmcid = art.get('pmc_id', 'N/A')
        articles_content_parts.append(f"""
Article {idx}:
Title: {art.get('title', 'Unknown')}
Journal: {journal} | Year: {art.get('year', 'N/A')}
Type: {art.get('paper_type', 'Unknown')}
Score: {art.get('score', 0)}
PMID: {pmid} | PMCID: {pmcid}

Full Text excerpt:
{content}
""")
    
    articles_content = "\n" + "="*80 + "\n".join(articles_content_parts)
    
    prompt = ANALYSIS_TEMPLATE.format(
        case_description=case_notes,
        primary_focus=cache["disease"],
        key_concepts=", ".join(cache["concepts"]),
        articles_content=articles_content
    )
    
    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(
            temperature=0.3,
            max_output_tokens=8192,
        ),
    )
    
    return table_section + response.text

synthesize_report_tool = FunctionTool(func=synthesize_report)