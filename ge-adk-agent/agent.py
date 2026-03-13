# Copyright 2025 Google LLC
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

"""
PubMed Medical Literature Analysis Agent for Vertex AI Agent Engine.
Analyzes patient case notes, searches PubMed via BigQuery vector search,
scores articles with dynamic criteria, and generates comprehensive
literature synthesis reports.
"""

import os
import re
import json
import math
import time
from datetime import datetime
from typing import Dict, List, Any

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext


def build_agent():
    """
    Build and return the PubMed literature analysis agent.
    All dependencies are inline to avoid module reference issues on Agent Engine.
    """

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    PROJECT_ID = os.environ.get("PROJECT_ID", "capricorn-gemini-enterprise")
    LOCATION = os.environ.get("LOCATION", "us-central1")
    BQ_LOCATION = "US"
    MODEL_ID = "gemini-2.5-flash"
    USER_DATASET = "pubmed"
    PUBMED_TABLE = "bigquery-public-data.pmc_open_access_commercial.articles"
    JOURNAL_IMPACT_CSV_URL = "https://raw.githubusercontent.com/google/pubmed-rag/main/scimagojr_2024.csv"

    # Default scoring criteria (matching the notebook configuration)
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

    # Pipeline parameters
    DEFAULT_ARTICLES_PER_BATCH = 10
    MIN_ARTICLES_PER_EVENT = 3
    MAX_ARTICLES_TO_SEARCH = 50

    # Extraction prompts
    DISEASE_EXTRACTION_PROMPT = """You are an expert pediatric oncologist analyzing patient case notes to identify the primary disease.

Task: Extract the initial diagnosis exactly as written in the case notes.

Examples:
- Input: "A now almost 4-year-old female diagnosed with KMT2A-rearranged AML and CNS2 involvement..."
  Output: AML

- Input: "18 y/o boy, diagnosed in November 2021 with T-ALL with CNS1..."
  Output: T-ALL

- Input: "A 10-year-old patient with relapsed B-cell acute lymphoblastic leukemia (B-ALL)..."
  Output: B-cell acute lymphoblastic leukemia (B-ALL)

Output only the disease name. No additional text or formatting."""

    EVENT_EXTRACTION_PROMPT = """You are an expert pediatric oncologist analyzing patient case notes to identify key disease concepts and clinical features for literature search.

Task: Extract 5 general medical concepts that would help find relevant literature. Focus on:
- Disease types and subtypes (e.g., "AML", "T-ALL", "B-ALL")
- Genetic alterations (gene names only, e.g., "KMT2A rearrangement", "FLT3 mutation", "TP53 mutation")
- Treatment modalities (e.g., "HSCT", "chemotherapy", "CAR-T therapy", "stem cell transplant")
- General complications (e.g., "relapse", "refractory disease", "CNS involvement", "MRD positive")
- Anatomical sites or disease features (e.g., "bone marrow", "extramedullary disease")

Instructions:
- Extract GENERAL CONCEPTS that appear in medical literature
- DO NOT include patient-specific details like percentages, timeframes, or specific protocol names
- Focus on searchable medical terms
- Output exactly 5 concepts

Example:
Input: "A 4-year-old female with KMT2A-rearranged AML and CNS2 involvement exhibited refractory disease after NOPHO protocol. MRD remained at 35%. She relapsed 10 months after cord blood HSCT with 33% blasts. WES showed KMT2A::MLLT3 fusion and NRAS mutation."

Output: "AML" "KMT2A rearrangement" "CNS involvement" "refractory disease" "HSCT relapse"

Output only 5 general medical concepts, one per line in quotes. No additional text or formatting."""

    # Analysis synthesis template
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

    # -------------------------------------------------------------------------
    # Lazy initialization
    # -------------------------------------------------------------------------
    _model = [None]
    _vertexai_initialized = [False]
    _bq_client = [None]
    _journal_dict = [None]
    _bq_setup_done = [False]
    # Note: No session state caching - Agent Engine is stateless across requests.
    # Follow-up questions are handled by the LLM from conversation history.

    def get_model():
        """Get or initialize the Gemini model via vertexai SDK."""
        if not _vertexai_initialized[0]:
            import vertexai
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            _vertexai_initialized[0] = True
        if _model[0] is None:
            from vertexai.generative_models import GenerativeModel
            _model[0] = GenerativeModel(MODEL_ID)
        return _model[0]

    def get_bq_client():
        """Get or initialize the BigQuery client."""
        if _bq_client[0] is None:
            from google.cloud import bigquery
            _bq_client[0] = bigquery.Client(project=PROJECT_ID)
        return _bq_client[0]

    # -------------------------------------------------------------------------
    # BigQuery setup (lazy, idempotent)
    # -------------------------------------------------------------------------
    def ensure_bq_setup():
        """Ensure BigQuery dataset, embedding model, and journal table exist."""
        if _bq_setup_done[0]:
            return

        from google.cloud import bigquery
        bq = get_bq_client()

        # Create dataset if needed
        dataset_ref = f"{PROJECT_ID}.{USER_DATASET}"
        try:
            bq.get_dataset(dataset_ref)
        except Exception:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = BQ_LOCATION
            bq.create_dataset(dataset, exists_ok=True)
            print(f"Created BigQuery dataset: {dataset_ref}")

        # Create embedding model if needed
        embedding_model = f"{PROJECT_ID}.{USER_DATASET}.textembed"
        create_model_sql = f"""
        CREATE MODEL IF NOT EXISTS `{embedding_model}`
          REMOTE WITH CONNECTION `{PROJECT_ID}.us.default`
          OPTIONS(endpoint='text-embedding-005');
        """
        try:
            bq.query(create_model_sql).result()
            print(f"Embedding model ready: {embedding_model}")
        except Exception as e:
            print(f"Warning: Could not create embedding model: {e}")

        # Create journal impact table if needed
        table_ref = f"{PROJECT_ID}.{USER_DATASET}.journal_impact"
        try:
            bq.get_table(table_ref)
            print(f"Journal impact table exists: {table_ref}")
        except Exception:
            _create_journal_table(bq, table_ref)

        _bq_setup_done[0] = True

    def _create_journal_table(bq, table_ref):
        """Download journal impact CSV and load into BigQuery."""
        import pandas as pd
        from google.cloud import bigquery

        try:
            print(f"Creating journal impact table: {table_ref}")
            df = pd.read_csv(JOURNAL_IMPACT_CSV_URL, sep=';')
            df['SJR_float'] = df['SJR'].apply(
                lambda x: float(str(x).replace(',', '')) if pd.notna(x) and str(x) != '' else None
            )
            columns_to_keep = {
                'Title': 'journal_title', 'SJR_float': 'sjr', 'Issn': 'issn',
                'SJR Best Quartile': 'sjr_best_quartile', 'H index': 'h_index',
                'Publisher': 'publisher', 'Categories': 'categories',
                'Country': 'country', 'Type': 'type',
            }
            df_clean = df[list(columns_to_keep.keys())].rename(columns=columns_to_keep)
            df_clean = df_clean[df_clean['sjr'].notna()]

            schema = [
                bigquery.SchemaField("journal_title", "STRING"),
                bigquery.SchemaField("sjr", "FLOAT64"),
                bigquery.SchemaField("issn", "STRING"),
                bigquery.SchemaField("sjr_best_quartile", "STRING"),
                bigquery.SchemaField("h_index", "INT64"),
                bigquery.SchemaField("publisher", "STRING"),
                bigquery.SchemaField("categories", "STRING"),
                bigquery.SchemaField("country", "STRING"),
                bigquery.SchemaField("type", "STRING"),
            ]
            job_config = bigquery.LoadJobConfig(
                schema=schema, write_disposition="WRITE_TRUNCATE"
            )
            bq.load_table_from_dataframe(df_clean, table_ref, job_config=job_config).result()
            print(f"Loaded {len(df_clean)} journals into {table_ref}")
        except Exception as e:
            print(f"Warning: Could not create journal impact table: {e}")

    def get_journal_dict():
        """Load journal SJR data from BigQuery into a dictionary."""
        if _journal_dict[0] is not None:
            return _journal_dict[0]

        bq = get_bq_client()
        try:
            query = f"""
            SELECT journal_title, sjr
            FROM `{PROJECT_ID}.{USER_DATASET}.journal_impact`
            WHERE sjr IS NOT NULL
            ORDER BY sjr DESC
            """
            results = bq.query(query).to_dataframe()
            _journal_dict[0] = dict(zip(results['journal_title'], results['sjr']))
            print(f"Loaded {len(_journal_dict[0])} journals for SJR lookup")
        except Exception as e:
            print(f"Warning: Could not load journal data: {e}")
            _journal_dict[0] = {}

        return _journal_dict[0]

    # -------------------------------------------------------------------------
    # Step 1: Medical Information Extraction
    # -------------------------------------------------------------------------
    def extract_medical_info(case_text):
        """Extract disease and actionable events from medical case notes."""
        from vertexai.generative_models import GenerationConfig

        model = get_model()
        results = {}

        for key, prompt in [("disease", DISEASE_EXTRACTION_PROMPT), ("events", EVENT_EXTRACTION_PROMPT)]:
            full_prompt = f"{prompt}\n\nCase notes:\n{case_text}"
            response = model.generate_content(
                full_prompt,
                generation_config=GenerationConfig(temperature=0),
            )
            results[key] = response.text.strip()

        # Parse events into list
        events_text = results.get('events', '')
        if '"' in events_text:
            events_list = re.findall(r'"([^"]+)"', events_text)
        else:
            events_list = [e.strip() for e in events_text.replace('\n', ',').split(',') if e.strip()]

        events_with_ids = {f"event_{i}": event for i, event in enumerate(events_list, 1)}

        return {
            'disease': results.get('disease', ''),
            'events_list': events_list,
            'events_with_ids': events_with_ids,
        }

    # -------------------------------------------------------------------------
    # Step 2: BigQuery Vector Search
    # -------------------------------------------------------------------------
    def search_pubmed_articles(disease, events_list, top_k=15, offset=0):
        """Search PubMed articles using BigQuery vector similarity."""
        bq = get_bq_client()
        embedding_model = f"{PROJECT_ID}.{USER_DATASET}.textembed"
        query_text = f"{disease} {' '.join(events_list)}"

        sql = f"""
        DECLARE query_text STRING;
        SET query_text = \"\"\"{query_text}\"\"\";

        WITH vector_results AS (
            SELECT base.pmc_id AS PMCID, base.pmid AS PMID,
                   base.article_text AS content, distance
            FROM VECTOR_SEARCH(
                TABLE `{PUBMED_TABLE}`,
                'ml_generate_embedding_result',
                (SELECT ml_generate_embedding_result
                 FROM ML.GENERATE_EMBEDDING(
                     MODEL `{embedding_model}`,
                     (SELECT query_text AS content)
                 )),
                top_k => {top_k + offset}
            )
        )
        SELECT * FROM vector_results
        ORDER BY distance
        LIMIT {top_k}
        OFFSET {offset}
        """
        return bq.query(sql).to_dataframe()

    # -------------------------------------------------------------------------
    # Step 3: Article Analysis & Scoring
    # -------------------------------------------------------------------------
    def analyze_single_article(article_row, disease, events_list, criteria):
        """Analyze a single article using Gemini to extract metadata."""
        from vertexai.generative_models import GenerationConfig

        model = get_model()
        content = article_row.get('content', '')[:3000]

        # Build criteria prompts (exclude special criteria)
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
            elif c['type'] == 'numeric':
                criteria_prompts.append(f"{field_num}. {c['name']}: {c['description']} (0-100 scale)")
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

    def normalize_journal_score(sjr, max_points):
        """Normalize journal SJR score using log scale."""
        if not sjr or sjr <= 0:
            return 0
        normalized = math.log(sjr + 1) * (max_points / 12)
        return min(normalized, max_points)

    def calculate_article_score(metadata, events_list):
        """Calculate article score based on metadata and default scoring criteria."""
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
                    # Try case-insensitive match
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

    # -------------------------------------------------------------------------
    # Step 4: Full Pipeline
    # -------------------------------------------------------------------------
    def process_medical_case(case_text):
        """Run the complete PubMed RAG pipeline on medical case notes."""
        import pandas as pd

        ensure_bq_setup()

        # Step 1: Extract medical information
        print("Extracting medical information...")
        medical_info = extract_medical_info(case_text)
        disease = medical_info['disease']
        events_list = medical_info['events_list']
        events_with_ids = medical_info['events_with_ids']
        print(f"Disease: {disease}")
        print(f"Events: {', '.join(events_list)}")

        # Step 2: Progressive search with event coverage tracking
        print("Searching PubMed articles via BigQuery vector search...")
        event_coverage = {eid: [] for eid in events_with_ids}
        total_searched = 0
        all_articles = []

        while total_searched < MAX_ARTICLES_TO_SEARCH:
            all_covered = all(
                len(pmcids) >= MIN_ARTICLES_PER_EVENT
                for pmcids in event_coverage.values()
            )
            if all_covered:
                print("All events have minimum coverage.")
                break

            print(f"  Searching articles {total_searched + 1}-{total_searched + DEFAULT_ARTICLES_PER_BATCH}...")
            articles_df = search_pubmed_articles(
                disease, events_list,
                top_k=DEFAULT_ARTICLES_PER_BATCH,
                offset=total_searched,
            )
            if articles_df.empty:
                print("  No more articles found.")
                break

            all_articles.append(articles_df)

            # Track event coverage
            for _, row in articles_df.iterrows():
                content = row.get('content', '')
                pmcid = row.get('PMCID')
                for eid, etext in events_with_ids.items():
                    if etext.lower() in content.lower():
                        if pmcid not in event_coverage[eid]:
                            event_coverage[eid].append(pmcid)

            total_searched += len(articles_df)

            for eid, etext in events_with_ids.items():
                count = len(event_coverage[eid])
                status = "ok" if count >= MIN_ARTICLES_PER_EVENT else "need more"
                print(f"    {etext}: {count}/{MIN_ARTICLES_PER_EVENT} ({status})")

        if not all_articles:
            return {
                'disease': disease, 'events': events_list,
                'articles': pd.DataFrame(), 'case_text': case_text,
            }

        articles_df = pd.concat(all_articles, ignore_index=True)
        print(f"Analyzing {len(articles_df)} articles...")

        # Step 3: Analyze articles individually
        analysis_criteria = [c for c in DEFAULT_CRITERIA if c['type'] != 'special']
        all_analyses = []

        for idx, (_, row) in enumerate(articles_df.iterrows()):
            pmid = row.get('PMID', 'N/A')
            print(f"  Analyzing article {idx + 1}/{len(articles_df)}: PMID {pmid}")
            analysis = analyze_single_article(row, disease, events_list, analysis_criteria)
            all_analyses.append(analysis)

        # Merge analysis metadata into articles
        for idx, analysis in enumerate(all_analyses):
            if idx < len(articles_df) and analysis:
                for key, value in analysis.items():
                    if key not in articles_df.columns:
                        articles_df[key] = None
                    if key == 'actionable_events' and isinstance(value, list):
                        matched_events = []
                        for event in value:
                            if isinstance(event, str):
                                matched = any(qe.lower() in event.lower() for qe in events_list)
                                matched_events.append({'event': event, 'matches_query': matched})
                            else:
                                matched_events.append(event)
                        articles_df.at[articles_df.index[idx], key] = matched_events
                    else:
                        articles_df.at[articles_df.index[idx], key] = value

        # Step 4: Score articles
        print("Calculating scores...")
        scores = []
        breakdowns = []
        for _, article in articles_df.iterrows():
            metadata = article.to_dict()
            sc, bd = calculate_article_score(metadata, events_list)
            scores.append(sc)
            breakdowns.append(bd)

        articles_df['score'] = scores
        articles_df['score_breakdown'] = breakdowns
        articles_df = articles_df.sort_values('score', ascending=False)

        print(f"Analysis complete. {len(articles_df)} articles scored.")
        return {
            'disease': disease,
            'events': events_list,
            'articles': articles_df,
            'case_text': case_text,
            'event_coverage': event_coverage,
            'total_searched': total_searched,
        }

    # -------------------------------------------------------------------------
    # Step 5: Literature Synthesis
    # -------------------------------------------------------------------------
    def format_article_for_analysis(article, idx):
        """Format a single article for the synthesis prompt."""
        metadata = article if isinstance(article, dict) else article
        journal = metadata.get('journal_title', metadata.get('journal', 'Unknown'))
        pmid = metadata.get('PMID', metadata.get('pmid', 'N/A'))
        pmcid = metadata.get('PMCID', metadata.get('pmcid', 'N/A'))
        events_found = metadata.get('actionable_events', 'None')

        if isinstance(events_found, list):
            events_str = ', '.join(
                e.get('event', str(e)) if isinstance(e, dict) else str(e)
                for e in events_found
            )
        elif isinstance(events_found, str):
            events_str = events_found
        else:
            events_str = "None identified"

        return f"""
Article {idx}:
Title: {metadata.get('title', 'Unknown')}
Journal: {journal} | Year: {metadata.get('year', 'N/A')}
Type: {metadata.get('paper_type', 'Unknown')}
Score: {metadata.get('score', 0)}
Key Concepts Found: {events_str}
PMID: {pmid} | PMCID: {pmcid}

Full Text:
{metadata.get('content', 'No content available')[:5000]}...
"""

    def generate_final_analysis(results, articles):
        """Generate comprehensive literature synthesis."""
        from vertexai.generative_models import GenerationConfig

        model = get_model()

        articles_content_parts = []
        for idx, article in enumerate(articles, 1):
            articles_content_parts.append(format_article_for_analysis(article, idx))

        articles_content = ("\n" + "=" * 80 + "\n").join(articles_content_parts)

        prompt = ANALYSIS_TEMPLATE.format(
            case_description=results['case_text'],
            primary_focus=results['disease'],
            key_concepts=', '.join(results['events']),
            articles_content=articles_content,
        )

        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.3,
                max_output_tokens=8192,
            ),
        )

        return response.text

    # -------------------------------------------------------------------------
    # ADK Tool Functions
    # -------------------------------------------------------------------------
    async def analyze_medical_literature(case_notes: str) -> str:
        """
        Analyze medical case notes by searching PubMed literature and generating
        a comprehensive synthesis report.

        This tool runs the full PubMed RAG pipeline:
        1. Extracts disease and key medical concepts from case notes
        2. Searches PubMed articles via BigQuery vector similarity
        3. Analyzes and scores each article for relevance
        4. Generates a comprehensive 11-section literature synthesis

        Args:
            case_notes: The patient's medical case description including diagnosis,
                       lab findings, genetic results, and clinical history.

        Returns:
            A comprehensive markdown literature analysis report.
        """
        try:
            # Run the full pipeline
            results = process_medical_case(case_notes)

            if results['articles'].empty:
                return f"""# Literature Analysis: {results['disease']}

No relevant articles were found in the PubMed database for this case.

**Disease:** {results['disease']}
**Search concepts:** {', '.join(results['events'])}

Please verify the case notes contain sufficient clinical information for a literature search."""

            # Generate synthesis
            all_articles = results['articles'].to_dict('records')
            print(f"Generating literature synthesis for {len(all_articles)} articles...")
            synthesis = generate_final_analysis(results, all_articles)

            # Build score summary header
            top_3 = results['articles'].head(3)
            score_summary = "\n### Top Scored Articles\n"
            for idx, (_, art) in enumerate(top_3.iterrows()):
                title = art.get('title', 'Unknown')
                if len(title) > 70:
                    title = title[:67] + "..."
                score_summary += f"{idx + 1}. **Score {art['score']:.1f}** - {title} (PMID: {art.get('PMID', 'N/A')})\n"

            return f"""# PubMed Literature Analysis Report

**Disease:** {results['disease']}
**Key Concepts:** {', '.join(results['events'])}
**Articles Analyzed:** {len(all_articles)}
**Articles Searched:** {results.get('total_searched', 'N/A')}

{score_summary}

---

{synthesis}"""

        except Exception as e:
            import traceback
            return f"Error during literature analysis: {str(e)}\n\nDetails:\n{traceback.format_exc()}"

    # -------------------------------------------------------------------------
    # Build ADK Tools and Agent
    # -------------------------------------------------------------------------
    analyze_tool = FunctionTool(func=analyze_medical_literature)

    agent = Agent(
        name="pubmed_literature_analyst",
        model="gemini-2.5-flash",
        instruction="""You are a PubMed Medical Literature Analysis Agent. You help clinicians and researchers analyze patient case notes by searching medical literature and generating comprehensive evidence-based reports.

## Your Capabilities

You have one tool:

**analyze_medical_literature** - Takes patient case notes and runs the full PubMed RAG pipeline:
   - Extracts the primary disease and 5 key medical concepts from case notes
   - Searches PubMed articles using BigQuery vector similarity on 5M+ open-access articles
   - Analyzes each article with Gemini for relevance, scoring on 16 criteria including journal impact, disease match, treatment evidence, and study type
   - Generates a comprehensive 11-section literature synthesis with PMID/PMCID citations

## How to Respond

When a user provides medical case notes or asks for a literature analysis:
1. Call `analyze_medical_literature` with their complete case notes
2. Return the full markdown report to the user
3. Let the user know they can ask follow-up questions about the analysis

When a user asks a follow-up question after an analysis:
- Answer directly from the analysis results already in your conversation history
- Reference specific PMIDs and findings from the report
- Provide evidence-based answers

## Important Guidelines
- Always pass the COMPLETE case notes to the analysis tool - do not summarize or truncate them
- The tool returns a markdown document - present it directly to the user
- If the user hasn't provided case notes yet, ask them to provide patient case information
- Be clear about what the analysis covers: disease extraction, PubMed search, article scoring, and literature synthesis""",
        description="Analyzes medical case notes by searching PubMed literature via BigQuery vector search, scoring articles on multiple criteria, and generating comprehensive literature synthesis reports with PMID citations.",
        tools=[analyze_tool],
    )

    return agent


root_agent = build_agent()
