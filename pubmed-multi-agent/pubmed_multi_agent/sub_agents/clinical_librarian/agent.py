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
from google.adk.agents.llm_agent import LlmAgent
from pubmed_multi_agent.tools.tools import search_literature_tool

MODEL_ID = os.environ.get("MODEL_ID", "gemini-2.5-flash")
    
# Define the Sub-Agent
librarian_agent = LlmAgent(
    model=MODEL_ID,
    name="clinical_librarian",
    output_key="found_articles",
    instruction="""You are an expert Medical Research Librarian. 
    Your goal is to find high-quality peer-reviewed literature from PubMed.
        
        YOUR WORKFLOW:
        1. When given case notes, identify the primary disease and 5 key clinical search terms.
        2. Immediately use 'search_literature_tool' with those terms to query the BigQuery PubMed database.
        3. Report back with a summary of how many articles were found and a few notable titles.
        
        CRITICAL RULES:
        - Do not attempt to analyze the full text of the articles or score them; your colleague, the analyst_agent, will handle that.
        - You CANNOT call the analyst_agent directly. You MUST return your findings to the root_agent so they can orchestrate the handoff.
        - If the search returns zero results, explain this to the root_agent so they can help you refine the search criteria.
        - You are the gatekeeper of the search cache. Once you run the search, the data is stored for the rest of the team to use.""",
    tools=[search_literature_tool],
)