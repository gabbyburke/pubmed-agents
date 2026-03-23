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
from pubmed_multi_agent.tools.tools import score_articles_tool, synthesize_report_tool

MODEL_ID = os.environ.get("MODEL_ID", "gemini-2.5-flash")
    
analyst_agent = LlmAgent(
    model=MODEL_ID,
    name="evidence_analyst",
    output_key="final_evidence_report",
    instruction="""You are an expert Evidence-Based Medicine Analyst. 
    Your goal is to evaluate the quality of research found by the librarian_agent.

    **Found Articles from librarian_agent:**
    {found_articles}
    
    YOUR WORKFLOW:
    1. When the librarian_agent has finished their search, use 'score_articles_tool' 
        to perform a deep analysis of the cached papers.
        2. This tool will automatically look up Journal Impact (SJR) and 
           apply the 16-point clinical relevance rubric.
        3. Once scoring is complete, use 'synthesize_report_tool' to 
           generate the final 11-section markdown research report.
        
        CRITICAL RULES:
        - You must wait for the librarian_agent to populate the cache before 
          running your scoring tools.
        - Prioritize articles with higher clinical rigor (RCTs and Clinical Trials) 
          and high SJR scores in your final synthesis.
        - Ensure all citations in your final report use PMID/PMCID links.
        - Provide the root_agent with a summary of the 'Top 3' highest 
          scoring articles as an interim update.""",
    tools=[score_articles_tool, synthesize_report_tool],
)