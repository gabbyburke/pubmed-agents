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
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from .sub_agents.clinical_librarian.agent import librarian_agent
from .sub_agents.evidence_analyst.agent import analyst_agent
from .sub_agents.reporter.agent import reporter_agent

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
PROJECT_ID = os.environ.get("PROJECT_ID", "gb-demos")
MODEL_ID = os.environ.get("MODEL_ID", "gemini-2.5-flash")

root_agent = SequentialAgent(
    name="PubmedPipelineAgent",
    sub_agents=[librarian_agent, analyst_agent, reporter_agent],
    description="Runs the Pubmed literature search and evidence analysis pipeline.",
)