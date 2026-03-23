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

MODEL_ID = os.environ.get("MODEL_ID", "gemini-2.5-flash")

reporter_agent = LlmAgent(
    name="ReporterAgent",
    model=MODEL_ID,
    instruction="""Read the {final_evidence_report} state and present it to the user. Your output is the final answer for the entire thread. Do not omit any sections.""",
    description="Reports the final synthesized findings to the user thread.",
    output_key="published_report"
)
