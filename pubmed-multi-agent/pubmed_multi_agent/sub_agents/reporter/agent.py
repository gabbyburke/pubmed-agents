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
