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
Deploy PubMed Multi-Agent to Vertex AI Agent Engine.
Preserves folder structures and package namespaces.
"""

import os
import sys
import vertexai
from vertexai import agent_engines
from vertexai.agent_engines.templates.adk import AdkApp as _BaseAdkApp

# Import the SequentialAgent graph from your package
from pubmed_multi_agent.agent import root_agent


# ---------------------------------------------------------------------------
# Patched AdkApp: Fixes Agentspace string message parsing bug (copied from ge-adk-agent)
# ---------------------------------------------------------------------------
class PatchedAdkApp(_BaseAdkApp):
    async def streaming_agent_run_with_events(self, request_json: str):
        import json as _json
        from google.genai import types
        from google.genai.errors import ClientError
        from vertexai.agent_engines.templates.adk import _StreamRunRequest

        request = _StreamRunRequest(**_json.loads(request_json))
        if not any(
            self._tmpl_attrs.get(service)
            for service in (
                "in_memory_runner", "runner",
                "in_memory_artifact_service", "artifact_service",
                "in_memory_session_service", "session_service",
                "in_memory_memory_service", "memory_service",
            )
        ):
            self.set_up()
        app = self._tmpl_attrs.get("app")

        if request.session_id:
            session_service = self._tmpl_attrs.get("session_service")
            artifact_service = self._tmpl_attrs.get("artifact_service")
            runner = self._tmpl_attrs.get("runner")
            session = None
            try:
                session = await session_service.get_session(
                    app_name=app.name if app else self._tmpl_attrs.get("app_name"),
                    user_id=request.user_id,
                    session_id=request.session_id,
                )
                if session:
                    await self._save_artifacts(
                        session_id=request.session_id,
                        artifact_service=artifact_service,
                        request=request,
                    )
            except ClientError:
                pass
            if not session:
                session = await self._init_session(
                    session_service=session_service,
                    artifact_service=artifact_service,
                    request=request,
                )
        else:
            session_service = self._tmpl_attrs.get("in_memory_session_service")
            artifact_service = self._tmpl_attrs.get("in_memory_artifact_service")
            runner = self._tmpl_attrs.get("in_memory_runner")
            session = await self._init_session(
                session_service=session_service,
                artifact_service=artifact_service,
                request=request,
            )
        if not session:
            raise RuntimeError("Session initialization failed.")

        if isinstance(request.message, str):
            message_for_agent = types.Content(
                role="user", parts=[types.Part(text=request.message)]
            )
        elif isinstance(request.message, dict):
            message_for_agent = types.Content.model_validate(request.message)
        else:
            message_for_agent = request.message

        try:
            async for event in runner.run_async(
                user_id=request.user_id,
                session_id=session.id,
                new_message=message_for_agent,
            ):
                converted_event = await self._convert_response_events(
                    user_id=request.user_id,
                    session_id=session.id,
                    events=[event],
                    artifact_service=artifact_service,
                )
                yield converted_event
        finally:
            if session and not request.session_id:
                app = self._tmpl_attrs.get("app")
                await session_service.delete_session(
                    app_name=app.name if app else self._tmpl_attrs.get("app_name"),
                    user_id=request.user_id,
                    session_id=session.id,
                )
                from vertexai.agent_engines.templates.adk import _force_flush_otel
                _ = await _force_flush_otel(
                    tracing_enabled=self._tracing_enabled(),
                    logging_enabled=bool(self._telemetry_enabled()),
                )


# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "gb-demos")
LOCATION = os.environ.get("LOCATION", "us-central1")
STAGING_BUCKET = f"gs://{PROJECT_ID}-agent-staging"

DISPLAY_NAME = "pubmed-multi-agent-sequential"
DESCRIPTION = "Sequential multi-agent graph with linear pipeline execution."

ENV_VARS = {
    "GOOGLE_GENAI_USE_VERTEXAI": "1",
    "PROJECT_ID": PROJECT_ID,
    "LOCATION": LOCATION,
}

REQUIREMENTS = [
    "google-cloud-aiplatform[agent_engines,a2a]>=1.127",
    "google-adk>=1.24.0",
    "vertexai",
    "google-cloud-bigquery",
    "google-cloud-bigquery-storage",
    "pandas>=2.2.3",
    "db-dtypes",
    "pydantic",
]


def main():
    print("Initializing Vertex AI...")
    vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

    print(f"Deploying {DISPLAY_NAME} to Agent Engine (this may take 5-10 mins)...")
    
    adk_app = PatchedAdkApp(
        agent=root_agent,
        enable_tracing=True,
    )

    remote_agent = agent_engines.create(
        adk_app,
        requirements=REQUIREMENTS,
        env_vars=ENV_VARS,
        display_name=DISPLAY_NAME,
        description=DESCRIPTION,
        extra_packages=["pubmed_multi_agent"],
    )

    print("\n✅ Deployment successful!")
    print(f"Resource Name: {remote_agent.resource_name}")
    resource_id = remote_agent.resource_name.split('/')[-1]
    print(f"Resource ID:   {resource_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
