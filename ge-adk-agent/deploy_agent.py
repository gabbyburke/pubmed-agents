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
Deploy PubMed Literature Analysis Agent to Vertex AI Agent Engine.
"""

import os
import sys
from agent import build_agent

import vertexai
from vertexai import agent_engines
from vertexai.agent_engines.templates.adk import AdkApp as _BaseAdkApp

# ---------------------------------------------------------------------------
# Patched AdkApp: Fix Agentspace string message parsing bug
# vertexai SDK bug: streaming_agent_run_with_events does
#   types.Content(**request.message) which crashes when message is a str.
# This subclass overrides the method with the fix.
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

        # FIX: Handle string messages from Agentspace (original bug: types.Content(**str) crashes)
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
PROJECT_ID = os.environ.get("PROJECT_ID", "capricorn-gemini-enterprise")
LOCATION = os.environ.get("LOCATION", "us-central1")
STAGING_BUCKET = f"gs://{PROJECT_ID}-agent-staging"

DISPLAY_NAME = "pubmed-literature-analyst"
DESCRIPTION = (
    "Analyzes medical case notes by searching PubMed literature via BigQuery "
    "vector search, scoring articles on multiple criteria, and generating "
    "comprehensive literature synthesis reports with PMID citations."
)

ENV_VARS = {
    "GOOGLE_GENAI_USE_VERTEXAI": "1",
    "PROJECT_ID": PROJECT_ID,
    "LOCATION": LOCATION,
}

REQUIREMENTS = [
    "google-cloud-aiplatform[agent_engines]>=1.112",
    "google-adk==1.17.0",
    "vertexai",
    "google-cloud-bigquery",
    "google-cloud-bigquery-storage",
    "pandas>=2.2.3",
    "db-dtypes",
    "pydantic",
]


def main():
    """Deploy the PubMed literature analysis agent to Vertex AI Agent Engine."""

    print("=" * 70)
    print("  PubMed Literature Analysis Agent - Deployment to Agent Engine")
    print("=" * 70)
    print()

    # Step 1: Initialize Vertex AI
    print("Step 1: Initializing Vertex AI...")
    try:
        vertexai.init(
            project=PROJECT_ID,
            location=LOCATION,
            staging_bucket=STAGING_BUCKET,
        )
        print(f"  Project:        {PROJECT_ID}")
        print(f"  Location:       {LOCATION}")
        print(f"  Staging Bucket: {STAGING_BUCKET}")
    except Exception as e:
        print(f"Failed to initialize Vertex AI: {e}")
        return 1

    print()

    # Step 2: Build the Agent
    print("Step 2: Building Agent...")
    try:
        agent = build_agent()
        print(f"  Agent: {agent.name}")
        print(f"  Model: {agent.model}")
        print(f"  Tools: {len(agent.tools)} configured")
    except Exception as e:
        print(f"Failed to build Agent: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()

    # Step 3: Deploy
    print("Step 3: Deploying to Agent Engine (this may take 5-10 minutes)...")
    print()

    try:
        adk_app = PatchedAdkApp(
            agent=agent,
            enable_tracing=True,
        )

        remote_agent = agent_engines.create(
            adk_app,
            requirements=REQUIREMENTS,
            env_vars=ENV_VARS,
            display_name=DISPLAY_NAME,
            description=DESCRIPTION,
        )

        print("Deployment successful!")
        print()
        print(f"  Resource Name: {remote_agent.resource_name}")
        resource_id = remote_agent.resource_name.split('/')[-1]
        print(f"  Resource ID:   {resource_id}")
        print(f"  Display Name:  {DISPLAY_NAME}")
        print()

        print("Supported Operations:")
        try:
            operations = remote_agent.operation_schemas()
            for op_name in operations:
                print(f"  - {op_name}")
        except Exception:
            pass
        print()

        print("Next Steps:")
        print(f"  1. Register in Gemini Enterprise with resource:")
        print(f"     projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{resource_id}")
        print(f"  2. Monitor: gcloud logging read \"resource.labels.service_name={resource_id}\"")
        print()

        return 0

    except Exception as e:
        print(f"Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())