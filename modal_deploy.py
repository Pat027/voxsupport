"""
Modal deploy script — spins up a public HTTPS endpoint for voxsupport with
a single GPU for Kyutai STT/TTS and a CPU container for the FastAPI app.

Usage:
    pip install modal
    modal setup              # one-time
    modal deploy modal_deploy.py

After deploy:
- The app URL is printed (https://{workspace}--voxsupport-app.modal.run).
- Open it in a browser for the demo, or point Twilio at /twilio/voice.

Secrets needed (create in the Modal dashboard or via `modal secret create`):
    voxsupport-secrets
        OPENAI_API_KEY          (optional)
        ANTHROPIC_API_KEY       (optional)
        LANGFUSE_HOST           (pointing to your Langfuse instance)
        LANGFUSE_PUBLIC_KEY
        LANGFUSE_SECRET_KEY
        TWILIO_ACCOUNT_SID      (week 5 phone demo)
        TWILIO_AUTH_TOKEN
        DATABASE_URL            (managed Postgres, e.g. Neon or Supabase)
        REDIS_URL               (managed Redis, e.g. Upstash)
"""

from __future__ import annotations

import modal

APP_NAME = "voxsupport"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir(".", remote_path="/app", copy=True)
    .workdir("/app")
)

app = modal.App(APP_NAME, image=image)
secrets = [modal.Secret.from_name("voxsupport-secrets")]


@app.function(
    gpu="A10G",              # GPU for Kyutai STT/TTS; A100 for vLLM Llama
    timeout=600,
    secrets=secrets,
    min_containers=1,        # warm container — cold start hurts voice UX
    max_containers=5,
    scaledown_window=60 * 5,
)
@modal.asgi_app()
def fastapi_app():
    """Return the FastAPI app as a Modal web endpoint."""
    from src.voice.server import app as fastapi_instance
    return fastapi_instance


@app.function(
    image=image,
    timeout=3600,
    secrets=secrets,
    gpu="A10G",
)
def run_benchmarks():
    """One-shot job: run the full benchmark suite on both architectures."""
    import subprocess
    subprocess.run(
        ["python", "benchmarks/run_benchmarks.py", "--all", "--runs", "3"],
        check=True,
    )


@app.function(
    image=image,
    timeout=600,
    secrets=secrets,
)
def ingest_knowledge_base():
    """One-shot job: embed and index the Acme Cloud docs into pgvector."""
    import asyncio
    import os

    from src.agent.rag import KnowledgeBase

    async def _run():
        kb = KnowledgeBase(os.environ["DATABASE_URL"].replace("+asyncpg", ""))
        total = await kb.ingest_directory("/app/data/acme_docs")
        print(f"Ingested {total} chunks")

    asyncio.run(_run())


if __name__ == "__main__":
    # Local test: `python modal_deploy.py` runs the local ASGI app.
    import uvicorn

    from src.voice.server import app as fastapi_instance
    uvicorn.run(fastapi_instance, host="0.0.0.0", port=8080)
