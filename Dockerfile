# LiveKit voice worker image (deployed with `lk agent create` from the repo root).
# Streamlit Community Cloud ignores this file; it keeps using cloud_lite/app.py.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Cloud turn detection is used; skip preloading the ~450 MB local fallback.
    LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE=0

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY cloud_lite/requirements-agent.txt ./requirements-agent.txt
RUN pip install --no-cache-dir -r requirements-agent.txt

# Only the modules the worker needs -- no Streamlit app, secrets or PDFs.
COPY cloud_lite/rag_core.py cloud_lite/rag_service.py cloud_lite/llm_client.py \
     cloud_lite/history_store.py cloud_lite/voice_bridge.py cloud_lite/voice_agent.py ./

USER appuser
CMD ["python", "voice_agent.py", "start"]
