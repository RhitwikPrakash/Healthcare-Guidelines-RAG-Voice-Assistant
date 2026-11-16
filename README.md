# Healthcare Guidelines RAG Assistant (with optional real-time voice)

A Streamlit Retrieval-Augmented Generation (RAG) assistant. It answers research
questions using only the healthcare guideline PDFs you upload, and shows
page-level references. An optional, interruptible **LiveKit voice assistant**
uses the same grounded pipeline.

> Educational and research use only. It does not diagnose or prescribe.

## Features

- Google sign-in (Streamlit OIDC: `st.login`, `st.user`, `st.logout`)
- Upload and process multiple PDFs (PyMuPDF), with overlapping chunks and a TF-IDF / cosine-similarity index
- Evidence-grounded answers: OpenAI (default `gpt-5.4-mini`) with Google Gemini fallback (`gemini-2.5-flash`)
- Validated citations mapped to real PDF pages, and abstention when evidence is insufficient
- Fast, Smart Auto and Deep Research answer modes
- Black, Dark Blue and Medium–Dark Brown themes
- Account-specific chat history in Supabase PostgreSQL, kept 72 hours after the last update; create, rename, delete and Markdown export
- **Optional voice assistant (OFF by default):** LiveKit Agents with STT, TTS, VAD,
  turn detection, barge-in and follow-up questions. Voice answers come from the
  **same** RAG pipeline. Only finalized text and references are saved, never audio.
  See **[VOICE_ASSISTANT.md](VOICE_ASSISTANT.md)**.

## Structure

```
cloud_lite/
  app.py             Streamlit UI (entrypoint)
  rag_core.py        chunking, TF-IDF retrieval, prompts, citation parsing
  rag_service.py     shared grounded-answer service (text + voice)
  llm_client.py      OpenAI primary / Gemini fallback
  pdf_utils.py       PyMuPDF extraction
  history_store.py   Supabase 72-hour chat history
  request_gate.py    provider request pacing
  voice_bridge.py    secure Streamlit <-> LiveKit agent bridge
  voice_ui.py        voice toggle, session lifecycle and sync
  voice_component/   browser voice panel (livekit-client)
  voice_agent.py     LiveKit Agents worker (deployed separately)
SUPABASE_SCHEMA.sql, SUPABASE_CRON.sql   chat history table + expiry job
SUPABASE_VOICE_SCHEMA.sql                optional temporary voice-session table
Dockerfile                               LiveKit worker image
tests/                                   pytest suite (all external APIs mocked)
```

## Local setup

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r cloud_lite/requirements-dev.txt
cd cloud_lite
..\.venv\Scripts\python -m streamlit run app.py
```

Create `cloud_lite/.streamlit/secrets.toml` from `secrets.toml.example`.
Never commit real keys. Run the tests with `.venv\Scripts\python -m pytest -q tests`.

## Streamlit Community Cloud

- Main file: `cloud_lite/app.py`
- Requirements: `cloud_lite/requirements.txt`
- Secrets: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `SUPABASE_URL`,
  `SUPABASE_SECRET_KEY` and the `[auth]` block. Voice also needs the `LIVEKIT_*` values and
  `VOICE_CONTEXT_SECRET`; see VOICE_ASSISTANT.md.

## Limitations

- Retrieval is lexical (TF-IDF without stemming) and may miss paraphrases.
- Answers depend on the uploaded evidence and the LLM's output.
- PDFs are not stored; re-upload them to continue research in a restored chat.
- Don't upload confidential patient data without authorization and a provider privacy review.

## License

MIT License.
