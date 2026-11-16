# Real-time Voice Assistant (LiveKit Agents)

An **optional**, interruptible, two-way voice assistant for the Healthcare
Guidelines RAG Assistant. It is **OFF by default**. With it off, the app behaves
exactly as before: no microphone prompt, no LiveKit connection, and no speech API calls.

> Educational and research use only. Voice answers follow the same document-only
> grounding and abstention rules as text answers. The assistant does not diagnose or prescribe.

---

## 1. Architecture

### Before

```
Browser ──► Streamlit Community Cloud (app.py)
              ├─ Google OIDC (st.login / st.user)
              ├─ PDF → PyMuPDF → chunks → TF-IDF (in st.session_state)
              ├─ rag_core prompt → llm_client (OpenAI → Gemini fallback)
              └─ history_store → Supabase rag_chat_history (72 h)
```

### After

```
                    ┌──────────── Streamlit Community Cloud (unchanged entrypoint) ────────────┐
Browser             │ app.py  ── text chat (unchanged behaviour) ──┐                           │
 ├ Streamlit UI ◄──►│ voice_ui.py (server-side)                    ▼                           │
 │                  │   1. verifies Google user (st.user `sub`)   rag_service.py ◄── SHARED ──┐ │
 │                  │   2. encrypts current PDF chunks ──────────► Supabase rag_voice_sessions│ │
 │                  │   3. mints 10-min single-room LiveKit token  (2 h TTL, service-role)   │ │
 │                  │   4. merges finalised voice turns ─────────► Supabase rag_chat_history │ │
 │                  └─────────────────────────────────────────────────────────────────────────┘ │
 │                                                                                              │
 └ voice_component (iframe, livekit-client 2.23.0)                                              │
        │  WebRTC audio + text streams                                                          │
        ▼                                                                                       │
   LiveKit Cloud room  ◄──► voice_agent.py (LiveKit Cloud agent deployment)                     │
                              VAD (Silero) · turn detector · STT (Deepgram Nova-3)              │
                              llm_node override ──► rag_service.answer_grounded ────────────────┘
                              TTS (Deepgram Aura-2) · barge-in · writes turns to rag_voice_sessions
```

**Speech pipeline:** mic → VAD → turn detection → STT → **existing RAG** → TTS → speaker.

### How a separate worker gets the right user's PDFs (the hard part)

The LiveKit worker cannot read `st.session_state`. I chose a short-lived,
session-bound, encrypted handoff through the Supabase project you already use.
It adds no new service.

1. When you press **Start voice session**, the Streamlit **server** serializes the
   chunks it already built for your active chat. That is text plus file name and
   page number; PDF bytes are never sent. It encrypts them with Fernet, using a key derived as
   `HMAC-SHA256(VOICE_CONTEXT_SECRET, session_id)`, and inserts one row in
   `rag_voice_sessions` with `owner_id` (your hashed Google `sub`), `chat_id`, a
   random `room_name` and a 2-hour expiry.
2. The server mints a LiveKit token that:
   - is valid for 10 minutes and joins **one** random room only;
   - may publish the **microphone only** and cannot publish data or change its own metadata;
   - carries an **agent dispatch** whose metadata is `session_id`, `owner_id`,
     `chat_id`, `room` and `answer_mode`.

   The JWT is signed with the LiveKit API secret, so the browser can't forge or edit
   the dispatch.
3. The agent trusts **only** that signed dispatch metadata. It checks:
   - the room name matches;
   - the row exists for that `id` + `owner_id` + `room_name` and is not expired or ended;
   - the `chat_id` matches;
   - the context decrypts and its embedded `session_id` and `owner_id` match.

   It then rebuilds the same TF-IDF index. Fitting is deterministic, so it gives the same ranking as text mode.
4. The agent answers only the bound participant identity
   (`RoomOptions.participant_identity`), and the room allows at most two participants.

What this guarantees:

- **No cross-account access.** Every query is filtered by the server-derived `owner_id`, and each session has its own encryption key.
- **No cross-conversation leakage.** Switching chats, processing new PDFs, clearing PDFs, toggling voice off or logging out all end the session.
- **No permanent document storage.** The context is wiped (`context_ciphertext = NULL`) when the session ends, and expired rows are deleted hourly.

**Alternatives considered:**

- *A dedicated RAG backend service.* It would be a third always-on service with its own auth, and cost more.
- *Relaying context through the browser.* The agent would then have to trust document text supplied by the browser.
- *Streamlit custom HTTP routes.* Not supported on Community Cloud.

### Why there is no "voice LLM"

LiveKit's pipeline schedules a reply through `Agent.llm_node`. The agent
overrides it, so every reply comes from `rag_service.answer_grounded`. That is the
**same** path the text chat uses: retrieval, ranking, prompt, OpenAI → Gemini,
`[[SUPPORTING_EVIDENCE]]` validation and abstention. A placeholder LLM object
satisfies the scheduler and raises if anything ever calls it.

The spoken text is a deterministic plain-text cleanup of the written answer
(`spoken_version`). If the answer is long, it speaks the leading whole sentences
and then says "The full answer and page references are shown on screen." No
model rewrites the medical content.

**Follow-up questions:** for short or pronoun-style follow-ups ("how frequently should
*those* tests be performed?"), the previous voice question is appended to the
**retrieval query only**. The last few turns go into the prompt in a block labelled
*"NOT evidence"*. Every answer still has to be supported by retrieved PDF
evidence. When `conversation_context` is empty, the text-mode prompt is
byte-for-byte unchanged, and a test checks this.

### Persistence (text only, never audio)

- The agent appends finalized turns to `rag_voice_sessions.turns`. Each turn has a
  `turn_id`, the text, references and evidence excerpts. The agent writes this
  **before** notifying the browser.
- The browser component only sends a `turn` event. The Streamlit server then
  **re-reads** the turns from Supabase and merges them into the active chat by
  `turn_id`. The merge is idempotent, so there are no duplicates across reruns,
  reconnects or repeated events.
- Streamlit stays the **only writer** of `rag_chat_history`, so there are no races, and
  the existing 72-hour policy applies unchanged.
- Partial STT transcripts are never saved. Only final user turns are saved, and only after the turn detector commits them.
- **Interruptions:**
  - *Barge-in while the assistant is speaking.* The written answer was already
    complete and grounded, so it is saved with `speech_interrupted` and shown as
    "spoken reply was interrupted; the complete written answer is shown".
  - *Barge-in before an answer exists.* The question is saved with the note
    "Interrupted before an answer was produced". No partial answer is ever saved.
- **Dropped connections or closed tabs.** At the next login,
  `recover_unsynced_sessions` merges any turns the agent persisted and wipes the row.

---

## 2. Files

| File | Status | Purpose |
|---|---|---|
| `cloud_lite/rag_service.py` | new | Shared grounded-answer service (text + voice), follow-up retrieval, speech cleanup |
| `cloud_lite/voice_bridge.py` | new | Settings, Fernet context encryption, signed metadata, token minting, `rag_voice_sessions` store, turn merging |
| `cloud_lite/voice_ui.py` | new | Streamlit toggle, session lifecycle, guards, sync, cleanup |
| `cloud_lite/voice_component/index.html` | new | Voice panel (custom component, `livekit-client@2.23.0`) |
| `cloud_lite/voice_agent.py` | new | LiveKit Agents worker |
| `cloud_lite/requirements-agent.txt`, `requirements-dev.txt` | new | Worker / test dependencies |
| `cloud_lite/.env.agent.example` | new | Worker secrets template (placeholders only) |
| `Dockerfile`, `.dockerignore` | new | Worker image for LiveKit Cloud |
| `SUPABASE_VOICE_SCHEMA.sql` | new | Approved temporary table + hourly cron |
| `tests/` | new | 43 automated tests |
| `cloud_lite/app.py` | changed | Uses `rag_service` (same behavior), voice panel, voice badges in chat/export, logout cleanup |
| `cloud_lite/rag_core.py` | changed | Optional `conversation_context` argument (default keeps the old prompt) |
| `cloud_lite/requirements.txt` | changed | `+ livekit-api==1.2.3`, `+ cryptography>=43` |
| `cloud_lite/.streamlit/secrets.toml.example` | changed | LiveKit placeholders |

`auth_utils.py` is legacy and unused (nothing imports it), so it was left untouched.

---

## 3. Credentials: which secret goes where

| Secret | Streamlit Cloud secrets | LiveKit agent secrets | Notes |
|---|:-:|:-:|---|
| `LIVEKIT_URL` (`wss://…livekit.cloud`) | ✅ | auto-injected | Public URL |
| `LIVEKIT_API_KEY` | ✅ | auto-injected | |
| `LIVEKIT_API_SECRET` | ✅ | auto-injected | Never sent to the browser; signs participant tokens |
| `VOICE_CONTEXT_SECRET` | ✅ | ✅ (same value) | New random 32+ character string. Encrypts the temporary PDF context. It's separate from the LiveKit secret because LiveKit Cloud injects its own key pair into agents |
| `LIVEKIT_AGENT_NAME` | optional | optional | Default `healthcare-rag-voice`; must match on both sides |
| `OPENAI_API_KEY`, `GOOGLE_API_KEY` | ✅ (existing) | ✅ | Same RAG providers |
| `OPENAI_MODEL`, `GEMINI_MODEL` | optional | optional | Defaults `gpt-5.4-mini`, `gemini-2.5-flash` |
| `SUPABASE_URL`, `SUPABASE_SECRET_KEY` | ✅ (existing) | ✅ | Server-side only, never in the browser |
| `VOICE_*` limits/models | optional | optional | See `.env.agent.example` |

The browser receives only the LiveKit URL and a 10-minute, single-room participant token.

---

## 4. Setup and deployment

> Nothing here was deployed or created automatically. Each step below needs your approval.

### 4.1 Supabase (once)
In the Supabase SQL Editor, run `SUPABASE_VOICE_SCHEMA.sql`. The existing table is untouched.

### 4.2 LiveKit Cloud project + CLI
```bash
winget install LiveKit.LiveKitCLI
lk cloud auth
```
Create a project on the free **Build** plan. Copy the URL, API key and secret from **Settings → API keys**.

### 4.3 Deploy the agent (from the repository root)
Copy `cloud_lite/.env.agent.example` to a local, git-ignored `cloud_lite/.env.agent`.
Fill in the OpenAI, Gemini, Supabase and `VOICE_CONTEXT_SECRET` values. **Delete the three `LIVEKIT_*`
credential lines**: LiveKit Cloud injects those and refuses them as secrets. Then:
```bash
lk agent create --secrets-file=cloud_lite/.env.agent
```
This builds the root `Dockerfile` and writes `livekit.toml`. For later code changes:
```bash
lk agent deploy
```
Pushing to GitHub does **not** redeploy the worker. To change secrets later:
`lk agent update-secrets --secrets "KEY=value"`. To list them: `lk agent secrets`.
To revert a deployment: `lk agent rollback`. Logs are in the LiveKit Cloud **Agents** dashboard.

### 4.4 Streamlit Community Cloud
Add `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` and `VOICE_CONTEXT_SECRET`
(the same value as on the agent) to the app's Secrets.
The entrypoint `cloud_lite/app.py` is unchanged. Without these secrets, the toggle
shows "not configured" and text chat is unaffected.

### 4.5 Local development
```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r cloud_lite/requirements-dev.txt
```
Terminal 1 (worker):
```bash
cd cloud_lite && ..\.venv\Scripts\python voice_agent.py dev
```
Terminal 2 (app, needs `.streamlit/secrets.toml` including the LiveKit keys):
```bash
cd cloud_lite && ..\.venv\Scripts\python -m streamlit run app.py
```

### 4.6 Tests
```bash
.venv\Scripts\python -m pytest -q tests
```

---

## 5. Cost estimate (3 users × 1 month)

**Assumptions:** 3 users × 10 sessions × 5 minutes = **150 voice-minutes per month**,
about one question per minute, about 700 spoken characters per answer, and about 30 questions per user.
Prices are as listed on livekit.com/pricing and /pricing/inference on 2026-11-16. Verify them before enabling.

| Item | Rate | Estimate |
|---|---|---|
| LiveKit Cloud Build plan | $0, no card; 1,000 agent-session min, 5,000 WebRTC min, 1 deployment, 5 concurrent sessions, **$2.50 inference credit** | $0 |
| Agent hosting | Included in Build (one deployment; no "always-on" option, so expect cold starts) | $0 |
| STT, Deepgram Nova-3 via LiveKit Inference | $0.0048/min (streams for the whole session) | 150 × 0.0048 ≈ **$0.72** |
| TTS, Deepgram Aura-2 | $30 per 1M chars | 150 × 700 × $0.00003 ≈ **$3.15** |
| Turn detector / VAD | Included for Cloud-deployed agents / runs locally | $0 |
| OpenAI `gpt-5.4-mini` (same as text) | ~4–5k input + ~400 output tokens per answer | ≈ 90 answers → **under $1** |
| Supabase | Existing project; a few KB to MB of short-lived rows | $0 |
| **Total** | | **≈ $4–5, of which ≈ $1.40 exceeds the $2.50 Build inference credit** |

What the pricing page does not say is whether Build hard-stops or allows overage
when the inference credit runs out. To stay at $0, keep usage at or below about 95
voice-minutes per month. If more is needed, the alternatives are LiveKit **Ship**
($50/month) or plugging in your own Deepgram key (new accounts get a trial
credit). **Cost guards built in:**
- 10-minute session cap;
- 20 questions per session;
- 6 sessions per user per hour;
- no preemptive generation, so no paid calls on partial speech;
- `max_participants=2`;
- rooms are deleted on end.

---

## 6. Privacy: what leaves the Streamlit app during voice

| Data | Goes to | Retention |
|---|---|---|
| Microphone audio (live) | LiveKit Cloud (WebRTC SFU) → agent → Deepgram STT via LiveKit Inference | Not stored by this app. `session.start(record=False)` disables LiveKit Cloud session recording/observability uploads |
| Spoken reply text | Deepgram TTS via LiveKit Inference | Not stored by this app |
| Retrieved excerpts + question | OpenAI (store=false) / Gemini, exactly as in text mode | Provider policy (unchanged) |
| Encrypted PDF chunk text | Supabase `rag_voice_sessions` | Wiped at session end; row deleted ≤ 2 h + 1 h cron |
| Finalized question/answer text + references | Supabase `rag_chat_history` | Existing 72-hour policy |

Logs contain only session IDs, timings and error classes, never transcripts,
answers or document text. Third-party processing by LiveKit, Deepgram, OpenAI
and Google is governed by their terms. Real-time speech processing cannot be
disabled while voice is used. Don't upload confidential patient data.

---

## 7. Test results and manual verification

**Automated (executed here, Python 3.12.10, Windows): 43 passed.** All external
services were mocked; no paid API was called.

| Area | Covered by |
|---|---|
| A: existing text workflow (real PDF bytes → PyMuPDF → TF-IDF → cited answer; real `app.py` via `streamlit.testing`) | `test_rag_service`, `test_app_streamlit` |
| C: voice uses the shared grounded pipeline with PDF evidence; placeholder LLM can't generate | `test_voice_agent` |
| D: real page references; hallucinated evidence IDs dropped | `test_rag_service` |
| E: unrelated question → abstention, no references | `test_rag_service` |
| F: follow-up retrieval and context labelled not-evidence | `test_rag_service`, `test_voice_agent` |
| G: interruption before an answer / during speech | `test_voice_agent` |
| H: toggle OFF by default, component not mounted; graceful "not configured" | `test_app_streamlit` |
| I: idempotent merge, restore through `HistoryStore`, recovery after closed tab | `test_voice_bridge`, `test_voice_ui_lifecycle` |
| J: owner-scoped queries, per-session keys, forged tokens rejected, other owner can't read or close, state reset on account switch | all four files |
| K: OpenAI/Gemini failure, Supabase outage, rate limits | `test_voice_agent`, `test_voice_ui_lifecycle` |
| L: Streamlit 1.67.0 + `livekit-api` + `cryptography` import together; agent CLI starts | `test_app_streamlit`, manual `voice_agent.py --help` |

The voice panel was also loaded in a browser: `livekit-client` 2.23.0 loads, and every API it uses
exists, with no console errors. Streamlit's component iframe policy includes `microphone`.

**Requires a real LiveKit project and microphone (NOT yet verified):**

1. **B (transcription)** and **real latency.** The worker logs `retrieval_s`,
   `generation_s` and `question_to_answer_s` per answer. LiveKit Cloud's agent
   dashboard shows STT, end-of-turn and TTS time-to-first-byte.
2. **G (barge-in).** Talk over a long answer. Audio should stop within about 0.5 s and the new question should be answered.
3. **K (microphone).** Deny mic permission and expect a friendly message. Disable
   the network for 10 s and expect "Reconnecting…". Stop the worker and expect the
   "agent did not join" message after 25 s.
4. **Cold start** on the Build plan (first session after idle).
5. **Two Google accounts** in two browsers. Confirm each sees only its own PDFs, voice answers and chats.

---

## 8. Known limitations

- Retrieval is the existing un-stemmed TF-IDF ("frequently" does not match
  "frequency"). Voice inherits this unchanged.
- The answer mode is fixed when a session starts. Restart the session to change it.
- Voice answers are generated in full before they are spoken, which favors
  grounding and citation validation over first-audio latency. A short
  non-medical filler ("Let me check the uploaded document") is spoken if
  generation takes more than 1.2 s.
- Typed questions asked during a voice session aren't part of the voice follow-up memory.
- If Streamlit re-mounts the page layout (for example, on processing new PDFs), the voice session ends by design.
- English STT only (`language="en"`).

## 9. Rollback

- **App:** don't merge the PR, or revert its merge commit. Removing the
  `LIVEKIT_*` Streamlit secrets also disables voice instantly without a redeploy.
- **Worker:** `lk agent delete`, or delete the deployment in the LiveKit dashboard.
- **Database:** run the commented rollback lines at the bottom of `SUPABASE_VOICE_SCHEMA.sql`.
  `rag_chat_history` is never modified by the schema change.
