"""LiveKit voice worker for the Healthcare Guidelines RAG Assistant.

Pipeline: browser mic -> LiveKit WebRTC -> VAD + turn detection -> STT ->
**shared grounded RAG (rag_service.answer_grounded)** -> TTS -> browser speaker.

No conversational LLM writes medical content. ``llm_node`` is overridden so
every reply is produced by the same retrieval + OpenAI/Gemini + citation
validation + abstention code used by the text chat. The placeholder LLM below
only exists because LiveKit's pipeline requires an LLM object to schedule a
reply; it is never called.

Run locally:   python voice_agent.py dev
Deploy:        see VOICE_ASSISTANT.md (``lk agent create`` from the repo root)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncIterable, Dict, List, Optional

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatContext,
    JobContext,
    ModelSettings,
    cli,
    inference,
    llm,
    room_io,
)

from llm_client import ProviderRateLimitError, ProviderUnavailableError
from rag_service import (
    NoDocumentsError,
    Turn,
    answer_grounded,
    rag_data_from_serialized_chunks,
    spoken_version,
)
from voice_bridge import (
    VoiceConfigError,
    VoiceContextError,
    VoiceSessionMeta,
    VoiceSessionStore,
    VoiceSettings,
    decrypt_context,
    make_turn,
)

try:  # Local development convenience; LiveKit Cloud injects env vars itself.
    from dotenv import load_dotenv

    load_dotenv(".env.local")
except ImportError:  # pragma: no cover
    pass

# Never log transcripts, answers or document text: only ids and timings.
logger = logging.getLogger("healthcare-rag-voice")
logger.setLevel(logging.INFO)

ANSWER_TOPIC = "rag.answer"
STATUS_TOPIC = "rag.status"
FILLER_AFTER_SECONDS = 1.2
GREETING = (
    "Voice assistant ready. Ask a question about your processed guideline documents."
)


class _PipelinePlaceholderLLM(llm.LLM):
    """Satisfies the STT-LLM-TTS scheduler; ``llm_node`` is overridden so this
    is never invoked. Raising guarantees no un-grounded generation can occur."""

    @property
    def model(self) -> str:
        return "grounded-rag"

    @property
    def provider(self) -> str:
        return "healthcare-rag"

    def chat(self, *, chat_ctx: ChatContext, tools=None, conn_options=None, **_: Any):
        raise RuntimeError("Free-form LLM generation is disabled for the voice assistant.")


class GroundedVoiceAgent(Agent):
    def __init__(
        self,
        *,
        rag_data: Dict[str, Any],
        meta: VoiceSessionMeta,
        store: VoiceSessionStore,
        room: rtc.Room,
        max_questions: int,
    ) -> None:
        super().__init__(
            instructions=(
                "Voice front-end for a document-grounded healthcare guideline RAG system. "
                "Replies are produced exclusively by the retrieval pipeline."
            )
        )
        self._rag_data = rag_data
        self._meta = meta
        self._store = store
        self._room = room
        self._max_questions = max_questions
        self._questions = 0
        self._history: List[Turn] = []
        self._turns: List[Dict[str, Any]] = []
        self._seq = 0
        self._awaiting_speech_turn: Optional[str] = None
        self._write_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Persistence helpers (text only; never audio)
    # ------------------------------------------------------------------ #
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def _persist(self) -> bool:
        async with self._write_lock:
            snapshot = list(self._turns)
            try:
                await asyncio.to_thread(
                    self._store.save_turns, self._meta.session_id, self._meta.owner_id, snapshot
                )
                return True
            except Exception:
                logger.warning("voice turn persistence failed", extra={"session": self._meta.session_id})
                return False

    async def _send(self, topic: str, payload: Dict[str, Any]) -> None:
        try:
            await self._room.local_participant.send_text(
                json.dumps(payload, ensure_ascii=False),
                topic=topic,
                destination_identities=[self._meta.participant_identity],
            )
        except Exception:
            logger.info("could not deliver %s message to browser", topic)

    async def send_status(self, state: str, message: str = "") -> None:
        await self._send(STATUS_TOPIC, {"state": state, "message": message})

    def mark_speech_interrupted(self) -> None:
        """Called when the assistant's spoken reply was cut off by the user."""
        turn_id = self._awaiting_speech_turn
        self._awaiting_speech_turn = None
        if not turn_id:
            return
        for turn in self._turns:
            if turn["turn_id"] == turn_id:
                turn["speech_interrupted"] = True
        asyncio.create_task(self._persist())

    def speech_finished(self) -> None:
        self._awaiting_speech_turn = None

    # ------------------------------------------------------------------ #
    # The grounded reply
    # ------------------------------------------------------------------ #
    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[str]:
        question = ""
        for item in reversed(chat_ctx.items):
            if getattr(item, "type", None) == "message" and item.role == "user":
                question = (item.text_content or "").strip()
                break
        if not question:
            return

        if self._questions >= self._max_questions:
            yield "This voice session reached its question limit. Please end it and start a new one."
            return
        self._questions += 1

        user_turn = make_turn(self._next_seq(), "user", question)
        timings: Dict[str, float] = {}
        started = time.perf_counter()
        answered = False
        failure_note = "Interrupted before an answer was produced."
        failure_speech = ""
        result = None
        await self.send_status("thinking")

        task = asyncio.create_task(
            asyncio.to_thread(
                answer_grounded,
                question,
                self._rag_data,
                self._meta.answer_mode,
                list(self._history[-6:]),
            )
        )
        try:
            done, _ = await asyncio.wait({task}, timeout=FILLER_AFTER_SECONDS)
            if not done:
                # Non-medical filler keeps the conversation natural while the
                # grounded answer is generated; interruptible like any speech.
                yield "Let me check the uploaded document. "
            try:
                result = await task
            except NoDocumentsError:
                failure_note = "No processed PDF was available."
                failure_speech = "Please upload and process at least one PDF first."
            except ProviderRateLimitError:
                failure_note = "Not answered: AI providers were rate-limited or out of credit."
                failure_speech = "The answer providers are rate-limited right now. Please try again shortly."
            except ProviderUnavailableError:
                failure_note = "Not answered: answer generation was temporarily unavailable."
                failure_speech = "Answer generation is temporarily unavailable. Please try again shortly."
            except Exception:
                logger.exception("grounded answer failed", extra={"session": self._meta.session_id})
                failure_note = "Not answered: an internal error occurred."
                failure_speech = "Sorry, the answer could not be generated. Please try again."
            if result is None:
                await self.send_status("error", failure_note)
                yield failure_speech
                return
            answered = True
        finally:
            if not answered:
                # Interrupted, cancelled or failed before an answer existed: keep
                # the finalised question, clearly marked; never save a partial
                # or invented answer.
                if not task.done():
                    task.cancel()
                user_turn["voice_note"] = failure_note
                self._turns.append(user_turn)
                asyncio.ensure_future(self._persist())

        timings.update(result.timings)
        timings["question_to_answer_s"] = round(time.perf_counter() - started, 3)
        assistant_turn = make_turn(
            self._next_seq(),
            "assistant",
            result.content,
            references=result.references,
            evidence_cards=result.evidence_cards,
            completion_warning=result.completion_warning,
            provider=result.provider,
            model=result.model,
        )
        self._turns.extend([user_turn, assistant_turn])
        self._history.extend([Turn("user", question), Turn("assistant", result.content)])
        self._awaiting_speech_turn = assistant_turn["turn_id"]

        # Persist BEFORE notifying the browser so Streamlit's sync always finds it.
        persisted = await self._persist()
        await self._send(
            ANSWER_TOPIC,
            {
                "turn_id": assistant_turn["turn_id"],
                "question": question,
                "answer": result.content,
                "references": result.references,
                "abstained": result.abstained,
                "persisted": persisted,
                "timings": timings,
            },
        )
        logger.info("grounded voice answer", extra={"session": self._meta.session_id, **timings})
        yield spoken_version(result.content)


# ---------------------------------------------------------------------- #
# Worker entrypoint
# ---------------------------------------------------------------------- #
server = AgentServer()


def _build_session() -> AgentSession:
    stt_model = os.getenv("VOICE_STT_MODEL", "deepgram/nova-3")
    stt_fallback = os.getenv("VOICE_STT_FALLBACK", "assemblyai/universal-streaming")
    tts_model = os.getenv("VOICE_TTS_MODEL", "deepgram/aura-2")
    tts_fallback = os.getenv("VOICE_TTS_FALLBACK", "cartesia/sonic-3")
    tts_kwargs: Dict[str, Any] = {"model": tts_model}
    if os.getenv("VOICE_TTS_VOICE"):
        tts_kwargs["voice"] = os.environ["VOICE_TTS_VOICE"]
    if tts_fallback:
        tts_kwargs["fallback"] = tts_fallback
    stt_kwargs: Dict[str, Any] = {"model": stt_model, "language": "en"}
    if stt_fallback:
        stt_kwargs["fallback"] = stt_fallback

    return AgentSession(
        stt=inference.STT(**stt_kwargs),
        tts=inference.TTS(**tts_kwargs),
        llm=_PipelinePlaceholderLLM(),
        vad=inference.VAD(),
        turn_handling={
            # Hosted model; if the gateway fails turns commit on endpointing delay.
            "turn_detection": inference.TurnDetector(version="v1", local_fallback=False),
            "interruption": {"enabled": True, "min_duration": 0.4},
            # Preemptive generation would start paid RAG calls on partial
            # transcripts and could persist half-heard questions.
            "preemptive_generation": {"enabled": False},
        },
        user_away_timeout=None,
    )


@server.rtc_session(agent_name=os.getenv("LIVEKIT_AGENT_NAME", "healthcare-rag-voice"))
async def entrypoint(ctx: JobContext) -> None:
    settings = VoiceSettings.from_config()
    store = VoiceSessionStore.from_config()

    # 1. Trust ONLY the dispatch metadata signed into the server-minted token.
    try:
        meta = VoiceSessionMeta.from_json(ctx.job.metadata)
        if ctx.room.name != meta.room:
            raise VoiceConfigError("Room does not match the signed session.")
        if not store.is_configured or len(settings.context_secret) < 32:
            raise VoiceConfigError("Voice storage is not configured on the worker.")
        row = await asyncio.to_thread(store.get_session, meta.session_id, meta.owner_id, meta.room)
        if not row or row.get("status") == "ended" or not row.get("context_ciphertext"):
            raise VoiceContextError("Voice session not found, ended or expired.")
        if str(row.get("chat_id")) != meta.chat_id:
            raise VoiceContextError("Voice session chat mismatch.")
        payload = decrypt_context(
            settings.context_secret,
            meta.session_id,
            row["context_ciphertext"],
            max_age_seconds=settings.context_ttl_minutes * 60,
        )
        if payload.get("session_id") != meta.session_id or payload.get("owner_id") != meta.owner_id:
            raise VoiceContextError("Voice context binding mismatch.")
        rag_data = rag_data_from_serialized_chunks(payload.get("chunks") or [])
    except (VoiceConfigError, VoiceContextError) as exc:
        logger.warning("rejecting voice job: %s", exc.__class__.__name__)
        ctx.shutdown(reason="invalid voice session")
        return

    session = _build_session()
    agent = GroundedVoiceAgent(
        rag_data=rag_data,
        meta=meta,
        store=store,
        room=ctx.room,
        max_questions=settings.max_questions_per_session,
    )

    @session.on("conversation_item_added")
    def _on_item(ev) -> None:
        item = ev.item
        if getattr(item, "type", None) == "message" and item.role == "assistant":
            if item.interrupted:
                agent.mark_speech_interrupted()
            else:
                agent.speech_finished()

    @session.on("agent_state_changed")
    def _on_state(ev) -> None:
        asyncio.create_task(agent.send_status(str(ev.new_state)))

    @session.on("error")
    def _on_error(ev) -> None:
        source = type(getattr(ev, "source", None)).__name__
        asyncio.create_task(agent.send_status("error", f"Voice provider issue ({source})."))

    async def _on_shutdown(reason: str = "") -> None:
        try:
            await asyncio.to_thread(store.set_status, meta.session_id, meta.owner_id, "closed")
        except Exception:
            pass

    ctx.add_shutdown_callback(_on_shutdown)

    await session.start(
        room=ctx.room,
        agent=agent,
        room_options=room_io.RoomOptions(
            participant_identity=meta.participant_identity,
            participant_kinds=[rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD],
            text_input=False,
            close_on_disconnect=True,
            delete_room_on_close=True,
        ),
        # Do not upload session audio/transcripts to LiveKit Cloud observability.
        record=False,
    )
    try:
        await asyncio.to_thread(store.set_status, meta.session_id, meta.owner_id, "active")
    except Exception:
        pass
    await agent.send_status("ready")
    session.say(GREETING)

    async def _enforce_time_limit() -> None:
        await asyncio.sleep(settings.max_session_minutes * 60)
        await agent.send_status("limit", "Session time limit reached.")
        handle = session.say("This voice session reached its time limit. Goodbye.", allow_interruptions=False)
        try:
            await handle.wait_for_playout()
        finally:
            session.shutdown()

    limit_task = asyncio.create_task(_enforce_time_limit())

    async def _stop_limit(reason: str = "") -> None:
        limit_task.cancel()

    ctx.add_shutdown_callback(_stop_limit)


if __name__ == "__main__":
    cli.run_app(server)
