"""Tests C, G, K for the LiveKit worker logic (LiveKit/STT/TTS/LLM mocked)."""
from __future__ import annotations

import asyncio
import json
import time
import uuid

import pytest

pytest.importorskip("livekit.agents")

from conftest import grounded_generator  # noqa: E402


class FakeStore:
    def __init__(self, fail=False):
        self.saved = []
        self.fail = fail

    def save_turns(self, session_id, owner_id, turns):
        if self.fail:
            raise RuntimeError("supabase down")
        self.saved.append([dict(t) for t in turns])


class FakeLocalParticipant:
    def __init__(self):
        self.sent = []

    async def send_text(self, text, *, topic="", destination_identities=None, **_):
        self.sent.append({"topic": topic, "payload": json.loads(text), "to": destination_identities})


class FakeRoom:
    def __init__(self):
        self.local_participant = FakeLocalParticipant()


def _agent(rag_data, owner, store=None, max_questions=20):
    import voice_agent
    from voice_bridge import new_session_meta

    meta = new_session_meta(owner, str(uuid.uuid4()), "Fast")
    room = FakeRoom()
    agent = voice_agent.GroundedVoiceAgent(
        rag_data=rag_data, meta=meta, store=store or FakeStore(), room=room, max_questions=max_questions
    )
    return agent, meta, room


def _ctx(*user_texts):
    from livekit.agents import llm

    ctx = llm.ChatContext.empty()
    for text in user_texts:
        ctx.add_message(role="user", content=text)
    return ctx


async def _collect(agent, ctx):
    chunks = [chunk async for chunk in agent.llm_node(ctx, [], None)]
    await asyncio.sleep(0.05)  # let background persistence tasks finish
    return chunks


def test_placeholder_llm_can_never_generate():
    import voice_agent

    with pytest.raises(RuntimeError):
        voice_agent._PipelinePlaceholderLLM().chat(chat_ctx=_ctx("hi"))


def test_llm_node_speaks_grounded_rag_answer_and_persists(rag_data, owner_a, monkeypatch):
    import rag_service
    import voice_agent

    gen = grounded_generator(labels="1")
    real = rag_service.answer_grounded
    monkeypatch.setattr(voice_agent, "answer_grounded", lambda *a, **k: real(*a, generator=gen, **k))
    store = FakeStore()
    agent, meta, room = _agent(rag_data, owner_a, store)

    spoken = asyncio.run(_collect(agent, _ctx("What follow-up tests are recommended after stage III colon cancer?")))
    assert spoken[-1].startswith("After treatment, follow-up includes CEA testing")
    assert "SUPPORTING_EVIDENCE" not in "".join(spoken)
    # The prompt came from the shared grounded pipeline with PDF evidence.
    assert "Source: colorectal_guideline.pdf" in gen.calls[0]["prompt"]

    turns = store.saved[-1]
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[1]["references"] and turns[1]["evidence_cards"][0]["page"] in (4, 5)
    # Persisted before the browser was told about it; only to the bound participant.
    answer_msgs = [m for m in room.local_participant.sent if m["topic"] == "rag.answer"]
    assert answer_msgs[0]["payload"]["turn_id"] == turns[1]["turn_id"]
    assert answer_msgs[0]["payload"]["persisted"] is True
    assert all(m["to"] == [meta.participant_identity] for m in room.local_participant.sent)


def test_follow_up_question_uses_previous_turns(rag_data, owner_a, monkeypatch):
    """Test F through the agent: history from earlier voice turns drives retrieval."""
    import rag_service
    import voice_agent

    gen = grounded_generator(labels="1")
    real = rag_service.answer_grounded
    monkeypatch.setattr(voice_agent, "answer_grounded", lambda *a, **k: real(*a, generator=gen, **k))
    agent, _, _ = _agent(rag_data, owner_a)
    asyncio.run(_collect(agent, _ctx("What are the recommended follow-up tests after stage three colon cancer treatment?")))
    asyncio.run(_collect(agent, _ctx("And how frequently should those tests be performed?")))
    second_prompt = gen.calls[1]["prompt"]
    assert "Recent conversation" in second_prompt and "Frequency of surveillance testing" in second_prompt


def test_interrupt_before_answer_saves_marked_question_only(rag_data, owner_a, monkeypatch):
    """Test G: barge-in while the answer is being generated."""
    import voice_agent

    def slow(*_a, **_k):
        time.sleep(0.5)
        raise AssertionError("result must be discarded")

    monkeypatch.setattr(voice_agent, "answer_grounded", slow)
    monkeypatch.setattr(voice_agent, "FILLER_AFTER_SECONDS", 0.05)
    store = FakeStore()
    agent, _, _ = _agent(rag_data, owner_a, store)

    async def scenario():
        gen = agent.llm_node(_ctx("What follow-up tests are recommended?"), [], None)
        first = await gen.__anext__()
        assert first == "Let me check the uploaded document. "
        await gen.aclose()  # the framework closes the generator on interruption
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    turns = store.saved[-1]
    assert len(turns) == 1 and turns[0]["role"] == "user"
    assert turns[0]["voice_note"] == "Interrupted before an answer was produced."


def test_interrupted_speech_flags_complete_written_answer(rag_data, owner_a, monkeypatch):
    """Test G: the reply text existed; only the speech was cut off."""
    import rag_service
    import voice_agent

    real = rag_service.answer_grounded
    monkeypatch.setattr(voice_agent, "answer_grounded", lambda *a, **k: real(*a, generator=grounded_generator(), **k))
    store = FakeStore()
    agent, _, _ = _agent(rag_data, owner_a, store)

    async def scenario():
        await _collect(agent, _ctx("What follow-up tests are recommended?"))
        agent.mark_speech_interrupted()
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    assert store.saved[-1][1]["speech_interrupted"] is True


@pytest.mark.parametrize(
    "exc_name, note",
    [
        ("ProviderRateLimitError", "rate-limited"),
        ("ProviderUnavailableError", "temporarily unavailable"),
    ],
)
def test_provider_failures_are_spoken_and_never_saved_as_answers(rag_data, owner_a, monkeypatch, exc_name, note):
    """Test K: OpenAI and Gemini failures."""
    import llm_client
    import voice_agent

    exc = getattr(llm_client, exc_name)

    def failing(*_a, **_k):
        raise exc("boom with secret sk-should-not-leak")

    monkeypatch.setattr(voice_agent, "answer_grounded", failing)
    store = FakeStore()
    agent, _, room = _agent(rag_data, owner_a, store)
    spoken = asyncio.run(_collect(agent, _ctx("question?")))
    assert note in spoken[-1]
    assert "sk-should-not-leak" not in json.dumps(room.local_participant.sent)
    asyncio.run(asyncio.sleep(0))
    assert all(t["role"] == "user" for t in store.saved[-1])
    assert note.split()[0] in store.saved[-1][0]["voice_note"]


def test_supabase_outage_still_answers_and_reports(rag_data, owner_a, monkeypatch):
    """Test K: history storage unavailable -> answer still spoken, UI warned."""
    import rag_service
    import voice_agent

    real = rag_service.answer_grounded
    monkeypatch.setattr(voice_agent, "answer_grounded", lambda *a, **k: real(*a, generator=grounded_generator(), **k))
    agent, _, room = _agent(rag_data, owner_a, FakeStore(fail=True))
    spoken = asyncio.run(_collect(agent, _ctx("follow-up tests?")))
    assert spoken and "CEA" in spoken[-1]
    msg = [m for m in room.local_participant.sent if m["topic"] == "rag.answer"][0]
    assert msg["payload"]["persisted"] is False


def test_question_limit(rag_data, owner_a, monkeypatch):
    import voice_agent

    calls = []
    monkeypatch.setattr(voice_agent, "answer_grounded", lambda *a, **k: calls.append(1))
    agent, _, _ = _agent(rag_data, owner_a, max_questions=0)
    spoken = asyncio.run(_collect(agent, _ctx("q?")))
    assert "question limit" in spoken[0] and calls == []


def test_session_builds_with_official_inference_models(fake_env):
    import voice_agent

    async def build():
        return voice_agent._build_session()

    session = asyncio.run(build())
    assert isinstance(session.llm, voice_agent._PipelinePlaceholderLLM)
    assert session.stt is not None and session.tts is not None
