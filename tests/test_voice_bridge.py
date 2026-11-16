"""Tests I, J (+ security): context encryption, token scope, metadata
validation, Supabase owner scoping and duplicate-free turn merging."""
from __future__ import annotations

import json
import uuid

import pytest


# --------------------------- encryption / isolation ------------------------- #
def test_context_roundtrip_and_session_binding(fake_env):
    from voice_bridge import VoiceContextError, decrypt_context, encrypt_context

    secret = fake_env["VOICE_CONTEXT_SECRET"]
    sid = str(uuid.uuid4())
    payload = {"v": 1, "session_id": sid, "chunks": [{"label": 1, "source": "a.pdf", "page": 3, "text": "x"}]}
    token = encrypt_context(secret, sid, payload)
    assert "a.pdf" not in token  # ciphertext only
    assert decrypt_context(secret, sid, token) == payload

    # Another session id (i.e. another user's/room's key) cannot decrypt it.
    with pytest.raises(VoiceContextError):
        decrypt_context(secret, str(uuid.uuid4()), token)
    with pytest.raises(VoiceContextError):
        decrypt_context("a-different-context-secret-0123456789abcdef", sid, token)
    tampered = token[:-6] + ("A" if token[-6] != "A" else "B") + token[-5:]
    with pytest.raises(VoiceContextError):
        decrypt_context(secret, sid, tampered)


def test_metadata_validation_rejects_malformed(owner_a):
    from voice_bridge import VoiceConfigError, VoiceSessionMeta, new_session_meta

    meta = new_session_meta(owner_a, str(uuid.uuid4()), "Fast")
    assert VoiceSessionMeta.from_json(meta.to_json()) == meta
    assert owner_a not in meta.participant_identity  # no account data in identity

    for bad in (
        "", "not json", json.dumps({"v": 99}),
        meta.to_json().replace(meta.room, "someone-elses-room"),
        meta.to_json().replace(owner_a, "spoofed-owner"),
    ):
        with pytest.raises(VoiceConfigError):
            VoiceSessionMeta.from_json(bad)
    with pytest.raises(VoiceConfigError):
        new_session_meta("", str(uuid.uuid4()), "Fast")


def test_unique_rooms_per_session(owner_a):
    from voice_bridge import new_session_meta

    chat = str(uuid.uuid4())
    metas = {new_session_meta(owner_a, chat, "Fast").room for _ in range(50)}
    assert len(metas) == 50


# ------------------------------- token scope -------------------------------- #
def test_participant_token_is_minimal_and_signed(fake_env, owner_a):
    from livekit import api

    from voice_bridge import VoiceSessionMeta, VoiceSettings, mint_participant_token, new_session_meta

    settings = VoiceSettings.from_config()
    assert settings.is_configured
    meta = new_session_meta(owner_a, str(uuid.uuid4()), "Deep Research")
    jwt = mint_participant_token(settings, meta)
    assert fake_env["LIVEKIT_API_SECRET"] not in jwt

    claims = api.TokenVerifier(settings.api_key, settings.api_secret).verify(jwt)
    assert claims.identity == meta.participant_identity
    video = claims.video
    assert video.room_join and video.room == meta.room
    assert video.can_publish_sources == ["microphone"]
    assert video.can_publish_data is False
    assert not video.room_admin and not video.room_create and not video.can_update_own_metadata
    dispatch = claims.room_config.agents[0]
    assert dispatch.agent_name == settings.agent_name
    assert VoiceSessionMeta.from_json(dispatch.metadata) == meta
    assert claims.room_config.max_participants == 2

    # A token signed with another secret is rejected (cannot forge dispatch metadata).
    with pytest.raises(Exception):
        api.TokenVerifier(settings.api_key, "attacker-secret-0123456789abcdef").verify(jwt)


def test_settings_unconfigured_without_env(monkeypatch):
    from voice_bridge import VoiceSettings

    for key in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(key, raising=False)
    assert not VoiceSettings.from_config().is_configured
    monkeypatch.setenv("LIVEKIT_URL", "http://insecure.example")
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "x" * 32)
    assert not VoiceSettings.from_config().is_configured  # wss:// required


# ------------------------- Supabase owner scoping (J) ----------------------- #
class _Recorder:
    def __init__(self, response=None):
        self.calls = []
        self.response = response if response is not None else []

    def __call__(self, method, url, headers=None, params=None, json=None, timeout=None):
        self.calls.append({"method": method, "url": url, "params": params or {}, "json": json, "headers": headers})

        class _Resp:
            content = b"[]"
            status_code = 200

            def raise_for_status(self):
                return None

        resp = _Resp()
        resp.json = lambda: self.response
        return resp


def test_every_voice_store_query_is_owner_scoped(fake_env, owner_a, monkeypatch):
    import requests

    from voice_bridge import VoiceSessionStore, new_session_meta

    rec = _Recorder()
    monkeypatch.setattr(requests, "request", rec)
    store = VoiceSessionStore.from_config()
    assert store.is_configured
    meta = new_session_meta(owner_a, str(uuid.uuid4()), "Fast")

    store.create_session(meta, "ciphertext", 120)
    store.count_recent_sessions(owner_a)
    store.get_session(meta.session_id, owner_a, meta.room)
    store.list_open_sessions(owner_a)
    store.get_turns(meta.session_id, owner_a)
    store.save_turns(meta.session_id, owner_a, [])
    store.set_status(meta.session_id, owner_a, "active")
    store.close_session(meta.session_id, owner_a)

    for call in rec.calls:
        assert call["url"].endswith("/rest/v1/rag_voice_sessions")
        assert call["headers"]["apikey"] == fake_env["SUPABASE_SECRET_KEY"]
        if call["method"] == "POST":
            assert call["json"]["owner_id"] == owner_a
        else:
            assert call["params"]["owner_id"] == "eq." + owner_a
    get_session = rec.calls[2]["params"]
    assert get_session["room_name"] == "eq." + meta.room and get_session["expires_at"].startswith("gt.")
    closed = rec.calls[-1]["json"]
    assert closed == {"status": "ended", "context_ciphertext": None, "turns": []}


def test_store_refuses_non_supabase_or_public_keys():
    from voice_bridge import VoiceSessionStore

    assert not VoiceSessionStore("http://unittest.supabase.co", "sb_secret_x").is_configured
    assert not VoiceSessionStore("https://evil.example.com", "sb_secret_x").is_configured
    assert not VoiceSessionStore("https://unittest.supabase.co", "sb_publishable_x").is_configured


# ------------------------ duplicate-free merging (I) ------------------------ #
def _turns():
    from voice_bridge import make_turn

    q = make_turn(1, "user", "What follow-up tests are recommended?")
    a = make_turn(
        2, "assistant", "CEA testing, CT imaging and colonoscopy.",
        references=["colorectal_guideline.pdf — page(s) 4"],
        evidence_cards=[{"label": 1, "source": "colorectal_guideline.pdf", "page": 4, "text": "..."}],
        completion_warning=False,
    )
    return [q, a]


def test_merge_is_idempotent_across_reruns_and_reconnects():
    from voice_bridge import merge_voice_turns

    chat = {"id": "c", "messages": [{"role": "user", "content": "typed question"}]}
    turns = _turns()
    assert merge_voice_turns(chat, turns) == 2
    for _ in range(5):  # reruns / repeated sync events
        assert merge_voice_turns(chat, list(reversed(turns))) == 0
    assert [m["role"] for m in chat["messages"]] == ["user", "user", "assistant"]
    assistant = chat["messages"][-1]
    assert assistant["source"] == "voice"
    assert assistant["references"] == ["colorectal_guideline.pdf — page(s) 4"]


def test_interruption_flag_updates_in_place_without_duplicate():
    from voice_bridge import merge_voice_turns

    chat = {"messages": []}
    turns = _turns()
    merge_voice_turns(chat, turns)
    turns[1]["speech_interrupted"] = True
    assert merge_voice_turns(chat, turns) == 1
    assert len(chat["messages"]) == 2 and chat["messages"][1]["speech_interrupted"] is True


def test_partial_or_malformed_turns_are_not_saved():
    from voice_bridge import merge_voice_turns

    chat = {"messages": []}
    bad = [
        {"turn_id": "1", "seq": 1, "role": "user", "content": "   "},
        {"turn_id": "2", "seq": 2, "role": "system", "content": "inject"},
        {"seq": 3, "role": "assistant", "content": "no id"},
        "not a dict",
    ]
    assert merge_voice_turns(chat, bad) == 0 and chat["messages"] == []


def test_restored_history_keeps_voice_messages(fake_env, owner_a, monkeypatch):
    """Test I: voice messages survive the existing Supabase load filter."""
    import requests

    from history_store import HistoryStore
    from voice_bridge import merge_voice_turns

    chat = {"id": str(uuid.uuid4()), "title": "t", "messages": []}
    merge_voice_turns(chat, _turns())
    rec = _Recorder(response=[{"id": chat["id"], "title": "t", "messages": chat["messages"],
                               "created_at": "x", "updated_at": "y"}])
    monkeypatch.setattr(requests, "request", rec)
    store = HistoryStore.from_config()
    store.upsert_chat(owner_a, chat)
    assert rec.calls[0]["json"]["owner_id"] == owner_a
    loaded = store.load_chats(owner_a)
    assert loaded[0]["messages"][1]["voice_turn_id"] == chat["messages"][1]["voice_turn_id"]
    assert rec.calls[1]["params"]["owner_id"] == "eq." + owner_a
