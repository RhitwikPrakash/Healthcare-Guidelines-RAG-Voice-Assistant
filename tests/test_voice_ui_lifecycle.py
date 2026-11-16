"""Streamlit-side voice lifecycle: start, rate limit, chat-switch guard,
end/logout cleanup and orphan recovery (Supabase + LiveKit mocked)."""
from __future__ import annotations

import json
import uuid

import pytest
import streamlit as st


class FakeVoiceDB:
    """Minimal in-memory PostgREST emulation for rag_voice_sessions."""

    def __init__(self):
        self.rows = {}
        self.calls = []

    def __call__(self, method, url, headers=None, params=None, json=None, timeout=None):
        params = params or {}
        self.calls.append((method, dict(params)))
        owner = params.get("owner_id", "")[3:]
        result = []
        if method == "POST":
            self.rows[json["id"]] = dict(json, created_at="now")
        elif method == "GET":
            rows = [r for r in self.rows.values() if r["owner_id"] == owner]
            if "id" in params:
                rows = [r for r in rows if r["id"] == params["id"][3:]]
            if params.get("status") == "neq.ended":
                rows = [r for r in rows if r["status"] != "ended"]
            result = rows
        elif method == "PATCH":
            row = self.rows.get(params["id"][3:])
            if row and row["owner_id"] == owner:
                row.update(json)

        class _Resp:
            content = b"x"

            def raise_for_status(self):
                return None

        resp = _Resp()
        resp.json = lambda: result
        return resp


@pytest.fixture
def env(fake_env, monkeypatch, rag_data, owner_a):
    import requests

    import voice_ui

    db = FakeVoiceDB()
    monkeypatch.setattr(requests, "request", db)
    deleted = []
    monkeypatch.setattr(voice_ui, "delete_room", lambda settings, room: deleted.append(room) or True)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    chat = {"id": str(uuid.uuid4()), "title": "New chat", "messages": []}
    st.session_state.chats = {chat["id"]: chat}
    st.session_state.active_chat_id = chat["id"]
    st.session_state.processed_key = "pdf-set-1"
    saved = []
    return {"db": db, "deleted": deleted, "chat": chat, "saved": saved, "save": saved.append,
            "rag": rag_data, "owner": owner_a, "ui": voice_ui}


def _start(env, request="req-1"):
    env["ui"]._start_voice_session(env["owner"], env["chat"], env["rag"], "Fast", request, env["save"])
    return st.session_state.get(env["ui"].SESSION_KEY)


def test_start_stores_encrypted_context_and_scoped_token(env):
    from livekit import api

    from voice_bridge import VoiceSettings, decrypt_context

    session = _start(env)
    assert session and session["request"] == "req-1" and session["chat_id"] == env["chat"]["id"]
    row = env["db"].rows[session["session_id"]]
    assert row["owner_id"] == env["owner"] and row["status"] == "pending"
    assert "CEA" not in row["context_ciphertext"]
    settings = VoiceSettings.from_config()
    payload = decrypt_context(settings.context_secret, session["session_id"], row["context_ciphertext"])
    assert payload["owner_id"] == env["owner"] and len(payload["chunks"]) == len(env["rag"]["chunks"])
    claims = api.TokenVerifier(settings.api_key, settings.api_secret).verify(session["token"])
    assert claims.video.room == session["room"] == row["room_name"]


def test_start_requires_processed_pdfs(env):
    env["ui"]._start_voice_session(env["owner"], env["chat"], {"chunks": []}, "Fast", "r", env["save"])
    assert env["ui"].current_session() is None
    assert "process at least one PDF" in st.session_state[env["ui"].NOTICE_KEY]["text"]


def test_hourly_rate_limit(env, monkeypatch):
    monkeypatch.setenv("VOICE_MAX_SESSIONS_PER_HOUR", "2")
    _start(env, "a")
    _start(env, "b")  # ends "a" (one live session) and starts "b"
    assert len(env["db"].rows) == 2
    assert _start(env, "c") is None  # third within the hour is refused
    assert "limit reached" in st.session_state[env["ui"].NOTICE_KEY]["text"]


def test_switching_chat_or_pdfs_ends_session_and_syncs_turns(env):
    from voice_bridge import make_turn

    session = _start(env)
    env["db"].rows[session["session_id"]]["turns"] = [
        make_turn(1, "user", "What follow-up tests?"),
        make_turn(2, "assistant", "CEA, CT and colonoscopy.", references=["g.pdf — page(s) 4"]),
    ]
    other = {"id": str(uuid.uuid4()), "title": "Other", "messages": []}
    st.session_state.chats[other["id"]] = other
    env["ui"].guard_session_binding(other["id"], env["save"])

    assert env["ui"].current_session() is None
    assert env["deleted"] == [session["room"]]
    # Turns landed in the ORIGINAL chat only; nothing leaked into the new one.
    assert [m["role"] for m in env["chat"]["messages"]] == ["user", "assistant"]
    assert other["messages"] == []
    assert env["chat"]["title"] == "What follow-up tests?"
    row = env["db"].rows[session["session_id"]]
    assert row["status"] == "ended" and row["context_ciphertext"] is None and row["turns"] == []

    session2 = _start(env, "req-2")
    st.session_state.processed_key = "pdf-set-2"
    env["ui"].guard_session_binding(env["chat"]["id"], env["save"])
    assert env["ui"].current_session() is None and session2["room"] in env["deleted"]


def test_sync_is_duplicate_free_across_repeated_events(env):
    from voice_bridge import make_turn

    session = _start(env)
    env["db"].rows[session["session_id"]]["turns"] = [make_turn(1, "user", "Q"), make_turn(2, "assistant", "A")]
    for _ in range(4):
        env["ui"].sync_active_session(env["save"])
    assert len(env["chat"]["messages"]) == 2
    assert len(env["saved"]) == 1  # saved only when something changed


def test_recover_orphaned_sessions_after_relogin(env):
    from voice_bridge import make_turn

    session = _start(env)
    row = env["db"].rows[session["session_id"]]
    row["turns"] = [make_turn(1, "user", "Q"), make_turn(2, "assistant", "A")]
    row["status"] = "closed"  # agent finished; browser tab was closed
    st.session_state.pop(env["ui"].SESSION_KEY)
    env["ui"].recover_unsynced_sessions(env["owner"], env["save"])
    assert len(env["chat"]["messages"]) == 2
    assert row["status"] == "ended" and row["context_ciphertext"] is None


def test_other_owner_cannot_read_or_close_session(env, owner_b):
    from voice_bridge import VoiceSessionStore

    session = _start(env)
    store = VoiceSessionStore.from_config()
    assert store.get_session(session["session_id"], owner_b, session["room"]) is None
    assert store.get_turns(session["session_id"], owner_b) == []
    store.close_session(session["session_id"], owner_b)
    assert env["db"].rows[session["session_id"]]["status"] == "pending"
