"""Tests A, H, J, L against the real Streamlit app using streamlit.testing.

Google OIDC is simulated by replacing ``st.user`` (the only identity source the
app trusts). OpenAI/Gemini and Supabase are mocked.
"""
from __future__ import annotations

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from conftest import APP_DIR, grounded_generator

APP = str(APP_DIR / "app.py")


class FakeUser:
    def __init__(self, sub=None, email="tester@example.com"):
        self._claims = {"sub": sub, "email": email, "name": "Tester"} if sub else {}

    @property
    def is_logged_in(self):
        return bool(self._claims)

    def to_dict(self):
        return dict(self._claims)

    def get(self, key, default=None):
        return self._claims.get(key, default)


@pytest.fixture
def no_external(monkeypatch):
    """No history storage, LiveKit configured or not per test."""
    for key in ("SUPABASE_URL", "SUPABASE_SECRET_KEY", "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(key, raising=False)
    import request_gate

    monkeypatch.setattr(request_gate, "wait_for_request_slot", lambda *a, **k: 0)


def _login(monkeypatch, sub="google-sub-a"):
    monkeypatch.setattr(st, "user", FakeUser(sub))


def test_signed_out_shows_google_login_only(monkeypatch, no_external):
    monkeypatch.setattr(st, "user", FakeUser(None))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any(b.label == "Continue with Google" for b in at.button)
    assert not at.toggle  # no voice controls before authentication


def test_voice_toggle_defaults_off_and_text_chat_works(monkeypatch, no_external, rag_data):
    """Test H + A: voice OFF by default; typed question -> grounded answer + refs."""
    import llm_client

    gen = grounded_generator(labels="1")
    monkeypatch.setattr(llm_client, "generate_answer", gen)
    _login(monkeypatch)

    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    toggle = at.toggle(key="voice_enabled")
    assert toggle.value is False
    assert not at.get("iframe")  # voice component not mounted -> no mic request

    at.session_state["rag_data"] = rag_data
    at.session_state["processed_key"] = "test-key"
    at.run()
    assert any("Active knowledge base" in i.value for i in at.info)
    at.chat_input[0].set_value("What follow-up tests are recommended after stage III colon cancer?").run()
    assert not at.exception
    assert len(gen.calls) == 1
    text = " ".join(m.value for m in at.markdown)
    assert "CEA testing" in text and "colorectal_guideline.pdf — page(s)" in text
    # Theme selector, answer modes and export remain available.
    assert {r.label for r in at.radio} >= {"Answer mode"}
    assert any(sb.label == "Interface color" for sb in at.selectbox)


def test_voice_toggle_without_configuration_degrades_gracefully(monkeypatch, no_external):
    _login(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.toggle(key="voice_enabled").set_value(True).run()
    assert not at.exception
    assert any("not configured" in c.value for c in at.caption)
    # Text input still present.
    assert at.chat_input


def test_session_state_is_reset_for_a_different_google_account(monkeypatch, no_external, rag_data):
    """Test J: user B in the same browser session never sees A's PDFs/chats/voice."""
    _login(monkeypatch, "google-sub-a")
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.session_state["rag_data"] = rag_data
    at.session_state["_voice_session"] = None
    at.run()
    assert at.session_state["rag_data"] is rag_data

    monkeypatch.setattr(st, "user", FakeUser("google-sub-b"))
    at.run()
    assert at.session_state["rag_data"] is None


def test_runtime_compatibility():
    """Test L: pinned Streamlit and the voice dependencies import together."""
    import importlib.metadata as md
    import sys

    assert md.version("streamlit") == "1.67.0"
    assert sys.version_info >= (3, 10)
    import livekit.api  # noqa: F401
    import cryptography  # noqa: F401
