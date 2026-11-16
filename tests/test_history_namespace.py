"""HISTORY_NAMESPACE separates a second deployment's chats in the same table."""
from __future__ import annotations

import hashlib


def test_unset_namespace_keeps_original_owner_ids(monkeypatch):
    from history_store import history_owner_from_google_claims

    monkeypatch.delenv("HISTORY_NAMESPACE", raising=False)
    expected = hashlib.sha256(b"google-oidc:user-123").hexdigest()
    assert history_owner_from_google_claims({"sub": "user-123"}) == expected


def test_namespace_gives_different_but_stable_owner_ids(monkeypatch):
    from history_store import history_owner_from_google_claims

    monkeypatch.delenv("HISTORY_NAMESPACE", raising=False)
    original = history_owner_from_google_claims({"sub": "user-123"})
    monkeypatch.setenv("HISTORY_NAMESPACE", "voice-app")
    voice = history_owner_from_google_claims({"sub": "user-123"})
    assert voice != original
    assert voice == history_owner_from_google_claims({"sub": "user-123"})
    assert len(voice) == 64  # still a valid owner id for the voice bridge
    assert history_owner_from_google_claims({"sub": "user-456"}) != voice
