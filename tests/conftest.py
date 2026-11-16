"""Shared fixtures. All external services (OpenAI, Gemini, Supabase, LiveKit)
are mocked: the suite never spends API credits or needs network access."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "cloud_lite"
sys.path.insert(0, str(APP_DIR))

# Deterministic fake configuration (NOT real credentials).
FAKE_ENV = {
    "LIVEKIT_URL": "wss://unit-test.livekit.cloud",
    "LIVEKIT_API_KEY": "APIunittestkey",
    "LIVEKIT_API_SECRET": "unit-test-secret-that-is-long-enough-0123456789",
    "VOICE_CONTEXT_SECRET": "unit-test-context-secret-0123456789abcdef",
    "SUPABASE_URL": "https://unittest.supabase.co",
    "SUPABASE_SECRET_KEY": "sb_secret_unit_test_only",
}

COLORECTAL_PAGES = [
    {
        "source": "colorectal_guideline.pdf",
        "page": 1,
        "text": "Colorectal cancer guideline overview. Scope and methods of this document.",
    },
    {
        "source": "colorectal_guideline.pdf",
        "page": 4,
        "text": (
            "Surveillance after curative treatment of stage III colon cancer.\n\n"
            "Recommended follow-up tests include CEA blood testing, CT imaging of the "
            "chest, abdomen and pelvis, and colonoscopy."
        ),
    },
    {
        "source": "colorectal_guideline.pdf",
        "page": 5,
        "text": (
            "Frequency of surveillance testing: CEA every 3 to 6 months for 2 years, "
            "then every 6 months up to 5 years. CT imaging every 6 to 12 months. "
            "Colonoscopy at 1 year after resection."
        ),
    },
]


@pytest.fixture
def fake_env(monkeypatch):
    for key, value in FAKE_ENV.items():
        monkeypatch.setenv(key, value)
    return FAKE_ENV


@pytest.fixture
def rag_data():
    from rag_service import build_rag_data

    return build_rag_data(COLORECTAL_PAGES)


@pytest.fixture
def owner_a():
    from history_store import history_owner_from_google_claims

    return history_owner_from_google_claims({"sub": "google-user-a"})


@pytest.fixture
def owner_b():
    from history_store import history_owner_from_google_claims

    return history_owner_from_google_claims({"sub": "google-user-b"})


def grounded_generator(labels="1", body=None):
    """Fake OpenAI/Gemini generator returning a grounded answer + metadata line."""
    calls = []

    def _gen(prompt, max_tokens=1600):
        calls.append({"prompt": prompt, "max_tokens": max_tokens})
        text = body or (
            "After treatment, follow-up includes CEA testing, CT imaging and colonoscopy."
        )
        return f"{text}\n[[SUPPORTING_EVIDENCE: {labels}]]", "OpenAI", "fake-model"

    _gen.calls = calls
    return _gen
