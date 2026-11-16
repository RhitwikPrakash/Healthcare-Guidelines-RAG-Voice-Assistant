"""Regression: the LiveKit worker image does NOT contain Streamlit.

Every module the worker imports (see Dockerfile) must load and read its
configuration from environment variables when ``import streamlit`` fails.
Runs in a subprocess so Streamlit is truly unavailable.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from conftest import APP_DIR, FAKE_ENV

WORKER_MODULES = ["rag_core", "rag_service", "llm_client", "history_store", "voice_bridge", "voice_agent"]


def test_worker_modules_load_and_configure_without_streamlit():
    script = textwrap.dedent(
        f"""
        import sys
        sys.modules["streamlit"] = None  # make `import streamlit` raise ImportError
        sys.path.insert(0, {str(APP_DIR)!r})
        for name in {WORKER_MODULES!r}:
            __import__(name)
        from voice_bridge import VoiceSettings, VoiceSessionStore
        from history_store import history_owner_from_google_claims
        s = VoiceSettings.from_config()
        assert s.is_configured, "settings not read from env"
        assert s.context_ttl_minutes == 120
        assert VoiceSessionStore.from_config().is_configured
        assert history_owner_from_google_claims({{"sub": "x"}})
        print("OK")
        """
    )
    env = {**os.environ, **FAKE_ENV}
    env.pop("HISTORY_NAMESPACE", None)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("OK")
