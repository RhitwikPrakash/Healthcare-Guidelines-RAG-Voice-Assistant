"""Streamlit side of the optional LiveKit voice assistant.

Everything security-sensitive happens here on the Streamlit SERVER: the owner
id comes only from Streamlit's verified Google OIDC claims, LiveKit tokens are
minted here, and document context is encrypted here. The browser component
only receives a short-lived single-room token and sends back UI events
(start / turn / ended). It never supplies content that is persisted: on a
"turn" event the server re-reads finalised turns from Supabase itself.

When the toggle is OFF nothing is rendered, no microphone permission is
requested and no LiveKit or speech API is contacted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import streamlit as st
import streamlit.components.v1 as components

from rag_service import serialize_chunks
from voice_bridge import (
    CONTEXT_VERSION,
    MAX_CONTEXT_CHUNKS,
    VoiceSessionStore,
    VoiceSettings,
    delete_room,
    encrypt_context,
    merge_voice_turns,
    mint_participant_token,
    new_session_meta,
    voice_feature_available,
)

_COMPONENT_DIR = Path(__file__).parent / "voice_component"
_voice_panel = components.declare_component("rag_voice_panel", path=str(_COMPONENT_DIR))

SESSION_KEY = "_voice_session"
HANDLED_KEY = "_voice_handled_events"
NOTICE_KEY = "_voice_notice"
TOGGLE_KEY = "voice_enabled"

SaveChat = Callable[[dict], None]


def _notice(text: str, error: bool = False) -> None:
    st.session_state[NOTICE_KEY] = {"text": text, "error": error}


def _handled() -> set:
    return st.session_state.setdefault(HANDLED_KEY, set())


def current_session() -> Optional[Dict]:
    return st.session_state.get(SESSION_KEY)


# --------------------------------------------------------------------------- #
# Sync finalised turns from Supabase into the bound chat
# --------------------------------------------------------------------------- #
def _merge_rows_into_chats(rows, chats: Dict[str, dict], save_chat: SaveChat) -> int:
    changed_total = 0
    for row in rows:
        chat = chats.get(str(row.get("chat_id")))
        turns = row.get("turns") if isinstance(row.get("turns"), list) else []
        if chat is None or not turns:
            continue
        changed = merge_voice_turns(chat, turns)
        if changed:
            if chat.get("title") == "New chat":
                first = next((t for t in turns if t.get("role") == "user"), None)
                if first:
                    compact = " ".join(str(first.get("content", "")).split())
                    chat["title"] = compact[:52] + ("…" if len(compact) > 52 else "")
            from datetime import datetime, timezone

            chat["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            save_chat(chat)
            changed_total += changed
    return changed_total


def sync_active_session(save_chat: SaveChat) -> int:
    session = current_session()
    if not session:
        return 0
    store = VoiceSessionStore.from_config()
    try:
        turns = store.get_turns(session["session_id"], session["owner_id"])
    except Exception:
        _notice("Voice turns could not be synced yet; they will be retried.", error=True)
        return 0
    return _merge_rows_into_chats(
        [{"chat_id": session["chat_id"], "turns": turns}], st.session_state.chats, save_chat
    )


def recover_unsynced_sessions(owner_id: Optional[str], save_chat: SaveChat) -> None:
    """After (re)login: merge finalised turns left by tabs that were closed
    mid-session, then wipe sessions whose agent has already finished."""
    if not owner_id or not voice_feature_available():
        return
    store = VoiceSessionStore.from_config()
    try:
        rows = store.list_open_sessions(owner_id)
        _merge_rows_into_chats(rows, st.session_state.chats, save_chat)
        for row in rows:
            if row.get("status") == "closed":
                store.close_session(str(row["id"]), owner_id)
    except Exception:
        pass  # Voice storage hiccups must never block the text application.


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
def end_voice_session(save_chat: SaveChat, reason: str = "") -> None:
    """Sync remaining turns, wipe server-side context and delete the room."""
    session = st.session_state.pop(SESSION_KEY, None)
    if not session:
        return
    store = VoiceSessionStore.from_config()
    try:
        turns = store.get_turns(session["session_id"], session["owner_id"])
        chats = st.session_state.get("chats") or {}
        _merge_rows_into_chats([{"chat_id": session["chat_id"], "turns": turns}], chats, save_chat)
    except Exception:
        pass
    try:
        store.close_session(session["session_id"], session["owner_id"])
    except Exception:
        pass  # Row still expires automatically (TTL + hourly cron).
    delete_room(VoiceSettings.from_config(), session["room"])
    if reason:
        _notice(reason)


def _start_voice_session(owner_id: str, chat: dict, rag_data: dict, answer_mode: str,
                         request_id: str, save_chat: SaveChat) -> None:
    settings = VoiceSettings.from_config()
    store = VoiceSessionStore.from_config()
    if not rag_data or not rag_data.get("chunks"):
        _notice("Please upload and process at least one PDF before starting voice.", error=True)
        return
    if len(rag_data["chunks"]) > MAX_CONTEXT_CHUNKS:
        _notice("These PDFs are too large for a voice session. Text chat still works.", error=True)
        return
    end_voice_session(save_chat)  # one live session per browser session
    try:
        if store.count_recent_sessions(owner_id, 60) >= settings.max_sessions_per_hour:
            _notice("Voice session limit reached for this hour. Text chat is still available.", error=True)
            return
        meta = new_session_meta(owner_id, chat["id"], answer_mode)
        ciphertext = encrypt_context(
            settings.context_secret,
            meta.session_id,
            {
                "v": CONTEXT_VERSION,
                "session_id": meta.session_id,
                "owner_id": owner_id,
                "chat_id": meta.chat_id,
                "chunks": serialize_chunks(rag_data["chunks"]),
            },
        )
        store.create_session(meta, ciphertext, settings.context_ttl_minutes)
        token = mint_participant_token(settings, meta)
    except Exception:
        _notice("A voice session could not be created right now. Text chat still works.", error=True)
        return
    st.session_state[SESSION_KEY] = {
        "session_id": meta.session_id,
        "owner_id": owner_id,
        "chat_id": meta.chat_id,
        "room": meta.room,
        "answer_mode": answer_mode,
        "processed_key": st.session_state.get("processed_key"),
        "token": token,
        "request": request_id,
        "url": settings.livekit_url,
    }
    st.session_state.pop(NOTICE_KEY, None)


def guard_session_binding(active_chat_id: str, save_chat: SaveChat) -> None:
    """End voice if the user switched chats or changed/cleared the PDF corpus,
    so a session never answers from another conversation's documents."""
    session = current_session()
    if not session:
        return
    if session["chat_id"] != active_chat_id:
        end_voice_session(save_chat, "Voice session ended because you switched chats.")
    elif session.get("processed_key") != st.session_state.get("processed_key"):
        end_voice_session(save_chat, "Voice session ended because the processed PDFs changed.")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_voice_panel(owner_id: Optional[str], chat: dict, rag_data: Optional[dict],
                       answer_mode: str, palette: dict, save_chat: SaveChat) -> None:
    enabled = st.toggle(
        "Enable Voice Assistant",
        key=TOGGLE_KEY,
        help="Real-time spoken questions and answers grounded in your processed PDFs.",
    )
    if not enabled:
        if current_session():
            end_voice_session(save_chat, "Voice session ended.")
        return

    if not owner_id:
        st.caption("Voice requires a verified Google account.")
        return
    if not voice_feature_available():
        st.caption("The voice assistant is not configured on this deployment. Text chat works normally.")
        return

    guard_session_binding(chat["id"], save_chat)
    session = current_session()
    notice = st.session_state.get(NOTICE_KEY) or {}
    event = _voice_panel(
        url=session["url"] if session else "",
        token=session["token"] if session else "",
        request=session["request"] if session else (notice.get("request") or ""),
        palette=palette,
        notice=notice.get("text", ""),
        notice_error=bool(notice.get("error")),
        key="rag_voice_panel",
        default=None,
    )

    if session:
        st.caption(
            f"Voice answer mode: **{session['answer_mode']}** (restart the session to change it). "
            "Only finalised text and references are saved, never audio."
        )
    elif not rag_data:
        st.caption("Process at least one PDF to use voice.")

    if not isinstance(event, dict) or not event.get("id") or event["id"] in _handled():
        return
    _handled().add(event["id"])
    kind = event.get("type")
    if kind == "start":
        request_id = str(event.get("request") or "")[:80]
        _start_voice_session(owner_id, chat, rag_data or {}, answer_mode, request_id, save_chat)
        if not current_session():
            notice = st.session_state.get(NOTICE_KEY) or {}
            notice["request"] = request_id
            st.session_state[NOTICE_KEY] = notice
        st.rerun()
    elif kind == "turn":
        # Messages are rendered after this panel in the same script run, so
        # the merged turns appear immediately without an extra rerun.
        sync_active_session(save_chat)
    elif kind == "ended":
        reason = str(event.get("reason") or "")
        end_voice_session(save_chat, "" if reason == "user_ended" else "Voice session disconnected.")
        st.rerun()
