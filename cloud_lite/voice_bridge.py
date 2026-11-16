"""Secure bridge between the Streamlit app and the separately hosted LiveKit agent.

Why this exists
---------------
The LiveKit worker runs in another process/host and cannot read Streamlit's
``st.session_state``. When a signed-in user starts a voice session, the
Streamlit SERVER:

1. Encrypts the user's currently processed PDF chunks (text + source/page only,
   never PDF bytes) with a per-session key derived from ``VOICE_CONTEXT_SECRET``
   and stores the ciphertext in ``public.rag_voice_sessions`` (2-hour TTL).
2. Mints a short-lived LiveKit participant token for one random room. The token
   embeds an agent dispatch whose metadata (session id, hashed owner id, chat id)
   is signed with the LiveKit API secret, so the browser cannot alter it.

The agent trusts only that signed dispatch metadata, re-checks it against the
database row (owner, room, expiry) and decrypts the context. Finalised voice
turns are written back to the same row; Streamlit (the single writer of
``rag_chat_history``) merges them into the user's chat, de-duplicated by turn id.

This module is imported by both sides and must not import Streamlit.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import re
import uuid
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from history_store import HistoryStore, _secret

AGENT_NAME_DEFAULT = "healthcare-rag-voice"
CONTEXT_VERSION = 1
METADATA_VERSION = 1
MAX_CONTEXT_CHUNKS = 3000
_ROOM_RE = re.compile(r"^ragvoice-[0-9a-f]{32}$")
_OWNER_RE = re.compile(r"^[0-9a-f]{64}$")


class VoiceConfigError(RuntimeError):
    """Voice is not configured or a value is invalid."""


class VoiceContextError(RuntimeError):
    """Document context could not be stored, found or decrypted."""


def _int_setting(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(_secret(name) or default)
    except ValueError:
        value = default
    return max(low, min(high, value))


@dataclass(frozen=True)
class VoiceSettings:
    livekit_url: str
    api_key: str
    api_secret: str
    # Shared ONLY between the Streamlit server and the agent; encrypts the PDF
    # context. Separate from LiveKit keys because LiveKit Cloud injects its
    # own (possibly different) API key pair into deployed agents.
    context_secret: str = ""
    agent_name: str = AGENT_NAME_DEFAULT
    context_ttl_minutes: int = 120
    token_ttl_minutes: int = 10
    max_session_minutes: int = 10
    max_sessions_per_hour: int = 6
    max_questions_per_session: int = 20

    @classmethod
    def from_config(cls) -> "VoiceSettings":
        return cls(
            livekit_url=_secret("LIVEKIT_URL"),
            api_key=_secret("LIVEKIT_API_KEY"),
            api_secret=_secret("LIVEKIT_API_SECRET"),
            context_secret=_secret("VOICE_CONTEXT_SECRET"),
            agent_name=_secret("LIVEKIT_AGENT_NAME") or AGENT_NAME_DEFAULT,
            # DB constraint allows <= 4h; cap lower to tolerate clock skew.
            context_ttl_minutes=_int_setting("VOICE_CONTEXT_TTL_MINUTES", 120, 15, 180),
            token_ttl_minutes=_int_setting("VOICE_TOKEN_TTL_MINUTES", 10, 2, 30),
            max_session_minutes=_int_setting("VOICE_MAX_SESSION_MINUTES", 10, 1, 60),
            max_sessions_per_hour=_int_setting("VOICE_MAX_SESSIONS_PER_HOUR", 6, 1, 60),
            max_questions_per_session=_int_setting("VOICE_MAX_QUESTIONS_PER_SESSION", 20, 1, 100),
        )

    @property
    def is_configured(self) -> bool:
        return bool(
            self.livekit_url.startswith("wss://")
            and self.api_key
            and len(self.api_secret) >= 16
            and len(self.context_secret) >= 32
        )

    @property
    def http_url(self) -> str:
        return "https://" + self.livekit_url[len("wss://"):]


def voice_feature_available() -> bool:
    """True only when LiveKit AND server-side Supabase storage are configured."""
    return VoiceSettings.from_config().is_configured and VoiceSessionStore.from_config().is_configured


# --------------------------------------------------------------------------- #
# Encryption of the transferred document context
# --------------------------------------------------------------------------- #
def _session_fernet(context_secret: str, session_id: str):
    from cryptography.fernet import Fernet

    if len(context_secret or "") < 32:
        raise VoiceConfigError("VOICE_CONTEXT_SECRET is missing or too short.")
    digest = hmac.new(
        context_secret.encode("utf-8"),
        b"healthcare-rag-voice-context-v1:" + session_id.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_context(context_secret: str, session_id: str, payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _session_fernet(context_secret, session_id).encrypt(zlib.compress(raw, 6)).decode("ascii")


def decrypt_context(
    context_secret: str, session_id: str, token: str, max_age_seconds: Optional[int] = None
) -> Dict[str, Any]:
    from cryptography.fernet import InvalidToken

    try:
        data = _session_fernet(context_secret, session_id).decrypt(
            token.encode("ascii"), ttl=max_age_seconds
        )
        payload = json.loads(zlib.decompress(data).decode("utf-8"))
    except (InvalidToken, ValueError, zlib.error, UnicodeError) as exc:
        raise VoiceContextError("Voice document context is invalid or expired.") from exc
    if not isinstance(payload, dict) or payload.get("v") != CONTEXT_VERSION:
        raise VoiceContextError("Unsupported voice document context.")
    return payload


# --------------------------------------------------------------------------- #
# Signed dispatch metadata
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VoiceSessionMeta:
    session_id: str
    owner_id: str
    chat_id: str
    room: str
    participant_identity: str
    answer_mode: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "v": METADATA_VERSION,
                "session_id": self.session_id,
                "owner_id": self.owner_id,
                "chat_id": self.chat_id,
                "room": self.room,
                "participant_identity": self.participant_identity,
                "answer_mode": self.answer_mode,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "VoiceSessionMeta":
        try:
            data = json.loads(raw or "")
        except ValueError as exc:
            raise VoiceConfigError("Invalid voice session metadata.") from exc
        if not isinstance(data, dict) or data.get("v") != METADATA_VERSION:
            raise VoiceConfigError("Unsupported voice session metadata.")
        try:
            meta = cls(
                session_id=str(uuid.UUID(str(data["session_id"]))),
                owner_id=str(data["owner_id"]),
                chat_id=str(uuid.UUID(str(data["chat_id"]))),
                room=str(data["room"]),
                participant_identity=str(data["participant_identity"]),
                answer_mode=str(data.get("answer_mode") or "Smart Auto"),
            )
        except (KeyError, ValueError) as exc:
            raise VoiceConfigError("Incomplete voice session metadata.") from exc
        if not _OWNER_RE.match(meta.owner_id) or not _ROOM_RE.match(meta.room):
            raise VoiceConfigError("Malformed voice session metadata.")
        return meta


def new_session_meta(owner_id: str, chat_id: str, answer_mode: str) -> VoiceSessionMeta:
    if not _OWNER_RE.match(owner_id or ""):
        raise VoiceConfigError("Voice requires a verified Google account.")
    session = uuid.uuid4()
    return VoiceSessionMeta(
        session_id=str(session),
        owner_id=owner_id,
        chat_id=str(uuid.UUID(chat_id)),
        room=f"ragvoice-{uuid.uuid4().hex}",
        # No email/name in LiveKit identities; random per session.
        participant_identity=f"user-{session.hex[:12]}",
        answer_mode=answer_mode,
    )


def mint_participant_token(settings: VoiceSettings, meta: VoiceSessionMeta) -> str:
    """Short-lived token: join ONE room, publish microphone only, no data publish."""
    from livekit import api

    token = (
        api.AccessToken(settings.api_key, settings.api_secret)
        .with_identity(meta.participant_identity)
        .with_name("Researcher")
        .with_ttl(timedelta(minutes=settings.token_ttl_minutes))
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=meta.room,
                can_publish=True,
                can_publish_sources=["microphone"],
                can_subscribe=True,
                can_publish_data=False,
                can_update_own_metadata=False,
            )
        )
        .with_room_config(
            api.RoomConfiguration(
                name=meta.room,
                empty_timeout=60,
                departure_timeout=15,
                max_participants=2,
                agents=[
                    api.RoomAgentDispatch(
                        agent_name=settings.agent_name,
                        metadata=meta.to_json(),
                    )
                ],
            )
        )
    )
    return token.to_jwt()


def delete_room(settings: VoiceSettings, room: str, timeout: float = 6.0) -> bool:
    """Best-effort server-side room deletion (ends the agent job too)."""
    if not settings.is_configured or not _ROOM_RE.match(room or ""):
        return False

    async def _run() -> None:
        from livekit import api

        lkapi = api.LiveKitAPI(settings.http_url, settings.api_key, settings.api_secret)
        try:
            await lkapi.room.delete_room(api.DeleteRoomRequest(room=room))
        finally:
            await lkapi.aclose()

    try:
        asyncio.run(asyncio.wait_for(_run(), timeout))
        return True
    except Exception:
        # Room may already be gone (empty_timeout) -- never surface details.
        return False


# --------------------------------------------------------------------------- #
# Supabase storage for short-lived voice sessions
# --------------------------------------------------------------------------- #
def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


class VoiceSessionStore(HistoryStore):
    """Server-only access to ``public.rag_voice_sessions`` (service role key)."""

    TABLE = "rag_voice_sessions"

    def create_session(
        self, meta: VoiceSessionMeta, ciphertext: str, ttl_minutes: int
    ) -> None:
        self._request(
            "POST",
            payload={
                "id": meta.session_id,
                "owner_id": meta.owner_id,
                "chat_id": meta.chat_id,
                "room_name": meta.room,
                "answer_mode": meta.answer_mode,
                "context_ciphertext": ciphertext,
                "turns": [],
                "status": "pending",
                "expires_at": _iso(datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)),
            },
            extra_headers={"Prefer": "return=minimal"},
        )

    def count_recent_sessions(self, owner_id: str, minutes: int = 60) -> int:
        since = _iso(datetime.now(timezone.utc) - timedelta(minutes=minutes))
        rows = self._request(
            "GET",
            params={
                "select": "id",
                "owner_id": "eq." + owner_id,
                "created_at": "gt." + since,
                "limit": "100",
            },
        )
        return len(rows) if isinstance(rows, list) else 0

    def get_session(self, session_id: str, owner_id: str, room: str) -> Optional[Dict[str, Any]]:
        rows = self._request(
            "GET",
            params={
                "select": "id,owner_id,chat_id,room_name,answer_mode,context_ciphertext,turns,status,expires_at",
                "id": "eq." + session_id,
                "owner_id": "eq." + owner_id,
                "room_name": "eq." + room,
                "expires_at": "gt." + _iso(datetime.now(timezone.utc)),
                "limit": "1",
            },
        )
        return rows[0] if isinstance(rows, list) and rows else None

    def list_open_sessions(self, owner_id: str) -> List[Dict[str, Any]]:
        rows = self._request(
            "GET",
            params={
                "select": "id,chat_id,room_name,turns,status",
                "owner_id": "eq." + owner_id,
                "status": "neq.ended",
                "expires_at": "gt." + _iso(datetime.now(timezone.utc)),
                "order": "created_at.asc",
                "limit": "20",
            },
        )
        return rows if isinstance(rows, list) else []

    def get_turns(self, session_id: str, owner_id: str) -> List[Dict[str, Any]]:
        rows = self._request(
            "GET",
            params={
                "select": "turns",
                "id": "eq." + session_id,
                "owner_id": "eq." + owner_id,
                "limit": "1",
            },
        )
        if isinstance(rows, list) and rows and isinstance(rows[0].get("turns"), list):
            return rows[0]["turns"]
        return []

    def _patch(self, session_id: str, owner_id: str, payload: Dict[str, Any]) -> None:
        self._request(
            "PATCH",
            params={"id": "eq." + session_id, "owner_id": "eq." + owner_id},
            payload=payload,
            extra_headers={"Prefer": "return=minimal"},
        )

    def save_turns(self, session_id: str, owner_id: str, turns: List[Dict[str, Any]]) -> None:
        """Agent-only writer of the ``turns`` column (one agent per room)."""
        self._patch(session_id, owner_id, {"turns": turns})

    def set_status(self, session_id: str, owner_id: str, status: str) -> None:
        if status not in ("pending", "active", "closed", "ended"):
            raise ValueError("invalid status")
        self._patch(session_id, owner_id, {"status": status})

    def close_session(self, session_id: str, owner_id: str) -> None:
        """Called by Streamlit after merging turns: wipe context and turns.

        The stub row is kept until expiry so the hourly rate limit can count it.
        """
        self._patch(
            session_id,
            owner_id,
            {"status": "ended", "context_ciphertext": None, "turns": []},
        )


# --------------------------------------------------------------------------- #
# Turn records and merging into chats (pure functions, unit tested)
# --------------------------------------------------------------------------- #
def make_turn(seq: int, role: str, content: str, **extra: Any) -> Dict[str, Any]:
    record = {
        "turn_id": str(uuid.uuid4()),
        "seq": int(seq),
        "role": role,
        "content": content,
        "created_at": _iso(datetime.now(timezone.utc)),
    }
    record.update(extra)
    return record


def turn_to_message(turn: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    role = turn.get("role")
    content = turn.get("content")
    turn_id = turn.get("turn_id")
    if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
        return None
    if not isinstance(turn_id, str):
        return None
    message: Dict[str, Any] = {
        "role": role,
        "content": content,
        "source": "voice",
        "voice_turn_id": turn_id,
    }
    if role == "assistant":
        message["references"] = [str(r) for r in turn.get("references") or []][:30]
        message["evidence_cards"] = [
            c for c in (turn.get("evidence_cards") or []) if isinstance(c, dict)
        ][:12]
        message["completion_warning"] = bool(turn.get("completion_warning"))
        message["provider"] = str(turn.get("provider") or "")
        message["model"] = str(turn.get("model") or "")
    if turn.get("speech_interrupted"):
        message["speech_interrupted"] = True
    if turn.get("voice_note"):
        message["voice_note"] = str(turn["voice_note"])[:200]
    return message


def merge_voice_turns(chat: Dict[str, Any], turns: List[Dict[str, Any]]) -> int:
    """Append unseen finalised voice turns to ``chat``; returns number of changes.

    Idempotent: already-merged ``voice_turn_id`` values are skipped, so UI
    reruns, reconnects and repeated syncs never duplicate messages. A turn that
    was already merged but later flagged as interrupted is updated in place.
    """
    messages = chat.setdefault("messages", [])
    by_id = {m.get("voice_turn_id"): m for m in messages if m.get("voice_turn_id")}
    added = 0
    ordered = sorted(
        (t for t in turns if isinstance(t, dict)), key=lambda t: int(t.get("seq") or 0)
    )
    for turn in ordered:
        message = turn_to_message(turn)
        if message is None:
            continue
        existing = by_id.get(message["voice_turn_id"])
        if existing is not None:
            for flag in ("speech_interrupted", "voice_note"):
                if flag in message and existing.get(flag) != message[flag]:
                    existing[flag] = message[flag]
                    added += 1
            continue
        messages.append(message)
        by_id[message["voice_turn_id"]] = message
        added += 1
    return added
