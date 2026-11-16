-- Healthcare Guidelines RAG Assistant: OPTIONAL voice-assistant bridge table.
-- Run once in Supabase Dashboard > SQL Editor ONLY if you enable the LiveKit
-- voice assistant. The existing public.rag_chat_history table is NOT changed.
--
-- Purpose: short-lived hand-off between the Streamlit server and the separately
-- hosted LiveKit agent worker.
--   * context_ciphertext: the user's currently processed PDF chunks (text and
--     source/page only, never PDF bytes), Fernet-encrypted with a per-session
--     key derived from VOICE_CONTEXT_SECRET. Wiped as soon as the session ends.
--   * turns: finalised voice transcript/answer TEXT + references (never audio),
--     written by the agent and merged into rag_chat_history by Streamlit.
-- Rows expire 2 hours after creation and are deleted hourly by cron.

create table if not exists public.rag_voice_sessions (
    id uuid primary key,
    owner_id text not null,
    chat_id uuid not null,
    room_name text not null unique,
    answer_mode text not null default 'Smart Auto',
    context_ciphertext text,
    turns jsonb not null default '[]'::jsonb,
    status text not null default 'pending'
        check (status in ('pending', 'active', 'closed', 'ended')),
    created_at timestamptz not null default now(),
    expires_at timestamptz not null default (now() + interval '2 hours'),
    constraint rag_voice_sessions_max_ttl
        check (expires_at <= created_at + interval '4 hours')
);

create index if not exists rag_voice_sessions_owner_created_idx
    on public.rag_voice_sessions (owner_id, created_at desc);
create index if not exists rag_voice_sessions_expiry_idx
    on public.rag_voice_sessions (expires_at);

-- Server-only access, exactly like rag_chat_history.
alter table public.rag_voice_sessions enable row level security;
revoke all on public.rag_voice_sessions from anon, authenticated;
grant select, insert, update, delete on public.rag_voice_sessions to service_role;

-- Hourly cleanup (requires the Supabase Cron module, already used for chat history).
select cron.schedule(
    'hourly-rag-voice-session-expiry',
    '15 * * * *',
    $$delete from public.rag_voice_sessions where expires_at <= now()$$
);

-- Rollback (removes ONLY the voice table and its job):
--   select cron.unschedule('hourly-rag-voice-session-expiry');
--   drop table if exists public.rag_voice_sessions;
