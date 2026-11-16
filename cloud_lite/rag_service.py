"""Shared grounded-answer service used by BOTH text chat and the voice agent.

There is exactly one RAG implementation: retrieval, ranking, prompt building,
OpenAI-primary/Gemini-fallback generation, citation validation and abstention
all come from ``rag_core`` / ``llm_client``. This module only packages those
steps so the Streamlit app and the separately hosted LiveKit worker call the
same code. Nothing here imports Streamlit.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from llm_client import generate_answer
from rag_core import (
    Chunk,
    build_answer_prompt,
    build_chunks,
    build_evidence_text,
    build_index,
    build_reference_lines,
    format_evidence_cards,
    looks_incomplete,
    parse_answer_evidence,
    retrieve_chunks,
)

ABSTAIN_PHRASE = "The uploaded document does not provide enough evidence to answer this."
ANSWER_MODES = ("Smart Auto", "Deep Research", "Fast")
TOKEN_LIMITS = {"Fast": 1100, "Smart Auto": 1600, "Deep Research": 2600}
RETRIEVAL_TOP_K = 8


def is_abstention(text: str) -> bool:
    return ABSTAIN_PHRASE.casefold().rstrip(".") in (text or "").casefold()


# --------------------------------------------------------------------------- #
# Index construction and (de)serialisation
# --------------------------------------------------------------------------- #
def build_rag_data(pages: List[Dict]) -> Dict[str, Any]:
    """Same structure the Streamlit app has always kept in session state."""
    chunks = build_chunks(pages)
    if not chunks:
        return {"pages": pages, "chunks": [], "vectorizer": None, "matrix": None}
    vectorizer, matrix = build_index(chunks)
    return {"pages": pages, "chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}


def serialize_chunks(chunks: Sequence[Chunk]) -> List[Dict[str, Any]]:
    return [
        {"label": c.label, "source": c.source, "page": c.page, "text": c.text}
        for c in chunks
    ]


def rag_data_from_serialized_chunks(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Rebuild the identical TF-IDF index from transferred chunks.

    ``TfidfVectorizer`` fitting is deterministic, so the voice worker ranks
    evidence exactly as the Streamlit text chat does for the same PDFs.
    """
    chunks = [
        Chunk(
            label=int(item["label"]),
            source=str(item["source"]),
            page=int(item["page"]),
            text=str(item["text"]),
        )
        for item in items
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]
    if not chunks:
        return {"pages": [], "chunks": [], "vectorizer": None, "matrix": None}
    vectorizer, matrix = build_index(chunks)
    return {"pages": [], "chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}


# --------------------------------------------------------------------------- #
# Follow-up handling (voice only; text-mode semantics are untouched)
# --------------------------------------------------------------------------- #
_FOLLOW_UP_HINTS = re.compile(
    r"^\s*(and|also|so|then|what about|how about)\b|"
    r"\b(it|its|they|them|their|those|these|that one|such|same|above|"
    r"previous|earlier|how often|how long|how frequently|which ones?)\b",
    re.IGNORECASE,
)


def looks_like_follow_up(question: str) -> bool:
    words = re.findall(r"[A-Za-z0-9']+", question or "")
    if not words:
        return False
    return len(words) <= 6 or bool(_FOLLOW_UP_HINTS.search(question))


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


def build_retrieval_query(question: str, history: Sequence[Turn] = ()) -> str:
    """Augment a follow-up with the previous user question for RETRIEVAL only.

    "How frequently should those tests be performed?" alone does not contain
    the words that locate the evidence; appending the previous question does.
    This changes which chunks are retrieved, never the facts in the answer.
    """
    if not history or not looks_like_follow_up(question):
        return question
    previous = [t.content for t in history if t.role == "user" and t.content.strip()]
    if not previous:
        return question
    # The current question is repeated so its terms dominate the TF-IDF query;
    # the previous question only supplies the missing topic words.
    return f"{question}\n{question}\n{previous[-1]}"


def format_conversation_context(history: Sequence[Turn], max_turns: int = 4) -> str:
    lines = []
    for turn in list(history)[-max_turns:]:
        text = " ".join(turn.content.split())
        if len(text) > 400:
            text = text[:400].rstrip() + "..."
        speaker = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Grounded request / answer
# --------------------------------------------------------------------------- #
@dataclass
class PreparedRequest:
    question: str
    answer_mode: str
    prompt: str
    evidence_cards: List[Dict[str, Any]]
    max_tokens: int
    retrieval_seconds: float = 0.0


@dataclass
class GroundedAnswer:
    content: str
    provider: str
    model: str
    references: List[str]
    evidence_cards: List[Dict[str, Any]]
    completion_warning: bool
    abstained: bool
    timings: Dict[str, float] = field(default_factory=dict)

    def to_message(self) -> Dict[str, Any]:
        """Assistant message in the exact shape app.py has always stored."""
        return {
            "role": "assistant",
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "references": self.references,
            "evidence_cards": self.evidence_cards,
            "completion_warning": self.completion_warning,
        }


class NoDocumentsError(RuntimeError):
    """Raised when no processed PDF evidence is available."""


def prepare_grounded_request(
    question: str,
    rag_data: Optional[Dict[str, Any]],
    answer_mode: str,
    history: Sequence[Turn] = (),
) -> PreparedRequest:
    if not rag_data or not rag_data.get("chunks"):
        raise NoDocumentsError("Please upload and process at least one PDF first.")
    mode = answer_mode if answer_mode in TOKEN_LIMITS else "Smart Auto"
    started = time.perf_counter()
    retrieved = retrieve_chunks(
        question=build_retrieval_query(question, history),
        chunks=rag_data["chunks"],
        vectorizer=rag_data["vectorizer"],
        matrix=rag_data["matrix"],
        top_k=RETRIEVAL_TOP_K,
    )
    evidence_text = build_evidence_text(retrieved)
    prompt = build_answer_prompt(
        question=question,
        evidence_text=evidence_text,
        answer_mode=mode,
        conversation_context=format_conversation_context(history) if history else "",
    )
    return PreparedRequest(
        question=question,
        answer_mode=mode,
        prompt=prompt,
        evidence_cards=format_evidence_cards(retrieved),
        max_tokens=TOKEN_LIMITS[mode],
        retrieval_seconds=time.perf_counter() - started,
    )


def finalize_grounded_answer(
    raw_answer: str, provider: str, model: str, prepared: PreparedRequest
) -> GroundedAnswer:
    clean_answer, supported_cards = parse_answer_evidence(raw_answer, prepared.evidence_cards)
    abstained = is_abstention(clean_answer)
    return GroundedAnswer(
        content=clean_answer,
        provider=provider,
        model=model,
        references=[] if abstained else build_reference_lines(supported_cards),
        evidence_cards=[] if abstained else supported_cards,
        completion_warning=looks_incomplete(clean_answer),
        abstained=abstained,
        timings={"retrieval_s": round(prepared.retrieval_seconds, 3)},
    )


def answer_grounded(
    question: str,
    rag_data: Optional[Dict[str, Any]],
    answer_mode: str,
    history: Sequence[Turn] = (),
    generator: Callable[..., Any] = generate_answer,
) -> GroundedAnswer:
    """Retrieve -> prompt -> OpenAI/Gemini -> validate citations. Raises the
    ``llm_client`` provider errors unchanged so callers keep their handling."""
    prepared = prepare_grounded_request(question, rag_data, answer_mode, history)
    started = time.perf_counter()
    raw, provider, model = generator(prepared.prompt, max_tokens=prepared.max_tokens)
    result = finalize_grounded_answer(raw, provider, model, prepared)
    result.timings["generation_s"] = round(time.perf_counter() - started, 3)
    return result


# --------------------------------------------------------------------------- #
# Speech rendering (deterministic; never changes the medical content)
# --------------------------------------------------------------------------- #
def spoken_version(answer: str, max_chars: int = 900) -> str:
    """Turn the grounded written answer into plain speakable text.

    Only formatting is removed (markdown, bullets, headings). If the answer is
    long, whole leading sentences are kept and the listener is pointed to the
    full on-screen answer, so nothing is paraphrased by another model.
    """
    text = answer or ""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)      # markdown links
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"^\s*(?:[-*+•]|\d+[.)])\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`~|>]+", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentences: List[str] = []
    for line in lines:
        sentences.append(line if line[-1] in ".!?:;" else line + ".")
    text = " ".join(" ".join(sentences).split())
    if len(text) <= max_chars:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text)
    kept, total = [], 0
    for part in parts:
        if total + len(part) > max_chars and kept:
            break
        kept.append(part)
        total += len(part) + 1
    return " ".join(kept) + " The full answer and page references are shown on screen."
