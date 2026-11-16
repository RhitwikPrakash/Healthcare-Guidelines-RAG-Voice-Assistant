"""Tests A, C, D, E, F: shared RAG pipeline used by text and voice."""
from __future__ import annotations

import fitz
import pytest

from conftest import COLORECTAL_PAGES, grounded_generator


def test_text_prompt_is_unchanged_without_conversation_context():
    """Existing text mode must produce the exact same prompt as before."""
    from rag_core import build_answer_prompt

    legacy = build_answer_prompt("What tests?", "EVIDENCE", "Fast")
    assert legacy == build_answer_prompt("What tests?", "EVIDENCE", "Fast", conversation_context="")
    assert "Recent conversation" not in legacy
    assert "Answer style:\nGive a direct answer" in legacy
    assert "\n\nQuestion:\nWhat tests?" in legacy


def test_prepare_matches_original_text_pipeline(rag_data):
    """prepare_grounded_request == the steps app.py used to inline."""
    from rag_core import build_answer_prompt, build_evidence_text, format_evidence_cards, retrieve_chunks
    from rag_service import prepare_grounded_request

    q = "What are the recommended follow-up tests after stage three colon cancer?"
    prepared = prepare_grounded_request(q, rag_data, "Smart Auto")
    retrieved = retrieve_chunks(q, rag_data["chunks"], rag_data["vectorizer"], rag_data["matrix"], top_k=8)
    assert prepared.prompt == build_answer_prompt(q, build_evidence_text(retrieved), "Smart Auto")
    assert prepared.evidence_cards == format_evidence_cards(retrieved)
    assert prepared.max_tokens == 1600


def test_end_to_end_pdf_to_cited_answer(tmp_path):
    """Test A/D: real PDF bytes -> PyMuPDF -> chunks -> TF-IDF -> answer with true page refs."""
    from pdf_utils import extract_pdf_pages
    from rag_service import answer_grounded, build_rag_data

    doc = fitz.open()
    for page in COLORECTAL_PAGES:
        p = doc.new_page()
        p.insert_text((50, 72), page["text"].replace("\n\n", "\n"), fontsize=9)
    pdf_bytes = doc.tobytes()
    pages = extract_pdf_pages(pdf_bytes, "colorectal_guideline.pdf")
    assert [p["page"] for p in pages] == [1, 2, 3]

    data = build_rag_data(pages)
    gen = grounded_generator(labels="1")
    result = answer_grounded("What follow-up tests are recommended after stage III colon cancer?", data, "Fast", generator=gen)
    assert result.references, "expected a page reference"
    top = result.evidence_cards[0]
    assert top["source"] == "colorectal_guideline.pdf"
    assert top["page"] == 2  # the surveillance-tests page
    assert "CEA" in top["text"]
    assert result.references == [f"colorectal_guideline.pdf — page(s) {top['page']}"]
    assert gen.calls[0]["max_tokens"] == 1100


def test_voice_uses_document_evidence_only(rag_data):
    """Test C: the generator only ever receives a prompt built from PDF evidence."""
    from rag_service import answer_grounded

    gen = grounded_generator()
    answer_grounded("Which follow-up tests are recommended?", rag_data, "Smart Auto", generator=gen)
    prompt = gen.calls[0]["prompt"]
    assert "Answer ONLY from the evidence supplied below." in prompt
    assert "Source: colorectal_guideline.pdf" in prompt
    assert "CEA blood testing" in prompt


def test_invalid_evidence_ids_never_become_citations(rag_data):
    """Test D: hallucinated evidence IDs are dropped, real ones map to real pages."""
    from rag_service import answer_grounded

    result = answer_grounded("follow-up tests", rag_data, "Fast", generator=grounded_generator(labels="1, 99"))
    assert len(result.evidence_cards) == 1
    assert result.evidence_cards[0]["page"] in {4, 5}


def test_insufficient_evidence_abstains_without_references(rag_data):
    """Test E: unrelated tuberculosis question -> abstention, no citations."""
    from rag_service import ABSTAIN_PHRASE, answer_grounded, spoken_version

    gen = grounded_generator(labels="NONE", body=ABSTAIN_PHRASE)
    result = answer_grounded("What is the first-line treatment for tuberculosis?", rag_data, "Fast", generator=gen)
    assert result.abstained
    assert result.references == [] and result.evidence_cards == []
    assert "does not provide enough evidence" in spoken_version(result.content)


def test_follow_up_retrieval_uses_previous_question(rag_data):
    """Test F: 'how frequently ... those tests' retrieves the frequency page."""
    from rag_service import Turn, build_retrieval_query, looks_like_follow_up, prepare_grounded_request

    history = [
        Turn("user", "What are the recommended follow-up tests after stage three colon cancer treatment?"),
        Turn("assistant", "CEA testing, CT imaging and colonoscopy."),
    ]
    follow = "And how frequently should those tests be performed?"
    assert looks_like_follow_up(follow)
    assert "stage three colon cancer" in build_retrieval_query(follow, history)
    assert not looks_like_follow_up("What are the screening recommendations for colorectal cancer in adults?")

    prepared = prepare_grounded_request(follow, rag_data, "Smart Auto", history)
    # Both the tests page and the frequency page reach the model. (Exact rank
    # order is limited by the existing un-stemmed TF-IDF: "frequently" does not
    # match "Frequency"; text-mode retrieval semantics are intentionally unchanged.)
    pages = [card["page"] for card in prepared.evidence_cards]
    assert pages[0] == 4 and 5 in pages
    # Conversation is provided for interpretation but explicitly marked not-evidence.
    assert "it is NOT evidence" in prepared.prompt
    assert "Question:\nAnd how frequently should those tests be performed?" in prepared.prompt


def test_no_documents_raises(rag_data):
    from rag_service import NoDocumentsError, prepare_grounded_request

    with pytest.raises(NoDocumentsError):
        prepare_grounded_request("anything", None, "Fast")
    with pytest.raises(NoDocumentsError):
        prepare_grounded_request("anything", {"chunks": []}, "Fast")


def test_chunk_serialisation_reproduces_identical_ranking(rag_data):
    """The worker rebuilds the same index from transferred chunks."""
    from rag_service import prepare_grounded_request, rag_data_from_serialized_chunks, serialize_chunks

    rebuilt = rag_data_from_serialized_chunks(serialize_chunks(rag_data["chunks"]))
    q = "colonoscopy frequency after resection"
    assert prepare_grounded_request(q, rebuilt, "Fast").prompt == prepare_grounded_request(q, rag_data, "Fast").prompt


def test_spoken_version_strips_formatting_without_rewriting():
    from rag_service import spoken_version

    answer = "## Follow-up\n- **CEA** testing every 3 to 6 months\n- CT imaging yearly"
    spoken = spoken_version(answer)
    assert spoken == "Follow-up. CEA testing every 3 to 6 months. CT imaging yearly."
    long_answer = " ".join(f"Sentence number {i} about surveillance." for i in range(80))
    short = spoken_version(long_answer, max_chars=200)
    assert short.endswith("The full answer and page references are shown on screen.")
    assert short.startswith("Sentence number 0 about surveillance.")
