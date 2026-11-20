from __future__ import annotations

import hashlib
import itertools
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import wait
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from arkive_rag import induction
from arkive_rag.compiler import compile_corpus
from arkive_rag.induction import (
    EVIDENCE_WINDOW_CHARS,
    FEEDBACK_BATCH_CHARS,
    MAX_CURATOR_NOTES_PROMPT_CHARS,
    MAX_GUIDE_TEXT_CHARS,
    MAX_PROMPT_CHARS,
    MAX_VERSION_PARTNERS,
    CanonicaliseTypes,
    Checked,
    CorpusView,
    ExtractFeedbackLessons,
    InductionExecutor,
    OrganiseCorpus,
    TypeLabel,
    WriteGlossary,
    WriteGuide,
    canonical_aliases,
    check_facets,
    check_types,
    generate_checked,
    induce,
    merge_facets,
    term_clusters,
)
from arkive_rag.llm import StructuredRequest
from arkive_rag.models import (
    CanonicalTypeDraft,
    CompiledCorpus,
    CorpusDocument,
    CuratorNote,
    DocumentDate,
    DocumentReading,
    DocumentRoleDraft,
    Facet,
    FacetDraft,
    FacetValue,
    FacetValueDraft,
    FeedbackLessonDraft,
    FeedbackLessonsDraft,
    GlossaryDraft,
    GlossaryEntryDraft,
    ObservationCard,
    OrganisationDraft,
    ReadSection,
    RejectedOutputError,
    SenseDraft,
    StatedAttribute,
    TermObservation,
    TypeCanonicalisationDraft,
    document_id,
)
from tests.support import (
    BATHYMETRY_CORPUS,
    ESTUARY_PROPOSAL,
    HARBOUR_MEMO,
    HOUSE_DISCLOSURE,
    LOYALTY_BRIEF,
    MARKETING_CORPUS,
    SPRING_BRIEF,
    SPRING_REVIEW,
    FakeGateway,
    sec,
)


def compile_files(
    root: Path, files: dict[str, str], gateway: FakeGateway | None = None
) -> CompiledCorpus:
    """Compile on one worker, so scripted responses meet requests in submission order."""
    for rel_path, text in files.items():
        (root / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (root / rel_path).write_text(text, encoding="utf-8")
    return compile_corpus(root, gateway or FakeGateway(), topic="t", title="T", max_concurrency=1)


def facets_by_doc(corpus: CompiledCorpus) -> dict[str, dict[str, tuple[str, ...]]]:
    return {d.rel_path: d.facets for d in corpus.documents}


def reading(
    markdowns: Sequence[str],
    terms: Sequence[TermObservation] = (),
    *,
    title: str = "Memo",
    deliverable_type: str = "memorandum",
    summary: str = "Memo.",
    date: str | None = None,
    sha256: str | None = None,
    attributes: Sequence[StatedAttribute] = (),
    headings: Sequence[str] | None = None,
) -> DocumentReading:
    """A reading whose source hash, unless given, is that of its text and title: readings that
    differ are different files, and the same text is the same bytes."""
    source = repr((tuple(markdowns), title, tuple(headings or ()))).encode()
    return DocumentReading(
        sha256=sha256 or hashlib.sha256(source).hexdigest(),
        format="md",
        is_curator_note=False,
        sections=tuple(
            ReadSection(
                ordinal=i,
                heading_path=headings[i - 1] if headings else f"Part {i}",
                level=1,
                page_start=1,
                page_end=1,
                markdown=markdown,
                pieces=(),
            )
            for i, markdown in enumerate(markdowns, start=1)
        ),
        pages=(),
        card=ObservationCard(
            title=title,
            doc_date=date,
            deliverable_type=deliverable_type,
            summary=summary,
            attributes=tuple(attributes),
            terms=tuple(terms),
            craft=(),
        ),
        date=None if date is None else DocumentDate(iso=date, basis="printed"),
        discarded_quotes=0,
    )


def view_of(*docs: tuple[str, DocumentReading]) -> CorpusView:
    """Documents in argument order, which must be path order."""
    return CorpusView(tuple(CorpusDocument(document_id(p), p, r) for p, r in docs))


def prompt_chars(request: Any) -> int:
    return len(request.instructions) + len(request.text())


# Validate and retry.


def test_generate_checked_reruns_once_naming_the_rejections_then_raises() -> None:
    feedbacks: list[str | None] = []

    def produce(feedback: str | None) -> int:
        feedbacks.append(feedback)
        return len(feedbacks)

    def accept_second(value: int) -> Checked[int]:
        return Checked.accept(value) if value == 2 else Checked.reject(["too early"])

    assert generate_checked("probe", produce, accept_second) == 2
    assert feedbacks[0] is None and feedbacks[1] is not None and "- too early" in feedbacks[1]

    feedbacks.clear()
    with pytest.raises(RejectedOutputError, match="probe still rejected after one rerun: never"):
        generate_checked("probe", produce, lambda value: Checked.reject(["never"]))
    assert len(feedbacks) == 2

    # A reduced result is accepted only after the rerun.
    feedbacks.clear()
    reduced = generate_checked("probe", produce, lambda v: Checked.reduce(["lost one"], v * 10))
    assert (reduced, len(feedbacks)) == (20, 2)


def test_guide_citation_rerun_lists_only_the_tried_documents_tokens_then_raises(
    tmp_path: Path,
) -> None:
    files = {
        "p1/memo.md": "# Results\n\nSeagrass mapped.",
        "p1/AGENTS.md": "# Note\n\nDraft.",
        "p2/other.md": "# Other\n\nKelp mapped.\n\n## More\n\nStill kelp.",
    }
    memo, note = document_id("p1/memo.md"), document_id("p1/AGENTS.md")
    valid = f"[{memo}:{sec('p1/memo.md')}]"
    gateway = FakeGateway(guide_script=[f"See [{memo}:bad_sec].", f"See {valid}."])

    corpus = compile_files(tmp_path / "ok", files, gateway)

    assert corpus.guide_markdown == f"See {valid}."
    first, rerun = gateway.of_type(WriteGuide)
    assert first.feedback is None and rerun.feedback is not None
    assert f"[{memo}:bad_sec]" in rerun.feedback and valid in rerun.feedback
    # Only the document the guide tried to cite: never every section of the corpus.
    assert document_id("p2/other.md") not in rerun.feedback
    # The curator note is not citable: citing it names no known document, so the rerun shows
    # sample catalog tokens instead.
    note_token = f"[{note}:{sec('p1/AGENTS.md')}]"
    stubborn = FakeGateway(guide_script=[f"See {note_token}.", f"Still {note_token}."])
    with pytest.raises(RejectedOutputError, match=re.escape(note_token)):
        compile_files(tmp_path / "bad", files, stubborn)
    feedback = stubborn.of_type(WriteGuide)[1].feedback
    assert feedback is not None and "unknown documents; valid tokens include" in feedback
    assert valid in feedback


@pytest.mark.parametrize(
    ("filler", "measured"),
    [
        # 9,000 three-byte characters: 27,000 bytes, which `get_topic_guide` could not serve.
        ("\u6587" * 9_000, "(27042 bytes as a JSON string)"),
        # 4,000 control characters are 4,000 UTF-8 bytes but 24,000 as JSON escapes.
        ("\u0001" * 4_000, "(24042 bytes as a JSON string)"),
    ],
)
def test_a_guide_over_the_json_byte_bound_is_rejected_though_under_the_character_bound(
    tmp_path: Path, filler: str, measured: str
) -> None:
    files = {"p1/memo.md": "# Results\n\nSeagrass mapped."}
    valid = f"[{document_id('p1/memo.md')}:{sec('p1/memo.md')}]"
    gateway = FakeGateway(guide_script=[f"{filler} {valid}", f"See {valid}."])

    corpus = compile_files(tmp_path, files, gateway)

    assert corpus.guide_markdown == f"See {valid}."
    rerun = gateway.of_type(WriteGuide)[1]
    assert rerun.feedback is not None and measured in rerun.feedback


def test_a_long_citation_rejection_reaches_the_rerun_with_its_repair_instruction_first() -> None:
    # Forty documents of 200 sections each, a guide citing each of them wrongly, and invalid
    # tokens of any length.
    view = view_of(
        *(
            (f"p{n:02d}/memo.md", reading([f"Part {k}." for k in range(200)], title=f"M{n}"))
            for n in range(40)
        )
    )
    bad = [f"[{d.doc_id}:{'x' * 3_000}]" for d in view.citable] + [
        "[doc_ffffffffff:doc_ffffffffff_sec_001]"
    ]

    rejection = induction.citation_feedback(bad, view)
    feedback = induction.rejection_feedback([rejection, "another rejection"])

    assert len(rejection) <= induction.MAX_CITATION_FEEDBACK_CHARS
    assert rejection.startswith("fix the citations: cite only catalog tokens")
    assert "unknown documents; valid tokens include" in rejection
    assert f"valid citations of {view.citable[0].doc_id}:" in rejection
    # The rerun shows the whole rejection, then the next one.
    assert f"- {rejection}\n- another rejection" in feedback
    assert len(feedback) <= induction.MAX_REJECTION_FEEDBACK_CHARS


@pytest.mark.parametrize(
    ("markdown", "bad"),
    [
        # A section ID of another document, and a pre-hash ID, never resolve.
        (f"See [{HARBOUR_MEMO}:{ESTUARY_PROPOSAL}_sec_001].", True),
        ("See [doc_001:doc_001_sec_001].", True),
        # Prose in brackets that only starts with "doc" is not a citation token.
        (f"Open the [docx] and [documents] files [{HARBOUR_MEMO}:{HARBOUR_MEMO}_sec_001].", False),
    ],
)
def test_guide_citation_tokens_name_a_section_of_their_own_document(
    markdown: str, bad: bool
) -> None:
    view = view_of(
        ("p101-harbour-lidar/harbour-memo.md", reading(["# A\n\nText."])),
        ("p102-estuary-multibeam/estuary-proposal.md", reading(["# A\n\nText."])),
    )

    assert bool(induction.invalid_citations(markdown, view)) is bad


# Organisation.


def value(name: str, *folders: str, aliases: Sequence[str] = ()) -> FacetValueDraft:
    return FacetValueDraft(
        value=name, description=name, aliases=list(aliases), folders=list(folders)
    )


def role(doc_id: str, deliverable_type: str = "memorandum", *reviewed: str) -> DocumentRoleDraft:
    return DocumentRoleDraft(
        doc_id=doc_id,
        deliverable_type=deliverable_type,
        is_feedback=bool(reviewed),
        feedback_for_types=list(reviewed),
    )


BATHYMETRY_ROLES = (role(HARBOUR_MEMO, "survey memorandum"), role(ESTUARY_PROPOSAL, "proposal"))


def organisation(
    *facets: FacetDraft, roles: Sequence[DocumentRoleDraft] = BATHYMETRY_ROLES
) -> OrganisationDraft:
    return OrganisationDraft(facets=list(facets), documents=list(roles))


BROKEN = organisation(
    FacetDraft(
        name="project",
        description="Folder.",
        values=[value("p101", "p101-harbour-lidar"), value("loose")],
    ),
    FacetDraft(
        name="client",
        description="Client.",
        values=[
            value("Northwind Marine", aliases=["NM"]),
            value("Contoso Ports", aliases=["nm"]),
        ],
    ),
    FacetDraft(name="Region", description="Region.", values=[value("north", "nowhere")]),
    roles=[
        role("doc_009"),
        role(HARBOUR_MEMO),
        role(HARBOUR_MEMO),
        DocumentRoleDraft(
            doc_id=ESTUARY_PROPOSAL, deliverable_type=" ", is_feedback=True, feedback_for_types=[]
        ),
    ],
)


def test_structurally_invalid_organisation_is_rerun_once_then_raises(tmp_path: Path) -> None:
    gateway = FakeGateway(organise_script=[BROKEN, BROKEN])

    with pytest.raises(RejectedOutputError) as raised:
        compile_files(tmp_path, BATHYMETRY_CORPUS, gateway)

    message = str(raised.value)
    assert "facet 'project' mixes values with folders and values without" in message
    assert "facet 'client': 'nm' names both 'Northwind Marine' and 'Contoso Ports'" in message
    assert "facet name 'Region' is not snake_case" in message
    assert "unknown folder 'nowhere'" in message
    assert "'doc_009' is not a document of this batch" in message
    assert f"'{HARBOUR_MEMO}' is classified twice" in message
    assert f"'{ESTUARY_PROPOSAL}' has a blank deliverable_type" in message
    assert f"'{ESTUARY_PROPOSAL}' is feedback but names no feedback_for_types" in message
    first, rerun = gateway.of_type(OrganiseCorpus)
    assert first.feedback is None and rerun.feedback is not None and "doc_009" in rerun.feedback
    # A batch missing a document is rejected too.
    missing = check_batch_of(organisation(roles=[role(HARBOUR_MEMO)]))
    assert any(f"no entry for the documents {ESTUARY_PROPOSAL}" in r for r in missing.rejections)


def check_batch_of(draft: OrganisationDraft) -> Checked[induction.BatchOrganisation]:
    return induction.check_batch(draft, [HARBOUR_MEMO, ESTUARY_PROPOSAL], frozenset())


def test_organisation_normalises_repetition_within_a_value_so_the_snapshot_stores_it(
    tmp_path: Path,
) -> None:
    from arkive_rag.snapshot import TopicSnapshot, write_snapshot

    clients = FacetDraft(
        name="client",
        description="Client.",
        values=[
            value(" Northwind Marine ", aliases=["nm", "NM", " ", "northwind marine", "NM "]),
            value("Contoso Ports", aliases=["CP"]),
        ],
    )
    gateway = FakeGateway(organise_script=[organisation(clients)])

    corpus = compile_files(tmp_path / "corpus", BATHYMETRY_CORPUS, gateway)
    write_snapshot(corpus, tmp_path / "t.arkive.db")

    assert gateway.calls.count("organise_corpus") == 1
    with TopicSnapshot.open(tmp_path / "t.arkive.db") as snap:
        (client,) = snap.facets
        assert [(v.value, v.aliases) for v in client.values] == [
            ("Northwind Marine", ("NM",)),
            ("Contoso Ports", ("CP",)),
        ]
        assert snap.resolve_filters({"client": "nm"}) == frozenset({HARBOUR_MEMO})


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([value("Contoso Ports"), value("Contoso Ports")], "gives the value 'Contoso Ports' twice"),
        (
            [value("Contoso Ports"), value("contoso ports ")],
            "gives the value 'contoso ports' twice",
        ),
        (
            [value("Northwind Marine", aliases=["NM"]), value("Contoso Ports", aliases=["nm"])],
            "'nm' names both 'Northwind Marine' and 'Contoso Ports'",
        ),
        (
            [value("Northwind Marine"), value("Contoso Ports", aliases=["NORTHWIND MARINE"])],
            "'northwind marine' names both 'Northwind Marine' and 'Contoso Ports'",
        ),
        ([value("  "), value("Contoso Ports")], "facet 'client' has a blank value"),
    ],
)
def test_organisation_rejects_values_a_filter_could_not_tell_apart(
    values: list[FacetValueDraft], message: str
) -> None:
    _, rejections = check_facets(
        [FacetDraft(name="client", description="Client.", values=values)], frozenset()
    )

    assert any(message in r for r in rejections), rejections


TYPE_NAMES = (
    "technical memorandum",
    "survey proposal",
    "campaign brief",
    "consultation response letter",
    "noise assessment report",
    "quotation",
    "slide deck",
    "data inventory",
    "field log",
    "method statement",
    "risk register",
    "post-project review",
)


def scale_corpus(documents: int = 260, sections: int = 30) -> CorpusView:
    """A synthetic corpus of `documents` citable documents of 13 deliverable types (one held by
    a single document), each with `sections` sections under long headings, long titles,
    summaries and stated attributes, and curator notes larger than a prompt may show."""
    docs = []
    for n in range(documents):
        dtype = "annual statement" if n == 0 else TYPE_NAMES[n % len(TYPE_NAMES)]
        rel_path = f"client-{n % 17:02d}/project-{n:04d}/{dtype.replace(' ', '-')}-{n}.md"
        bodies = [
            f"# Section {k} of document {n}\n\nWord{n}x{k} "
            + " ".join(f"w{n}y{k}z{i}" for i in range(12))
            for k in range(sections)
        ]
        headings = [
            f"Document {n} > " + "Long heading words " * 40 + str(k) for k in range(sections)
        ]
        docs.append(
            (
                rel_path,
                reading(
                    bodies,
                    title=f"{dtype.title()} for client {n % 17} on site {n} " + "about " * 30,
                    deliverable_type=dtype,
                    summary="A summary sentence that runs on and on. " * 12,
                    attributes=[StatedAttribute(name="client", value=f"Client {n % 17}")],
                    headings=headings,
                ),
            )
        )
    return view_of(*docs)


def test_organisation_scales_in_bounded_batches_and_keeps_every_deliverable_type() -> None:
    view = scale_corpus()
    notes = [
        CuratorNote(scope_path=f"client-{i:02d}/AGENTS.md", content="Curator intent. " * 400)
        for i in range(8)
    ]
    gateway = FakeGateway()

    knowledge = induce(view.documents, notes, gateway, topic="t", max_concurrency=4)

    sections = sum(len(d.reading.sections) for d in view.citable)
    assert len(view.citable) == 260 and sections == 7_800
    batches = gateway.of_type(OrganiseCorpus)
    assert len(batches) >= 2
    assert {d.doc_id for b in batches for d in b.digests} == set(view.citable_order)
    # One canonicalisation over every batch's labels; no cap on how many types survive, and the
    # single-document type is a type of its own.
    (canonical,) = gateway.of_type(CanonicaliseTypes)
    assert [t.name for t in knowledge.deliverable_types] == sorted(
        [*TYPE_NAMES, "annual statement"]
    )
    assert sum(r.deliverable_type == "annual statement" for r in knowledge.roles.values()) == 1
    assert len(knowledge.roles) == 260
    # Every prompt of the build stays bounded, and so does every judge request's evidence.
    structured = [r for r in gateway.requests if hasattr(r, "instructions")]
    assert {type(r).__name__ for r in structured} >= {
        "OrganiseCorpus",
        "CanonicaliseTypes",
        "WriteGuide",
        "ExtractFeedbackLessons",
    }
    assert max(prompt_chars(r) for r in structured) <= MAX_PROMPT_CHARS
    (guide,) = gateway.of_type(WriteGuide)
    assert len(guide.text()) <= MAX_GUIDE_TEXT_CHARS
    assert all(len(evidence) <= EVIDENCE_WINDOW_CHARS for _, evidence, _ in gateway.judged)
    # The notes are cut to the prompt's share, and the cut is reported.
    assert batches[0].notes.count("Curator intent.") * 16 <= MAX_CURATOR_NOTES_PROMPT_CHARS
    assert knowledge.report.truncated_curator_notes >= 1
    assert len(canonical.labels) == 13


def test_merged_facets_never_mix_bindings_and_union_values_across_batches() -> None:
    def path_facet(*folders: str) -> Facet:
        return Facet(
            name="project",
            description="Folder.",
            binding="path",
            values=tuple(
                FacetValue(value=f, description=f, aliases=(), folders=(f,)) for f in folders
            ),
        )

    content_project = Facet(
        name="project",
        description="Read.",
        binding="content",
        values=(FacetValue(value="Harbour", description="h", aliases=(), folders=()),),
    )
    client_a = Facet(
        name="client",
        description="Client.",
        binding="content",
        values=(FacetValue(value="Northwind", description="n", aliases=("NW",), folders=()),),
    )
    client_b = client_a.model_copy(
        update={
            "values": (
                FacetValue(value="northwind", description="n2", aliases=("NWM",), folders=()),
                FacetValue(value="Contoso", description="c", aliases=("NW",), folders=()),
            )
        }
    )

    merged = merge_facets(
        [[path_facet("p1"), client_a], [content_project, client_b], [path_facet("p2")]]
    )

    project, client = merged
    # Two batches proposed `project` path-bound and one content-bound: path-bound wins, and
    # the content-bound value is dropped.
    assert (project.binding, [v.value for v in project.values]) == ("path", ["p1", "p2"])
    # Values merge by folded label; an alias naming two values is dropped from both.
    assert [(v.value, v.aliases) for v in client.values] == [
        ("Northwind", ("NWM",)),
        ("Contoso", ()),
    ]
    assert merge_facets([[client_a]]) == (client_a,)


def test_aliases_two_batches_spell_differently_merge_into_one_the_snapshot_stores(
    tmp_path: Path,
) -> None:
    from arkive_rag.snapshot import TopicSnapshot, write_snapshot

    corpus = compile_files(tmp_path / "corpus", BATHYMETRY_CORPUS)
    (client,) = [f for f in corpus.facets if f.name == "client"]

    def spelled(alias: str) -> Facet:
        return client.model_copy(
            update={
                "values": tuple(
                    v.model_copy(update={"aliases": (alias,)})
                    if v.value == "Northwind Marine"
                    else v
                    for v in client.values
                )
            }
        )

    (merged,) = merge_facets([[spelled("NM")], [spelled("nm")]])
    # One spelling per folded alias: the snapshot keys aliases by their folded form.
    assert [(v.value, v.aliases) for v in merged.values if v.value == "Northwind Marine"] == [
        ("Northwind Marine", ("NM",))
    ]
    facets = tuple(merged if f.name == "client" else f for f in corpus.facets)
    write_snapshot(replace(corpus, facets=facets), tmp_path / "t.arkive.db")
    with TopicSnapshot.open(tmp_path / "t.arkive.db") as snap:
        assert snap.resolve_filters({"client": "nm"}) == frozenset({HARBOUR_MEMO})


def test_type_labels_must_each_map_to_exactly_one_canonical_type() -> None:
    labels = (TypeLabel("Tech memo", 2, ()), TypeLabel("technical memorandum", 1, ()))
    good = TypeCanonicalisationDraft(
        types=[
            CanonicalTypeDraft(
                name="technical memorandum",
                description="A memo.",
                labels=["tech memo", "Technical Memorandum"],
            )
        ]
    )
    bad = TypeCanonicalisationDraft(
        types=[
            CanonicalTypeDraft(name="memo", description="", labels=["Tech memo", "proposal"]),
            CanonicalTypeDraft(name="Memo ", description="", labels=["Tech memo"]),
        ]
    )

    accepted = check_types(good, labels).result
    rejected = check_types(bad, labels)

    assert accepted is not None and accepted.canonical == {
        "tech memo": "technical memorandum",
        "technical memorandum": "technical memorandum",
    }
    assert rejected.result is None
    assert "type 'memo': 'proposal' is not a given label" in rejected.rejections
    assert "type 'Memo' is given twice" in rejected.rejections
    assert "no type lists the labels 'technical memorandum'" in rejected.rejections


def test_feedback_documents_keep_their_own_type_and_name_the_types_they_review(
    tmp_path: Path,
) -> None:
    merged = TypeCanonicalisationDraft(
        types=[
            CanonicalTypeDraft(
                name="Campaign brief", description="A brief.", labels=["campaign brief"]
            ),
            CanonicalTypeDraft(name="Review notes", description="Notes.", labels=["review notes"]),
        ]
    )
    gateway = FakeGateway(types_script=[merged])

    corpus = compile_files(tmp_path, MARKETING_CORPUS, gateway)

    roles = {d.doc_id: d.role for d in corpus.documents}
    assert [t.name for t in corpus.deliverable_types] == ["Campaign brief", "Review notes"]
    assert roles[SPRING_REVIEW].deliverable_type == "Review notes"
    assert roles[SPRING_REVIEW].is_feedback
    assert roles[SPRING_REVIEW].feedback_for_types == ("Campaign brief",)
    assert roles[LOYALTY_BRIEF].deliverable_type == "Campaign brief"
    assert not roles[LOYALTY_BRIEF].is_feedback
    # The organisation request shows digests, not cards: one line per document.
    (request,) = gateway.of_type(OrganiseCorpus)
    assert f"{SPRING_REVIEW} | spring-launch/review.md | md |" in request.text()
    # The disclosure both briefs end with is a recurring unit of the corpus.
    (disclosure,) = [c for c in corpus.unit_clusters if c.text.startswith("Every sponsored post")]
    assert disclosure.text == HOUSE_DISCLOSURE.split("\n\n")[1].strip()
    assert disclosure.section_ids == (f"{LOYALTY_BRIEF}_sec_004", f"{SPRING_BRIEF}_sec_004")


# Facet binding.


def test_path_bound_facets_cover_curator_notes_and_content_bound_facets_are_judged(
    tmp_path: Path,
) -> None:
    gateway = FakeGateway()
    corpus = compile_files(tmp_path, BATHYMETRY_CORPUS, gateway)

    project, client = corpus.facets
    assert (project.name, project.binding, client.name, client.binding) == (
        "project",
        "path",
        "client",
        "content",
    )
    assert facets_by_doc(corpus) == {
        "p101-harbour-lidar/harbour-memo.md": {
            "project": ("p101-harbour-lidar",),
            "client": ("Northwind Marine",),
        },
        "p102-estuary-multibeam/estuary-proposal.md": {
            "project": ("p102-estuary-multibeam",),
            "client": ("Contoso Ports",),
        },
        # A curator note is bound by its folder and never judged.
        "p103-reef-sonar/AGENTS.md": {"project": ("p103-reef-sonar",)},
    }
    judged = [evidence for operation, evidence, _ in gateway.judged if operation == "bind_facets"]
    assert len(judged) == 2 and not any("AGENTS.md" in evidence for evidence in judged)


def test_path_bound_values_also_answer_to_their_folder_paths(tmp_path: Path) -> None:
    projects = FacetDraft(
        name="project",
        description="Folder.",
        values=[
            value("Harbour", "p101-harbour-lidar/", aliases=["p101"]),
            value("Estuary", "p102-estuary-multibeam"),
            # A folder path that already names another value stays that value's name only.
            value("p102-estuary-multibeam", "p103-reef-sonar", aliases=["Reef"]),
        ],
    )
    gateway = FakeGateway(organise_script=[organisation(projects)])

    corpus = compile_files(tmp_path, BATHYMETRY_CORPUS, gateway)

    (facet,) = corpus.facets
    assert [(v.value, v.folders, v.aliases) for v in facet.values] == [
        ("Harbour", ("p101-harbour-lidar",), ("p101", "p101-harbour-lidar")),
        ("Estuary", ("p102-estuary-multibeam",), ()),
        ("p102-estuary-multibeam", ("p103-reef-sonar",), ("Reef", "p103-reef-sonar")),
    ]
    assert facets_by_doc(corpus)["p103-reef-sonar/AGENTS.md"] == {
        "project": ("p102-estuary-multibeam",)
    }


class RefusingGateway(FakeGateway):
    """Refuses every binding question about Contoso Ports and affirms every other one."""

    def judge(self, operation: str, evidence: str, predicates: Sequence[str]) -> list[float | None]:
        answers = super().judge(operation, evidence, predicates)
        if operation != "bind_facets":
            return answers
        return [None if "Contoso" in p else 0.9 for p in predicates]


def test_binding_assigns_several_values_skips_refusals_and_counts_them(tmp_path: Path) -> None:
    gateway = RefusingGateway()

    corpus = compile_files(tmp_path, BATHYMETRY_CORPUS, gateway)

    assert corpus.build_report.judge_refusals["bind_facets"] == 2
    # Contoso Ports was refused everywhere, so only one client value is populated and the
    # facet is pruned along with the empty value.
    assert [f.name for f in corpus.facets] == ["project"]
    assert corpus.build_report.pruned_facet_values == ["client=Contoso Ports"]
    assert corpus.build_report.pruned_facets == ["client"]
    assert facets_by_doc(corpus)["p101-harbour-lidar/harbour-memo.md"] == {
        "project": ("p101-harbour-lidar",)
    }
    binding_questions = [p for op, _, ps in gateway.judged if op == "bind_facets" for p in ps]
    assert len(binding_questions) == 4  # two citable documents, two client values each


def test_values_no_document_holds_are_pruned_and_documents_may_hold_several(
    tmp_path: Path,
) -> None:
    audiences = FacetDraft(
        name="audience",
        description="Readers.",
        values=[value("Social"), value("Email"), value("Print")],
    )
    campaigns = FacetDraft(
        name="campaign",
        description="Folder.",
        values=[
            value("loyalty-relaunch", "loyalty-relaunch"),
            value("spring-launch", "spring-launch"),
        ],
    )

    class EveryChannel(FakeGateway):
        def judge(
            self, operation: str, evidence: str, predicates: Sequence[str]
        ) -> list[float | None]:
            answers = super().judge(operation, evidence, predicates)
            if operation == "bind_facets" and "brief" in evidence.split("\n")[0]:
                return [0.1 if '"Print"' in p else 0.9 for p in predicates]
            return answers

    roles = [
        role(LOYALTY_BRIEF, "campaign brief"),
        role(SPRING_BRIEF, "campaign brief"),
        role(SPRING_REVIEW, "review notes", "campaign brief"),
    ]
    gateway = EveryChannel(organise_script=[organisation(campaigns, audiences, roles=roles)])

    corpus = compile_files(tmp_path, MARKETING_CORPUS, gateway)

    bound = facets_by_doc(corpus)
    assert bound["spring-launch/brief.md"]["audience"] == ("Email", "Social")
    assert bound["spring-launch/review.md"]["audience"] == ("Social",)
    assert "audience" not in bound["AGENTS.md"]
    assert corpus.build_report.pruned_facet_values == ["audience=Print"]
    assert [v.value for v in corpus.facets[1].values] == ["Social", "Email"]


def test_binding_evidence_bounds_every_heading_and_the_outline() -> None:
    (doc,) = scale_corpus(documents=1, sections=400).citable

    evidence = induction.binding_evidence(doc)

    assert len(evidence) <= induction.MAX_BINDING_OUTLINE_CHARS + 2_000
    assert "more sections not listed" in evidence


# Version families.


def memo_text(extra: str = "") -> list[str]:
    body = " ".join(f"Sounding line {i} was run across the harbour approach." for i in range(80))
    return [f"# Harbour survey\n\n{body}\n\n{extra}"]


def test_version_families_join_drafts_and_copies_but_keep_phases_apart() -> None:
    view = view_of(
        (
            "archive/harbour-report-copy.md",
            reading(
                memo_text("Final."), title="Harbour Report v1.0", sha256="b" * 64, date="2025-04-01"
            ),
        ),
        (
            "p1/drafts/harbour-report-v0.9.md",
            reading(
                memo_text("Draft."), title="Harbour Report v0.9", sha256="a" * 64, date="2025-03-01"
            ),
        ),
        (
            "p1/harbour-report.md",
            reading(
                memo_text("Final."), title="Harbour Report v1.0", sha256="b" * 64, date="2025-04-01"
            ),
        ),
        (
            "p2/phase-1.md",
            reading(memo_text("Phase one."), title="Estuary Study Phase 1", sha256="c" * 64),
        ),
        (
            "p2/phase-2.md",
            reading(memo_text("Phase two."), title="Estuary Study Phase 2", sha256="d" * 64),
        ),
    )
    copy, draft, final, phase1, phase2 = (d.doc_id for d in view.citable)
    roles = {d: induction.ProposedRole("report", False, ()) for d in view.citable_order}
    gateway = FakeGateway()

    with InductionExecutor(2) as pool:
        families = induction.version_families(gateway, view, roles, pool)

    # The two byte-identical copies join without a question. The other four share most of
    # their text, so every pair of them is a candidate; the decision model keeps only the
    # drafts of one deliverable together, and every superseded member names the latest.
    assert families.members == {
        copy: (copy, "latest", None),
        draft: (copy, "superseded", copy),
        final: (copy, "superseded", copy),
        phase1: (phase1, "sole", None),
        phase2: (phase2, "sole", None),
    }
    # Dates ordered the two distinct members; no ordering question was needed.
    assert not [op for op, *_ in gateway.judged if op == "order_versions"]
    confirm = [p for op, _, ps in gateway.judged if op == "confirm_versions" for p in ps]
    assert len(confirm) == 6 and "not a follow-on phase" in confirm[0]


def test_byte_identical_documents_are_one_family_and_every_member_names_the_latest() -> None:
    first = reading(["# Export\n\nTwo words."], title="Export")
    v1 = reading(memo_text("One."), title="Survey Report v1", date="2025-01-10")
    v2 = reading(memo_text("Two."), title="Survey Report v2", date="2025-02-10")
    view = view_of(
        ("a/report-v1.md", v1),
        ("archive/archive.md", first),
        ("b/report-v1-copy.md", v1),
        ("c/report-v2.md", v2),
        ("exports/export.md", first),
    )
    report_v1, archive, copy, report_v2, export = (d.doc_id for d in view.citable)
    roles = {d: induction.ProposedRole("report", False, ()) for d in view.citable_order}
    gateway = FakeGateway()

    with InductionExecutor(1) as pool:
        members = induction.version_families(gateway, view, roles, pool).members

    # Two words share no 4-gram and their names no word, yet the same bytes are one artifact.
    assert members[archive] == (archive, "latest", None)
    assert members[export] == (archive, "superseded", archive)
    # The copy of the first version names the latest version, not the version it copies.
    assert members[report_v1] == (report_v1, "superseded", report_v2)
    assert members[copy] == (report_v1, "superseded", report_v2)
    assert members[report_v2] == (report_v1, "latest", None)
    # Byte-identical pairs reach no decision question.
    confirm = [p for op, _, ps in gateway.judged if op == "confirm_versions" for p in ps]
    assert not [p for p in confirm if export in p or copy in p]


def test_a_review_is_never_a_version_candidate_of_what_it_reviews() -> None:
    quoted = memo_text("Final.")
    view = view_of(
        ("p1/report.md", reading(quoted, title="Harbour Report")),
        ("p1/review.md", reading([*quoted, "Reviewer: cite the datum."], title="Harbour Report")),
    )
    report, review = (d.doc_id for d in view.citable)
    roles = {
        report: induction.ProposedRole("report", False, ()),
        review: induction.ProposedRole("review notes", True, ("report",)),
    }

    assert induction.version_candidates(view, roles) == []


def test_version_order_without_dates_asks_the_judge_and_stays_unordered_without_a_winner() -> None:
    undated = view_of(
        ("a/report.md", reading(memo_text("One."), title="Survey Report v1", sha256="a" * 64)),
        ("b/report.md", reading(memo_text("Two."), title="Survey Report v2", sha256="b" * 64)),
    )
    unversioned = view_of(
        ("a/report.md", reading(memo_text("One."), title="Survey Report", sha256="a" * 64)),
        (
            "b/report.md",
            reading(memo_text("Two."), title="Survey Report", sha256="b" * 64, date="2025-04"),
        ),
    )

    def families(view: CorpusView) -> Mapping[str, tuple[str, str, str | None]]:
        roles = {d: induction.ProposedRole("report", False, ()) for d in view.citable_order}
        with InductionExecutor(1) as pool:
            return induction.version_families(FakeGateway(), view, roles, pool).members

    first, second = (d.doc_id for d in undated.citable)
    assert families(undated) == {
        first: (first, "superseded", second),
        second: (first, "latest", None),
    }
    # One member is undated, so dates cannot decide; no version token lets the judge decide.
    one, two = (d.doc_id for d in unversioned.citable)
    assert families(unversioned) == {one: (one, "unordered", None), two: (one, "unordered", None)}


def test_version_candidates_keep_a_bounded_number_of_partners_per_document() -> None:
    view = view_of(
        *(
            (
                f"p{n:02d}/report.md",
                reading(memo_text(f"Copy {n}."), title="Same report", sha256=f"{n:064d}"),
            )
            for n in range(20)
        )
    )
    roles = {d: induction.ProposedRole("report", False, ()) for d in view.citable_order}

    pairs = induction.version_candidates(view, roles)

    partners: dict[str, int] = {}
    for a, b in pairs:
        partners[a] = partners.get(a, 0) + 1
        partners[b] = partners.get(b, 0) + 1
    # Every document keeps its best MAX_VERSION_PARTNERS; a pair kept by either side stays.
    assert len(pairs) <= 20 * MAX_VERSION_PARTNERS
    assert all(a < b for a, b in ((view.citable_order[x], view.citable_order[y]) for x, y in pairs))
    assert min(partners.values()) >= MAX_VERSION_PARTNERS


# Feedback lessons.


def test_feedback_lessons_are_located_and_confirmed_one_request_per_section(
    tmp_path: Path,
) -> None:
    comments = "\n\n".join(f"Reviewer: State the datum used for item {i}." for i in range(30))
    files = {
        "p1/memo.md": "# Memo\n\nDeliverable: memo\n\nThe datum is LAT.",
        "p1/review.md": f"# Review\n\nDeliverable: review notes\n\n## Comments\n\n{comments}\n",
    }
    gateway = FakeGateway()

    corpus = compile_files(tmp_path, files, gateway)

    review = document_id("p1/review.md")
    assert len(corpus.feedback_lessons) == 30
    assert {
        (lesson.source_doc_id, lesson.section_id, lesson.target_types)
        for lesson in corpus.feedback_lessons
    } == {(review, sec("p1/review.md", 2), ("memo",))}
    # Thirty lessons of one section are confirmed by one decision request, not thirty.
    checks = [ps for op, _, ps in gateway.judged if op == "check_feedback_lessons"]
    assert [len(ps) for ps in checks] == [30]
    assert corpus.build_report.dropped_feedback_lessons == 0


class DoubtsLessons(FakeGateway):
    def judge(self, operation: str, evidence: str, predicates: Sequence[str]) -> list[float | None]:
        answers = super().judge(operation, evidence, predicates)
        if operation == "check_feedback_lessons":
            return [None if "jargon" in p else 0.2 for p in predicates]
        return answers


def test_unsupported_or_unlocated_feedback_lessons_are_dropped_and_counted(
    tmp_path: Path,
) -> None:
    lost = FeedbackLessonsDraft(
        lessons=[
            FeedbackLessonDraft(statement="Name the audience.", quote="not in the review at all"),
            FeedbackLessonDraft(
                statement="Cut the jargon from the opening line.",
                quote="Cut the jargon from the opening",
            ),
        ]
    )
    gateway = FakeGateway(lesson_script=[lost, lost])

    corpus = compile_files(tmp_path / "unlocated", MARKETING_CORPUS, gateway)

    _, rerun = gateway.of_type(ExtractFeedbackLessons)
    assert rerun.feedback is not None and "not in the review at all" in rerun.feedback
    assert [lesson.statement for lesson in corpus.feedback_lessons] == [
        "Cut the jargon from the opening line."
    ]
    assert corpus.build_report.discarded_unlocated_quotes == 1

    doubted = compile_files(tmp_path / "doubted", MARKETING_CORPUS, DoubtsLessons())
    assert doubted.feedback_lessons == ()
    assert doubted.build_report.dropped_feedback_lessons == 1
    assert doubted.build_report.judge_refusals["check_feedback_lessons"] == 1


def test_an_oversized_feedback_table_is_read_in_batches_that_repeat_its_header() -> None:
    header = "| Comment | Response |\n| --- | --- |"
    rows = "\n".join(
        f"| Reviewer asked for item {i} to be clarified at length. | Accepted; clarified. |"
        for i in range(900)
    )
    markdown = f"## Comments\n\n{header}\n{rows}"
    view = view_of(("p1/review.md", reading([markdown], title="Review")))
    (doc,) = view.citable

    batches = induction.feedback_batches(doc)

    assert len(markdown) > 2 * FEEDBACK_BATCH_CHARS and len(batches) >= 3
    pieces = [s.markdown for batch in batches for s in batch]
    assert pieces[0].startswith("## Comments")
    assert all(piece.startswith(header) for piece in pieces[1:])
    assert all(len(piece) <= FEEDBACK_BATCH_CHARS + len(header) + 1 for piece in pieces)
    # Every row is read once, apart from the repeated header.
    assert sum(piece.count("Accepted; clarified.") for piece in pieces) == 900


def test_markdown_pieces_keep_line_order_and_repeat_pipeless_table_headers() -> None:
    long_line = "b" * 30
    assert induction._split_markdown(f"{'a' * 10}\n{long_line}\nccccc", 20) == [
        "a" * 10,
        "b" * 20,
        "b" * 10,
        "ccccc",
    ]
    header = "Comment | Response\n--- | ---"
    rows = "\n".join(f"Row {i} | Accepted" for i in range(40))
    pieces = induction._split_markdown(f"{header}\n{rows}", 120)
    assert len(pieces) > 2 and all(len(p) <= 120 for p in pieces)
    assert all(p.startswith(header) for p in pieces)
    body = [line for p in pieces for line in p.split("\n") if line.startswith("Row")]
    assert body == rows.split("\n")
    # A row too long for any piece is cut as text; the next row still comes under the header.
    oversized = f"{header}\nRow 0 | Accepted\nRow 1 | {'x' * 200}\nRow 2 | Accepted"
    pieces = induction._split_markdown(oversized, 120)
    assert pieces[-1] == f"{header}\nRow 2 | Accepted"


def test_feedback_batches_count_each_section_label_within_the_batch_size() -> None:
    # Two hundred short comments under very long headings: their labels, not their text,
    # make up most of a batch.
    markdowns = [f"Reviewer: comment {i}." for i in range(200)]
    headings = ["Comments > " + "Long heading words " * 40 + str(i) for i in range(200)]
    view = view_of(("p1/review.md", reading(markdowns, title="Review", headings=headings)))
    (doc,) = view.citable

    batches = induction.feedback_batches(doc)

    assert len(batches) > 1
    for batch in batches:
        assert len("\n\n".join(s.render() for s in batch)) <= FEEDBACK_BATCH_CHARS
        request = ExtractFeedbackLessons("Review", ("memo",), batch, "x" * 6_000)
        assert prompt_chars(request) <= MAX_PROMPT_CHARS
    assert [s.markdown for b in batches for s in b] == markdowns


def test_lessons_with_blank_statements_or_none_located_are_rejected() -> None:
    sections = (induction.LessonSection("doc_a_sec_001", "Comments", "Reviewer: cite the datum."),)
    blank = FeedbackLessonsDraft(
        lessons=[
            FeedbackLessonDraft(statement=" ", quote="cite the datum"),
            FeedbackLessonDraft(statement="Cite the datum.", quote="cite the datum"),
        ]
    )
    lost = FeedbackLessonsDraft(
        lessons=[FeedbackLessonDraft(statement="Cite the datum.", quote="not there")]
    )

    reduced = induction.locate_lessons(blank, sections, rerun_of_unlocated=False)
    nothing = induction.locate_lessons(lost, sections, rerun_of_unlocated=False)
    empty_rerun = induction.locate_lessons(
        FeedbackLessonsDraft(lessons=[]), sections, rerun_of_unlocated=True
    )

    assert reduced.result is not None and reduced.result.lessons == (
        ("Cite the datum.", "doc_a_sec_001"),
    )
    assert reduced.rejections == ("the lesson quoting 'cite the datum' has a blank statement",)
    # Lessons proposed and none located is a broken reading, and so is an empty rerun of it.
    assert nothing.result is None
    assert (
        empty_rerun.result is None and "none of them could be located" in empty_rerun.rejections[0]
    )
    assert "accepted or resolved, not one they rejected" in induction.lesson_predicate("x")


def test_a_superseded_review_gives_no_lessons(tmp_path: Path) -> None:
    def review(version: str, date: str, comment: str) -> str:
        return (
            f"# Spring Review {version}\n\nDeliverable: review notes\nDate: {date}\n\n"
            "## Comments\n\nThe social brief was read by the whole panel this week.\n\n"
            f"Reviewer: {comment}\n"
        )

    files = {
        "p1/brief.md": "# Brief\n\nDeliverable: campaign brief\n\nA brief.",
        "p1/review-v1.md": review("v1", "2025-01-10", "Add more jargon to the opening line."),
        "p1/review-v2.md": review("v2", "2025-02-10", "Cut the jargon from the opening line."),
    }
    gateway = FakeGateway()

    corpus = compile_files(tmp_path, files, gateway)

    first = document_id("p1/review-v1.md")
    roles = {d.doc_id: d.role for d in corpus.documents}
    assert roles[first].version_status == "superseded"
    assert [lesson.statement for lesson in corpus.feedback_lessons] == [
        "Cut the jargon from the opening line."
    ]
    assert [r.doc_title for r in gateway.of_type(ExtractFeedbackLessons)] == ["Spring Review v2"]


def test_a_review_quoting_a_deliverable_does_not_make_its_text_recur() -> None:
    passage = "Survey results describe the seabed on the survey dates only and certify nothing."
    view = view_of(
        ("p1/memo.md", reading([passage], title="Memo")),
        ("p1/review.md", reading([f"The memo says: {passage}"], title="Review")),
    )
    memo, review = (d.doc_id for d in view.citable)
    roles = {
        memo: induction.ProposedRole("memo", False, ()),
        review: induction.ProposedRole("review notes", True, ("memo",)),
    }

    assert induction.unit_clusters(view, roles) == ()
    as_deliverable = {**roles, review: induction.ProposedRole("memo", False, ())}
    assert len(induction.unit_clusters(view, as_deliverable)) == 1


# Every induction request on its rerun.


def test_every_induction_request_stays_within_the_prompt_bound_on_its_rerun() -> None:
    feedback = induction.rejection_feedback([f"quote {'x' * 5_000} {n}" for n in range(200)])
    # A term with a 5,000-character kind, 300 long aliases, and an alias of 20,000 characters
    # that another document's section holds verbatim.
    huge_alias = "datum " * 3_400
    odd = TermObservation(
        term="Survey datum",
        kind="kind " * 1_000,
        aliases=(huge_alias, *(f"alias {n} " * 40 for n in range(300))),
        quote="The survey datum is defined here.",
        defines=True,
        section_ordinal=1,
    )
    (odd_cluster,) = term_clusters(
        view_of(
            ("p1/a.md", reading(["The survey datum is defined here."], [odd], title="A")),
            ("p2/b.md", reading([f"Words. {huge_alias} more words."], title="B")),
        )
    )
    notes = induction.notes_text(
        [CuratorNote(scope_path=f"n{i}/AGENTS.md", content="Intent. " * 2_000) for i in range(9)]
    ).text
    digests = [
        induction.DocumentDigest(
            f"doc_{n:010d}",
            "/".join(["folder" * 30] * 3) + f"/{n}.md",
            "pdf",
            "1 January 2025",
            "type " * 40,
            "title " * 40,
            "summary " * 60,
            (StatedAttribute(name="client", value="client " * 30),),
        )
        for n in range(400)
    ]
    folders = tuple(f"{'folder' * 20}/{n}" for n in range(2_000))
    labels = tuple(
        TypeLabel(f"label {n} " + "word " * 10, 3, ("title " * 20,) * 3, ("review " * 20,) * 3)
        for n in range(700)
    )
    catalog = tuple(
        induction.GuideEntry(
            f"doc_{n:010d}",
            "title " * 40,
            None,
            "type",
            "summary " * 80,
            f"doc_{n:010d}_sec_001",
            tuple((f"doc_{n:010d}_sec_{k:03d}", "heading " * 30) for k in range(2, 8)),
        )
        for n in range(600)
    )
    facet = Facet(
        name="client",
        description="d " * 200,
        binding="content",
        values=tuple(
            FacetValue(value=f"v{n} " * 20, description="d", aliases=(), folders=())
            for n in range(300)
        ),
    )
    quotes = [
        TermObservation(
            term="Gauge datum",
            kind="concept",
            aliases=tuple(f"alias {n} " * 5 for n in range(3)),
            quote="The gauge datum is " + "long " * 200,
            defines=n % 7 == 0,
            section_ordinal=1,
        )
        for n in range(350)
    ]
    glossary_view = view_of(
        *(
            (f"p{n:03d}/memo.md", reading([f"Gauge datum note {n}."], [quotes[n]], title=f"M{n}"))
            for n in range(350)
        )
    )
    (cluster,) = term_clusters(glossary_view)
    requests: list[Any] = [
        OrganiseCorpus(folders, notes, induction.digest_batches(digests)[0], "batch 1 of 9"),
        CanonicaliseTypes(labels),
        WriteGuide(
            "t",
            notes,
            catalog,
            (facet,) * 6,
            tuple("term " * 5 for _ in range(2_000)),
            tuple(induction.TypeSummaryLine(f"type {n}", "d " * 100, 3, 1) for n in range(400)),
        ),
        WriteGlossary((cluster,)),
        WriteGlossary((odd_cluster,)),
        ExtractFeedbackLessons(
            "Review",
            tuple(f"deliverable type {n} " + "z" * 60 for n in range(1_300)),
            (induction.LessonSection("doc_a_sec_001", "Comments", "c" * FEEDBACK_BATCH_CHARS),),
        ),
        WriteGuide(
            "topic " * 10_000,
            notes,
            catalog,
            (facet,) * 6,
            tuple("term " * 5 for _ in range(2_000)),
            tuple(induction.TypeSummaryLine(f"type {n}", "d " * 100, 3, 1) for n in range(400)),
        ),
    ]

    # A term found in 350 documents keeps a bounded number of quotes, defining ones first.
    assert len(cluster.quotes) == induction.MAX_QUOTES_PER_TERM
    # A binding question reads a bounded card and outline, and quotes bounded descriptions.
    (bound_doc,) = view_of(
        (
            "p1/memo.md",
            reading(
                [f"Part {k}." for k in range(2_000)],
                title="title " * 2_000,
                summary="summary " * 10_000,
                attributes=[StatedAttribute(name="client", value="c " * 500)] * 300,
            ),
        )
    ).citable
    wide_facet = facet.model_copy(update={"description": "d " * 10_000})
    longest = max(
        len(
            induction.binding_predicate(
                wide_facet, v.model_copy(update={"description": "v " * 9_000})
            )
        )
        for v in facet.values
    )
    assert len(induction.binding_evidence(bound_doc)) + longest <= MAX_PROMPT_CHARS
    # However long its kind, aliases or matched alias, a term renders within a bound.
    (occurrence,) = odd_cluster.occurrences
    assert len(occurrence.snippet) == induction.OCCURRENCE_CHARS
    assert len(odd_cluster.render()) < 12_000
    assert all(q.defines for q in cluster.quotes)
    assert len(feedback) <= induction.MAX_REJECTION_FEEDBACK_CHARS
    for request in requests:
        for rerun in (request, replace(request, feedback=feedback)):
            assert prompt_chars(rerun) <= MAX_PROMPT_CHARS, type(request).__name__


# Glossary.


def senses(corpus: CompiledCorpus, term: str) -> list[tuple[str | None, tuple[str, ...]]]:
    (entry,) = [e for e in corpus.glossary if e.term == term]
    return [(s.definition, s.citations) for s in entry.senses]


def test_glossary_keeps_defined_terms_separate_senses_and_undefined_terms(
    tmp_path: Path,
) -> None:
    corpus = compile_files(tmp_path, MARKETING_CORPUS)

    assert [e.term for e in corpus.glossary] == ["Churn", "CTA", "ICP"]
    assert senses(corpus, "Churn") == [
        (
            "Lapsed members. Churn is the share of members who stop buying within a year.",
            (f"{LOYALTY_BRIEF}_sec_002",),
        )
    ]
    # Two documents define CTA differently: two senses, each citing its own section.
    assert senses(corpus, "CTA") == [
        (
            "CTA (call to action) is the button that closes every email.",
            (f"{LOYALTY_BRIEF}_sec_003",),
        ),
        (
            "CTA (call to action) means the single step a reader should take next.",
            (f"{SPRING_BRIEF}_sec_003",),
        ),
    ]
    (cta,) = [e for e in corpus.glossary if e.term == "CTA"]
    assert cta.aliases == ("call to action",)
    # Used in two documents, defined in neither: discoverable, with no definition.
    assert senses(corpus, "ICP") == [
        (None, (f"{SPRING_BRIEF}_sec_003", f"{SPRING_REVIEW}_sec_002"))
    ]
    assert corpus.build_report.unsupported_definitions == 0


class DoubtsChurn(FakeGateway):
    def judge(self, operation: str, evidence: str, predicates: Sequence[str]) -> list[float | None]:
        answers = super().judge(operation, evidence, predicates)
        if operation == "check_definitions" and '"Churn"' in predicates[0]:
            return [0.2]
        if operation == "check_definitions" and "button" in predicates[0]:
            return [None]
        return answers


def test_definitions_the_judge_does_not_support_become_undefined_and_are_counted(
    tmp_path: Path,
) -> None:
    corpus = compile_files(tmp_path, MARKETING_CORPUS, DoubtsChurn())

    assert senses(corpus, "Churn") == [(None, (f"{LOYALTY_BRIEF}_sec_002",))]
    assert senses(corpus, "CTA")[0] == (None, (f"{LOYALTY_BRIEF}_sec_003",))
    assert senses(corpus, "CTA")[1][0] is not None
    assert corpus.build_report.unsupported_definitions == 2
    assert corpus.build_report.judge_refusals == {"check_definitions": 1}


def glossary_draft(
    term: str, citations: list[list[str]], aliases: Sequence[str] = ()
) -> GlossaryDraft:
    return GlossaryDraft(
        entries=[
            GlossaryEntryDraft(
                term=term,
                kind="acronym",
                aliases=list(aliases),
                senses=[SenseDraft(definition=None, citations=c) for c in citations],
            )
        ]
    )


def test_every_glossary_citation_must_contain_a_form_the_entry_publishes(tmp_path: Path) -> None:
    # The proposal's second section says only "LAT"; the entry publishes neither "LAT" nor an
    # alias for it.
    unpublished = glossary_draft("Lowest astronomical tide", [[f"{ESTUARY_PROPOSAL}_sec_002"]])
    gateway = FakeGateway(glossary_script=[unpublished, unpublished])

    with pytest.raises(RejectedOutputError, match=f"{ESTUARY_PROPOSAL}_sec_002 contains neither"):
        compile_files(tmp_path, BATHYMETRY_CORPUS, gateway)
    first, rerun = gateway.of_type(WriteGlossary)
    assert [c.term for c in first.clusters] == ["LAT"]
    assert rerun.feedback is not None and "add the form used there as an alias" in rerun.feedback

    published = glossary_draft(
        "Lowest astronomical tide", [[f"{ESTUARY_PROPOSAL}_sec_002"]], ["LAT"]
    )
    corpus = compile_files(
        tmp_path / "ok", BATHYMETRY_CORPUS, FakeGateway(glossary_script=[published])
    )
    assert [(e.term, e.aliases) for e in corpus.glossary] == [
        ("Lowest astronomical tide", ("LAT",))
    ]


def test_glossary_drafts_without_senses_or_citations_are_rejected_and_rerun(
    tmp_path: Path,
) -> None:
    no_senses = glossary_draft("LAT", [])
    no_citations = glossary_draft("LAT", [[]])
    gateway = FakeGateway(glossary_script=[no_senses, no_citations])

    with pytest.raises(RejectedOutputError, match="a sense of 'LAT' has no citations"):
        compile_files(tmp_path, BATHYMETRY_CORPUS, gateway)
    rerun = gateway.of_type(WriteGlossary)[1]
    assert rerun.feedback is not None and "entry 'LAT' has no senses" in rerun.feedback


def test_aliases_are_stripped_one_per_folded_form_in_a_total_order() -> None:
    forms = ["Chart Datum", "chart datum", " ", "LAT", "lat ", "Lowest Tide", "lowest tide"]
    expected = ("Chart Datum", "Lowest Tide")

    # Every input order gives the same aliases: the order does not depend on set iteration.
    assert {canonical_aliases("LAT", list(p)) for p in itertools.permutations(forms)} == {expected}


def test_term_clusters_do_not_depend_on_observation_order() -> None:
    observations = [
        TermObservation(
            term=term, kind=kind, aliases=aliases, quote=quote, defines=False, section_ordinal=n
        )
        for term, kind, aliases, quote, n in (
            ("Datum", "concept", ("chart datum",), "Datum is fixed.", 1),
            ("datum", "acronym", ("Chart Datum",), "The datum moved.", 2),
        )
    ]
    sections = ("Datum is fixed.", "The datum moved.")

    views = [
        CorpusView((CorpusDocument("doc_001", "m.md", reading(sections, p)),))
        for p in itertools.permutations(observations)
    ]
    clusters = {tuple(term_clusters(view)) for view in views}

    (only,) = clusters
    (cluster,) = only
    assert (cluster.term, cluster.kind, cluster.aliases) == ("Datum", "acronym", ("Chart Datum",))
    assert [q.section_id for q in cluster.quotes] == ["doc_001_sec_001", "doc_001_sec_002"]


def test_induction_orders_documents_and_sections_by_path_not_by_document_id() -> None:
    paths = ("p1/a.md", "p2/b.md", "p3/c.md")
    ids = [document_id(p) for p in paths]
    assert sorted(ids) == ids[::-1], "the paths' IDs must sort against path order"
    term = TermObservation(
        term="LAT", kind="acronym", aliases=(), quote="LAT", defines=False, section_ordinal=1
    )
    view = view_of(*((p, reading(["LAT is the datum."], [term])) for p in paths))
    sections = [sec(p) for p in paths]

    (cluster,) = term_clusters(view)
    assert [q.section_id for q in cluster.quotes] == sections
    ((first, *_),) = [
        [d.doc_id for d in b]
        for b in induction.digest_batches(
            [
                induction.DocumentDigest(d.doc_id, d.rel_path, "md", None, "memo", "T", "S", ())
                for d in view.citable
            ]
        )
    ]
    assert first == ids[0]


def test_definitions_are_judged_over_complete_sections_in_windows() -> None:
    filler = "\n\n".join(
        " ".join(f"Line {p}.{i} was surveyed twice." for i in range(30)) for p in range(140)
    )
    view = view_of(
        (
            "p1/memo.md",
            reading([f"# Datum\n\n{filler}\n\n**LAT** means the lowest astronomical tide."]),
        )
    )
    gateway = FakeGateway()

    windows = view.evidence([sec("p1/memo.md")])
    (probability,) = induction.judge_support(
        gateway,
        "check_definitions",
        windows,
        [induction.definition_predicate("LAT", "LAT means the lowest astronomical tide.")],
    )

    assert len(filler) > 2 * EVIDENCE_WINDOW_CHARS
    assert len(windows) >= 3 and all(len(w) <= EVIDENCE_WINDOW_CHARS for w in windows)
    assert [e for _, e, _ in gateway.judged] == list(windows)
    # Only the last window holds the definition; the highest probability decides.
    assert "lowest astronomical tide" in windows[-1] and "lowest astronomical" not in windows[0]
    assert probability == 0.9


@pytest.mark.parametrize("heading_chars", [45_000, 46_000, 200_000])
def test_a_section_with_a_very_long_heading_is_judged_in_few_windows(
    tmp_path: Path, heading_chars: int
) -> None:
    files = {"p1/memo.md": f"# {'H' * heading_chars}\n\n**LAT** means lowest astronomical tide.\n"}
    gateway = FakeGateway()

    corpus = compile_files(tmp_path, files, gateway)

    (entry,) = corpus.glossary
    assert entry.senses[0].definition == "LAT means lowest astronomical tide."
    windows = [e for op, e, _ in gateway.judged if op == "check_definitions"]
    # The heading line is part of the section, so it alone sets how many windows it needs.
    assert 1 <= len(windows) <= heading_chars // 40_000 + 2
    assert all(len(w) <= EVIDENCE_WINDOW_CHARS for w in windows)
    assert "lowest astronomical tide" in windows[-1]
    # Every induction prompt still shows the heading cut to its label length.
    induction_requests = (OrganiseCorpus, CanonicaliseTypes, WriteGlossary, WriteGuide)
    assert (
        max(prompt_chars(r) for r in gateway.requests if isinstance(r, induction_requests)) < 40_000
    )


def test_a_notes_only_corpus_gets_a_guide_without_citation_tokens(tmp_path: Path) -> None:
    files = {"AGENTS.md": "# Library\n\nNothing is filed yet.", "p1/AGENTS.md": "# P1\n\nEmpty."}
    gateway = FakeGateway(guide_script=["See [doc_001:doc_001_sec_001].", "Nothing is filed yet."])

    corpus = compile_files(tmp_path, files, gateway)

    assert corpus.guide_markdown == "Nothing is filed yet."
    rerun = gateway.of_type(WriteGuide)[1]
    assert rerun.feedback is not None and "valid citations of" not in rerun.feedback
    assert "no citable sections, so the guide must contain no citation tokens" in rerun.feedback
    # Nothing to organise, so no organisation or canonicalisation request is made.
    assert gateway.of_type(OrganiseCorpus) == [] and gateway.of_type(CanonicaliseTypes) == []


class TimedGateway(FakeGateway):
    """Counts provider calls in flight; each call takes long enough for calls to overlap."""

    def __init__(self, **scripts: Any) -> None:
        super().__init__(**scripts)
        self.in_flight = self.peak = 0
        self._flight = threading.Lock()

    def _timed[R](self, call: Callable[[], R], seconds: float = 0.03) -> R:
        with self._flight:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        try:
            time.sleep(seconds)
            return call()
        finally:
            with self._flight:
                self.in_flight -= 1

    def run[T: BaseModel](self, request: StructuredRequest[T]) -> T:
        return self._timed(lambda: super(TimedGateway, self).run(request))

    def judge(self, operation: str, evidence: str, predicates: Sequence[str]) -> list[float | None]:
        return self._timed(lambda: super(TimedGateway, self).judge(operation, evidence, predicates))


def test_one_limit_bounds_glossary_definition_and_lesson_calls_together(tmp_path: Path) -> None:
    for rel_path, text in MARKETING_CORPUS.items():
        (tmp_path / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel_path).write_text(text, encoding="utf-8")
    gateway = TimedGateway()

    corpus = compile_corpus(tmp_path, gateway, topic="t", title="T", max_concurrency=2)

    assert gateway.peak == 2
    assert [e.term for e in corpus.glossary] == ["Churn", "CTA", "ICP"]
    assert [t.name for t in corpus.deliverable_types] == ["campaign brief", "review notes"]
    with pytest.raises(ValueError, match="max_workers"):
        induce((), (), FakeGateway(), topic="t", max_concurrency=0)


def test_no_queued_induction_task_starts_after_a_task_failed() -> None:
    release_first, release_second = threading.Event(), threading.Event()
    started: list[str] = []

    def fail(name: str, gate: threading.Event) -> None:
        started.append(name)
        gate.wait()
        raise RuntimeError(name)

    with InductionExecutor(2) as pool:
        first = pool.submit(fail, "first", release_first)
        second = pool.submit(fail, "second", release_second)
        # Both workers are busy, so this task queues behind them.
        queued = pool.submit(started.append, "queued")
        release_first.set()
        # The first failure frees a worker before the coordinator looks at any result; the
        # queued task must still not start.
        wait([first])
        release_second.set()
        wait([first, second, queued])

        assert sorted(started) == ["first", "second"]
        with pytest.raises(RuntimeError, match=r"^first$"):
            pool.results([first, second, queued])
