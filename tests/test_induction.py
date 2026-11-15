from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import wait
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from arkive_rag import induction
from arkive_rag.compiler import compile_corpus
from arkive_rag.induction import (
    EVIDENCE_WINDOW_CHARS,
    MAX_DELIVERABLE_TYPES,
    Checked,
    CorpusView,
    CraftLead,
    InductionExecutor,
    OrganiseCorpus,
    WriteBlueprint,
    WriteGlossary,
    WriteGuide,
    canonical_aliases,
    check_organisation,
    generate_checked,
    induce,
    term_clusters,
)
from arkive_rag.llm import AgentRequest, StructuredRequest
from arkive_rag.models import (
    BlueprintDraft,
    CompiledCorpus,
    CorpusDocument,
    DeliverableTypeDraft,
    DocumentReading,
    FacetDraft,
    FacetValueDraft,
    GlossaryDraft,
    GlossaryEntryDraft,
    NotFoundError,
    ObservationCard,
    OrganisationDraft,
    ReadSection,
    RejectedOutputError,
    SenseDraft,
    TermObservation,
)
from arkive_rag.snapshot import TopicSnapshot, write_snapshot
from tests.support import BATHYMETRY_CORPUS, MARKETING_CORPUS, FakeGateway


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


def test_guide_citations_get_one_rerun_listing_the_valid_tokens_then_raise(
    tmp_path: Path,
) -> None:
    files = {"p1/memo.md": "# Results\n\nSeagrass mapped.", "p1/AGENTS.md": "# Note\n\nDraft."}
    gateway = FakeGateway(guide_script=["See [doc_002:bad_sec].", "See [doc_002:doc_002_sec_001]."])

    corpus = compile_files(tmp_path / "ok", files, gateway)

    assert corpus.guide_markdown == "See [doc_002:doc_002_sec_001]."
    first, rerun = gateway.of_type(WriteGuide)
    assert first.feedback is None and rerun.feedback is not None
    assert "[doc_002:bad_sec]" in rerun.feedback and "[doc_002:doc_002_sec_001]" in rerun.feedback
    # The curator note (doc_001) is not citable.
    assert "doc_001_sec_001" not in rerun.feedback.split("cite only")[1]
    stubborn = FakeGateway(guide_script=["No citations.", "Still [doc_001:doc_001_sec_001]."])
    with pytest.raises(RejectedOutputError, match=r"\[doc_001:doc_001_sec_001\]"):
        compile_files(tmp_path / "bad", files, stubborn)


# Organisation.


def value(name: str, *folders: str, aliases: Sequence[str] = ()) -> FacetValueDraft:
    return FacetValueDraft(
        value=name, description=name, aliases=list(aliases), folders=list(folders)
    )


def organisation(
    *facets: FacetDraft, types: Sequence[DeliverableTypeDraft] = ()
) -> OrganisationDraft:
    return OrganisationDraft(facets=list(facets), deliverable_types=list(types))


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
    types=[
        DeliverableTypeDraft(
            name="memo", description="Memo.", exemplar_doc_ids=["doc_009"], feedback_doc_ids=[]
        )
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
    assert "unknown document ID 'doc_009'" in message
    first, rerun = gateway.of_type(OrganiseCorpus)
    assert first.feedback is None and rerun.feedback is not None and "doc_009" in rerun.feedback


def test_organisation_normalises_repetition_within_a_value_so_the_snapshot_stores_it(
    tmp_path: Path,
) -> None:
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
        assert snap.resolve_filters({"client": "nm"}) == frozenset({"doc_001"})


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
    draft = organisation(FacetDraft(name="client", description="Client.", values=values))

    checked = check_organisation(draft, frozenset(), frozenset())

    assert checked.result is None
    assert any(message in r for r in checked.rejections), checked.rejections


def test_more_recurring_deliverable_types_than_the_cap_are_rejected_and_rerun(
    tmp_path: Path,
) -> None:
    many = organisation(
        types=[
            DeliverableTypeDraft(
                name=f"type {i}",
                description="A type.",
                exemplar_doc_ids=["doc_001"],
                feedback_doc_ids=[],
            )
            for i in range(MAX_DELIVERABLE_TYPES + 1)
        ]
    )
    gateway = FakeGateway(organise_script=[many, many])

    with pytest.raises(RejectedOutputError, match="9 deliverable types; give at most 8"):
        compile_files(tmp_path, BATHYMETRY_CORPUS, gateway)
    rerun = gateway.of_type(OrganiseCorpus)[1]
    assert rerun.feedback is not None and "give at most 8" in rerun.feedback
    assert "at most 8 kinds of deliverable" in OrganiseCorpus.instructions


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
        super().judge(operation, evidence, predicates)
        if operation != "bind_facets":
            return [0.9] * len(predicates)
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

    gateway = EveryChannel(organise_script=[organisation(campaigns, audiences)])

    corpus = compile_files(tmp_path, MARKETING_CORPUS, gateway)

    bound = facets_by_doc(corpus)
    assert bound["spring-launch/brief.md"]["audience"] == ("Email", "Social")
    assert bound["spring-launch/review.md"]["audience"] == ("Social",)
    assert "audience" not in bound["AGENTS.md"]
    assert corpus.build_report.pruned_facet_values == ["audience=Print"]
    assert [v.value for v in corpus.facets[1].values] == ["Social", "Email"]


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
            ("doc_002_sec_002",),
        )
    ]
    # Two documents define CTA differently: two senses, each citing its own section.
    assert senses(corpus, "CTA") == [
        ("CTA (call to action) is the button that closes every email.", ("doc_002_sec_003",)),
        (
            "CTA (call to action) means the single step a reader should take next.",
            ("doc_003_sec_003",),
        ),
    ]
    (cta,) = [e for e in corpus.glossary if e.term == "CTA"]
    assert cta.aliases == ("call to action",)
    # Used in two documents, defined in neither: discoverable, with no definition.
    assert senses(corpus, "ICP") == [(None, ("doc_003_sec_003", "doc_004_sec_002"))]
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

    assert senses(corpus, "Churn") == [(None, ("doc_002_sec_002",))]
    assert senses(corpus, "CTA")[0] == (None, ("doc_002_sec_003",))
    assert senses(corpus, "CTA")[1][0] is not None
    assert corpus.build_report.unsupported_definitions == 2
    assert corpus.build_report.judge_refusals == {"check_definitions": 1}


LAT_CLUSTER_SECTIONS = ("doc_001_sec_003", "doc_002_sec_002")


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
    # doc_002_sec_002 says only "LAT"; the entry publishes neither "LAT" nor an alias for it.
    unpublished = glossary_draft("Lowest astronomical tide", [["doc_002_sec_002"]])
    gateway = FakeGateway(glossary_script=[unpublished, unpublished])

    with pytest.raises(RejectedOutputError, match="doc_002_sec_002 contains neither"):
        compile_files(tmp_path, BATHYMETRY_CORPUS, gateway)
    first, rerun = gateway.of_type(WriteGlossary)
    assert [c.term for c in first.clusters] == ["LAT"]
    assert rerun.feedback is not None and "add the form used there as an alias" in rerun.feedback

    published = glossary_draft("Lowest astronomical tide", [["doc_002_sec_002"]], ["LAT"])
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


def reading(markdowns: Sequence[str], terms: Sequence[TermObservation]) -> DocumentReading:
    return DocumentReading(
        sha256="0" * 64,
        format="md",
        is_curator_note=False,
        sections=tuple(
            ReadSection(
                ordinal=i,
                heading_path=f"Part {i}",
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
            title="Memo",
            doc_date=None,
            deliverable_type="memorandum",
            summary="Memo.",
            attributes=(),
            terms=tuple(terms),
            craft=(),
        ),
        discarded_quotes=0,
    )


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


def test_definitions_are_judged_over_complete_sections_in_windows(tmp_path: Path) -> None:
    filler = "\n\n".join(
        " ".join(f"Line {p}.{i} was surveyed twice." for i in range(30)) for p in range(140)
    )
    files = {
        "p1/memo.md": f"# Datum\n\n{filler}\n\n**LAT** means the lowest astronomical tide.\n",
    }
    gateway = FakeGateway()

    corpus = compile_files(tmp_path, files, gateway)

    assert len(filler) > 2 * EVIDENCE_WINDOW_CHARS
    (entry,) = corpus.glossary
    assert entry.senses[0].definition == "LAT means the lowest astronomical tide."
    assert corpus.build_report.unsupported_definitions == 0
    windows = [e for op, e, _ in gateway.judged if op == "check_definitions"]
    assert len(windows) >= 3 and all(len(w) <= EVIDENCE_WINDOW_CHARS for w in windows)
    # Only the last window holds the definition; the highest probability decides.
    assert "lowest astronomical tide" in windows[-1] and "lowest astronomical" not in windows[0]


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


# Blueprints.


def test_blueprints_cite_resolvable_sections_and_exemplars(tmp_path: Path) -> None:
    gateway = FakeGateway()

    corpus = compile_files(tmp_path, MARKETING_CORPUS, gateway)

    brief, review = corpus.blueprints
    assert (brief.deliverable_type, review.deliverable_type) == ("campaign brief", "review notes")
    assert brief.exemplar_doc_ids == ("doc_002", "doc_003")
    assert brief.feedback_doc_ids == ("doc_004",)
    sections = {s.section_id for d in corpus.documents for s in d.sections}
    cited = [c for s in brief.outline for c in s.citations]
    cited += [c for g in (*brief.conventions, *brief.pitfalls) for c in g.citations]
    assert cited and set(cited) <= sections
    assert [g.statement for g in brief.conventions] == [
        "Lead with the customer benefit before the product name."
    ]
    assert brief.conventions[0].citations == ("doc_002_sec_003", "doc_003_sec_003")
    assert [g.statement for g in brief.pitfalls] == ["Cut the jargon from the opening line."]
    assert [s.heading for s in brief.outline] == ["Title and summary", "Audience", "Messaging"]
    # The agent run used its tools; its cache identity covers everything they can read.
    request = gateway.of_type(WriteBlueprint)[0]
    # Craft observations of the exemplars and the feedback document are leads to verify.
    assert request.craft_leads == (
        CraftLead(
            "doc_002_sec_003",
            "convention",
            "Lead with the customer benefit before the product name.",
        ),
        CraftLead(
            "doc_003_sec_003",
            "convention",
            "Lead with the customer benefit before the product name.",
        ),
        CraftLead("doc_004_sec_002", "reviewer_feedback", "Cut the jargon from the opening line."),
    )
    assert "[doc_004_sec_002] reviewer_feedback: Cut the jargon" in request.text()
    assert "leads, not evidence" in WriteBlueprint.instructions
    assert {t.name for t in request.tools} == {"outline", "read_section", "grep"}
    assert len(request.source_digest) == 64


def test_blueprint_tools_cannot_read_curator_notes_and_their_revision_is_in_the_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway = FakeGateway()
    corpus = compile_files(tmp_path, MARKETING_CORPUS, gateway)
    assert corpus.documents[0].is_curator_note and corpus.documents[0].doc_id == "doc_001"
    request = gateway.of_type(WriteBlueprint)[0]
    tools = {t.name: t for t in request.tools}

    with pytest.raises(NotFoundError, match="doc_001"):
        tools["outline"].invoke({"doc_id": "doc_001"})
    with pytest.raises(NotFoundError, match="doc_001_sec_001"):
        tools["read_section"].invoke({"section_id": "doc_001_sec_001"})
    assert tools["grep"].invoke({"pattern": "autumn recap"}) == "[]"
    with pytest.raises(ValueError, match="blank"):
        tools["grep"].invoke({"pattern": " "})
    for name, args in (
        ("grep", {"pattern": 3}),
        ("outline", {"doc_id": ["doc_002"]}),
        ("read_section", {}),
    ):
        with pytest.raises(ValueError, match="must be a string"):
            tools[name].invoke(args)

    monkeypatch.setattr(induction, "SOURCE_TOOLS_REVISION", "arkive-source-tools/test")
    revised = FakeGateway()
    compile_files(tmp_path, {}, revised)
    assert revised.of_type(WriteBlueprint)[0].source_digest != request.source_digest


class DoubtsEverything(FakeGateway):
    def judge(self, operation: str, evidence: str, predicates: Sequence[str]) -> list[float | None]:
        answers = super().judge(operation, evidence, predicates)
        return [0.3] * len(predicates) if operation == "check_blueprint" else answers


def test_blueprint_guidance_the_judge_does_not_support_is_dropped_and_counted(
    tmp_path: Path,
) -> None:
    corpus = compile_files(tmp_path, MARKETING_CORPUS, DoubtsEverything())

    brief = corpus.blueprints[0]
    assert (brief.conventions, brief.pitfalls) == ((), ())
    assert brief.outline
    assert corpus.build_report.dropped_blueprint_statements == 2


def test_a_blueprint_with_an_empty_outline_fails_after_one_rerun(tmp_path: Path) -> None:
    empty = BlueprintDraft(
        description="Nothing.",
        exemplar_doc_ids=["doc_002"],
        outline=[],
        conventions=[],
        pitfalls=[],
    )
    gateway = FakeGateway(blueprint_script=[empty, empty])

    with pytest.raises(
        RejectedOutputError, match=r"blueprint for 'campaign brief'.*outline is empty"
    ):
        compile_files(tmp_path, MARKETING_CORPUS, gateway)
    first, rerun = gateway.of_type(WriteBlueprint)[:2]
    assert first.feedback is None and rerun.feedback is not None
    assert "the outline is empty" in rerun.feedback


def test_a_notes_only_corpus_gets_a_guide_without_citation_tokens(tmp_path: Path) -> None:
    files = {"AGENTS.md": "# Library\n\nNothing is filed yet.", "p1/AGENTS.md": "# P1\n\nEmpty."}
    gateway = FakeGateway(guide_script=["See [doc_001:doc_001_sec_001].", "Nothing is filed yet."])

    corpus = compile_files(tmp_path, files, gateway)

    assert corpus.guide_markdown == "Nothing is filed yet."
    rerun = gateway.of_type(WriteGuide)[1]
    assert rerun.feedback is not None and "cite only" not in rerun.feedback
    assert "no citable sections, so the guide must contain no citation tokens" in rerun.feedback


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

    def run_agent[T: BaseModel](self, request: AgentRequest[T]) -> T:
        return self._timed(lambda: super(TimedGateway, self).run_agent(request))

    def judge(self, operation: str, evidence: str, predicates: Sequence[str]) -> list[float | None]:
        return self._timed(lambda: super(TimedGateway, self).judge(operation, evidence, predicates))


def test_one_limit_bounds_glossary_definition_and_blueprint_calls_together(tmp_path: Path) -> None:
    for rel_path, text in MARKETING_CORPUS.items():
        (tmp_path / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel_path).write_text(text, encoding="utf-8")
    gateway = TimedGateway()

    corpus = compile_corpus(tmp_path, gateway, topic="t", title="T", max_concurrency=2)

    assert gateway.peak == 2
    assert [e.term for e in corpus.glossary] == ["Churn", "CTA", "ICP"]
    assert [b.deliverable_type for b in corpus.blueprints] == ["campaign brief", "review notes"]
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
