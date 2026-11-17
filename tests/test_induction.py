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
    HOUSE_TEXT_OPERATION,
    HOUSE_TEXT_PREDICATE,
    MAX_DELIVERABLE_TYPES,
    Checked,
    CorpusView,
    CraftLead,
    DeliverableType,
    InductionExecutor,
    OrganiseCorpus,
    WriteBlueprint,
    WriteGlossary,
    WriteGuide,
    canonical_aliases,
    check_organisation,
    exemplar_documents,
    generate_checked,
    induce,
    term_clusters,
)
from arkive_rag.llm import AgentRequest, StructuredRequest
from arkive_rag.models import (
    BlueprintDraft,
    BlueprintSectionDraft,
    CompiledCorpus,
    ConditionalPassage,
    CorpusDocument,
    DeliverableTypeDraft,
    DocumentReading,
    FacetDraft,
    FacetValueDraft,
    GlossaryDraft,
    GlossaryEntryDraft,
    GuidanceDraft,
    HouseSectionNomination,
    NominatedUnitRecord,
    NotFoundError,
    ObservationCard,
    OrganisationDraft,
    PassageSelection,
    PassageVariant,
    ProviderError,
    ReadSection,
    RecurringCandidateRecord,
    RecurringKind,
    RejectedOutputError,
    SenseDraft,
    SingleExemplarPassage,
    StandingPassage,
    TermObservation,
)
from arkive_rag.passages import standing_candidates, word_candidates
from arkive_rag.snapshot import TopicSnapshot, write_snapshot
from tests.support import (
    BATHYMETRY_CORPUS,
    DATA_HANDLING,
    DATED_REPORTS,
    HOUSE_DISCLOSURE,
    HOUSE_STANDARD,
    MARKETING_CORPUS,
    REDUCTION,
    FakeGateway,
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
        ),
        DeliverableTypeDraft(
            name="proposal",
            description="Proposal.",
            exemplar_doc_ids=["doc_001", "doc_002"],
            feedback_doc_ids=["doc_002"],
        ),
        DeliverableTypeDraft(
            name="  ", description="Nothing.", exemplar_doc_ids=["doc_001"], feedback_doc_ids=[]
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
    assert "unknown document ID 'doc_009'" in message
    # A feedback document of a type must not also supply its exemplar text.
    assert "'proposal': 'doc_002' is both an exemplar and a feedback document" in message
    # A snapshot resolves a type by its fold, so a type that folds to nothing names nothing.
    assert "deliverable type '  ' is blank" in message
    first, rerun = gateway.of_type(OrganiseCorpus)
    assert first.feedback is None and rerun.feedback is not None and "doc_009" in rerun.feedback
    assert "'doc_002' is both an exemplar and a feedback document" in rerun.feedback


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
        date=None,
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
    assert [s.heading for s in brief.outline] == [
        "Title and summary",
        "Audience",
        "Messaging",
        "Disclosure",
    ]
    # Both briefs share the convention line and the disclosure; the fake leaves the
    # `Convention:` line unselected. The single review-notes exemplar has no candidates.
    request, review_request = gateway.of_type(WriteBlueprint)
    text = request.text()
    assert "[candidate_01] found in doc_002_sec_003, doc_003_sec_003;" in text
    assert "[candidate_02] found in doc_002_sec_004, doc_003_sec_004;" in text
    assert "[candidate_03]" not in text
    (disclosure,) = brief.passages
    assert (disclosure.kind, disclosure.label) == ("standing", "Disclosure")
    assert disclosure.text == HOUSE_DISCLOSURE.split("\n\n")[1].strip()
    assert disclosure.citations == ("doc_002_sec_004", "doc_003_sec_004")
    assert review.passages == ()
    assert "(none: this type has one exemplar document" in review_request.text()
    assert corpus.build_report.dropped_blueprint_statements == 0
    # The agent run used its tools; its cache identity covers everything they can read.
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
    # Only the unsupported convention and pitfall count; recurring passages are never dropped.
    assert brief.passages
    assert corpus.build_report.dropped_blueprint_statements == 2


def test_a_blueprint_with_an_empty_outline_fails_after_one_rerun(tmp_path: Path) -> None:
    empty = BlueprintDraft(
        description="Nothing.",
        outline=[],
        conventions=[],
        pitfalls=[],
        passages=[],
        house_sections=[],
    )
    gateway = FakeGateway(blueprint_script=[empty, empty])

    with pytest.raises(
        RejectedOutputError, match=r"blueprint for 'campaign brief'.*outline is empty"
    ):
        compile_files(tmp_path, MARKETING_CORPUS, gateway)
    first, rerun = gateway.of_type(WriteBlueprint)[:2]
    assert first.feedback is None and rerun.feedback is not None
    assert "the outline is empty" in rerun.feedback


# Standing passage candidates offered to the blueprint agent.


def candidate_view(*docs: tuple[str, Sequence[str]]) -> CorpusView:
    """Documents doc_001, doc_002, ... in argument order; each section's Markdown as given."""
    return CorpusView(
        tuple(
            CorpusDocument(f"doc_{n:03d}", rel_path, reading(markdowns, ()))
            for n, (rel_path, markdowns) in enumerate(docs, start=1)
        )
    )


def test_the_blueprint_request_says_why_it_lists_no_candidates() -> None:
    def request(*exemplars: str) -> str:
        dtype = DeliverableType("memo", "A memo.", exemplars, ())
        return WriteBlueprint(dtype, (), (), (), (), (), 0, (), "0" * 64).text()

    assert "(none: this type has one exemplar document" in request("doc_001")
    assert "(none: no block of this type's exemplars recurs" in request("doc_001", "doc_002")


def test_candidates_beyond_the_cap_keep_those_in_the_most_documents_and_the_request_says_so() -> (
    None
):
    assert induction.MAX_STANDING_CANDIDATES == 60
    # Sixty-one distinct sentences recur between doc_001 and doc_002; the last two also recur
    # in doc_003, so they outrank the rest, and the tie among the others keeps canonical order.
    sentences = [
        f"Standing clause number {n} of the house terms binds every survey we carry out."
        for n in range(1, 62)
    ]
    view = candidate_view(
        ("p1/a.md", sentences), ("p2/b.md", sentences), ("p3/c.md", sentences[-2:])
    )
    exemplars = ("doc_001", "doc_002", "doc_003")
    documents = exemplar_documents(view, exemplars)
    found = standing_candidates(documents)

    offered = induction.offered_candidates(view, found)

    assert len(found) == 61
    assert [c.candidate_id for c in offered] == [
        f"candidate_{n:02d}" for n in (*range(1, 59), 60, 61)
    ]
    dtype = DeliverableType("terms", "Terms.", exemplars, ())
    worded = word_candidates(documents, offered)

    cut = WriteBlueprint(dtype, (), (), (), (), worded, len(found), (), "0" * 64)
    assert "(this list was cut from 61 candidates to the 60 that recur in the most" in cut.text()
    whole = WriteBlueprint(dtype, (), (), (), (), worded, 60, (), "0" * 64)
    assert "was cut" not in whole.text()


Selection = (
    tuple[str, str]
    | tuple[str, str, RecurringKind, str | None]
    | tuple[str, str, RecurringKind, str | None, str]
)


def selecting(*selections: Selection, house: Sequence[tuple[str, str]] = ()) -> BlueprintDraft:
    """A blueprint answer selecting `(candidate_id, label)` as standing, or with an explicit
    kind and condition and optionally the holder section whose wording to take, and
    nominating `(section_id, label)` house sections."""
    return BlueprintDraft(
        description="A brief.",
        outline=[
            BlueprintSectionDraft(heading="Audience", purpose="Who.", citations=["doc_002_sec_002"])
        ],
        conventions=[],
        pitfalls=[],
        passages=[
            PassageSelection(
                candidate_id=s[0],
                label=s[1],
                kind=s[2] if len(s) >= 4 else "standing",
                condition=s[3] if len(s) >= 4 else None,
                holder_section_id=s[4] if len(s) == 5 else None,
            )
            for s in selections
        ],
        house_sections=[HouseSectionNomination(section_id=i, label=label) for i, label in house],
    )


def test_unknown_or_repeated_passage_selections_and_nominations_are_rerun_then_rejected(
    tmp_path: Path,
) -> None:
    bad = selecting(
        ("candidate_09", "Made up"),
        ("candidate_02", "A"),
        ("candidate_02", "B"),
        ("candidate_01", "  "),
        # doc_001 is the curator note and doc_004 the feedback document: neither is an
        # exemplar of the campaign brief.
        house=[
            ("doc_001_sec_001", "Library"),
            ("doc_004_sec_002", "Comments"),
            ("doc_002_sec_002", "Audience"),
            ("doc_002_sec_002", "Again"),
            ("doc_003_sec_002", " "),
        ],
    )
    good = selecting(("candidate_02", "Disclosure"))
    gateway = FakeGateway(blueprint_script=[bad, good])

    corpus = compile_files(tmp_path / "rerun", MARKETING_CORPUS, gateway)

    first, rerun = gateway.of_type(WriteBlueprint)[:2]
    assert first.feedback is None and rerun.feedback is not None
    assert "'candidate_09' is not a listed candidate_id" in rerun.feedback
    assert "'candidate_02' is selected twice" in rerun.feedback
    assert "'candidate_01' has a blank label" in rerun.feedback
    for section_id in ("doc_001_sec_001", "doc_004_sec_002"):
        assert (
            f"house section {section_id!r} is not a section of an exemplar document of this type"
            in rerun.feedback
        )
    assert "house section 'doc_002_sec_002' is listed twice" in rerun.feedback
    assert "house section 'doc_003_sec_002' has a blank label" in rerun.feedback
    (passage,) = corpus.blueprints[0].passages
    assert passage.label == "Disclosure"
    assert passage.text == HOUSE_DISCLOSURE.split("\n\n")[1].strip()
    # The convention line the answer left out is declined.
    assert [
        (r.candidate_id, r.disposition)
        for r in corpus.candidate_records
        if isinstance(r, RecurringCandidateRecord)
    ] == [("candidate_01", "declined"), ("candidate_02", "selected_standing")]
    stubborn = FakeGateway(blueprint_script=[bad, bad])
    with pytest.raises(RejectedOutputError, match="'candidate_09' is not a listed candidate_id"):
        compile_files(tmp_path / "stubborn", MARKETING_CORPUS, stubborn)


# Survey report sections in `DATED_REPORTS`: _001 the title block, _002 Method, _003 Results,
# and _004 Data handling, which only the newest report has.
NOTE = "# Reports\n\nThe layout with a data handling section is the current one."


def test_the_blueprint_request_carries_curator_notes_dates_wordings_and_variants(
    tmp_path: Path,
) -> None:
    gateway = FakeGateway()

    compile_files(tmp_path / "dated", {"AGENTS.md": NOTE, **DATED_REPORTS}, gateway)

    (request,) = gateway.of_type(WriteBlueprint)
    text = request.text()
    # Notes travel in the request as stated intent; the agent may not cite them.
    assert f"CURATOR NOTES (stated intent; never cite them):\n### AGENTS.md\n{NOTE}\n\n" in text
    assert "a note is not\na source" in WriteBlueprint.instructions
    assert "outranks dates" in WriteBlueprint.instructions
    # Each exemplar outline carries its document's date and the date's basis.
    for doc_id, name, date in (
        ("doc_002", "North", "2024-03-01"),
        ("doc_003", "East", "2025-03-12"),
        ("doc_004", "South", "2026-01-20"),
    ):
        assert f"{doc_id}: {name} Survey Report\nDate: {date} (printed)\n" in text
    # Every candidate shows the wording a passage would store and why; the reduction sentence
    # also shows the older holders' wordings with their dates.
    holders = "found in doc_002_sec_002, doc_003_sec_002, doc_004_sec_002; wording from"
    assert (
        f"[candidate_01] {holders} doc_004_sec_002 (the newest dated holder)\n{HOUSE_STANDARD}"
        f"\n\n[candidate_02] {holders} doc_004_sec_002 (the newest dated holder)\n"
        f"{REDUCTION.format('fortnight')}\n"
        f"Variant in doc_002_sec_002 (2024-03-01, printed):\n{REDUCTION.format('week')}\n"
        f"Variant in doc_003_sec_002 (2025-03-12, printed):\n{REDUCTION.format('month')}"
    ) in text
    # Without dates no order is claimed, and the request says so.
    undated = FakeGateway()
    compile_files(tmp_path / "undated", MARKETING_CORPUS, undated)
    brief_request = undated.of_type(WriteBlueprint)[0]
    assert "doc_002: Loyalty Relaunch Brief\nDate: none known\n" in brief_request.text()
    assert (
        "wording from doc_002_sec_004 (the first holder, because the holders' dates do not show"
        " which is newest)"
    ) in brief_request.text()
    # A note is not a source: an outline citing it is rejected and rerun.
    cites_note = selecting().model_copy(
        update={
            "outline": [
                BlueprintSectionDraft(heading="Notes", purpose="?", citations=["doc_001_sec_001"])
            ]
        }
    )
    stubborn = FakeGateway(blueprint_script=[cites_note, cites_note])
    with pytest.raises(RejectedOutputError, match="'doc_001_sec_001' is not a citable section"):
        compile_files(tmp_path / "cites-note", {"AGENTS.md": NOTE, **DATED_REPORTS}, stubborn)


CONDITION = "The project names the period whose tide gauge records reduce its soundings."


def test_a_selection_is_standing_or_conditional_and_only_a_conditional_one_has_a_condition(
    tmp_path: Path,
) -> None:
    bad = selecting(
        ("candidate_01", "Standard", "conditional", "  "),
        ("candidate_02", "Reduction", "standing", CONDITION),
    )
    good = selecting(
        ("candidate_01", "Standard", "standing", ""),
        ("candidate_02", "Reduction", "conditional", CONDITION),
    )
    gateway = FakeGateway(blueprint_script=[bad, good])

    corpus = compile_files(tmp_path / "rerun", DATED_REPORTS, gateway)

    rerun = gateway.of_type(WriteBlueprint)[1].feedback
    assert rerun is not None
    # The standard sentence has no variants, so it may be made standing instead.
    assert (
        "passage 'candidate_01' is conditional without a condition; state the fact of the new"
        " project that makes it apply and the part that varies, or make it standing"
    ) in rerun
    assert "passage 'candidate_02' is standing but has a condition" in rerun
    standard, reduction = corpus.blueprints[0].passages
    assert standard == StandingPassage(
        kind="standing",
        label="Standard",
        text=HOUSE_STANDARD,
        citations=("doc_001_sec_002", "doc_002_sec_002", "doc_003_sec_002"),
        source_section_id="doc_003_sec_002",
        wording_from="newest_holder",
    )
    assert reduction == ConditionalPassage(
        kind="conditional",
        label="Reduction",
        text=REDUCTION.format("fortnight"),
        citations=("doc_001_sec_002", "doc_002_sec_002", "doc_003_sec_002"),
        source_section_id="doc_003_sec_002",
        wording_from="newest_holder",
        condition=CONDITION,
        variants=(
            PassageVariant(
                section_id="doc_001_sec_002",
                doc_date_iso="2024-03-01",
                text=REDUCTION.format("week"),
            ),
            PassageVariant(
                section_id="doc_002_sec_002",
                doc_date_iso="2025-03-12",
                text=REDUCTION.format("month"),
            ),
        ),
    )
    # The provider schema parses a mismatch, so it reaches the rerun above instead of failing
    # as malformed output.
    parsed = BlueprintDraft.model_validate_json(bad.model_dump_json())
    assert parsed.passages[0].kind == "conditional"
    stubborn = FakeGateway(blueprint_script=[bad, bad])
    with pytest.raises(RejectedOutputError, match="conditional without a condition"):
        compile_files(tmp_path / "stubborn", DATED_REPORTS, stubborn)
    # The reduction sentence has variants, so it cannot be standing: the rejection offers
    # stating the condition or leaving it out.
    unstated = selecting(("candidate_02", "Reduction", "conditional", None))
    varied = FakeGateway(blueprint_script=[unstated, unstated])
    with pytest.raises(RejectedOutputError) as raised:
        compile_files(tmp_path / "varied", DATED_REPORTS, varied)
    assert (
        "passage 'candidate_02' is conditional without a condition; state the fact of the new"
        " project that makes it apply and the part that varies, or leave it out"
    ) in str(raised.value)
    assert "make it standing" not in str(raised.value)


def test_a_candidate_its_holders_word_differently_cannot_be_standing(tmp_path: Path) -> None:
    # The reduction sentence has variants: its holders word it differently.
    standing = selecting(("candidate_01", "Standard"), ("candidate_02", "Reduction"))
    gateway = FakeGateway(blueprint_script=[standing, standing])

    with pytest.raises(RejectedOutputError) as raised:
        compile_files(tmp_path, DATED_REPORTS, gateway)

    assert (
        "passage 'candidate_02' is standing, but its holders word it differently (it has"
        " variants); make it conditional with a condition, or leave it out"
    ) in str(raised.value)
    assert "'candidate_01'" not in str(raised.value)
    assert (
        "A candidate listed with variants is worded differently by its holders, so it cannot be"
        " standing."
    ) in " ".join(WriteBlueprint.instructions.split())


def test_a_curator_note_can_choose_which_holders_wording_a_passage_stores(
    tmp_path: Path,
) -> None:
    note = "# Reports\n\nThe East report's wording of the reduction method is the current one."
    files = {"AGENTS.md": note, **DATED_REPORTS}
    # With the note, doc_001 is the note and the reports are doc_002 (North), doc_003 (East)
    # and doc_004 (South).
    bad = selecting(
        ("candidate_01", "Standard", "standing", None, "doc_003_sec_003"),
        ("candidate_02", "Reduction", "conditional", CONDITION, "doc_003_sec_009"),
    )
    good = selecting(("candidate_02", "Reduction", "conditional", CONDITION, "doc_003_sec_002"))
    gateway = FakeGateway(blueprint_script=[bad, good])

    corpus = compile_files(tmp_path / "corpus", files, gateway)

    rerun = gateway.of_type(WriteBlueprint)[1].feedback
    assert rerun is not None
    for cid, section_id in (
        ("candidate_01", "doc_003_sec_003"),
        ("candidate_02", "doc_003_sec_009"),
    ):
        assert (
            f"passage {cid!r}: holder_section_id {section_id!r} is not a section that holds"
            f" {cid}; name a section it is found in that holds it in one unit"
        ) in rerun
    (reduction,) = corpus.blueprints[0].passages
    assert reduction == ConditionalPassage(
        kind="conditional",
        label="Reduction",
        text=REDUCTION.format("month"),
        citations=("doc_002_sec_002", "doc_003_sec_002", "doc_004_sec_002"),
        source_section_id="doc_003_sec_002",
        wording_from="selected_holder",
        condition=CONDITION,
        variants=(
            PassageVariant(
                section_id="doc_002_sec_002",
                doc_date_iso="2024-03-01",
                text=REDUCTION.format("week"),
            ),
            PassageVariant(
                section_id="doc_004_sec_002",
                doc_date_iso="2026-01-20",
                text=REDUCTION.format("fortnight"),
            ),
        ),
    )
    assert "holder_section_id is null, unless a curator note states which wording" in (
        WriteBlueprint.instructions
    )
    # Only null means no choice: a blank holder is rejected and rerun.
    blank = selecting(("candidate_02", "Reduction", "conditional", CONDITION, " \t"))
    stubborn = FakeGateway(blueprint_script=[blank, blank])
    with pytest.raises(
        RejectedOutputError, match="passage 'candidate_02' has a blank holder_section_id"
    ):
        compile_files(tmp_path / "blank", files, stubborn)
    # The snapshot stores and serves the chosen wording.
    write_snapshot(corpus, tmp_path / "t.arkive.db")
    with TopicSnapshot.open(tmp_path / "t.arkive.db") as snap:
        assert snap.blueprints[0].passages == (reduction,)


# Three dated reports: the newest one's sentence says the same as each older one, but the two
# older ones do not say the same as each other, so they are two candidates both held by it.
SHARED_NEWEST = (
    "Every survey we deliver is reduced to the stated vertical datum, checked against an"
    " independent crossline, reviewed by a second surveyor and archived with its raw files"
    " for seven years."
)
SHARED_EARLY = SHARED_NEWEST.replace("stated", "declared")
SHARED_LATE = SHARED_NEWEST.replace("raw", "source")


def test_two_candidates_held_by_the_newest_unit_store_two_wordings(tmp_path: Path) -> None:
    files = {
        f"p{n}/report.md": (
            f"# {name} Report\n\nDeliverable: survey report\nDate: {date}\n\n## Method\n\n{text}\n"
        )
        for n, (name, date, text) in enumerate(
            (
                ("North", "2024-03-01", SHARED_EARLY),
                ("East", "2025-03-01", SHARED_LATE),
                ("South", "2026-03-01", SHARED_NEWEST),
            ),
            start=1,
        )
    }
    gateway = FakeGateway()

    corpus = compile_files(tmp_path, files, gateway)

    # The first candidate takes the newest report's unit; the second keeps its own wording.
    assert [p.text for p in corpus.blueprints[0].passages] == [SHARED_NEWEST, SHARED_LATE]
    (request,) = gateway.of_type(WriteBlueprint)
    assert (
        "[candidate_02] found in doc_002_sec_002, doc_003_sec_002; wording from doc_002_sec_002"
        " (the first holder, because the newest holder's wording is already another"
        f" candidate's)\n{SHARED_LATE}\n"
    ) in request.text()
    # A selection that names the newest holder for the second candidate too is rerun.
    both_newest = selecting(
        ("candidate_01", "Datum", "conditional", "The project names its datum."),
        (
            "candidate_02",
            "Archive",
            "conditional",
            "The project names its archive.",
            "doc_003_sec_002",
        ),
    )
    stubborn = FakeGateway(blueprint_script=[both_newest, both_newest])
    with pytest.raises(RejectedOutputError, match="passage 'candidate_02' would store the same"):
        compile_files(tmp_path / "stubborn", files, stubborn)
    # A fourth project holds the newest wording too. The second candidate's chosen holder is
    # another section than the first candidate's wording, but the wording is the same, and
    # the later candidate in candidate order is told to change, whatever the answer's order.
    four = {
        **files,
        "p4/report.md": files["p3/report.md"].replace("South", "West").replace("2026", "2027"),
    }
    chosen_twice = selecting(
        (
            "candidate_02",
            "Archive",
            "conditional",
            "The project names its archive.",
            "doc_003_sec_002",
        ),
        ("candidate_01", "Datum", "conditional", "The project names its datum."),
    )
    stubborn = FakeGateway(blueprint_script=[chosen_twice, chosen_twice])
    with pytest.raises(
        RejectedOutputError,
        match="passage 'candidate_02' would store the same wording as passage 'candidate_01'",
    ):
        compile_files(tmp_path / "four", four, stubborn)


def test_guidance_cites_only_the_types_own_exemplar_and_feedback_sections(
    tmp_path: Path,
) -> None:
    # doc_002 and doc_003 are the campaign brief's exemplars and doc_004 its feedback
    # document; doc_004 is also the review notes' only exemplar, and they have no feedback.
    def citing(section_id: str) -> BlueprintDraft:
        return selecting().model_copy(
            update={
                "outline": [
                    BlueprintSectionDraft(heading="Notes", purpose="?", citations=[section_id])
                ],
                "conventions": [GuidanceDraft(statement="Be brief.", citations=[section_id])],
                "pitfalls": [GuidanceDraft(statement="Jargon.", citations=[section_id])],
            }
        )

    # Outline items, conventions and pitfalls may all cite the type's feedback document, and
    # none may cite another type's exemplar.
    brief, review = citing("doc_004_sec_002"), citing("doc_002_sec_002")
    gateway = FakeGateway(blueprint_script=[brief, review, review])

    with pytest.raises(RejectedOutputError, match="blueprint for 'review notes'") as raised:
        compile_files(tmp_path, MARKETING_CORPUS, gateway)

    requests = gateway.of_type(WriteBlueprint)
    assert [(r.deliverable_type.name, r.feedback is None) for r in requests] == [
        ("campaign brief", True),
        ("review notes", True),
        ("review notes", False),
    ]
    for kind, label in (("outline", "Notes"), ("convention", "Be brief."), ("pitfall", "Jargon.")):
        assert (
            f"{kind} {label!r}: 'doc_002_sec_002' is not a section of this type's exemplar or"
            " feedback documents"
        ) in str(raised.value)


def test_code_takes_a_nominated_sections_units_except_short_recurring_or_overlapping_ones(
    tmp_path: Path,
) -> None:
    candidate = "Every survey is checked against the house quality manual before it ships."
    shared_in_project = "The project team archives field sheets with the crew lead every evening."
    genuine = "Personal data collected on site is deleted thirty days after the final report."
    policy = (
        "## Policy\n\n"
        f"{candidate} Clients may ask for the manual at any time during the project.\n\n"
        f"{shared_in_project}\n\n"
        "Short policy line here.\n\n"
        "| Policy | Owner |\n| --- | --- |\n| Data | Studio |\n\n"
        f"{genuine}\n"
    )
    # Recurrence for candidates needs another project folder, so text shared within p1 is no
    # candidate, but a nominated unit another document holds is still not single-exemplar.
    files = {
        "p1/a.md": (
            f"# A\n\nDeliverable: memo\n\n## Quality\n\n{candidate}\n\n{shared_in_project}\n"
        ),
        "p1/b.md": f"# B\n\nDeliverable: memo\n\n## Quality\n\n{candidate}\n\n{policy}",
        "p2/c.md": f"# C\n\nDeliverable: memo\n\n## Quality\n\n{candidate}\n",
    }
    gateway = FakeGateway(house_headings=("Policy",))

    corpus = compile_files(tmp_path, files, gateway)

    (recurring,) = [r for r in corpus.candidate_records if isinstance(r, RecurringCandidateRecord)]
    assert recurring.text == candidate
    # The first unit overlaps the recurring candidate; the second is held by p1/a.md; the
    # third is short; the fourth is a table header. Only the fifth is taken.
    nominated = [r for r in corpus.candidate_records if isinstance(r, NominatedUnitRecord)]
    assert nominated == [
        NominatedUnitRecord(
            deliverable_type="memo",
            section_id="doc_002_sec_003",
            unit_ordinal=5,
            disposition="nominated_kept",
            text=genuine,
        )
    ]
    house = corpus.blueprints[0].passages[-1]
    assert house == SingleExemplarPassage(
        kind="single_exemplar",
        label="Policy",
        text=genuine,
        citations=("doc_002_sec_003",),
        heading_path="B > Policy",
    )


class HouseJudge(FakeGateway):
    """Answers the house-text check by the unit it judges: `answers` maps a unit's text to a
    probability, None for a refusal, or an exception to raise."""

    def __init__(self, answers: dict[str, float | Exception | None], **kwargs: Any) -> None:
        super().__init__(house_headings=("Data handling",), **kwargs)
        self.answers = answers

    def judge(self, operation: str, evidence: str, predicates: Sequence[str]) -> list[float | None]:
        answers = super().judge(operation, evidence, predicates)
        if operation != HOUSE_TEXT_OPERATION:
            return answers
        answer = self.answers[evidence.split("THIS TEXT:\n")[1]]
        if isinstance(answer, Exception):
            raise answer
        return [answer]


def test_the_house_text_check_judges_each_unit_in_its_whole_section_and_drops_the_rest(
    tmp_path: Path,
) -> None:
    kept, dropped, refused = (
        DATA_HANDLING,
        "Sonar files for the South channel survey were handed to the client on 20 January.",
        "Survey crews log each archive transfer in the studio register before they leave.",
    )
    section = f"## Data handling\n\n{kept}\n\n{dropped}\n\n{refused}\n"
    files = dict(DATED_REPORTS)
    files["p3-south/report.md"] = files["p3-south/report.md"].replace(
        f"## Data handling\n\n{DATA_HANDLING}\n", section
    )
    gateway = HouseJudge({kept: 0.9, dropped: 0.2, refused: None})

    corpus = compile_files(tmp_path / "judged", files, gateway)

    judged = [(e, p) for op, e, p in gateway.judged if op == HOUSE_TEXT_OPERATION]
    # One question per unit, in one window holding the document's card as binding sees it,
    # the heading path, the whole section and the unit itself; the question covers any
    # project, not only the one the document reports.
    assert len(judged) == 3
    assert "whether the one this document reports or an earlier one" in HOUSE_TEXT_PREDICATE
    for (evidence, predicates), unit in zip(judged, (kept, dropped, refused), strict=True):
        assert predicates == [HOUSE_TEXT_PREDICATE]
        assert evidence.startswith(
            "DOCUMENT TITLE: South Survey Report\nDELIVERABLE TYPE: survey report\n"
            f"DATE: 2026-01-20\nSUMMARY: {HOUSE_STANDARD}\nATTRIBUTES:\n"
            "- deliverable: survey report\n- date: 2026-01-20\n"
            "HEADING PATH: South Survey Report > Data handling\nSECTION:\n## Data handling"
        )
        assert f"{kept}\n\n{dropped}\n\n{refused}" in evidence
        assert evidence.endswith(f"THIS TEXT:\n{unit}")
    nominated = [
        (r.unit_ordinal, r.disposition)
        for r in corpus.candidate_records
        if isinstance(r, NominatedUnitRecord)
    ]
    assert nominated == [(1, "nominated_kept"), (2, "nominated_dropped"), (3, "nominated_refused")]
    assert [p.text for p in corpus.blueprints[0].passages if p.kind == "single_exemplar"] == [kept]
    report = corpus.build_report
    assert report.judge_refusals == {HOUSE_TEXT_OPERATION: 1}
    assert report.passage_dispositions == {
        "survey report": {
            "nominated_dropped": 1,
            "nominated_kept": 1,
            "nominated_refused": 1,
            "selected_conditional": 1,
            "selected_standing": 1,
        }
    }
    # A provider failure is no verdict: it ends the build.
    failing = HouseJudge({kept: ProviderError("decisions unavailable"), dropped: 0.2, refused: 0.9})
    with pytest.raises(ProviderError, match="decisions unavailable"):
        compile_files(tmp_path / "failing", files, failing)
    # The fake gateway doubts a unit it is told to: that unit is dropped, the others kept.
    doubting = FakeGateway(house_headings=("Data handling",), doubted=(dropped,))
    doubted = compile_files(tmp_path / "doubted", files, doubting)
    assert [
        (r.unit_ordinal, r.disposition)
        for r in doubted.candidate_records
        if isinstance(r, NominatedUnitRecord)
    ] == [(1, "nominated_kept"), (2, "nominated_dropped"), (3, "nominated_kept")]


def test_every_candidate_and_nominated_unit_gets_one_disposition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Both candidates recur in three documents, so the cap of one keeps the first.
    monkeypatch.setattr(induction, "MAX_STANDING_CANDIDATES", 1)
    gateway = FakeGateway(house_headings=("Data handling",))

    corpus = compile_files(tmp_path, DATED_REPORTS, gateway)

    reports = ("doc_001_sec_002", "doc_002_sec_002", "doc_003_sec_002")
    assert corpus.candidate_records == (
        RecurringCandidateRecord(
            deliverable_type="survey report",
            candidate_id="candidate_01",
            disposition="selected_standing",
            citations=reports,
            text=HOUSE_STANDARD,
        ),
        # A candidate the request never showed keeps its first holder's wording.
        RecurringCandidateRecord(
            deliverable_type="survey report",
            candidate_id="candidate_02",
            disposition="not_offered",
            citations=reports,
            text=REDUCTION.format("week"),
        ),
        NominatedUnitRecord(
            deliverable_type="survey report",
            section_id="doc_003_sec_004",
            unit_ordinal=1,
            disposition="nominated_kept",
            text=DATA_HANDLING,
        ),
    )
    assert corpus.build_report.passage_dispositions == {
        "survey report": {"nominated_kept": 1, "not_offered": 1, "selected_standing": 1}
    }


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
