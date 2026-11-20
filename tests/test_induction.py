from __future__ import annotations

import itertools
import re
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
    CorpusDocument,
    CraftObservation,
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
    StandingPassageSelection,
    TermObservation,
    document_id,
)
from arkive_rag.passages import standing_candidates
from arkive_rag.snapshot import TopicSnapshot, write_snapshot
from tests.support import (
    BATHYMETRY_CORPUS,
    CAMPAIGN_NOTE,
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
    memo, note = document_id("p1/memo.md"), document_id("p1/AGENTS.md")
    valid = f"[{memo}:{sec('p1/memo.md')}]"
    gateway = FakeGateway(guide_script=[f"See [{memo}:bad_sec].", f"See {valid}."])

    corpus = compile_files(tmp_path / "ok", files, gateway)

    assert corpus.guide_markdown == f"See {valid}."
    first, rerun = gateway.of_type(WriteGuide)
    assert first.feedback is None and rerun.feedback is not None
    assert f"[{memo}:bad_sec]" in rerun.feedback and valid in rerun.feedback
    # The curator note is not citable.
    assert sec("p1/AGENTS.md") not in rerun.feedback.split("cite only")[1]
    note_token = f"[{note}:{sec('p1/AGENTS.md')}]"
    stubborn = FakeGateway(guide_script=["No citations.", f"Still {note_token}."])
    with pytest.raises(RejectedOutputError, match=re.escape(note_token)):
        compile_files(tmp_path / "bad", files, stubborn)


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
    documents = tuple(
        CorpusDocument(document_id(path), path, reading(["# A\n\nText."], ()))
        for path in (
            "p101-harbour-lidar/harbour-memo.md",
            "p102-estuary-multibeam/estuary-proposal.md",
        )
    )

    assert bool(induction.invalid_citations(markdown, CorpusView(documents))) is bad


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
            exemplar_doc_ids=[HARBOUR_MEMO, ESTUARY_PROPOSAL],
            feedback_doc_ids=[ESTUARY_PROPOSAL],
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
    assert f"'proposal': '{ESTUARY_PROPOSAL}' is both an exemplar and a feedback" in message
    first, rerun = gateway.of_type(OrganiseCorpus)
    assert first.feedback is None and rerun.feedback is not None and "doc_009" in rerun.feedback
    assert f"'{ESTUARY_PROPOSAL}' is both an exemplar and a feedback document" in rerun.feedback


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
    draft = organisation(FacetDraft(name="client", description="Client.", values=values))

    checked = check_organisation(draft, frozenset(), {})

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
                exemplar_doc_ids=[HARBOUR_MEMO],
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


LAT_CLUSTER_SECTIONS = (f"{HARBOUR_MEMO}_sec_003", f"{ESTUARY_PROPOSAL}_sec_002")


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


def reading(
    markdowns: Sequence[str],
    terms: Sequence[TermObservation],
    craft: Sequence[CraftObservation] = (),
) -> DocumentReading:
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
            craft=tuple(craft),
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


def test_induction_orders_documents_and_sections_by_path_not_by_document_id() -> None:
    paths = ("p1/a.md", "p2/b.md", "p3/c.md")
    ids = [document_id(p) for p in paths]
    assert sorted(ids) == ids[::-1], "the paths' IDs must sort against path order"
    term = TermObservation(
        term="LAT", kind="acronym", aliases=(), quote="LAT", defines=False, section_ordinal=1
    )
    craft = CraftObservation(aspect="tone", statement="Plain.", quote="LAT", section_ordinal=1)
    view = CorpusView(
        tuple(
            CorpusDocument(document_id(p), p, reading(["LAT is the datum."], [term], [craft]))
            for p in paths
        )
    )
    sections = [sec(p) for p in paths]

    (cluster,) = term_clusters(view)
    assert [q.section_id for q in cluster.quotes] == sections
    assert [lead.section_id for lead in induction.craft_leads(view, ids[::-1])] == sections
    assert [d.doc_id for d in exemplar_documents(view, ids[::-1])] == ids
    draft = organisation(
        types=[
            DeliverableTypeDraft(
                name="memo",
                description="Memo.",
                exemplar_doc_ids=[ids[2], ids[0], ids[2]],
                feedback_doc_ids=[ids[1]],
            )
        ]
    )
    accepted = check_organisation(draft, frozenset(), view.citable_order).result
    assert accepted is not None
    (memo,) = accepted.deliverable_types
    assert (memo.exemplar_doc_ids, memo.feedback_doc_ids) == ((ids[0], ids[2]), (ids[1],))


def test_definitions_are_judged_over_complete_sections_in_windows() -> None:
    filler = "\n\n".join(
        " ".join(f"Line {p}.{i} was surveyed twice." for i in range(30)) for p in range(140)
    )
    view = candidate_view(
        ("p1/memo.md", [f"# Datum\n\n{filler}\n\n**LAT** means the lowest astronomical tide."])
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


# Blueprints.


def test_blueprints_cite_resolvable_sections_and_exemplars(tmp_path: Path) -> None:
    gateway = FakeGateway()

    corpus = compile_files(tmp_path, MARKETING_CORPUS, gateway)

    brief, review = corpus.blueprints
    assert (brief.deliverable_type, review.deliverable_type) == ("campaign brief", "review notes")
    assert brief.exemplar_doc_ids == (LOYALTY_BRIEF, SPRING_BRIEF)
    assert brief.feedback_doc_ids == (SPRING_REVIEW,)
    sections = {s.section_id for d in corpus.documents for s in d.sections}
    cited = [c for s in brief.outline for c in s.citations]
    cited += [c for g in (*brief.conventions, *brief.pitfalls) for c in g.citations]
    assert cited and set(cited) <= sections
    assert [g.statement for g in brief.conventions] == [
        "Lead with the customer benefit before the product name."
    ]
    assert brief.conventions[0].citations == (f"{LOYALTY_BRIEF}_sec_003", f"{SPRING_BRIEF}_sec_003")
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
    assert [(c.candidate_id, c.citations) for c in request.standing_candidates] == [
        ("candidate_01", (f"{LOYALTY_BRIEF}_sec_003", f"{SPRING_BRIEF}_sec_003")),
        ("candidate_02", (f"{LOYALTY_BRIEF}_sec_004", f"{SPRING_BRIEF}_sec_004")),
    ]
    (disclosure,) = brief.standing_passages
    assert disclosure.label == "Disclosure"
    assert disclosure.text == HOUSE_DISCLOSURE.split("\n\n")[1].strip()
    assert disclosure.citations == (f"{LOYALTY_BRIEF}_sec_004", f"{SPRING_BRIEF}_sec_004")
    assert (
        f"[candidate_02] found in {LOYALTY_BRIEF}_sec_004, {SPRING_BRIEF}_sec_004" in request.text()
    )
    assert review.standing_passages == () and review_request.standing_candidates == ()
    assert "(none: this type has one exemplar document" in review_request.text()
    assert corpus.build_report.dropped_blueprint_statements == 0
    # The agent run used its tools; its cache identity covers everything they can read.
    # Craft observations of the exemplars and the feedback document are leads to verify.
    assert request.craft_leads == (
        CraftLead(
            f"{LOYALTY_BRIEF}_sec_003",
            "convention",
            "Lead with the customer benefit before the product name.",
        ),
        CraftLead(
            f"{SPRING_BRIEF}_sec_003",
            "convention",
            "Lead with the customer benefit before the product name.",
        ),
        CraftLead(
            f"{SPRING_REVIEW}_sec_002", "reviewer_feedback", "Cut the jargon from the opening line."
        ),
    )
    assert f"[{SPRING_REVIEW}_sec_002] reviewer_feedback: Cut the jargon" in request.text()
    assert "leads, not evidence" in WriteBlueprint.instructions
    assert {t.name for t in request.tools} == {"outline", "read_section", "grep"}
    assert len(request.source_digest) == 64


def test_blueprint_tools_cannot_read_curator_notes_and_their_revision_is_in_the_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway = FakeGateway()
    corpus = compile_files(tmp_path, MARKETING_CORPUS, gateway)
    assert corpus.documents[0].is_curator_note and corpus.documents[0].doc_id == CAMPAIGN_NOTE
    request = gateway.of_type(WriteBlueprint)[0]
    tools = {t.name: t for t in request.tools}

    with pytest.raises(NotFoundError, match=CAMPAIGN_NOTE):
        tools["outline"].invoke({"doc_id": CAMPAIGN_NOTE})
    with pytest.raises(NotFoundError, match=f"{CAMPAIGN_NOTE}_sec_001"):
        tools["read_section"].invoke({"section_id": f"{CAMPAIGN_NOTE}_sec_001"})
    assert tools["grep"].invoke({"pattern": "autumn recap"}) == "[]"
    with pytest.raises(ValueError, match="blank"):
        tools["grep"].invoke({"pattern": " "})
    for name, args in (
        ("grep", {"pattern": 3}),
        ("outline", {"doc_id": [LOYALTY_BRIEF]}),
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
    # Only the unsupported convention and pitfall count; standing passages are never dropped.
    assert brief.standing_passages
    assert corpus.build_report.dropped_blueprint_statements == 2


def test_a_blueprint_with_an_empty_outline_fails_after_one_rerun(tmp_path: Path) -> None:
    empty = BlueprintDraft(
        description="Nothing.",
        outline=[],
        conventions=[],
        pitfalls=[],
        standing_passages=[],
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
    """Documents in argument order, which must be path order; each section's Markdown as
    given."""
    return CorpusView(
        tuple(
            CorpusDocument(document_id(rel_path), rel_path, reading(markdowns, ()))
            for rel_path, markdowns in docs
        )
    )


def test_the_blueprint_request_says_why_it_lists_no_candidates() -> None:
    def request(*exemplars: str) -> str:
        dtype = DeliverableType("memo", "A memo.", exemplars, ())
        return WriteBlueprint(dtype, (), (), (), (), 0, (), "0" * 64).text()

    assert "(none: this type has one exemplar document" in request("doc_001")
    assert "(none: no block of this type's exemplars recurs" in request("doc_001", "doc_002")


def test_candidates_beyond_the_cap_keep_those_in_the_most_documents_and_the_request_says_so() -> (
    None
):
    assert induction.MAX_STANDING_CANDIDATES == 60
    # Sixty-one distinct sentences recur between a.md and b.md; the last two also recur in
    # c.md, so they outrank the rest, and the tie among the others keeps canonical order.
    sentences = [
        f"Standing clause number {n} of the house terms binds every survey we carry out."
        for n in range(1, 62)
    ]
    view = candidate_view(
        ("p1/a.md", sentences), ("p2/b.md", sentences), ("p3/c.md", sentences[-2:])
    )
    exemplars = tuple(document_id(p) for p in ("p1/a.md", "p2/b.md", "p3/c.md"))
    found = standing_candidates(exemplar_documents(view, exemplars))

    offered = induction.offered_candidates(view, found)

    assert len(found) == 61
    assert [c.candidate_id for c in offered] == [
        f"candidate_{n:02d}" for n in (*range(1, 59), 60, 61)
    ]
    dtype = DeliverableType("terms", "Terms.", exemplars, ())
    cut = WriteBlueprint(dtype, (), (), (), offered, len(found), (), "0" * 64).text()
    assert "(this list was cut from 61 candidates to the 60 that recur in the most" in cut
    whole = WriteBlueprint(dtype, (), (), (), found[:60], 60, (), "0" * 64).text()
    assert "was cut" not in whole


def selecting(*selections: tuple[str, str]) -> BlueprintDraft:
    return BlueprintDraft(
        description="A brief.",
        outline=[
            BlueprintSectionDraft(
                heading="Audience", purpose="Who.", citations=[f"{LOYALTY_BRIEF}_sec_002"]
            )
        ],
        conventions=[],
        pitfalls=[],
        standing_passages=[
            StandingPassageSelection(candidate_id=i, label=label) for i, label in selections
        ],
    )


def test_unknown_or_repeated_standing_passage_selections_are_rerun_then_rejected(
    tmp_path: Path,
) -> None:
    bad = selecting(
        ("candidate_09", "Made up"),
        ("candidate_02", "A"),
        ("candidate_02", "B"),
        ("candidate_01", "  "),
    )
    good = selecting(("candidate_02", "Disclosure"))
    gateway = FakeGateway(blueprint_script=[bad, good])

    corpus = compile_files(tmp_path / "rerun", MARKETING_CORPUS, gateway)

    first, rerun = gateway.of_type(WriteBlueprint)[:2]
    assert first.feedback is None and rerun.feedback is not None
    assert "'candidate_09' is not a listed candidate_id" in rerun.feedback
    assert "'candidate_02' is selected twice" in rerun.feedback
    assert "'candidate_01' has a blank label" in rerun.feedback
    (passage,) = corpus.blueprints[0].standing_passages
    assert passage.label == "Disclosure"
    assert passage.text == HOUSE_DISCLOSURE.split("\n\n")[1].strip()
    stubborn = FakeGateway(blueprint_script=[bad, bad])
    with pytest.raises(RejectedOutputError, match="'candidate_09' is not a listed candidate_id"):
        compile_files(tmp_path / "stubborn", MARKETING_CORPUS, stubborn)


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
