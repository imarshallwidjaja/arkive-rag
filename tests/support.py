"""Deterministic test doubles and synthetic source documents."""

from __future__ import annotations

import base64
import json
import re
import threading
import zipfile
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from xml.sax.saxutils import escape

import httpx2
import numpy as np
import openai
from numpy.typing import NDArray
from pydantic import BaseModel

from arkive_rag.compiler import (
    ConsolidateCard,
    ObserveSections,
    PageTranscription,
    compile_corpus,
    parse_markdown_blocks,
)
from arkive_rag.induction import (
    MAX_FACETS,
    CanonicaliseFacets,
    CanonicaliseTypes,
    ExtractFeedbackLessons,
    OrganiseCorpus,
    PartitionVersions,
    WriteGlossary,
    WriteGuide,
)
from arkive_rag.llm import OpenAIGateway, StructuredRequest
from arkive_rag.models import (
    CanonicalTypeDraft,
    DocumentRoleDraft,
    FacetCanonicalisationDraft,
    FacetDraft,
    FacetValueDraft,
    FeedbackLessonDraft,
    FeedbackLessonsDraft,
    GlossaryDraft,
    GlossaryEntryDraft,
    GuideDraft,
    ObservationDraft,
    OrganisationDraft,
    PageExtraction,
    ProposedVersionFamily,
    SenseDraft,
    StatedAttribute,
    TermDraft,
    TypeCanonicalisationDraft,
    VersionPartitionDraft,
    document_id,
    fold_facet_label,
    fold_for_quote,
)
from arkive_rag.snapshot import write_snapshot

TEST_BASE_URL = "https://api.openai.test/v1"
SCRIPTED_EMBEDDING_DIMENSIONS = 8
# Synthetic documents state what the fake gateway observes in plain lines: `Key: value`
# attributes quoting their whole line (`Convention:`, `Reviewer:` and `Caveat:` lines are
# writing rules, not attributes; a `Reviewer:` line is also a feedback lesson), and
# `**Term** (alias) means ...` terms.
_KEY_LINE = re.compile(r"([A-Z][A-Za-z]{1,19}): (\S.*)")
_RULE_KEYS = {"Convention", "Reviewer", "Caveat"}
_BOLD_TERM = re.compile(r"\*\*([^*]+)\*\*(?: \(([^)]+)\))?")
_QUOTED = re.compile(r'"([^"]+)"')
_FOLDER_FACET = re.compile(r"^Folders: (\w+)$", re.MULTILINE)
# Attributes that describe the document rather than group the collection.
_NOT_FACETS = {"deliverable", "date"}


def sec(rel_path: str, ordinal: int = 1) -> str:
    """The section ID the build gives section `ordinal` of the document at `rel_path`."""
    return f"{document_id(rel_path)}_sec_{ordinal:03d}"


def hash_embedding(text: str, dimensions: int) -> NDArray[np.float32]:
    """Bag-of-words vector: texts sharing words have positive cosine similarity."""
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in re.findall(r"\w+", text.lower()):
        vector[zlib.crc32(token.encode()) % dimensions] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        vector[0] = 1.0
        return vector
    return vector / norm


# A version token in a title, such as `v0.9` or `V2`.
_VERSION_TOKEN = re.compile(r"\bv(\d+(?:\.\d+)*)\b", re.IGNORECASE)
_PRINTED_DATE = re.compile(r"(\d{4}-\d{2}(?:-\d{2})?) \(printed\)")


def _version(title: str) -> tuple[int, ...] | None:
    match = _VERSION_TOKEN.search(title)
    return tuple(int(p) for p in match[1].split(".")) if match else None


def _strictly_greatest(keys: Mapping[str, Any]) -> str | None:
    """The member whose key is strictly greatest, when every member has one."""
    if None in keys.values():
        return None
    ranked = sorted(keys, key=keys.__getitem__, reverse=True)
    return ranked[0] if keys[ranked[0]] != keys[ranked[1]] else None


def partition_versions(request: PartitionVersions) -> VersionPartitionDraft:
    """Fake partition: documents are one family when their titles are equal once a version
    token is removed. The latest member has the highest version token, else the latest
    printed date; a family that neither orders has no latest member."""
    families: dict[str, list[str]] = {}
    for card in request.cards:
        stem = " ".join(_VERSION_TOKEN.sub("", card.title).casefold().split())
        families.setdefault(stem, []).append(card.doc_id)
    versions = {c.doc_id: _version(c.title) for c in request.cards}
    printed = {
        c.doc_id: (m[1] if (m := _PRINTED_DATE.fullmatch(c.date)) else None) for c in request.cards
    }
    proposed = []
    for members in families.values():
        latest = None
        if len(members) > 1:
            latest = _strictly_greatest({m: versions[m] for m in members}) or _strictly_greatest(
                {m: printed[m] for m in members}
            )
        proposed.append(ProposedVersionFamily(doc_ids=members, latest_doc_id=latest))
    return VersionPartitionDraft(families=proposed)


def canonicalise_facets(request: CanonicaliseFacets) -> FacetCanonicalisationDraft:
    """Fake facet canonicalisation: facets merge by name, each keeping the binding of its first
    proposal; values merge by folded label, uniting aliases and folders; the first
    `MAX_FACETS` names are kept."""
    merged: dict[str, FacetDraft] = {}
    for proposal in request.proposals:
        facet = proposal.facet
        prior = merged.get(facet.name)
        binding = (
            facet.binding if prior is None else ("path" if prior.values[0].folders else "content")
        )
        if binding != facet.binding:
            continue
        values = {fold_facet_label(v.value): v for v in (prior.values if prior else [])}
        for value in facet.values:
            key = fold_facet_label(value.value)
            old = values.get(key)
            values[key] = FacetValueDraft(
                value=old.value if old else value.value,
                description=old.description if old else value.description,
                aliases=sorted({*(old.aliases if old else []), *value.aliases}),
                folders=sorted({*(old.folders if old else []), *value.folders}),
            )
        merged[facet.name] = FacetDraft(
            name=facet.name,
            description=prior.description if prior else facet.description,
            values=list(values.values()),
        )
    return FacetCanonicalisationDraft(facets=list(merged.values())[:MAX_FACETS])


class FakeGateway:
    """A `LanguageGateway` that answers every compiler request type deterministically.

    It reads typed request fields and source text, never rendered prompts. Transcription echoes
    the native text layer as Markdown blocks. Observation reads the synthetic line conventions
    above. Organisation makes one path-bound facet from the top-level folders (named by a
    `Folders: <name>` line in a root curator note, else `project`), one content-bound facet per
    attribute with two or more values, gives each document its observed type, and makes a
    document whose type says "review" feedback for the batch's other types. Canonicalisation
    keeps every label as its own type. Facet canonicalisation is `canonicalise_facets` and
    version partition `partition_versions`. Feedback lessons are the `Reviewer:` lines.
    `judge` answers 0.9 when every double-quoted span of a predicate occurs in the evidence,
    else 0.1.

    Each `*_script` queues responses consumed before the deterministic answer. Every request
    is kept in `requests`, its operation name in `calls`, and judge questions in `judged`.
    """

    embedding_model = "fake-hash-embedding"
    embedding_dimensions = 64

    def __init__(
        self,
        *,
        page_script: Sequence[PageExtraction] = (),
        observe_script: Sequence[ObservationDraft] = (),
        organise_script: Sequence[OrganisationDraft] = (),
        types_script: Sequence[TypeCanonicalisationDraft] = (),
        facets_script: Sequence[FacetCanonicalisationDraft] = (),
        versions_script: Sequence[VersionPartitionDraft] = (),
        lesson_script: Sequence[FeedbackLessonsDraft] = (),
        glossary_script: Sequence[GlossaryDraft] = (),
        guide_script: Sequence[str] = (),
    ) -> None:
        self.page_script = list(page_script)
        self.observe_script = list(observe_script)
        self.organise_script = list(organise_script)
        self.types_script = list(types_script)
        self.facets_script = list(facets_script)
        self.versions_script = list(versions_script)
        self.lesson_script = list(lesson_script)
        self.glossary_script = list(glossary_script)
        self.guide_script = list(guide_script)
        self.requests: list[object] = []
        self.calls: list[str] = []
        self.judged: list[tuple[str, str, list[str]]] = []
        self._lock = threading.Lock()

    @property
    def transcribe_calls(self) -> list[tuple[str, tuple[str, ...]]]:
        return [
            (r.effort, r.missing_values) for r in self.requests if isinstance(r, PageTranscription)
        ]

    def of_type[R](self, kind: type[R]) -> list[R]:
        return [r for r in self.requests if isinstance(r, kind)]

    def describe(self) -> dict[str, str]:
        return {"model": "fake-model", "embedding_model": self.embedding_model}

    def _record(self, request: object, operation: str) -> None:
        with self._lock:
            self.requests.append(request)
            self.calls.append(operation)

    def _pop[R](self, script: list[R]) -> R | None:
        with self._lock:
            return script.pop(0) if script else None

    def run[T: BaseModel](self, request: StructuredRequest[T]) -> T:
        self._record(request, request.operation)
        return cast(T, self.answer(request))

    def answer(self, request: object) -> BaseModel:
        match request:
            case PageTranscription(native_text=native_text, image_webp=webp):
                assert webp[:4] == b"RIFF", "pages must be rendered as WebP"
                return self._pop(self.page_script) or parse_markdown_blocks(native_text)
            case ObserveSections():
                return self._pop(self.observe_script) or self.observe(request)
            case ConsolidateCard(headers=headers):
                return headers[0]
            case OrganiseCorpus():
                return self._pop(self.organise_script) or self.organise(request)
            case CanonicaliseTypes(labels=labels):
                scripted = self._pop(self.types_script)
                return scripted or TypeCanonicalisationDraft(
                    types=[
                        CanonicalTypeDraft(
                            name=t.label, description=f"A {t.label}.", labels=[t.label]
                        )
                        for t in labels
                    ]
                )
            case CanonicaliseFacets():
                return self._pop(self.facets_script) or canonicalise_facets(request)
            case PartitionVersions():
                return self._pop(self.versions_script) or partition_versions(request)
            case ExtractFeedbackLessons(sections=sections):
                scripted_lessons = self._pop(self.lesson_script)
                return scripted_lessons or FeedbackLessonsDraft(
                    lessons=[
                        FeedbackLessonDraft(statement=line.removeprefix("Reviewer: "), quote=line)
                        for section in sections
                        for line in section.markdown.splitlines()
                        if line.startswith("Reviewer: ")
                    ]
                )
            case WriteGlossary():
                return self._pop(self.glossary_script) or self.glossary(request)
            case WriteGuide(topic=topic, catalog=catalog):
                markdown = self._pop(self.guide_script)
                if markdown is None:
                    cited = " ".join(f"[{e.doc_id}:{e.first_section_id}]" for e in catalog[:2])
                    markdown = f"# {topic}\n\nThe corpus covers several documents {cited}."
                return GuideDraft(guide_markdown=markdown)
        raise AssertionError(f"unexpected request {request!r}")

    def observe(self, request: ObserveSections) -> ObservationDraft:
        attributes: list[StatedAttribute] = []
        terms: list[TermDraft] = []
        prose: list[str] = []
        for section in request.sections:
            for line in section.markdown.splitlines():
                key = _KEY_LINE.fullmatch(line.strip())
                if key:
                    if key[1] not in _RULE_KEYS:
                        attributes.append(
                            StatedAttribute(name=key[1].lower(), value=key[2], quote=line.strip())
                        )
                elif line.strip() and not line.startswith(("#", "|", ">")):
                    prose.append(line.strip().replace("**", ""))
                for match in _BOLD_TERM.finditer(line):
                    rest = line[match.end() :]
                    terms.append(
                        TermDraft(
                            term=match[1],
                            kind="acronym" if match[1].isupper() else "concept",
                            aliases=[match[2]] if match[2] else [],
                            quote=line.strip().replace("**", ""),
                            defines=rest.startswith((" means ", " is ")),
                        )
                    )
        stated = {a.name: a.value for a in attributes}
        title = request.sections[0].heading_path.split(" > ")[0]
        return ObservationDraft(
            title=title,
            doc_date=stated.get("date"),
            deliverable_type=stated.get("deliverable", "document"),
            summary=prose[0] if prose else f"About {title}.",
            attributes=attributes,
            terms=terms,
        )

    def organise(self, request: OrganiseCorpus) -> OrganisationDraft:
        facets: list[FacetDraft] = []
        top = sorted({f for f in request.folders if "/" not in f})
        named = [m[1] for m in [_FOLDER_FACET.search(request.notes)] if m]
        if len(top) >= 2:
            name = named[0] if named else "project"
            values = [
                FacetValueDraft(value=f, description=f"The {f} folder.", aliases=[], folders=[f])
                for f in top
            ]
            facets.append(FacetDraft(name=name, description=f"The {name} folder.", values=values))
        stated: dict[str, set[str]] = {}
        for digest in request.digests:
            for attribute in digest.attributes:
                stated.setdefault(attribute.name, set()).add(attribute.value)
        for name in sorted(stated):
            if name in _NOT_FACETS or len(stated[name]) < 2:
                continue
            firsts = [v.split()[0] for v in sorted(stated[name])]
            values = [
                FacetValueDraft(
                    value=v,
                    description=f"Documents whose {name} is {v}.",
                    aliases=[v.split()[0]] if " " in v and firsts.count(v.split()[0]) == 1 else [],
                    folders=[],
                )
                for v in sorted(stated[name])
            ]
            facets.append(
                FacetDraft(name=name, description=f"The {name} a document states.", values=values)
            )
        reviewed = sorted(
            {d.observed_type for d in request.digests if "review" not in d.observed_type}
        )
        documents = [
            DocumentRoleDraft(
                doc_id=d.doc_id,
                deliverable_type=d.observed_type,
                is_feedback="review" in d.observed_type,
                feedback_for_types=(reviewed or [d.observed_type])
                if "review" in d.observed_type
                else [],
            )
            for d in request.digests
        ]
        return OrganisationDraft(facets=facets, documents=documents)

    def glossary(self, request: WriteGlossary) -> GlossaryDraft:
        entries: list[GlossaryEntryDraft] = []
        for cluster in request.clusters:
            defining = {q.text: q.section_id for q in cluster.quotes if q.defines}
            if defining:
                senses = [SenseDraft(definition=t, citations=[s]) for t, s in defining.items()]
            else:
                seen = [q.section_id for q in cluster.quotes]
                seen += [o.section_id for o in cluster.occurrences]
                senses = [SenseDraft(definition=None, citations=list(dict.fromkeys(seen))[:3])]
            entries.append(
                GlossaryEntryDraft(
                    term=cluster.term,
                    kind=cluster.kind,
                    aliases=list(cluster.aliases),
                    senses=senses,
                )
            )
        return GlossaryDraft(entries=entries)

    def judge(self, operation: str, evidence: str, predicates: Sequence[str]) -> list[float | None]:
        with self._lock:
            self.calls.append(operation)
            self.judged.append((operation, evidence, list(predicates)))
        haystack = fold_for_quote(evidence)
        return [
            0.9
            if all(fold_for_quote(span) in haystack for span in _QUOTED.findall(predicate))
            else 0.1
            for predicate in predicates
        ]

    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self._record(list(texts), "embed_texts")
        rows = [hash_embedding(text, self.embedding_dimensions) for text in texts]
        return np.vstack(rows) if rows else np.zeros((0, self.embedding_dimensions), np.float32)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        return hash_embedding(query, self.embedding_dimensions)


# Scripted OpenAI HTTP API: the pinned SDK clients with their real request and response
# handling, served by a local transport.


def usage(input_tokens: int = 100, output_tokens: int = 10, cached: int = 0) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }


def message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def function_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def response_body(
    output: Sequence[dict[str, Any]],
    status: str = "completed",
    reason: str | None = None,
    tokens: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A Responses API body; `tokens=None` gives the default usage of 100 in / 10 out."""
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-6.1-sol",
        "status": status,
        "incomplete_details": {"reason": reason} if reason else None,
        "output": list(output),
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": tokens if tokens is not None else usage(),
    }


def structured_body(output: BaseModel, **kwargs: Any) -> dict[str, Any]:
    return response_body([message(output.model_dump_json())], **kwargs)


def refusal_body(text: str = "I can't help with that.") -> dict[str, Any]:
    """A completed response whose message refuses instead of giving structured output."""
    refusal = message("") | {"content": [{"type": "refusal", "refusal": text}]}
    return response_body([refusal])


def decision_body(answers: Sequence[float | None], input_tokens: int = 50) -> dict[str, Any]:
    return {
        "answers": [
            {"type": "refusal", "name": None}
            if p is None
            else {"type": "predicate", "name": None, "probability": p}
            for p in answers
        ],
        "model": "gpt-6-luna",
        "usage": usage(input_tokens, 1),
    }


_CATALOG_CITATION = re.compile(r"\[doc_[0-9a-f]{10}:doc_[0-9a-f]{10}_sec_\d{3}\]")


def _default_structured(schema_name: str, body: dict[str, Any]) -> BaseModel:
    if schema_name == "ObservationDraft":
        return ObservationDraft(
            title="Memo",
            doc_date="2025-10",
            deliverable_type="memorandum",
            summary="Bathymetry memo.",
            attributes=[],
            terms=[],
        )
    text = body["input"][0]["content"][0]["text"]
    if schema_name == "OrganisationDraft":
        # Every document of the batch, from its digest line, as a memorandum.
        doc_ids = re.findall(r"^(doc_[0-9a-f]{10}) \|", text, re.MULTILINE)
        roles = [
            DocumentRoleDraft(
                doc_id=d, deliverable_type="memorandum", is_feedback=False, feedback_for_types=[]
            )
            for d in doc_ids
        ]
        return OrganisationDraft(facets=[], documents=roles)
    if schema_name == "TypeCanonicalisationDraft":
        labels = re.findall(r"^- (.+?) \(", text, re.MULTILINE)
        return TypeCanonicalisationDraft(
            types=[CanonicalTypeDraft(name=t, description=t, labels=[t]) for t in labels]
        )
    if schema_name == "GuideDraft":
        # The first citation token the request's catalog lists.
        cited = _CATALOG_CITATION.search(json.dumps(body))
        return GuideDraft(guide_markdown=f"Overview {cited[0] if cited else ''}.")
    raise AssertionError(f"no default response for schema {schema_name!r}; script one")


@dataclass
class Exchange:
    path: str
    body: dict[str, Any]
    read_timeout: float | None


Outcome = dict[str, Any] | httpx2.Response | Exception


class ScriptedOpenAI:
    """Serves `/responses`, `/decisions` and `/embeddings` to real SDK clients from scripts.

    A scripted outcome is a JSON body (status 200), an `httpx2.Response`, or a transport
    exception to raise (`httpx2.ConnectError` reaches the SDK as `APIConnectionError`). When
    a path's script is empty, a structured request gets `defaults[schema name]`, a model or a
    function of the request body (agent runs ask for the schema name "final_output"), else
    a valid default for observation, organisation and guide requests; decisions answer 0.75
    to every question and embeddings return positive vectors.
    """

    def __init__(self) -> None:
        self.defaults: dict[str, BaseModel | Callable[[dict[str, Any]], BaseModel]] = {}
        self._lock = threading.Lock()
        self.scripts: dict[str, list[Outcome]] = {
            "/responses": [],
            "/decisions": [],
            "/embeddings": [],
        }
        self.exchanges: list[Exchange] = []

    def script(self, path: str, *outcomes: Outcome) -> None:
        self.scripts[path] += outcomes

    def requests(self, path: str) -> list[dict[str, Any]]:
        return [e.body for e in self.exchanges if e.path == path]

    @property
    def request_count(self) -> int:
        return len(self.exchanges)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path.removeprefix("/v1")
        body = json.loads(request.content)
        timeout = request.extensions.get("timeout", {}).get("read")
        with self._lock:
            self.exchanges.append(Exchange(path, body, timeout))
            script = self.scripts[path]
            outcome = script.pop(0) if script else self._default(path, body)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, httpx2.Response):
            return outcome
        return httpx2.Response(200, json=outcome)

    def _default(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if path == "/responses":
            name = body["text"]["format"]["name"]
            default = self.defaults.get(name) or _default_structured(name, body)
            return structured_body(default if isinstance(default, BaseModel) else default(body))
        if path == "/decisions":
            return decision_body([0.75] * len(body["questions"]))
        rows = [
            {
                "object": "embedding",
                "index": i,
                "embedding": base64.b64encode(
                    np.full(SCRIPTED_EMBEDDING_DIMENSIONS, i + 1, dtype=np.float32).tobytes()
                ).decode(),
            }
            for i in reversed(range(len(body["input"])))
        ]
        tokens = {"prompt_tokens": len(body["input"]), "total_tokens": len(body["input"])}
        return {"object": "list", "data": rows, "model": body["model"], "usage": tokens}

    def client(self) -> openai.OpenAI:
        return openai.OpenAI(
            api_key="test-key",
            base_url=TEST_BASE_URL,
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(self._handle)),
        )

    def async_client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key="test-key",
            base_url=TEST_BASE_URL,
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self._handle)),
        )

    def gateway(self, **kwargs: Any) -> OpenAIGateway:
        options: dict[str, Any] = {
            "embedding_dimensions": SCRIPTED_EMBEDDING_DIMENSIONS,
            "retry_delays": (0.0, 0.0, 0.0, 0.0),
        }
        return OpenAIGateway(
            client=self.client(), async_client_factory=self.async_client, **(options | kwargs)
        )


def write_pdf(
    path: Path, pages: Sequence[Sequence[str]], info: Mapping[str, str] | None = None
) -> Path:
    """Write a minimal Helvetica PDF with one text line per entry, and an info dictionary
    (such as `{"CreationDate": "D:20250630164500Z"}`) when `info` is given."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b""]
    page_ids: list[int] = []
    font_id = 3
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for lines in pages:
        text = "".join(
            f"({line.replace(chr(92), r'\\').replace('(', r'\(').replace(')', r'\)')}) Tj T* "
            for line in lines
        )
        stream = f"BT /F1 11 Tf 14 TL 60 780 Td {text}ET".encode("latin-1")
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        content_id = len(objects)
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents %d 0 R"
            b" /Resources << /Font << /F1 %d 0 R >> >> >>" % (content_id, font_id)
        )
        page_ids.append(len(objects))
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()
    info_ref = b""
    if info:
        entries = " ".join(f"/{key} ({value})" for key, value in info.items())
        objects.append(f"<< {entries} >>".encode("latin-1"))
        info_ref = b" /Info %d 0 R" % len(objects)
    body = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body += b"%d 0 obj\n%s\nendobj\n" % (number, obj)
    xref = len(body)
    body += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    body += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    body += b"trailer\n<< /Size %d /Root 1 0 R%s >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        info_ref,
        xref,
    )
    path.write_bytes(bytes(body))
    return path


def write_docx(
    path: Path, paragraphs: Sequence[str], core_dates: Mapping[str, str] | None = None
) -> Path:
    """Write a minimal WordprocessingML document with one paragraph per entry, and a
    `docProps/core.xml` holding `core_dates` (`{"modified": ..., "created": ...}` W3CDTF
    values) when given."""
    body = "".join(f"<w:p><w:r><w:t>{escape(text)}</w:t></w:r></w:p>" for text in paragraphs)
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?><Types xmlns='
            '"http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType='
            '"application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns='
            '"http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{rel}" Target="word/document.xml"/></Relationships>',
        )
        archive.writestr(
            "word/document.xml",
            f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{ns}">'
            f"<w:body>{body}</w:body></w:document>",
        )
        if core_dates is not None:
            dates = "".join(
                f'<dcterms:{name} xsi:type="dcterms:W3CDTF">{value}</dcterms:{name}>'
                for name, value in core_dates.items()
            )
            archive.writestr(
                "docProps/core.xml",
                '<?xml version="1.0" encoding="UTF-8"?><cp:coreProperties xmlns:cp='
                '"http://schemas.openxmlformats.org/package/2006/metadata/core-properties"'
                ' xmlns:dcterms="http://purl.org/dc/terms/"'
                ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
                f"{dates}</cp:coreProperties>",
            )
    return path


def write_flat_presentation(path: Path, slides: Sequence[Sequence[str]]) -> Path:
    """Write a flat OpenDocument presentation (.fodp) that LibreOffice can turn into .pptx."""
    pages = "".join(
        f'<draw:page draw:name="s{i}"><draw:frame svg:x="1cm" svg:y="1cm" svg:width="24cm"'
        ' svg:height="14cm"><draw:text-box>'
        + "".join(f"<text:p>{escape(line)}</text:p>" for line in lines)
        + "</draw:text-box></draw:frame></draw:page>"
        for i, lines in enumerate(slides, start=1)
    )
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<office:document xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"'
        ' xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"'
        ' xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"'
        ' xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"'
        ' office:version="1.3" office:mimetype="application/vnd.oasis.opendocument.presentation">'
        f"<office:body><office:presentation>{pages}</office:presentation></office:body>"
        "</office:document>",
        encoding="utf-8",
    )
    return path


BATHYMETRY_CORPUS = {
    "p101-harbour-lidar/harbour-memo.md": (
        "# Rainfall\n\nThe chart shows the Nine Year Median = 330.9 mm reference line for"
        " 2016\u20132017. Sentinel-2 imagery was not used; airborne lidar was.\n\n"
        "## Statistics\n\nThe p-value: 0.05 threshold applied. Area covered 59,494 km\u00b2"
        " of the harbour approaches.\n\n| Layer | File |\n| --- | --- |\n| Dry season |"
        " harbour_dry_depth_thresh.tif |\n\n# Bathymetry\n\nSoundings were reduced with"
        " formula C12-SUM(C13:C15) in the workbook. **LAT** (lowest astronomical tide) is the"
        " vertical datum of every sounding.\n\nConvention: Report depths in metres below the"
        " stated datum.\n\nDeliverable: survey memorandum\nClient: Northwind Marine\n"
    ),
    "p102-estuary-multibeam/estuary-proposal.md": (
        "# Scope\n\nEstuary multibeam bathymetry using Sentinel-2 and tide gauge depth.\n\n"
        "# Budget\n\nTotal fee 120,000 AUD. Soundings will be reduced to **LAT**.\n\n"
        "Deliverable: proposal\nClient: Contoso Ports\n"
    ),
    "p103-reef-sonar/AGENTS.md": (
        "# Status\n\nThe reef Sentinel-2 case study has not been written yet.\n"
    ),
}

HARBOUR_MEMO = document_id("p101-harbour-lidar/harbour-memo.md")
ESTUARY_PROPOSAL = document_id("p102-estuary-multibeam/estuary-proposal.md")
REEF_NOTE = document_id("p103-reef-sonar/AGENTS.md")

# A standing section every campaign brief ends with, whatever the campaign.
HOUSE_DISCLOSURE = (
    "## Disclosure\n\nEvery sponsored post carries the label *Paid partnership* in its first"
    " line, and the studio keeps each signed creator agreement for two years.\n"
)

# A campaign library organised by campaign folder, with authors and channels stated in the
# documents: a different organisation from the project folders above.
MARKETING_CORPUS = {
    "AGENTS.md": (
        "# Campaign library\n\nFolders: campaign\n\nEach folder holds one campaign's brief"
        " and its review notes. The autumn recap deck is still missing.\n"
    ),
    "loyalty-relaunch/brief.md": (
        "# Loyalty Relaunch Brief\n\nDeliverable: campaign brief\nChannel: Email\n"
        "Author: Dana Reyes\n\n## Audience\n\nLapsed members. **Churn** is the share of"
        " members who stop buying within a year.\n\n## Messaging\n\n"
        "Convention: Lead with the customer benefit before the product name.\n\n"
        "**CTA** (call to action) is the button that closes every email.\n\n"
        f"{HOUSE_DISCLOSURE}"
    ),
    "spring-launch/brief.md": (
        "# Spring Launch Brief\n\nDeliverable: campaign brief\nChannel: Social\n"
        "Author: Dana Reyes\n\n## Audience\n\nFirst-time buyers aged 25 to 34.\n\n"
        "## Messaging\n\nConvention: Lead with the customer benefit before the product name."
        "\n\n**CTA** (call to action) means the single step a reader should take next.\n\n"
        "Write every line for the **ICP**.\n\n"
        f"{HOUSE_DISCLOSURE}"
    ),
    "spring-launch/review.md": (
        "# Spring Brief Review\n\nDeliverable: review notes\nChannel: Social\n"
        "Author: Sam Okafor\n\n## Comments\n\nReviewer: Cut the jargon from the opening line."
        "\n\nThe **CTA** was buried in paragraph three. The **ICP** was never named.\n"
    ),
}


CAMPAIGN_NOTE = document_id("AGENTS.md")
LOYALTY_BRIEF = document_id("loyalty-relaunch/brief.md")
SPRING_BRIEF = document_id("spring-launch/brief.md")
SPRING_REVIEW = document_id("spring-launch/review.md")


def compile_snapshot(
    root: Path,
    files: dict[str, str],
    *,
    topic: str = "bathymetry",
    pdfs: dict[str, Sequence[Sequence[str]]] | None = None,
    gateway: FakeGateway | None = None,
) -> Path:
    """Compile Markdown (and optional synthetic PDF) sources into `root/<topic>.arkive.db`."""
    corpus_dir = root / f"{topic}-corpus"
    for rel_path, text in files.items():
        (corpus_dir / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (corpus_dir / rel_path).write_text(text, encoding="utf-8")
    for rel_path, pages in (pdfs or {}).items():
        (corpus_dir / rel_path).parent.mkdir(parents=True, exist_ok=True)
        write_pdf(corpus_dir / rel_path, pages)
    corpus = compile_corpus(corpus_dir, gateway or FakeGateway(), topic=topic, title=topic.upper())
    output = root / f"{topic}.arkive.db"
    write_snapshot(corpus, output)
    return output
