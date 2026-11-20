"""Layer 1: retrieval ablation on one snapshot.

Three arms rank raw chunk channels (`TopicSnapshot.rank_channels`): BM25 only, dense only and
their hybrid RRF fusion, with no family diversity. The fourth, `served_search`, is the ranking
agents get: `TopicSnapshot.search`, which leaves out a superseded document whose replacement
passes the case's filters and takes at most `SEARCH_FAMILY_CAP` chunks of one version family
before filling the list with held-back chunks. All four come from one query embedding per
case, made when the first ranking has validated the case.

A case is *complete* at rank r when the top-r prefix jointly covers every required fact group
and every expected document; multi-fact cases may need several chunks before they complete.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from arkive_rag.snapshot import QueryEmbedder, RankingChannel, TopicSnapshot
from evals.schema import RetrievalCase, satisfies_groups

TOP_K = 5
RetrievalArm = RankingChannel | Literal["served_search"]
ARMS: tuple[RetrievalArm, ...] = ("bm25_only", "dense_only", "hybrid_rrf", "served_search")


class RankedText(Protocol):
    @property
    def doc_id(self) -> str: ...
    @property
    def content(self) -> str: ...


def first_satisfying_rank(case: RetrievalCase, ranking: Sequence[RankedText]) -> int | None:
    """Smallest r in 1..TOP_K whose prefix ranking[:r] holds every group and expected doc."""
    for r in range(1, min(TOP_K, len(ranking)) + 1):
        prefix = ranking[:r]
        docs = {hit.doc_id for hit in prefix}
        if all(doc in docs for doc in case.expected_docs) and satisfies_groups(
            "\n".join(hit.content for hit in prefix), case.required_groups
        ):
            return r
    return None


@dataclass(frozen=True)
class RankSummary:
    cases: int
    completion_at_1: float
    completion_at_3: float
    completion_at_5: float
    completion_mrr_at_5: float


def summarize_ranks(ranks: Sequence[int | None]) -> RankSummary:
    if not ranks:
        raise ValueError("cannot summarize zero cases")

    def completion_rate(k: int) -> float:
        return sum(r is not None and r <= k for r in ranks) / len(ranks)

    mrr = sum(1.0 / r for r in ranks if r is not None) / len(ranks)
    return RankSummary(len(ranks), completion_rate(1), completion_rate(3), completion_rate(5), mrr)


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    category: str
    ranks: dict[RetrievalArm, int | None]
    top_docs: dict[RetrievalArm, list[str]]


def run_layer1(
    snapshot_path: Path, embed_query: QueryEmbedder, cases: Sequence[RetrievalCase]
) -> dict[str, object]:
    results: list[CaseResult] = []
    with TopicSnapshot.open(snapshot_path) as snapshot:
        for case in cases:
            vectors: list[NDArray[np.float32]] = []

            def embedded(
                query: str, case: RetrievalCase = case, vectors: list[NDArray[np.float32]] = vectors
            ) -> NDArray[np.float32]:
                # Both rankings embed exactly the case's query, once, on first use: each ranking
                # validates the query, filters and top_k before it embeds, so an invalid case
                # fails before any provider call.
                if query != case.query:
                    raise ValueError(f"case {case.case_id!r} embedded another query: {query!r}")
                if not vectors:
                    vectors.append(embed_query(query))
                return vectors[0]

            rankings = snapshot.rank_channels(
                case.query, embedded, filters=case.filters, top_k=TOP_K
            )
            served = snapshot.search(case.query, embedded, filters=case.filters, top_k=TOP_K)
            arms: dict[RetrievalArm, Sequence[RankedText]] = {a: h for a, h in rankings.items()}
            arms["served_search"] = served
            results.append(
                CaseResult(
                    case.case_id,
                    case.category,
                    {arm: first_satisfying_rank(case, arms[arm]) for arm in ARMS},
                    {arm: [hit.doc_id for hit in arms[arm]] for arm in ARMS},
                )
            )
    categories = sorted({r.category for r in results})
    return {
        "top_k": TOP_K,
        "overall": {arm: summarize_ranks([r.ranks[arm] for r in results]) for arm in ARMS},
        "by_category": {
            category: {
                arm: summarize_ranks([r.ranks[arm] for r in results if r.category == category])
                for arm in ARMS
            }
            for category in categories
        },
        "cases": results,
    }
