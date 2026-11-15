"""Layer 1: retrieval ablation of BM25-only, dense-only and hybrid RRF ranking on one snapshot.

A case is *complete* at rank r when the top-r prefix jointly covers every required fact group
and every expected document; multi-fact cases may need several chunks before they complete.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from arkive_rag.snapshot import QueryEmbedder, RankingChannel, TopicSnapshot
from evals.schema import RetrievalCase, satisfies_groups

TOP_K = 5
ARMS: tuple[RankingChannel, ...] = ("bm25_only", "dense_only", "hybrid_rrf")


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
    ranks: dict[RankingChannel, int | None]
    top_docs: dict[RankingChannel, list[str]]


def run_layer1(
    snapshot_path: Path, embed_query: QueryEmbedder, cases: Sequence[RetrievalCase]
) -> dict[str, object]:
    results: list[CaseResult] = []
    with TopicSnapshot.open(snapshot_path) as snapshot:
        for case in cases:
            rankings = snapshot.rank_channels(
                case.query, embed_query, filters=case.filters, top_k=TOP_K
            )
            results.append(
                CaseResult(
                    case.case_id,
                    case.category,
                    {arm: first_satisfying_rank(case, rankings[arm]) for arm in ARMS},
                    {arm: [hit.doc_id for hit in rankings[arm]] for arm in ARMS},
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
