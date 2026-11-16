"""Evaluation harness for Arkive snapshots: retrieval, grounded QA and holdout drafting.

The harness runs an eval pack (`schema.py`) against a snapshot and its corpus. Packs with real
corpus facts stay outside this repository; it is not part of the `arkive_rag` package.
"""
