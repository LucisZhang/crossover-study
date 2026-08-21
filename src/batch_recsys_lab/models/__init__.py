"""Recommender models (Phase 2, T4; docs/engineering-log/UPGRADE_PLAN.md §8).

``Recommender`` (``models.base``) is the protocol the eval harness scores
against. This package currently ships three baselines (random, popularity,
popularity-per-category); item-kNN lives alongside in ``models.item_knn``.
"""
