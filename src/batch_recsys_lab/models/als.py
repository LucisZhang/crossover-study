"""Implicit-ALS recommender — Step B: pure-numpy rescoring of persisted factors
(Phase 2, T2/T3; UPGRADE_PLAN.md §8 "Architecture", two-process eval design).

This module contains NO ``pyspark`` import. It is the eval-time half of a
two-step design:

* **Step A** (``models.als_train``) trains Spark MLlib ALS once and persists the
  dense factor matrices ``U`` (n_users × rank) and ``V`` (n_items × rank) plus an
  ``als_manifest.json`` under a snapshot-and-param-keyed artifact directory.
* **Step B** (this module) is the ``Recommender`` the harness fits: it loads the
  factors and scores ``U @ Vᵀ``. The harness stays contractually Spark-free.

Determinism stance (mirrors ``als_train``): Spark ALS retraining is *not*
bit-stable (float reduction order varies with partitioning/parallelism), so
reproducibility is not claimed at the training step. It is claimed instead via
(a) the persisted artifact — its factor sha256s are recorded in the run record
and rescoring ``U @ Vᵀ`` is bit-deterministic — and (b) recorded seed/params
with a 3-seed mean±sd bounding the residual stochastic variance.

Cold-start / segment-0 collapse (intentional, documented): users with no TRAIN
interactions and items never seen in TRAIN receive an all-zero factor row from
Step A, hence a score of exactly 0 for every catalog item. Combined with the
harness's deterministic index tie-break this floors cold users to the bottom of
every ranking — the same behavior item-kNN exhibits (segment "0" collapses to
~popularity-of-zero). This is expected, not a bug: ALS has no signal for a user
it never saw, and the crossover analysis relies on that honest collapse.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from batch_recsys_lab.eval.dataset import EvalDataset

# Full catalog table name the eval cache is keyed by (see eval/extract.py:
# cache subdir == this table's snapshot id). The artifact directory is keyed by
# the SAME id so an artifact and the cache it was trained from always correspond.
FIVE_CORE_TABLE = "local.gold.interactions_5core"

# The exact param set that defines an ALS artifact's identity (order-independent).
# ``half_life_days`` is a CONDITIONAL seventh key — see canonical_params.
HASH_KEYS = ("rank", "reg_param", "alpha", "max_iter", "weighting", "seed")
TIME_DECAY = "time_decay"
WEIGHTINGS = ("binary", "rating", TIME_DECAY)


# --- shared helpers (numpy-only; imported by Step A without circularity) ------


def canonical_params(
    rank: int,
    reg_param: float,
    alpha: float,
    max_iter: int,
    weighting: str,
    seed: int,
    half_life_days: float | None = None,
) -> dict:
    """Normalize the identity params to canonical types.

    Both Step A and Step B build this dict from the same eval config, so the
    normalized types must be pinned (int/float/str/int) for the param hash and
    the fit-time equality assertions to agree byte-for-byte.

    ``half_life_days`` (Phase 8, T8-2) enters the canonical dict — and therefore
    the param hash and the artifact path — ONLY when ``weighting == "time_decay"``.
    For ``binary``/``rating`` it is dropped even if supplied, so every artifact,
    manifest and param hash recorded before T8-2 keeps its identity byte-for-byte.
    A ``time_decay`` config without a positive, finite half-life is an error, not
    a default: the half-life is the one tuned quantity of the arm.
    """
    canon = {
        "rank": int(rank),
        "reg_param": float(reg_param),
        "alpha": float(alpha),
        "max_iter": int(max_iter),
        "weighting": str(weighting),
        "seed": int(seed),
    }
    if canon["weighting"] == TIME_DECAY:
        if half_life_days is None:
            raise ValueError(
                "weighting='time_decay' requires params.half_life_days (the T8-2 "
                "preregistered grid is {90, 365, 1460} days)."
            )
        hl = float(half_life_days)
        if not math.isfinite(hl) or hl <= 0:
            raise ValueError(f"half_life_days must be finite and > 0, got {half_life_days!r}")
        canon["half_life_days"] = hl
    return canon


def time_decay_confidence(age_days: np.ndarray, half_life_days: float) -> np.ndarray:
    """``r = 2^(-age_days / half_life_days)`` as float32 (Phase 8, T8-2).

    ``age_days`` is the per-TRAIN-pair staleness at the train cutoff written by
    ``eval/extract_age.py``. ``r`` is fed to Spark ALS as ``ratingCol`` under
    ``implicitPrefs=True``, i.e. confidence ``c = 1 + α·r`` with α unchanged: a
    pair observed exactly at ``train_end`` has ``r = 1`` and is therefore
    identical to the binary baseline, and decay only ever REMOVES stale
    confidence (``0 < r <= 1``, monotonically decreasing in age).
    """
    hl = float(half_life_days)
    if not math.isfinite(hl) or hl <= 0:
        raise ValueError(f"half_life_days must be finite and > 0, got {half_life_days!r}")
    age = np.asarray(age_days, dtype=np.float32)
    if age.size and (not np.isfinite(age).all() or (age < 0).any()):
        raise ValueError("age_days must be finite and non-negative")
    return np.exp2(-(age / np.float32(hl))).astype(np.float32, copy=False)


def als_param_hash(params: dict) -> str:
    """First 12 hex chars of ``sha256`` over canonical JSON of exactly the six
    identity keys — seven for ``weighting == "time_decay"`` (``sorted_keys``,
    compact separators).

    Order-independent (keys are sorted) and type-normalized (via
    :func:`canonical_params`), so two configs that name the same values in a
    different order or with ``int``/``float`` drift hash identically. A
    ``half_life_days`` entry on a binary/rating param dict is ignored, which is
    what keeps every pre-T8-2 hash unchanged.
    """
    canon = canonical_params(
        rank=params["rank"],
        reg_param=params["reg_param"],
        alpha=params["alpha"],
        max_iter=params["max_iter"],
        weighting=params["weighting"],
        seed=params["seed"],
        half_life_days=params.get("half_life_days"),
    )
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def five_core_snapshot_id(manifest: dict, five_core_table: str = FIVE_CORE_TABLE) -> int:
    """The five-core snapshot id the cache (and thus the artifact) is keyed by.

    ``five_core_table`` defaults to the Amazon lane's table name
    (``FIVE_CORE_TABLE``) so every pre-existing call site is unaffected; a
    caller running against a different lane (e.g. ML-32M's
    ``local.gold_ml32m.interactions_5core``) passes that table name so the
    lookup hits the right key in ``manifest["snapshot_ids"]``."""
    return int(manifest["snapshot_ids"][five_core_table])


def artifact_dir(factors_root: str | Path, snapshot_id: int, param_hash: str) -> Path:
    """``<factors_root>/<five_core_snapshot_id>/<param_hash>/`` — the directory
    holding ``user_factors.npy``, ``item_factors.npy``, ``als_manifest.json``."""
    return Path(factors_root) / str(snapshot_id) / str(param_hash)


def sha256_file(path: str | Path) -> str:
    """Raw hex ``sha256`` of a file's bytes (no ``sha256:`` prefix — the ALS
    manifest stores bare hex; the run record surfaces it as-is)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- Step B recommender -------------------------------------------------------


class ALSRecommender:
    """Implicit-ALS scorer over persisted factors (``Recommender`` protocol).

    Instantiated from the eval config's ``model.params`` plus ``seeds.model``;
    ``fit`` loads the artifact Step A wrote for the cache's snapshot and this
    param set. ``score_batch`` returns ``U[user_idx] @ Vᵀ``.

    Cold users / untrained items have zero factor rows and therefore score 0 for
    every item — see the module docstring's "segment-0 collapse" note; this is
    intentional and matches item-kNN's precedent.

    ``half_life_days`` (Phase 8, T8-2) is carried for identity only: the time
    decay is applied by Step A when it builds ``ratingCol``, so Step B just has
    to name the right artifact.
    """

    name = "als"

    def __init__(
        self,
        rank: int,
        reg_param: float,
        alpha: float,
        max_iter: int,
        weighting: str,
        seed: int,
        factors_root: str | Path = "data/eval/als",
        half_life_days: float | None = None,
        five_core_table: str = FIVE_CORE_TABLE,
    ):
        self.rank = int(rank)
        self.reg_param = float(reg_param)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.weighting = str(weighting)
        self.seed = int(seed)
        self.factors_root = str(factors_root)
        # Not part of self.params / the param hash — identity is defined by the
        # model params only; the table name just tells fit() which manifest key
        # to resolve the snapshot id from (default preserves Amazon-lane behavior).
        self.five_core_table = str(five_core_table)
        # Identity-bearing only under weighting='time_decay' (see canonical_params);
        # Step B itself never applies the decay — it scores persisted factors.
        self.half_life_days = None if half_life_days is None else float(half_life_days)
        # JSON-serializable, echoed into the run record. fit() augments this with
        # param_hash and the factor sha256s (read from the manifest, not
        # recomputed) so provenance lands in results/runs.jsonl.
        self.params = canonical_params(
            rank=rank,
            reg_param=reg_param,
            alpha=alpha,
            max_iter=max_iter,
            weighting=weighting,
            seed=seed,
            half_life_days=half_life_days,
        )
        self._U: np.ndarray | None = None
        self._V: np.ndarray | None = None

    def fit(self, ds: EvalDataset) -> "ALSRecommender":
        """Resolve + load the persisted artifact for this cache snapshot + params.

        Raises a clear error (pointing at ``make als-train``) if no artifact
        exists, and a mismatch error if the on-disk manifest's params/seed/
        snapshot or the factor shapes disagree with what this instance expects.
        """
        manifest = ds.manifest
        snap = five_core_snapshot_id(manifest, self.five_core_table)
        param_hash = als_param_hash(self.params)
        adir = artifact_dir(self.factors_root, snap, param_hash)
        man_path = adir / "als_manifest.json"

        if not man_path.exists():
            raise FileNotFoundError(
                f"ALS artifact not found at {adir} (no als_manifest.json). Train it "
                f"first:  make als-train CONFIG=<your eval_als_*.yaml>  (Step A trains "
                f"Spark MLlib ALS once and persists the factors this scorer loads)."
            )

        am = json.loads(man_path.read_text())

        # Identity guards: the artifact must be the one this instance names.
        if am.get("param_hash") != param_hash:
            raise ValueError(
                f"ALS artifact param_hash mismatch at {adir}: manifest "
                f"{am.get('param_hash')!r} != expected {param_hash!r}."
            )
        if am.get("params") != self.params:
            raise ValueError(
                f"ALS artifact param/seed mismatch at {adir}: manifest params "
                f"{am.get('params')!r} != expected {self.params!r}."
            )
        if int(am.get("five_core_snapshot_id", -1)) != snap:
            raise ValueError(
                f"ALS artifact snapshot mismatch at {adir}: manifest five-core "
                f"snapshot {am.get('five_core_snapshot_id')!r} != cache snapshot {snap!r}."
            )

        n_users = len(ds.user_ids)
        n_items = len(ds.item_ids)
        U = np.load(adir / "user_factors.npy", allow_pickle=False).astype(np.float32, copy=False)
        V = np.load(adir / "item_factors.npy", allow_pickle=False).astype(np.float32, copy=False)
        if U.shape != (n_users, self.rank):
            raise ValueError(
                f"ALS user_factors shape {U.shape} != (n_users={n_users}, rank={self.rank})."
            )
        if V.shape != (n_items, self.rank):
            raise ValueError(
                f"ALS item_factors shape {V.shape} != (n_items={n_items}, rank={self.rank})."
            )
        self._U = U
        self._V = V

        # Provenance into the run record — sha256s are read from the manifest, not
        # recomputed here (rehashing ~800MB of factors at fit time is wasteful).
        self.params = {
            **self.params,
            "param_hash": param_hash,
            "user_factors_sha256": am.get("user_factors_sha256"),
            "item_factors_sha256": am.get("item_factors_sha256"),
        }
        return self

    def score_batch(self, user_idx: np.ndarray) -> np.ndarray:
        """Score every catalog item for a batch of users -> (B, I) float32.

        ``scores = U[user_idx] @ Vᵀ``, returned as a fresh, writable float32 array
        (the harness masks TRAIN-seen items in place afterward). Cold users and
        untrained items carry zero factor rows and therefore score 0 for every
        item — the intentional segment-0 collapse documented at module level.
        """
        if self._U is None or self._V is None:
            raise RuntimeError("ALSRecommender.score_batch called before fit().")
        user_idx = np.asarray(user_idx)
        # U[user_idx] fancy-indexes a fresh copy; matmul yields a fresh array;
        # astype(copy=True default) guarantees a writable, non-aliasing buffer.
        return (self._U[user_idx] @ self._V.T).astype(np.float32)
