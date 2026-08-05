"""``EvalDataset`` loader — reads the cache built by ``eval/extract.py`` into
numpy/scipy structures for Step B (scoring; UPGRADE_PLAN.md §8 "Architecture").

This module never starts Spark. Load order and attribute names are a pinned
interface consumed by ``eval/harness.py`` and the model modules built in
parallel — do not rename attributes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp

POP_AS_OF_LABELS = ("train_end", "val_end")
POP_WINDOWS = (0, 30, 90, 365)
GT_SPLITS = ("val", "test")


@dataclass
class GroundTruth:
    """CSR-style ragged ground truth for one eval split.

    ``user_idx[k]`` has GT item indices ``item_idx[indptr[k]:indptr[k+1]]``.
    Only users with >=1 GT pair appear in ``user_idx`` (ascending, unique).
    """

    user_idx: np.ndarray  # int32
    indptr: np.ndarray  # int64, len == len(user_idx) + 1
    item_idx: np.ndarray  # int32, concatenated GT item indices


@dataclass
class EvalDataset:
    cache_dir: Path
    manifest: dict
    item_ids: np.ndarray  # object/str, len I, sorted (catalog order)
    user_ids: np.ndarray  # object/str, len U, sorted
    n_train: np.ndarray  # int32, len U
    train_csr: sp.csr_matrix  # U x I float32 binary
    pop: dict = field(default_factory=dict)  # (as_of_label, window) -> float32 len I
    item_category_codes: np.ndarray = None  # int32, len I
    category_names: list = field(default_factory=list)
    gt: dict = field(default_factory=dict)  # "val"|"test" -> GroundTruth


def _read_string_column(path: Path) -> np.ndarray:
    table = pq.read_table(path)
    col = table.column(0)
    return np.array(col.to_pylist(), dtype=object)


def _build_gt(user_idx: np.ndarray, item_idx: np.ndarray) -> GroundTruth:
    """Sort (user_idx, item_idx) pairs by user_idx and build a CSR-style ragged
    array. Assumes ``user_idx``/``item_idx`` are aligned int32 pair arrays; may
    contain duplicate users (multiple GT items) but not duplicate pairs (silver
    dedup guarantees one row per (user, item))."""
    if len(user_idx) == 0:
        return GroundTruth(
            user_idx=np.array([], dtype=np.int32),
            indptr=np.array([0], dtype=np.int64),
            item_idx=np.array([], dtype=np.int32),
        )
    order = np.argsort(user_idx, kind="stable")
    sorted_users = user_idx[order]
    sorted_items = item_idx[order]

    unique_users, first_pos, counts = np.unique(
        sorted_users, return_index=True, return_counts=True
    )
    indptr = np.zeros(len(unique_users) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return GroundTruth(
        user_idx=unique_users.astype(np.int32),
        indptr=indptr,
        item_idx=sorted_items.astype(np.int32),
    )


def load_dataset(cache_dir: str | Path) -> EvalDataset:
    """Load a cache directory (built by ``eval.extract.extract``) into an
    ``EvalDataset``."""
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "cache_manifest.json").read_text())

    item_ids = _read_string_column(cache_dir / "item_ids.parquet")
    user_ids = _read_string_column(cache_dir / "user_ids.parquet")
    n_items = len(item_ids)
    n_users = len(user_ids)

    n_train = np.load(cache_dir / "n_train.npy", allow_pickle=False)

    train_user_idx = np.load(cache_dir / "train_user_idx.npy", allow_pickle=False)
    train_item_idx = np.load(cache_dir / "train_item_idx.npy", allow_pickle=False)
    # Defensive dedup: COO -> CSR construction sums duplicate (user, item) pairs;
    # clip to binary since a duplicate pair is a defect, not a repeat-purchase
    # signal, for this exclusion-mask use.
    data = np.ones(len(train_user_idx), dtype=np.float32)
    train_coo = sp.coo_matrix(
        (data, (train_user_idx, train_item_idx)), shape=(n_users, n_items), dtype=np.float32
    )
    train_csr = train_coo.tocsr()
    train_csr.data[:] = 1.0
    train_csr.eliminate_zeros()

    pop: dict = {}
    for label in POP_AS_OF_LABELS:
        for w in POP_WINDOWS:
            path = cache_dir / f"pop_{label}_{w}.npy"
            if path.exists():
                pop[(label, w)] = np.load(path, allow_pickle=False)

    item_category_codes = np.load(cache_dir / "item_category_codes.npy", allow_pickle=False)
    category_names = json.loads((cache_dir / "item_category_names.json").read_text())

    gt: dict = {}
    for split in GT_SPLITS:
        u_path = cache_dir / f"{split}_user_idx.npy"
        i_path = cache_dir / f"{split}_item_idx.npy"
        u = np.load(u_path, allow_pickle=False) if u_path.exists() else np.array([], dtype=np.int32)
        i = np.load(i_path, allow_pickle=False) if i_path.exists() else np.array([], dtype=np.int32)
        gt[split] = _build_gt(u, i)

    return EvalDataset(
        cache_dir=cache_dir,
        manifest=manifest,
        item_ids=item_ids,
        user_ids=user_ids,
        n_train=n_train,
        train_csr=train_csr,
        pop=pop,
        item_category_codes=item_category_codes,
        category_names=category_names,
        gt=gt,
    )
