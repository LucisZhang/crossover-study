"""Deterministic 30-user pick for the pick-a-shopper exhibit (Phase 6, T28).

Implements — and nothing but — the curation rule pre-declared in
``EXPERIMENT_LOG.md`` § "Phase 6 T28 — shopper curation rule (pre-declared)",
which was written before this module was run (select-then-look discipline):

* universe = the TEST-eval users, i.e. the rows of the **blend** per-user
  artifact named by the headline record; segment = that artifact's own
  ``segment`` column, never recomputed;
* 6 users per segment × the frozen five segments = 30;
* per segment, prefer the users with ``>=min_test_gt`` TEST ground-truth items
  (fallback to the whole segment, recorded);
* draw ``min_blend_hits`` users from the segment's **hit stratum**
  (``hitrate@10 == 1.0`` in the blend artifact) and the rest from its **miss
  stratum**, without replacement, with
  ``default_rng([seed, segment_ordinal, 1])`` / ``default_rng([seed,
  segment_ordinal, 0])``;
* display identity is ``HMAC-SHA256(salt, user_id)[:12]``; the salt and the
  ``user_id -> shopper_id`` mapping stay under gitignored ``data/``.

The stratification is the **v2** rule (log entry "curation rule v1 FAILED,
superseded by a stratified draw"): v1's uniform-draw-and-redraw predicate is
unsatisfiable — blend's recorded ``hitrate@10`` is 1.5–4% per segment, so
P(≥2 hits in 6) ≈ 6·10⁻³ — and raising the redraw cap would have let the seed,
not the rule, choose the pick. Stratifying satisfies the predicate in one pass
and pays for it with disclosure: the export carries the draw counts *and* the
true per-segment hit rate, traced to each model's record, so the exhibit can
say out loud that 2-of-6 hits is by design and ~1-in-50 is the truth.

Nothing here consults a model, a threshold or a metric other than the two
membership tests above, and nothing feeds back into model or policy choice: the
frozen TEST split is being *displayed*, not tuned against.

    uv run python -m batch_recsys_lab.demo.select_shoppers --config configs/shoppers_export.yaml

Outputs (both gitignored): ``data/demo_export/shopper_selection.json`` and
``data/demo_export/shopper_map.parquet``.
"""

from __future__ import annotations

import argparse
import hmac
import json
import secrets
from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from batch_recsys_lab.demo.export_core import index_runs, load_export_config, sha256_file
from batch_recsys_lab.eval.protocol import SEGMENT_LABELS

__all__ = [
    "SEGMENT_LABELS",
    "Context",
    "load_config",
    "resolve_context",
    "select_shoppers",
    "shopper_id_for",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]

FIVE_CORE_KEY = "local.gold.interactions_5core"
SELECTION_SCHEMA_VERSION = 1
# Bumped whenever the pre-declared rule changes (which needs a superseding
# EXPERIMENT_LOG entry first). Recorded in the selection artifact.
RULE_ID = "phase6-t28-v2-stratified"


# --- config + context ---------------------------------------------------------


def load_config(path: str | Path) -> dict:
    """Load ``configs/shoppers_export.yaml`` and check the knobs are sane."""
    cfg = yaml.safe_load(Path(path).read_text())
    required = (
        "demo_export_config",
        "models",
        "cold_collapse_models",
        "seed",
        "per_segment",
        "min_test_gt",
        "min_blend_hits",
        "min_blend_misses",
        "max_attempts",
        "salt_path",
        "work_dir",
    )
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    if cfg["models"][0] != "blend":
        raise ValueError(f"{path}: models[0] must be 'blend' (the curated arm)")
    if cfg["min_blend_hits"] + cfg["min_blend_misses"] > cfg["per_segment"]:
        raise ValueError(f"{path}: predicate is unsatisfiable for per_segment={cfg['per_segment']}")
    unknown = set(cfg["cold_collapse_models"]) - set(cfg["models"])
    if unknown:
        raise ValueError(f"{path}: cold_collapse_models not in models: {sorted(unknown)}")
    return cfg


class Context:
    """Everything the three T28 stages resolve the same way, resolved once.

    Model run_ids come from ``configs/demo_export.yaml`` (never re-pinned here),
    the per-user artifact paths come from those records, and the eval cache is
    keyed by the headline record's pinned 5-core snapshot id — so a cache that
    is not the cache the runs were scored against cannot be picked up silently.
    """

    def __init__(self, cfg: dict, *, repo_root: Path | None = None) -> None:
        self.cfg = cfg
        self.root = Path(repo_root) if repo_root else _REPO_ROOT
        self.demo_cfg = load_export_config(self._p(cfg["demo_export_config"]))
        self.runs = index_runs(self._p(self.demo_cfg["runs_log"]))
        self.model_keys: list[str] = list(cfg["models"])

        by_key = {m["key"]: m for m in self.demo_cfg["models"]}
        unknown = [k for k in self.model_keys if k not in by_key]
        if unknown:
            raise ValueError(f"models {unknown} are not in {cfg['demo_export_config']}")
        self.run_ids = {k: by_key[k]["run_id"] for k in self.model_keys}
        self.labels = {k: by_key[k]["label"] for k in self.model_keys}

        self.headline_run_id: str = self.demo_cfg["headline_run_id"]
        self.artifacts: dict[str, Path] = {}
        for key, rid in self.run_ids.items():
            rec = self.runs.get(rid)
            if rec is None:
                raise KeyError(f"model {key!r}: run_id {rid} is not in the runs log")
            rel = rec.get("per_user_artifact")
            if not rel:
                raise KeyError(f"model {key!r}: run {rid} has no per_user_artifact")
            path = self._p(rel)
            if not path.exists():
                raise FileNotFoundError(f"model {key!r}: per-user artifact {rel} is missing")
            self.artifacts[key] = path

        self.snapshot_ids: dict[str, int] = {
            str(t): int(s)
            for t, s in self.runs[self.headline_run_id]["iceberg_snapshots"].items()
        }
        cache_dir = cfg.get("cache_dir")
        if cache_dir:
            self.cache_dir = self._p(cache_dir)
        else:
            self.cache_dir = self.root / "data" / "eval" / "cache" / str(self.snapshot_ids[FIVE_CORE_KEY])
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"eval cache {self.cache_dir} does not exist")
        manifest = json.loads((self.cache_dir / "cache_manifest.json").read_text())
        cached = {str(t): int(s) for t, s in manifest["snapshot_ids"].items()}
        if cached != self.snapshot_ids:
            raise ValueError(
                f"eval cache {self.cache_dir} was built at snapshots {cached}, but the headline "
                f"record {self.headline_run_id} was scored at {self.snapshot_ids}"
            )
        self.work_dir = self._p(cfg["work_dir"])

    def _p(self, rel: str | Path) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def artifact_rel(self, key: str) -> str:
        """The artifact path exactly as the record spells it (manifest citations
        must match ``per_user_artifact`` verbatim)."""
        return str(self.runs[self.run_ids[key]]["per_user_artifact"])

    # -- cache reads --

    def item_ids(self) -> np.ndarray:
        """Catalog order: index -> item_id (== ``parent_asin``)."""
        return np.array(
            pq.read_table(self.cache_dir / "item_ids.parquet").column(0).to_pylist(), dtype=object
        )

    def user_ids(self) -> np.ndarray:
        return np.array(
            pq.read_table(self.cache_dir / "user_ids.parquet").column(0).to_pylist(), dtype=object
        )

    def n_train(self) -> np.ndarray:
        return np.load(self.cache_dir / "n_train.npy", allow_pickle=False)

    def test_gt(self) -> dict[int, list[int]]:
        """user_idx -> its TEST ground-truth catalog indices (ascending)."""
        u = np.load(self.cache_dir / "test_user_idx.npy", allow_pickle=False)
        i = np.load(self.cache_dir / "test_item_idx.npy", allow_pickle=False)
        order = np.argsort(u, kind="stable")
        out: dict[int, list[int]] = {}
        for uu, ii in zip(u[order].tolist(), i[order].tolist()):
            out.setdefault(int(uu), []).append(int(ii))
        return {k: sorted(v) for k, v in out.items()}


# --- re-hash ------------------------------------------------------------------


def load_or_create_salt(path: str | Path) -> bytes:
    """Read the local HMAC salt, generating a 32-byte one on first use.

    ``data/`` is gitignored, so the salt never leaves the machine; the exported
    ``shopper_id``s are therefore not reversible by anyone holding the repo.
    """
    p = Path(path)
    if p.exists():
        text = p.read_text().strip()
        if len(text) < 32:
            raise ValueError(f"{p}: salt is too short ({len(text)} chars); delete it to regenerate")
    else:
        text = secrets.token_hex(32)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n")
        p.chmod(0o600)
    return text.encode("utf-8")


def shopper_id_for(salt: bytes, user_id: str, *, length: int = 12) -> str:
    """``HMAC-SHA256(salt, user_id)`` truncated to ``length`` hex chars."""
    return hmac.new(salt, user_id.encode("utf-8"), sha256).hexdigest()[:length]


# --- the pre-declared rule ----------------------------------------------------


def select_shoppers(ctx: Context) -> dict:
    """Execute the pre-declared rule. Returns the selection document."""
    cfg = ctx.cfg
    seed = int(cfg["seed"])
    per_segment = int(cfg["per_segment"])
    min_gt = int(cfg["min_test_gt"])
    min_hits = int(cfg["min_blend_hits"])
    min_misses = int(cfg["min_blend_misses"])
    max_attempts = int(cfg["max_attempts"])

    blend_path = ctx.artifacts["blend"]
    table = pq.read_table(blend_path, columns=["user_id", "user_idx", "segment", "hitrate@10"])
    user_idx = np.asarray(table.column("user_idx").to_numpy(zero_copy_only=False), dtype=np.int64)
    user_id = np.array(table.column("user_id").to_pylist(), dtype=object)
    segment = np.array(table.column("segment").to_pylist(), dtype=object)
    hitrate = np.asarray(table.column("hitrate@10").to_numpy(zero_copy_only=False), dtype=np.float64)

    n_users_total = int(user_idx.max()) + 1
    row_of = np.full(n_users_total, -1, dtype=np.int64)
    row_of[user_idx] = np.arange(len(user_idx), dtype=np.int64)

    test_u = np.load(ctx.cache_dir / "test_user_idx.npy", allow_pickle=False)
    n_test_by_user = np.bincount(np.asarray(test_u, dtype=np.int64), minlength=n_users_total)
    n_train_all = ctx.n_train()

    salt = load_or_create_salt(ctx._p(cfg["salt_path"]))

    segments_out: dict[str, dict] = {}
    order: list[str] = []
    seen_ids: dict[str, str] = {}

    for ordinal, seg in enumerate(SEGMENT_LABELS):
        seg_users = np.sort(user_idx[segment == seg])
        if len(seg_users) < per_segment:
            raise ValueError(f"segment {seg!r} has only {len(seg_users)} eval users; need {per_segment}")
        pool = seg_users[n_test_by_user[seg_users] >= min_gt]
        fallback = len(pool) < per_segment
        if fallback:
            pool = seg_users

        # v2: stratify, do not fish. The hit/miss split is on blend's recorded
        # per-user hitrate@10 (1.0 iff >=1 TEST GT item in its top 10).
        pool_hit_mask = hitrate[row_of[pool]] == 1.0
        hit_pool = pool[pool_hit_mask]
        miss_pool = pool[~pool_hit_mask]
        n_hit = min_hits
        n_miss = per_segment - n_hit
        if len(hit_pool) < n_hit or len(miss_pool) < n_miss:
            raise RuntimeError(
                f"segment {seg!r}: stratum too small — need {n_hit} of "
                f"{len(hit_pool)} blend-hit and {n_miss} of {len(miss_pool)} blend-miss "
                f"users. The rule is not relaxed — see EXPERIMENT_LOG.md T28 (v2)."
            )
        if n_hit < min_hits or n_miss < min_misses:
            raise AssertionError(
                f"segment {seg!r}: stratified draw ({n_hit} hit / {n_miss} miss) would violate "
                f"the pre-declared predicate (>={min_hits} hit, >={min_misses} miss)"
            )
        attempts = 1  # deterministic in one pass, by construction
        drawn_hit = np.random.default_rng([seed, ordinal, 1]).choice(
            hit_pool, size=n_hit, replace=False
        )
        drawn_miss = np.random.default_rng([seed, ordinal, 0]).choice(
            miss_pool, size=n_miss, replace=False
        )
        chosen = np.sort(np.concatenate([drawn_hit, drawn_miss]))

        members = []
        for u in chosen.tolist():
            row = int(row_of[u])
            uid = str(user_id[row])
            sid = shopper_id_for(salt, uid)
            if sid in seen_ids:
                raise RuntimeError(
                    f"shopper_id collision {sid!r} between {seen_ids[sid]!r} and {uid!r} — "
                    "regenerate data/demo_salt.txt"
                )
            seen_ids[sid] = uid
            order.append(sid)
            members.append(
                {
                    "shopper_id": sid,
                    "user_id": uid,
                    "user_idx": int(u),
                    "artifact_row": row,
                    "segment": str(segment[row]),
                    "n_train": int(n_train_all[u]),
                    "n_test_gt": int(n_test_by_user[u]),
                    "blend_hit_at_10": bool(hitrate[row] == 1.0),
                }
            )

        segments_out[seg] = {
            "segment": seg,
            "segment_ordinal": ordinal,
            "eval_users": int(len(seg_users)),
            "pool_size": int(len(pool)),
            "pool_fallback_to_all_users": bool(fallback),
            "attempts": attempts,
            "sub_seed": seed + attempts - 1,
            "hit_stratum_size": int(len(hit_pool)),
            "miss_stratum_size": int(len(miss_pool)),
            "blend_hits": n_hit,
            "blend_misses": n_miss,
            "members": members,
        }

    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "rule_id": RULE_ID,
        "rule_declared_in": "EXPERIMENT_LOG.md#phase-6-t28--shopper-curation-rule-pre-declared",
        "seed": seed,
        "per_segment": per_segment,
        "min_test_gt": min_gt,
        "min_blend_hits": min_hits,
        "min_blend_misses": min_misses,
        "max_attempts": max_attempts,
        "segments": list(SEGMENT_LABELS),
        "models": ctx.model_keys,
        "run_ids": dict(ctx.run_ids),
        "headline_run_id": ctx.headline_run_id,
        "iceberg_snapshots": ctx.snapshot_ids,
        "cache_dir": str(ctx.cache_dir.relative_to(ctx.root)),
        "blend_artifact": ctx.artifact_rel("blend"),
        "blend_artifact_sha256": sha256_file(blend_path),
        "shopper_order": order,
        "total_attempts": sum(s["attempts"] for s in segments_out.values()),
        "by_segment": segments_out,
    }


def write_selection(ctx: Context, selection: dict) -> tuple[Path, Path]:
    out_dir = ctx.work_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "shopper_selection.json"
    json_path.write_text(json.dumps(selection, indent=2) + "\n")

    rows = [m for seg in selection["by_segment"].values() for m in seg["members"]]
    table = pa.table(
        {
            "shopper_id": pa.array([r["shopper_id"] for r in rows], pa.string()),
            "user_id": pa.array([r["user_id"] for r in rows], pa.string()),
            "user_idx": pa.array([r["user_idx"] for r in rows], pa.int64()),
            "segment": pa.array([r["segment"] for r in rows], pa.string()),
            "n_train": pa.array([r["n_train"] for r in rows], pa.int64()),
            "n_test_gt": pa.array([r["n_test_gt"] for r in rows], pa.int64()),
        }
    )
    map_path = out_dir / "shopper_map.parquet"
    pq.write_table(table, map_path)
    return json_path, map_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/shoppers_export.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ctx = Context(cfg)
    selection = select_shoppers(ctx)
    json_path, map_path = write_selection(ctx, selection)

    print(f"selected {len(selection['shopper_order'])} shoppers · rule {selection['rule_id']}")
    for seg in selection["segments"]:
        s = selection["by_segment"][seg]
        print(
            f"  segment {seg:>5}: {len(s['members'])} users · attempts={s['attempts']} "
            f"(sub_seed={s['sub_seed']}) · blend hits={s['blend_hits']} misses={s['blend_misses']} "
            f"· pool={s['pool_size']}/{s['eval_users']} "
            f"(strata {s['hit_stratum_size']} hit / {s['miss_stratum_size']} miss)"
            + (" [FALLBACK: <6 users with >=2 TEST GT]" if s["pool_fallback_to_all_users"] else "")
        )
    print(f"wrote {json_path}\nwrote {map_path}")


if __name__ == "__main__":
    main()
