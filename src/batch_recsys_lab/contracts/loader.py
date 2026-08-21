"""Contract YAML → frozen dataclasses (Phase 1, T1; docs/engineering-log/UPGRADE_PLAN.md §8).

A contract is a table's data-quality specification: its expected columns
(name / dtype / nullability) and an ordered list of checks. The check order is
load-bearing — it *is* the fixed priority used to pick a quarantined row's
``primary_reason`` (D5). The loader preserves declaration order verbatim.

The set of check kinds and actions is CLOSED. Unknown kinds/actions are rejected
at load time with a clear error, so a typo in a YAML never silently disables a
check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# Closed vocabularies. Anything outside these is a load-time error.
VALID_KINDS: frozenset[str] = frozenset(
    {
        "not_null",
        "allowed_values",
        "forbidden_values",
        "range",
        "no_control_chars",
        "no_all_null",
        "orphan_rate",
        "unknown_share",
    }
)
VALID_ACTIONS: frozenset[str] = frozenset({"quarantine", "fail", "measure"})


@dataclass(frozen=True)
class ColumnSpec:
    """One expected column: name, declared dtype, and nullability.

    ``dtype`` is the contract's canonical spelling (e.g. ``long``, not Spark's
    ``bigint``); audit normalizes it when comparing to a published schema.
    """

    name: str
    dtype: str
    nullable: bool


@dataclass(frozen=True)
class Check:
    """One data-quality check.

    ``columns`` is the ordered tuple of columns the check applies to (a single
    ``column:`` key in YAML is normalized to a one-element tuple). The remaining
    fields are per-kind parameters; unused ones stay ``None``.
    """

    check_id: str
    kind: str
    action: str
    columns: tuple[str, ...] = ()
    values: tuple[object, ...] | None = None  # allowed_values / forbidden_values
    min: object | None = None  # range lower bound (inclusive)
    max: object | None = None  # range upper bound (inclusive)
    max_exclusive: object | None = None  # range upper bound (exclusive)
    ref_table: str | None = None  # orphan_rate reference table
    ref_column: str | None = None  # orphan_rate reference column
    value: object | None = None  # unknown_share sentinel


@dataclass(frozen=True)
class Contract:
    """A table contract: identity plus expected columns and ordered checks."""

    name: str
    version: int
    table: str
    columns: tuple[ColumnSpec, ...]
    checks: tuple[Check, ...]


def _parse_column(raw: dict) -> ColumnSpec:
    return ColumnSpec(
        name=raw["name"],
        dtype=raw["dtype"],
        nullable=bool(raw.get("nullable", True)),
    )


def _parse_check(raw: dict) -> Check:
    check_id = raw["id"]
    kind = raw["kind"]
    action = raw["action"]
    if kind not in VALID_KINDS:
        raise ValueError(
            f"check {check_id!r}: unknown check kind {kind!r}; "
            f"allowed kinds are {sorted(VALID_KINDS)}"
        )
    if action not in VALID_ACTIONS:
        raise ValueError(
            f"check {check_id!r}: unknown action {action!r}; "
            f"allowed actions are {sorted(VALID_ACTIONS)}"
        )

    if "columns" in raw:
        columns = tuple(raw["columns"])
    elif "column" in raw:
        columns = (raw["column"],)
    else:
        columns = ()

    values = tuple(raw["values"]) if "values" in raw else None

    check = Check(
        check_id=check_id,
        kind=kind,
        action=action,
        columns=columns,
        values=values,
        min=raw.get("min"),
        max=raw.get("max"),
        max_exclusive=raw.get("max_exclusive"),
        ref_table=raw.get("ref_table"),
        ref_column=raw.get("ref_column"),
        value=raw.get("value"),
    )
    _validate_check_params(check)
    return check


def _validate_check_params(check: Check) -> None:
    """Reject structurally incomplete checks (wrong params for the kind)."""
    kind = check.kind
    if kind in {"not_null", "no_control_chars"} and not check.columns:
        raise ValueError(f"check {check.check_id!r}: {kind} requires column(s)")
    if kind in {"allowed_values", "forbidden_values"}:
        if len(check.columns) != 1:
            raise ValueError(f"check {check.check_id!r}: {kind} requires exactly one column")
        if not check.values:
            raise ValueError(f"check {check.check_id!r}: {kind} requires a non-empty values list")
    if kind == "range":
        if len(check.columns) != 1:
            raise ValueError(f"check {check.check_id!r}: range requires exactly one column")
        if check.min is None and check.max is None and check.max_exclusive is None:
            raise ValueError(
                f"check {check.check_id!r}: range requires at least one of min / max / max_exclusive"
            )
    if kind == "orphan_rate":
        if len(check.columns) != 1 or not check.ref_table or not check.ref_column:
            raise ValueError(
                f"check {check.check_id!r}: orphan_rate requires column, ref_table, ref_column"
            )
    if kind == "unknown_share":
        if len(check.columns) != 1 or check.value is None:
            raise ValueError(f"check {check.check_id!r}: unknown_share requires column and value")


def parse_contract(doc: dict) -> Contract:
    """Build a :class:`Contract` from an already-parsed YAML mapping."""
    return Contract(
        name=doc["name"],
        version=int(doc["version"]),
        table=doc["table"],
        columns=tuple(_parse_column(c) for c in doc["columns"]),
        checks=tuple(_parse_check(c) for c in doc["checks"]),
    )


def load_contract(path: str | Path) -> Contract:
    """Load and validate a contract from a YAML file path."""
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"contract {path!s}: top-level YAML must be a mapping")
    return parse_contract(doc)
