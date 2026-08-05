"""Contract engine (Phase 1, T1): YAML contracts → gate + audit + dq_results.

Public surface:

* :func:`load_contract` and the :class:`Contract` / :class:`ColumnSpec` /
  :class:`Check` dataclasses (``loader``).
* :func:`gate` / :func:`audit` / :func:`write_dq_results` and the
  :class:`GateResult` / :class:`DqResult` records (``engine``).
"""

from __future__ import annotations

from batch_recsys_lab.contracts.engine import (
    DqResult,
    GateResult,
    audit,
    gate,
    write_dq_results,
)
from batch_recsys_lab.contracts.loader import (
    Check,
    ColumnSpec,
    Contract,
    load_contract,
    parse_contract,
)

__all__ = [
    "Check",
    "ColumnSpec",
    "Contract",
    "DqResult",
    "GateResult",
    "audit",
    "gate",
    "load_contract",
    "parse_contract",
    "write_dq_results",
]
