# SPDX-License-Identifier: Apache-2.0
"""frontier-prep: deterministic, auditable data-preparation operations for physics datasets.

The contract (``Op`` protocol, ``Plan`` / ``Preview`` / ``Result`` / ``Explanation``, registry,
tool-schema export) lives in :mod:`frontier_prep.op` and is specified in
``docs/spec/data-prep.md``. First-party ops are in :mod:`frontier_prep.ops`; recipes in
:mod:`frontier_prep.recipes`; the CLI is ``python -m frontier_prep``.
"""

from frontier_prep.op import (
    CONTRACT_VERSION,
    ENTRY_POINT_GROUP,
    REGISTRY,
    Actor,
    Assertion,
    Change,
    Context,
    Explanation,
    FieldDiff,
    Judgment,
    Op,
    OpInfo,
    OpWarning,
    Params,
    Plan,
    Preview,
    Render,
    Result,
    Selection,
    export_tools,
    get_op,
    register,
    tool_schema,
)

__all__ = [
    "CONTRACT_VERSION",
    "ENTRY_POINT_GROUP",
    "REGISTRY",
    "Actor",
    "Assertion",
    "Change",
    "Context",
    "Explanation",
    "FieldDiff",
    "Judgment",
    "Op",
    "OpInfo",
    "OpWarning",
    "Params",
    "Plan",
    "Preview",
    "Render",
    "Result",
    "Selection",
    "export_tools",
    "get_op",
    "register",
    "tool_schema",
]
