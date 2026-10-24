# SPDX-License-Identifier: Apache-2.0
"""Inspect installed extension compatibility without importing extension code."""

import argparse
import json
from importlib.metadata import entry_points

from frontier_plugins import API_VERSIONS, ENTRY_POINT_APIS, check_entry_point


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the machine-readable report")
    parser.add_argument("--data-adapters", action="store_true", help="List data capabilities")
    args = parser.parse_args()
    if args.data_adapters:
        from frontier_plugins.data_adapters import catalog

        adapters = catalog()
        print(json.dumps({"data_adapters": adapters}, indent=2))
        return 0 if all(row["available"] for row in adapters) else 1
    rows = []
    for group, api in ENTRY_POINT_APIS.items():
        for ep in entry_points(group=group):
            checked = check_entry_point(ep)
            rows.append(
                {
                    "name": ep.name,
                    "group": group,
                    "api": api,
                    "distribution": ep.dist.metadata["Name"] if ep.dist else None,
                    "version": ep.dist.version if ep.dist else None,
                    "compatible": checked.compatible,
                    "errors": checked.errors,
                    "warnings": checked.warnings,
                }
            )
    ok = all(row["compatible"] for row in rows)
    if args.json:
        print(json.dumps({"ok": ok, "apis": API_VERSIONS, "extensions": rows}, indent=2))
    else:
        for row in rows:
            state = "compatible" if row["compatible"] else "incompatible"
            print(f"{row['group']} / {row['name']}: {state} ({row['version']})")
            for note in row["errors"] + row["warnings"]:
                print(f"  {note}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
