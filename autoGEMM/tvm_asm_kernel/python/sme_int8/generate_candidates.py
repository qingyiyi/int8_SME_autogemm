#!/usr/bin/env python3
"""Generate INT8 SME candidate bundles from ``baseline_config.json``.

The batch dimensions are defined by ``search_space`` in the JSON config:
``p`` × ``r`` × paired ``thread_groups``.  P/R command-line options are only
backward-compatible subset filters; they cannot add values not declared in the
JSON configuration.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from generate_driver import (
    DEFAULT_CONFIG,
    candidate_id,
    configured_search_space,
    read_json,
    search_space_json,
    sha256,
    validate_config,
    write_bundle,
    write_json,
)


def parse_values(raw: str, allowed: Sequence[int], field: str) -> Tuple[int, ...]:
    """Parse a backward-compatible P/R subset, constrained by the JSON list."""
    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise ValueError("%s must be a comma-separated integer list" % field) from error
    if not values:
        raise ValueError("%s must not be empty" % field)
    if len(set(values)) != len(values):
        raise ValueError("%s must not contain duplicates" % field)
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(
            "%s contains values not declared in baseline_config.json: %r; configured=%r" %
            (field, invalid, tuple(allowed))
        )
    return values


def selected_values(raw: str | None, configured: Sequence[int], field: str) -> Tuple[int, ...]:
    if raw is None:
        return tuple(configured)
    return parse_values(raw, configured, field)


def candidate_config(
    base: Dict[str, Any], p: int, r: int, threads_m: int, threads_n: int
) -> Dict[str, Any]:
    """Create one candidate while preserving the complete JSON search contract."""
    config = deepcopy(base)
    config["driver"]["p"] = p
    config["driver"]["r"] = r
    config["driver"]["threads_m"] = threads_m
    config["driver"]["threads_n"] = threads_n
    config["name"] = "%s_p%d_r%d_t%dx%d" % (
        base.get("name", "sme_int8"), p, r, threads_m, threads_n
    )
    validate_config(config)
    return config


def fixed_driver_fields(base: Dict[str, Any]) -> Dict[str, Any]:
    """Fields shared by every candidate in this search (not sweep variables)."""
    return {
        field: base["driver"][field]
        for field in ("q", "jblock", "region_align", "b_packing")
    }


def candidate_index(
    config_path: Path,
    base: Dict[str, Any],
    reference_root: str,
    p_values: Sequence[int],
    r_values: Sequence[int],
    output: Path,
    force: bool,
) -> Dict[str, Any]:
    """Render all P/R/thread-group combinations and return candidates.json data."""
    configured = configured_search_space(base)
    candidates = []
    for p in p_values:
        for r in r_values:
            for threads_m, threads_n in configured["thread_groups"]:
                config = candidate_config(base, p, r, threads_m, threads_n)
                identifier = candidate_id(config)
                bundle = output / identifier
                manifest = write_bundle(config, bundle, reference_root, force)
                candidates.append({
                    "candidate_id": identifier,
                    "bundle_dir": identifier,
                    "config": config["driver"],
                    "manifest": "%s/manifest.json" % identifier,
                    "buffer_contract": manifest["buffer_contract"],
                })

    search_space = search_space_json(base)
    search_space["p"] = list(p_values)
    search_space["r"] = list(r_values)
    return {
        "schema_version": 1,
        "source_config": str(config_path),
        "source_config_sha256": sha256(config_path),
        "reference_root": reference_root,
        "candidate_count": len(candidates),
        "fixed_driver": fixed_driver_fields(base),
        "search_space": search_space,
        "candidates": candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="baseline-compatible backend JSON")
    parser.add_argument("--output", type=Path, required=True,
                        help="directory containing one build bundle per candidate")
    parser.add_argument("--reference-root", default="/path/to/int8gemm-kblas",
                        help="recorded reference root; each Makefile may override REF_ROOT")
    parser.add_argument("--p-values", default=None,
                        help="optional comma-separated subset of search_space.p")
    parser.add_argument("--r-values", default=None,
                        help="optional comma-separated subset of search_space.r")
    parser.add_argument("--force", action="store_true",
                        help="allow updating an existing candidate output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    try:
        base = read_json(config_path)
        validate_config(base)
        configured = configured_search_space(base)
        p_values = selected_values(args.p_values, configured["p"], "--p-values")
        r_values = selected_values(args.r_values, configured["r"], "--r-values")
        if output.exists() and not output.is_dir():
            raise ValueError("output exists and is not a directory: %s" % output)
        if output.exists() and any(output.iterdir()) and not args.force:
            raise ValueError("output exists and is not empty; pass --force to update it")
        output.mkdir(parents=True, exist_ok=True)
        index = candidate_index(
            config_path, base, args.reference_root, p_values, r_values, output, args.force
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error

    write_json(output / "candidates.json", index)
    print("generated %d SME candidate bundles in %s" % (index["candidate_count"], output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
