#!/usr/bin/env python3
"""Generate safe P/R candidate bundles for the unchanged INT8 SME backend."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from generate_driver import (
    ALLOWED_P,
    ALLOWED_R,
    DEFAULT_CONFIG,
    candidate_id,
    read_json,
    sha256,
    validate_config,
    write_bundle,
    write_json,
)


def parse_values(raw: str, allowed: Sequence[int], field: str) -> Tuple[int, ...]:
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
        raise ValueError("%s contains unsupported values %r; allowed=%r" % (field, invalid, tuple(allowed)))
    return values


def candidate_config(base: Dict[str, Any], p: int, r: int) -> Dict[str, Any]:
    config = deepcopy(base)
    config["driver"]["p"] = p
    config["driver"]["r"] = r
    config["name"] = "%s_p%d_r%d" % (base.get("name", "sme_int8"), p, r)
    validate_config(config)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="baseline-compatible backend JSON")
    parser.add_argument("--output", type=Path, required=True,
                        help="directory containing one build bundle per candidate")
    parser.add_argument("--reference-root", default="/path/to/int8gemm-kblas",
                        help="recorded reference root; each Makefile may override REF_ROOT")
    parser.add_argument("--p-values", default=",".join(str(value) for value in ALLOWED_P),
                        help="safe P candidates, comma-separated")
    parser.add_argument("--r-values", default=",".join(str(value) for value in ALLOWED_R),
                        help="safe R candidates, comma-separated")
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
        p_values = parse_values(args.p_values, ALLOWED_P, "--p-values")
        r_values = parse_values(args.r_values, ALLOWED_R, "--r-values")
        if output.exists() and not output.is_dir():
            raise ValueError("output exists and is not a directory: %s" % output)
        if output.exists() and any(output.iterdir()) and not args.force:
            raise ValueError("output exists and is not empty; pass --force to update it")
        output.mkdir(parents=True, exist_ok=True)

        candidates = []
        for p in p_values:
            for r in r_values:
                config = candidate_config(base, p, r)
                identifier = candidate_id(config)
                bundle = output / identifier
                manifest = write_bundle(config, bundle, args.reference_root, args.force)
                candidates.append({
                    "candidate_id": identifier,
                    "bundle_dir": identifier,
                    "config": config["driver"],
                    "manifest": "%s/manifest.json" % identifier,
                    "buffer_contract": manifest["buffer_contract"],
                })
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    index = {
        "schema_version": 1,
        "source_config": str(config_path),
        "source_config_sha256": sha256(config_path),
        "reference_root": args.reference_root,
        "candidate_count": len(candidates),
        "fixed_driver": {
            field: base["driver"][field]
            for field in ("threads_m", "threads_n", "q", "jblock", "region_align", "b_packing")
        },
        "search_space": {"p": list(p_values), "r": list(r_values)},
        "candidates": candidates,
    }
    write_json(output / "candidates.json", index)
    print("generated %d SME candidate bundles in %s" % (len(candidates), output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
