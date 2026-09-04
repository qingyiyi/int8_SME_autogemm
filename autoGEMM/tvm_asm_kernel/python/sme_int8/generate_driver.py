#!/usr/bin/env python3
"""Render constrained INT8 SME CBLAS driver candidates and build bundles.

The generated driver owns the public CBLAS ABI.  The packers and SME kernel
remain source files in the supplied reference root and are compiled unchanged
by the emitted Makefile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from string import Template
from typing import Any, Dict, Tuple


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "baseline_config.json"
DRIVER_TEMPLATE = MODULE_DIR / "driver.c.tmpl"
MAKEFILE_TEMPLATE = MODULE_DIR / "Makefile.tmpl"

EXPECTED_K = 2048
MULTIPLE = 2048
MAX_TARGET_N = 8192
FIXED_DRIVER = {
    "threads_m": 32,
    "threads_n": 1,
    "q": 2048,
    "jblock": 32,
    "region_align": 4,
    "b_packing": "reference_two_phase",
}
ALLOWED_P = (64, 128, 256)
ALLOWED_R = (2048, 4096, 8192)
PACK_ALIGNMENT = 16
SA_DOUBLE_BUFFER_FACTOR = 2
SB_REFERENCE_ALLOCATION_FACTOR = 4
INT8_BYTES = 1


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="ascii") as source:
        return json.load(source)


def as_shape(value: Any) -> Tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3 or not all(isinstance(x, int) for x in value):
        raise ValueError("each shape must be an integer [M, N, K] list")
    return value[0], value[1], value[2]


def as_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("driver.%s must be an integer" % field)
    return value


def validate_driver(driver: Dict[str, Any]) -> None:
    for field, expected in FIXED_DRIVER.items():
        if driver.get(field) != expected:
            raise ValueError(
                "this SME backend fixes driver.%s=%r; got %r" %
                (field, expected, driver.get(field))
            )

    p = as_int(driver.get("p"), "p")
    r = as_int(driver.get("r"), "r")
    if p not in ALLOWED_P:
        raise ValueError("driver.p must be one of %r; got %r" % (ALLOWED_P, p))
    if r not in ALLOWED_R:
        raise ValueError("driver.r must be one of %r; got %r" % (ALLOWED_R, r))
    if p % PACK_ALIGNMENT or r % PACK_ALIGNMENT:
        raise ValueError("driver.p and driver.r must be multiples of %d" % PACK_ALIGNMENT)

    # The two-phase B pack first reserves Jblock columns for every M worker,
    # then divides the remaining work among those workers.  Keep both phases
    # non-empty in the initial search space.
    if r < 2 * driver["jblock"] * driver["threads_m"]:
        raise ValueError(
            "driver.r must be at least 2 * jblock * threads_m for the reference B packing"
        )


def validate_config(config: Dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("unsupported schema_version")

    abi = config.get("abi")
    if not isinstance(abi, dict) or abi.get("export") != "cblas_gemm_s8s8s32":
        raise ValueError("the baseline backend only supports cblas_gemm_s8s8s32")
    for field in ("kernel", "a_packer", "b_packer"):
        if not isinstance(abi.get(field), str) or not abi[field]:
            raise ValueError("abi.%s must be a non-empty symbol name" % field)

    driver = config.get("driver")
    if not isinstance(driver, dict):
        raise ValueError("driver must be an object")
    validate_driver(driver)

    shapes = config.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        raise ValueError("at least one target shape is required")
    for m, n, k in map(as_shape, shapes):
        if k != EXPECTED_K or m <= 0 or n <= 0 or m % MULTIPLE or n % MULTIPLE:
            raise ValueError(
                "only M/N multiples of 2048 and K=2048 are supported: %r" %
                ((m, n, k),)
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_id(config: Dict[str, Any]) -> str:
    driver = config["driver"]
    return "sme-int8-k%d-p%d-r%d-j%d-t%dx%d" % (
        driver["q"],
        driver["p"],
        driver["r"],
        driver["jblock"],
        driver["threads_m"],
        driver["threads_n"],
    )


def buffer_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    """Describe the external buffers required by the unchanged driver.

    The CBLAS ABI supplies pointers but no capacities, so this is metadata for
    the caller and remote sweep runner rather than a runtime check.  The
    formulas deliberately mirror the current reference driver and test.
    """
    driver = config["driver"]
    q = driver["q"]
    p = driver["p"]
    r = driver["r"]
    threads_m = driver["threads_m"]
    threads_n = driver["threads_n"]
    workers = threads_m * threads_n
    sa_stride = q * p * INT8_BYTES * SA_DOUBLE_BUFFER_FACTOR * threads_n
    return {
        "caller_allocates_buffers": True,
        "sa": {
            "per_worker_stride_bytes": sa_stride,
            "candidate_bytes": sa_stride * workers,
            "reference_capacity_bytes": (
                q * max(ALLOWED_P) * INT8_BYTES * SA_DOUBLE_BUFFER_FACTOR *
                threads_n * workers
            ),
        },
        "sb": {
            "candidate_reference_allocation_bytes": (
                q * r * INT8_BYTES * SB_REFERENCE_ALLOCATION_FACTOR
            ),
            "reference_capacity_bytes": (
                q * max(ALLOWED_R) * INT8_BYTES * SB_REFERENCE_ALLOCATION_FACTOR
            ),
        },
    }


def needs_b_ready_reset(config: Dict[str, Any]) -> bool:
    """Whether a call can reuse sb across multiple N/K blocks."""
    driver = config["driver"]
    # The public ABI accepts the target family of shapes, not only the subset
    # listed in a particular tuning run.  Any reduced R/Q therefore needs the
    # reset protocol even when a config lists just one representative shape.
    return driver["r"] < MAX_TARGET_N or driver["q"] < EXPECTED_K


def b_ready_template_values(config: Dict[str, Any]) -> Dict[str, str]:
    if not needs_b_ready_reset(config):
        return {
            "B_READY_PREPARE": "#pragma omp barrier",
            "B_READY_PUBLISH_FLUSH": "",
            "B_READY_POLL_FLUSH": "",
        }
    return {
        "B_READY_PREPARE": (
            "            /* The first barrier prevents a fast M worker from clearing a\n"
            "             * prior block's ready flag while a slower worker still polls it.\n"
            "             * The second makes every new-block flag visibly empty before any\n"
            "             * worker starts asynchronous two-phase B packing. */\n"
            "#pragma omp barrier\n"
            "            if (mask & GEMM_PB_MASK) {\n"
            "                bufferB[mypos] = NULL;\n"
            "            }\n"
            "#pragma omp barrier"
        ),
        "B_READY_PUBLISH_FLUSH": "#pragma omp flush",
        "B_READY_POLL_FLUSH": "#pragma omp flush",
    }


def render(config: Dict[str, Any]) -> str:
    driver = config["driver"]
    abi = config["abi"]
    values = {
        "EXPORT": abi["export"],
        "KERNEL": abi["kernel"],
        "THREADS_M": str(driver["threads_m"]),
        "THREADS_N": str(driver["threads_n"]),
        "P": str(driver["p"]),
        "Q": str(driver["q"]),
        "R": str(driver["r"]),
        "JBLOCK": str(driver["jblock"]),
        "REGION_ALIGN": str(driver["region_align"]),
    }
    values.update(b_ready_template_values(config))
    return Template(DRIVER_TEMPLATE.read_text(encoding="ascii")).substitute(values)


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


def build_manifest(config_path: Path, config: Dict[str, Any], reference_root: str) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "generator": str(Path(__file__).relative_to(MODULE_DIR.parent.parent)),
        "candidate_id": candidate_id(config),
        "reference_root": reference_root,
        "config_sha256": sha256(config_path),
        "driver_template_sha256": sha256(DRIVER_TEMPLATE),
        "makefile_template_sha256": sha256(MAKEFILE_TEMPLATE),
        "expected_dynamic_export": config["abi"]["export"],
        "kernel_sources_are_unmodified": True,
        "target_shapes": config["shapes"],
        "driver": config["driver"],
        "driver_synchronization": (
            {
                "b_ready_state": "reset_before_each_nk_block",
                "block_entry_barriers": 2,
                "publish_flush": True,
                "poll_flush": True,
            }
            if needs_b_ready_reset(config)
            else {
                "b_ready_state": "reference_single_nk_block",
                "block_entry_barriers": 1,
                "publish_flush": False,
                "poll_flush": False,
            }
        ),
        "buffer_contract": buffer_contract(config),
    }


def write_bundle(
    config: Dict[str, Any],
    output: Path,
    reference_root: str,
    force: bool,
) -> Dict[str, Any]:
    validate_config(config)
    if output.exists() and any(output.iterdir()) and not force:
        raise ValueError("output exists and is not empty; pass --force to update it")
    output.mkdir(parents=True, exist_ok=True)

    (output / "cblas_gemm_s8s8s32_autogemm.c").write_text(
        render(config), encoding="ascii"
    )
    (output / "Makefile").write_text(
        MAKEFILE_TEMPLATE.read_text(encoding="ascii"), encoding="ascii"
    )
    write_json(output / "config.json", config)
    manifest = build_manifest(output / "config.json", config, reference_root)
    write_json(output / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="backend JSON configuration")
    parser.add_argument("--output", type=Path, required=True,
                        help="directory for the generated source and Makefile")
    parser.add_argument("--reference-root", default="/path/to/int8gemm-kblas",
                        help="recorded reference root; make may override REF_ROOT")
    parser.add_argument("--force", action="store_true",
                        help="allow writing into an existing output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    validate_config(config)

    output = args.output.resolve()
    try:
        write_bundle(config, output, args.reference_root, args.force)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
