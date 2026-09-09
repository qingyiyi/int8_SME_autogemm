#!/usr/bin/env python3
"""Render constrained INT8 SME CBLAS driver candidates and build bundles.

The generated driver owns the public CBLAS ABI.  The packers and SME kernel
remain source files in the supplied reference root and are compiled unchanged
by the emitted Makefile.

All candidate-search knobs live in ``baseline_config.json``.  In particular,
``search_space.p``, ``search_space.r``, and the paired
``search_space.thread_groups`` determine the candidates emitted by the batch
builders.
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

# These are implementation constraints of the current SME driver, rather than
# hidden tuning choices.  The tuneable tile and thread-group values are read
# from baseline_config.json.
EXPECTED_K = 2048
MULTIPLE = 2048
PACK_ALIGNMENT = 16
KERNEL_TILE_ALIGNMENT = 4
INT8_BYTES = 1
THREAD_TOTAL = 32
SUPPORTED_THREAD_GROUPS = ((32, 1), (16, 2))
REFERENCE_B_PACKING = "reference_full_panel_barrier"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="ascii") as source:
        return json.load(source)


def as_shape(value: Any) -> Tuple[int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(isinstance(x, int) and not isinstance(x, bool) for x in value)
    ):
        raise ValueError("each shape must be an integer [M, N, K] list")
    return value[0], value[1], value[2]


def as_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("%s must be an integer" % field)
    return value


def as_positive_int(value: Any, field: str) -> int:
    result = as_int(value, field)
    if result <= 0:
        raise ValueError("%s must be greater than zero" % field)
    return result


def tile_value(value: Any, field: str) -> int:
    """Validate a P/R packing-panel size without imposing a hidden whitelist."""
    result = as_positive_int(value, field)
    if result % PACK_ALIGNMENT:
        raise ValueError("%s must be a multiple of %d" % (field, PACK_ALIGNMENT))
    return result


def integer_list(value: Any, field: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("%s must be a non-empty integer list" % field)
    result = tuple(tile_value(item, "%s[%d]" % (field, index)) for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ValueError("%s must not contain duplicates" % field)
    return result


def thread_group(value: Any, field: str) -> Tuple[int, int]:
    """Read one indivisible M/N worker-grid configuration.

    ``threads_m`` and ``threads_n`` cannot be independently swept: changing
    their Cartesian product would create unsupported worker grids.  The
    current driver supports exactly 32x1 and 16x2, and both contain 32 OpenMP
    workers in total.
    """
    if not isinstance(value, dict):
        raise ValueError("%s must be an object with threads_m and threads_n" % field)
    threads_m = as_positive_int(value.get("threads_m"), "%s.threads_m" % field)
    threads_n = as_positive_int(value.get("threads_n"), "%s.threads_n" % field)
    if threads_m * threads_n != THREAD_TOTAL:
        raise ValueError(
            "%s must contain exactly %d worker threads; got %d x %d" %
            (field, THREAD_TOTAL, threads_m, threads_n)
        )
    group = (threads_m, threads_n)
    if group not in SUPPORTED_THREAD_GROUPS:
        allowed = ", ".join("%d x %d" % pair for pair in SUPPORTED_THREAD_GROUPS)
        raise ValueError("%s must be one of (%s); got %d x %d" % (
            field, allowed, threads_m, threads_n
        ))
    return group


def configured_search_space(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the validated, normalized candidate search-space from JSON."""
    raw = config.get("search_space")
    if not isinstance(raw, dict):
        raise ValueError("search_space must be an object")

    p_values = integer_list(raw.get("p"), "search_space.p")
    r_values = integer_list(raw.get("r"), "search_space.r")
    groups_raw = raw.get("thread_groups")
    if not isinstance(groups_raw, list) or not groups_raw:
        raise ValueError("search_space.thread_groups must be a non-empty list")
    groups = tuple(
        thread_group(value, "search_space.thread_groups[%d]" % index)
        for index, value in enumerate(groups_raw)
    )
    if len(set(groups)) != len(groups):
        raise ValueError("search_space.thread_groups must not contain duplicates")
    return {
        "p": p_values,
        "r": r_values,
        "thread_groups": groups,
    }


def search_space_json(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the normalized search space in JSON-serializable form."""
    space = configured_search_space(config)
    return {
        "p": list(space["p"]),
        "r": list(space["r"]),
        "thread_groups": [
            {"threads_m": threads_m, "threads_n": threads_n}
            for threads_m, threads_n in space["thread_groups"]
        ],
    }


def validate_driver(driver: Dict[str, Any], search_space: Dict[str, Any]) -> None:
    if not isinstance(driver, dict):
        raise ValueError("driver must be an object")

    threads_m = as_positive_int(driver.get("threads_m"), "driver.threads_m")
    threads_n = as_positive_int(driver.get("threads_n"), "driver.threads_n")
    group = thread_group(
        {"threads_m": threads_m, "threads_n": threads_n},
        "driver thread group",
    )
    if group not in search_space["thread_groups"]:
        raise ValueError(
            "driver thread group %d x %d is not present in search_space.thread_groups" % group
        )

    p = tile_value(driver.get("p"), "driver.p")
    r = tile_value(driver.get("r"), "driver.r")
    if p not in search_space["p"]:
        raise ValueError("driver.p must be present in search_space.p; got %d" % p)
    if r not in search_space["r"]:
        raise ValueError("driver.r must be present in search_space.r; got %d" % r)

    q = tile_value(driver.get("q"), "driver.q")
    if q != EXPECTED_K:
        raise ValueError(
            "this SME backend currently requires driver.q=%d; got %d" % (EXPECTED_K, q)
        )

    jblock = as_positive_int(driver.get("jblock"), "driver.jblock")
    if jblock % KERNEL_TILE_ALIGNMENT:
        raise ValueError(
            "driver.jblock must be a multiple of %d" % KERNEL_TILE_ALIGNMENT
        )

    region_align = as_positive_int(driver.get("region_align"), "driver.region_align")
    if region_align % KERNEL_TILE_ALIGNMENT:
        raise ValueError(
            "driver.region_align must be a multiple of %d" % KERNEL_TILE_ALIGNMENT
        )

    if driver.get("b_packing") != REFERENCE_B_PACKING:
        raise ValueError(
            "this driver template requires driver.b_packing=%r; got %r" %
            (REFERENCE_B_PACKING, driver.get("b_packing"))
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

    search_space = configured_search_space(config)
    driver = config.get("driver")
    validate_driver(driver, search_space)

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
    the caller and remote sweep runner rather than a runtime check.  ``sa``
    has one Q*P slice per OpenMP worker.  ``sb`` has one Q*R panel per N-thread
    group; this second factor is required for the 16x2 thread grid.

    ``reference_capacity_bytes`` is the conservative buffer capacity needed to
    test every candidate currently listed in ``search_space``.
    """
    driver = config["driver"]
    search_space = configured_search_space(config)
    q = driver["q"]
    p = driver["p"]
    r = driver["r"]
    threads_m = driver["threads_m"]
    threads_n = driver["threads_n"]
    workers = threads_m * threads_n
    sa_stride = q * p * INT8_BYTES
    return {
        "caller_allocates_buffers": True,
        "sa": {
            "per_worker_stride_bytes": sa_stride,
            "required_bytes": sa_stride * workers,
            "reference_capacity_bytes": (
                q * max(search_space["p"]) * THREAD_TOTAL * INT8_BYTES
            ),
        },
        "sb": {
            "required_bytes": q * r * threads_n * INT8_BYTES,
            "reference_capacity_bytes": (
                q
                * max(search_space["r"])
                * max(group[1] for group in search_space["thread_groups"])
                * INT8_BYTES
            ),
        },
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
        "driver_synchronization": {
            "protocol": config["driver"]["b_packing"],
            "barriers_per_nk_block": 2,
            "b_ready_snoop": False,
            "ready_state_reset": False,
            "openmp_flush": False,
            "b_packing_compute_overlap": False,
        },
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
