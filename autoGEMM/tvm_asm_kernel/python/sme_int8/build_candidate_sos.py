#!/usr/bin/env python3
"""Generate and build INT8 SME candidate shared libraries only.

This command deliberately never starts ``test_unigemm`` and never refers to
the test project's source tree.  It emits one generated CBLAS driver per
P/R/thread-group candidate declared in ``baseline_config.json``, compiles the
validated SME sources bundled with autoGEMM, and publishes a flat
``libraries/*.so`` directory for a separate test step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from generate_candidates import candidate_config, fixed_driver_fields, selected_values
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


MODULE_DIR = Path(__file__).resolve().parent
GENERATED_LIBRARY_NAME = "libint8gemm_autogemm.so"
MAX_CAPTURE_BYTES = 128 * 1024


def tail(text: str) -> str:
    """Keep logs useful without allowing a failed compiler to fill the disk."""
    if len(text) <= MAX_CAPTURE_BYTES:
        return text
    return "...[truncated]...\n" + text[-MAX_CAPTURE_BYTES:]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_export(library: Path) -> None:
    """Require the ABI symbol that the existing test binary resolves."""
    try:
        result = subprocess.run(
            ["nm", "-D", str(library)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )
    except OSError as error:
        raise RuntimeError("无法执行 nm -D: %s" % error) from error
    if result.returncode != 0:
        raise RuntimeError("nm -D 失败: %s" % tail(result.stderr))
    for line in result.stdout.splitlines():
        fields = line.split()
        if (
            len(fields) >= 3
            and fields[-1] == "cblas_gemm_s8s8s32"
            and fields[-2] in {"T", "t"}
        ):
            return
    raise RuntimeError("候选库没有导出 T cblas_gemm_s8s8s32")


def make_build_command(
    bundle: Path,
    reference_root: Path,
    compiler: str | None,
) -> List[str]:
    """Build one self-contained source bundle without starting the test binary."""
    command = [
        "make",
        "-C",
        str(bundle),
        "REF_ROOT=%s" % reference_root,
    ]
    if compiler:
        command.append("SME_CC=%s" % compiler)
    command.extend(["all", "check"])
    return command


def run_command(command: Sequence[str], timeout: float) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=timeout,
        )
        return {
            "command": list(command),
            "returncode": result.returncode,
            "stdout": tail(result.stdout),
            "stderr": tail(result.stderr),
            "wall_seconds": time.monotonic() - started,
        }
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        error_output = error.stderr or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        if isinstance(error_output, bytes):
            error_output = error_output.decode(errors="replace")
        return {
            "command": list(command),
            "returncode": None,
            "stdout": tail(output),
            "stderr": tail(error_output),
            "wall_seconds": time.monotonic() - started,
            "timeout": timeout,
        }
    except OSError as error:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
            "wall_seconds": time.monotonic() - started,
        }


def write_log(path: Path, result: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = result.get("command", [])
    path.write_text(
        "# command\n%s\n\n# returncode\n%s\n\n# stdout\n%s\n\n# stderr\n%s\n" % (
            shlex.join(command),
            result.get("returncode"),
            result.get("stdout", ""),
            result.get("stderr", ""),
        ),
        encoding="utf-8",
    )


def copy_atomically(source: Path, destination: Path) -> None:
    """Publish a flat candidate artifact without exposing a partial .so."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        ".%s.autogemm-tmp-%d" % (destination.name, os.getpid())
    )
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def build_index(
    config_path: Path,
    base: Dict[str, Any],
    reference_root: Path,
    p_values: Sequence[int],
    r_values: Sequence[int],
    output: Path,
    force: bool,
) -> Dict[str, Any]:
    """Render bundles for every configured P/R/thread-group candidate."""
    configured = configured_search_space(base)
    candidates: List[Dict[str, Any]] = []
    for p in p_values:
        for r in r_values:
            for threads_m, threads_n in configured["thread_groups"]:
                config = candidate_config(base, p, r, threads_m, threads_n)
                identifier = candidate_id(config)
                bundle = output / identifier
                manifest = write_bundle(config, bundle, str(reference_root), force)
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
        "reference_root": str(reference_root),
        "candidate_count": len(candidates),
        "fixed_driver": fixed_driver_fields(base),
        "search_space": search_space,
        "candidates": candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="SME backend JSON configuration")
    parser.add_argument("--output", type=Path, required=True,
                        help="candidate source, build log, and .so output directory")
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="int8gemm-kblas root containing int8_gemm.h")
    parser.add_argument("--sme-cc", default=None,
                        help="optional BiSheng-clang path passed to generated Makefiles")
    parser.add_argument("--p-values", default=None,
                        help="optional comma-separated subset of search_space.p")
    parser.add_argument("--r-values", default=None,
                        help="optional comma-separated subset of search_space.r")
    parser.add_argument("--build-timeout", type=float, default=3600.0,
                        help="maximum seconds for each candidate Make invocation")
    parser.add_argument("--force", action="store_true",
                        help="allow refreshing an existing generated output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    reference_root = args.reference_root.resolve()
    config_path = args.config.resolve()

    try:
        if args.build_timeout <= 0:
            raise ValueError("--build-timeout must be greater than zero")
        if not reference_root.is_dir():
            raise ValueError("reference root is not a directory: %s" % reference_root)
        if output.exists() and not output.is_dir():
            raise ValueError("output exists and is not a directory: %s" % output)
        if output.exists() and any(output.iterdir()) and not args.force:
            raise ValueError("output exists and is not empty; pass --force to update it")
        output.mkdir(parents=True, exist_ok=True)

        base = read_json(config_path)
        validate_config(base)
        configured = configured_search_space(base)
        p_values = selected_values(args.p_values, configured["p"], "--p-values")
        r_values = selected_values(args.r_values, configured["r"], "--r-values")
        index = build_index(
            config_path, base, reference_root, p_values, r_values, output, args.force
        )
        write_json(output / "candidates.json", index)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print("候选库生成启动失败：%s" % error, file=sys.stderr)
        return 2

    summary: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "sme_int8_candidate_build",
        "test_binary_was_not_started": True,
        "reference_root": str(reference_root),
        "kernel_source_mode": "autogemm_embedded_assembly",
        "compiler": args.sme_cc,
        "output": str(output),
        "libraries_dir": str(output / "libraries"),
        "candidates": [],
    }
    failures = 0
    for candidate in index["candidates"]:
        identifier = candidate["candidate_id"]
        bundle = output / candidate["bundle_dir"]
        library = bundle / GENERATED_LIBRARY_NAME
        flat_library = output / "libraries" / (identifier + ".so")
        command = make_build_command(bundle, reference_root, args.sme_cc)
        print("构建候选 %s" % identifier, flush=True)
        result = run_command(command, args.build_timeout)
        log = output / "logs" / (identifier + ".build.log")
        write_log(log, result)
        entry: Dict[str, Any] = {
            "candidate_id": identifier,
            "bundle_dir": str(bundle),
            "command": result["command"],
            "build_log": str(log),
            "returncode": result["returncode"],
            "wall_seconds": result["wall_seconds"],
        }
        if result.get("returncode") != 0:
            entry["status"] = "build_failed"
            failures += 1
            print("  构建失败，日志：%s" % log, file=sys.stderr, flush=True)
        else:
            try:
                if not library.is_file():
                    raise RuntimeError("Make 成功但候选库不存在: %s" % library)
                check_export(library)
                copy_atomically(library, flat_library)
                entry.update({
                    "status": "built",
                    "bundle_library": str(library),
                    "library": str(flat_library),
                    "library_sha256": sha256_file(flat_library),
                })
                print("  已生成：%s" % flat_library, flush=True)
            except (OSError, RuntimeError) as error:
                entry.update({"status": "artifact_failed", "error": str(error)})
                failures += 1
                print("  候选工件检查失败：%s" % error, file=sys.stderr, flush=True)
        summary["candidates"].append(entry)

    summary["successful_count"] = sum(
        entry["status"] == "built" for entry in summary["candidates"]
    )
    summary["failed_count"] = failures
    summary["status"] = "complete" if not failures else "completed_with_failures"
    summary["finished_at_unix"] = time.time()
    write_json(output / "build_summary.json", summary)
    print("构建汇总：%s" % (output / "build_summary.json"))
    print("候选 .so 目录：%s" % (output / "libraries"))
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
