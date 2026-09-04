#!/usr/bin/env python3
"""Build and measure generated INT8 SME candidates on the target machine.

This is intentionally a thin remote orchestration layer.  It does not modify
the reference project or the SME assembly objects; each candidate gets its own
build and result log so a failed build or measurement cannot contaminate the
other candidates.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


MODULE_DIR = Path(__file__).resolve().parent
EVALUATOR = MODULE_DIR / "evaluate_candidate.py"
MAX_CAPTURE_BYTES = 128 * 1024
PROCESS_OVERHEAD_SECONDS = 60.0


def tail(text: str) -> str:
    if len(text) <= MAX_CAPTURE_BYTES:
        return text
    return "...[truncated]\n" + text[-MAX_CAPTURE_BYTES:]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="ascii") as source:
        return json.load(source)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


def parse_shape(value: str) -> Tuple[int, int, int]:
    fields = value.split(",")
    if len(fields) != 3:
        raise ValueError("shape must be M,N,K, got %r" % value)
    try:
        shape = tuple(int(field.strip()) for field in fields)
    except ValueError as error:
        raise ValueError("shape must contain three integers, got %r" % value) from error
    if any(dimension <= 0 for dimension in shape):
        raise ValueError("shape dimensions must be positive, got %r" % (shape,))
    return shape  # type: ignore[return-value]


def run_command(
    command: Sequence[str],
    env: Optional[Dict[str, str]],
    timeout: float,
) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            env=env,
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


def valid_records(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get("status") == "valid":
                records.append(value)
    return records


def record_for_shape(records: Iterable[Dict[str, Any]], shape: Tuple[int, int, int]) -> Optional[Dict[str, Any]]:
    for record in records:
        if tuple(record.get("shape", ())) == shape:
            return record
    return None


def rank_quick_results(results: Iterable[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    measured = [
        result for result in results
        if result.get("quick") and result["quick"].get("gflops") is not None
    ]
    measured.sort(key=lambda result: float(result["quick"]["gflops"]), reverse=True)
    return measured[:top_k]


def candidate_entries(index: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = index.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates.json has no candidates")
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate.get("candidate_id"):
            raise ValueError("invalid candidate entry in candidates.json")
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-root", type=Path, required=True,
                        help="directory produced by generate_candidates.py")
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="int8gemm-kblas source/reference root")
    parser.add_argument("--kernel-objects", type=Path, nargs="+", required=True,
                        help="validated SME/packer .o files to link into every candidate")
    parser.add_argument("--test-bin", type=Path, required=True,
                        help="remote int8/test_unigemm executable")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="directory for build logs and JSONL measurements")
    parser.add_argument("--quick-shape", default="2048,2048,2048",
                        help="first-pass shape, M,N,K; must be in each candidate config")
    parser.add_argument("--top-k", type=int, default=3,
                        help="number of quick winners to measure on all configured shapes")
    parser.add_argument("--build-timeout", type=float, default=3600.0,
                        help="maximum seconds for one candidate build")
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="maximum seconds per shape in the evaluator")
    parser.add_argument("--skip-build", action="store_true",
                        help="reuse existing .so files; still records and measures candidates")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.candidates_root.resolve()
    reference_root = args.reference_root.resolve()
    test_bin = args.test_bin.resolve()
    results_dir = args.results_dir.resolve()
    kernel_objects = [path.resolve() for path in args.kernel_objects]
    quick_shape = parse_shape(args.quick_shape)

    if args.top_k <= 0:
        raise SystemExit("--top-k must be greater than zero")
    if args.build_timeout <= 0 or args.timeout <= 0:
        raise SystemExit("timeouts must be greater than zero")
    index_path = root / "candidates.json"
    if not index_path.is_file():
        raise SystemExit("missing candidates index: %s" % index_path)
    if not reference_root.is_dir():
        raise SystemExit("reference root is not a directory: %s" % reference_root)
    if not test_bin.is_file() or not os.access(test_bin, os.X_OK):
        raise SystemExit("test binary is missing or not executable: %s" % test_bin)
    missing_objects = [str(path) for path in kernel_objects if not path.is_file()]
    if missing_objects:
        raise SystemExit("kernel object does not exist: %s" % ", ".join(missing_objects))

    try:
        index = read_json(index_path)
        candidates = candidate_entries(index)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error

    if results_dir.exists() and any(results_dir.iterdir()):
        raise SystemExit(
            "results directory must be empty to avoid mixing sweep runs: %s" % results_dir
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "sweep_summary.json"
    run_inputs = {
        "candidates_root": str(root),
        "reference_root": str(reference_root),
        "kernel_objects": [str(path) for path in kernel_objects],
        "test_bin": str(test_bin),
        "quick_shape": list(quick_shape),
        "top_k": args.top_k,
        "build_timeout": args.build_timeout,
        "measurement_timeout": args.timeout,
        "skip_build": args.skip_build,
    }
    write_json(results_dir / "sweep_inputs.json", run_inputs)
    quick_results: List[Dict[str, Any]] = []
    for candidate in candidates:
        identifier = str(candidate["candidate_id"])
        bundle = root / str(candidate.get("bundle_dir", identifier))
        config_path = bundle / "config.json"
        library = bundle / "libint8gemm_autogemm.so"
        entry: Dict[str, Any] = {
            "candidate_id": identifier,
            "bundle_dir": str(bundle),
            "config": str(config_path),
            "library": str(library),
            "kernel_objects": run_inputs["kernel_objects"],
        }

        if not config_path.is_file():
            entry["status"] = "missing_config"
            quick_results.append(entry)
            continue

        if not args.skip_build:
            build_env = os.environ.copy()
            build_env["KERNEL_OBJECTS"] = " ".join(str(path) for path in kernel_objects)
            build_command = [
                "make", "-C", str(bundle), "REF_ROOT=%s" % reference_root,
                "all", "check",
            ]
            build = run_command(build_command, build_env, args.build_timeout)
            entry["build"] = build
            (results_dir / (identifier + ".build.log")).write_text(
                build["stdout"] + "\n--- stderr ---\n" + build["stderr"], encoding="utf-8"
            )
            if build.get("returncode") != 0 or not library.is_file():
                entry["status"] = "build_failed"
                quick_results.append(entry)
                continue
        elif not library.is_file():
            entry["status"] = "missing_library"
            quick_results.append(entry)
            continue

        quick_path = results_dir / (identifier + ".quick.valid.jsonl")
        evaluator_command = [
            sys.executable, str(EVALUATOR),
            "--library", str(library),
            "--test-bin", str(test_bin),
            "--config", str(config_path),
            "--results", str(quick_path),
            "--mode", "kblas",
            "--shape", "%d,%d,%d" % quick_shape,
            "--timeout", str(args.timeout),
        ]
        measured = run_command(
            evaluator_command, os.environ.copy(), args.timeout + PROCESS_OVERHEAD_SECONDS
        )
        entry["quick_evaluation"] = measured
        (results_dir / (identifier + ".quick.log")).write_text(
            measured["stdout"] + "\n--- stderr ---\n" + measured["stderr"], encoding="utf-8"
        )
        quick_record = record_for_shape(valid_records(quick_path), quick_shape)
        if quick_record is None:
            entry["status"] = "quick_measurement_failed"
        else:
            entry["status"] = "quick_measured"
            entry["quick"] = {
                "shape": quick_record.get("shape"),
                "latency_seconds": quick_record.get("latency_seconds"),
                "gflops": quick_record.get("gflops"),
                "results": str(quick_path),
            }
        quick_results.append(entry)
        write_json(summary_path, {
            "schema_version": 1,
            "phase": "quick",
            "inputs": run_inputs,
            "candidates": quick_results,
        })

    winners = rank_quick_results(quick_results, args.top_k)
    full_results: List[Dict[str, Any]] = []
    for entry in winners:
        identifier = str(entry["candidate_id"])
        library = Path(entry["library"])
        config_path = Path(entry["config"])
        config = read_json(config_path)
        expected_shapes = [tuple(shape) for shape in config["shapes"]]
        full_path = results_dir / (identifier + ".full.valid.jsonl")
        evaluator_command = [
            sys.executable, str(EVALUATOR),
            "--library", str(library),
            "--test-bin", str(test_bin),
            "--config", str(config_path),
            "--results", str(full_path),
            "--mode", "kblas",
            "--timeout", str(args.timeout),
        ]
        measured = run_command(
            evaluator_command,
            os.environ.copy(),
            args.timeout * len(expected_shapes) + PROCESS_OVERHEAD_SECONDS,
        )
        records = valid_records(full_path)
        measured_shapes = {tuple(record.get("shape", ())) for record in records}
        missing_shapes = [list(shape) for shape in expected_shapes if shape not in measured_shapes]
        full_entry = {
            "candidate_id": identifier,
            "evaluation": measured,
            "results": str(full_path),
            "expected_shape_count": len(expected_shapes),
            "measured_shape_count": len(measured_shapes),
            "missing_shapes": missing_shapes,
            "status": "full_measured" if not missing_shapes else "full_measurement_incomplete",
            "records": records,
        }
        full_results.append(full_entry)
        (results_dir / (identifier + ".full.log")).write_text(
            measured["stdout"] + "\n--- stderr ---\n" + measured["stderr"], encoding="utf-8"
        )

    write_json(summary_path, {
        "schema_version": 1,
        "phase": "complete",
        "inputs": run_inputs,
        "candidates": quick_results,
        "full_evaluations": full_results,
    })
    print("quickly measured %d candidates; full evaluations: %d" % (len(quick_results), len(full_results)))
    return 0 if any(entry.get("quick") for entry in quick_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
