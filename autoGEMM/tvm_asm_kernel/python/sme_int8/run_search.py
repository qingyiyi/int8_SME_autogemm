#!/usr/bin/env python3
"""Generate, verify, benchmark, and publish the best INT8 SME candidate.

This is the normal remote entry point for the constrained SME backend.  It
keeps the existing SME kernel and packer objects unchanged, generates all P/R
candidate drivers, and uses the existing test program through ``LD_PRELOAD``.

The default is deliberately correctness-gated:

1. generate all candidates;
2. build every candidate and check the exported CBLAS ABI;
3. verify every configured shape for every candidate;
4. measure every configured shape only after that candidate passes verification;
5. rank fully measured candidates and copy the winner to ``best/``.

``--skip-verify`` exists only for local debugging.  It never publishes a
``best/libint8gemm_autogemm.so`` because an unverified measurement is not a
final result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from generate_driver import DEFAULT_CONFIG, as_shape, read_json, sha256, validate_config


MODULE_DIR = Path(__file__).resolve().parent
GENERATOR = MODULE_DIR / "generate_candidates.py"
EVALUATOR = MODULE_DIR / "evaluate_candidate.py"
LIBRARY_NAME = "libint8gemm_autogemm.so"
EXPECTED_EXPORT = "cblas_gemm_s8s8s32"
MAX_CAPTURE_BYTES = 128 * 1024
PROCESS_OVERHEAD_SECONDS = 60.0
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def tail(text: str) -> str:
    """Keep result JSON bounded while retaining the end of diagnostics."""
    if len(text) <= MAX_CAPTURE_BYTES:
        return text
    return "...[truncated]\n" + text[-MAX_CAPTURE_BYTES:]


def run_command(
    command: Sequence[str],
    env: Optional[Dict[str, str]],
    timeout: float,
) -> Dict[str, Any]:
    """Run a remote-side command and return serializable diagnostics."""
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


def command_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep summary JSON small; full stdout/stderr lives in the log file."""
    summary = {
        "command": result.get("command", []),
        "returncode": result.get("returncode"),
        "wall_seconds": result.get("wall_seconds"),
    }
    if "timeout" in result:
        summary["timeout"] = result["timeout"]
    return summary


def write_log(path: Path, result: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        result.get("stdout", "") + "\n--- stderr ---\n" + result.get("stderr", ""),
        encoding="utf-8",
    )


def atomic_write_json(path: Path, value: Any) -> None:
    """Write a complete JSON document even if the process is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def candidate_entries(index: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = index.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates.json has no candidates")
    checked: List[Dict[str, Any]] = []
    identifiers = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("invalid candidate entry in candidates.json")
        identifier = candidate.get("candidate_id")
        bundle = candidate.get("bundle_dir", identifier)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("candidate entry has no candidate_id")
        if not isinstance(bundle, str) or not bundle:
            raise ValueError("candidate entry has no bundle_dir")
        if not RUN_ID_RE.fullmatch(identifier):
            raise ValueError("candidate_id contains unsafe characters: %s" % identifier)
        if identifier in identifiers:
            raise ValueError("duplicate candidate_id in candidates.json: %s" % identifier)
        identifiers.add(identifier)
        checked.append(candidate)
    return checked


def path_inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def dynamic_export_check(library: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Perform the same public-symbol check as the generated Makefile."""
    if not library.is_file():
        return {
            "passed": False,
            "reason": "candidate library does not exist",
        }, {"command": [], "stdout": "", "stderr": ""}
    result = run_command(["nm", "-D", str(library)], os.environ.copy(), 60.0)
    check: Dict[str, Any] = {
        "passed": False,
        "command": command_summary(result),
    }
    if result.get("returncode") != 0:
        check["reason"] = "nm -D failed"
        return check, result
    for line in result.get("stdout", "").splitlines():
        fields = line.split()
        if (
            len(fields) >= 3
            and fields[-1] == EXPECTED_EXPORT
            and fields[-2] in {"T", "t"}
        ):
            check["passed"] = True
            check["symbol_type"] = fields[-2]
            return check, result
    check["reason"] = "missing dynamic export %s" % EXPECTED_EXPORT
    return check, result


def summary_for_measurements(
    records: Iterable[Dict[str, Any]],
    shapes: Sequence[Tuple[int, int, int]],
    require_correctness: bool,
) -> Dict[str, Any]:
    """Validate that exactly one usable measurement exists for every shape."""
    expected = set(shapes)
    by_shape: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for record in records:
        if record.get("status") != "valid":
            continue
        raw_shape = record.get("shape")
        if not isinstance(raw_shape, list) or len(raw_shape) != 3:
            continue
        try:
            shape = tuple(int(value) for value in raw_shape)
        except (TypeError, ValueError):
            continue
        if shape in expected:
            by_shape[shape] = record  # Files are fresh per run; last one is safest on resume.

    missing = [list(shape) for shape in shapes if shape not in by_shape]
    wrong_correctness = [
        list(shape)
        for shape in shapes
        if shape in by_shape and by_shape[shape].get("correctness") != "passed"
    ]
    bad_metrics: List[List[int]] = []
    for shape in shapes:
        record = by_shape.get(shape)
        if record is None:
            continue
        try:
            latency = float(record["latency_seconds"])
            gflops = float(record["gflops"])
        except (KeyError, TypeError, ValueError):
            bad_metrics.append(list(shape))
            continue
        if not math.isfinite(latency) or not math.isfinite(gflops) or latency <= 0 or gflops <= 0:
            bad_metrics.append(list(shape))

    ordered_records = [by_shape[shape] for shape in shapes if shape in by_shape]
    complete = not missing and not bad_metrics
    correctness_passed = not wrong_correctness
    return {
        "expected_shape_count": len(shapes),
        "measured_shape_count": len(by_shape),
        "missing_shapes": missing,
        "wrong_correctness_shapes": wrong_correctness,
        "invalid_metric_shapes": bad_metrics,
        "complete": complete,
        "correctness_passed": correctness_passed if require_correctness else False,
        "records": ordered_records,
    }


def metrics_for_records(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Return both supported global objectives from full shape measurements."""
    if not records:
        raise ValueError("cannot calculate metrics without measurements")
    latencies = [float(record["latency_seconds"]) for record in records]
    gflops_values = [float(record["gflops"]) for record in records]
    if any(value <= 0 or not math.isfinite(value) for value in latencies + gflops_values):
        raise ValueError("measurements must contain finite positive latency and gflops")
    return {
        "total_latency_seconds": sum(latencies),
        "geomean_gflops": math.exp(sum(math.log(value) for value in gflops_values) / len(gflops_values)),
    }


def rank_qualified_candidates(
    candidates: Iterable[Dict[str, Any]],
    objective: str,
) -> List[Dict[str, Any]]:
    """Rank only fully verified, fully measured candidates deterministically."""
    qualified = [
        candidate for candidate in candidates
        if candidate.get("status") == "qualified" and isinstance(candidate.get("metrics"), dict)
    ]
    if objective == "geomean_gflops":
        qualified.sort(
            key=lambda item: (-float(item["metrics"]["geomean_gflops"]), str(item["candidate_id"]))
        )
        objective_key = "geomean_gflops"
    elif objective == "total_latency":
        qualified.sort(
            key=lambda item: (float(item["metrics"]["total_latency_seconds"]), str(item["candidate_id"]))
        )
        objective_key = "total_latency_seconds"
    else:
        raise ValueError("unsupported objective: %s" % objective)

    ranking: List[Dict[str, Any]] = []
    for rank, candidate in enumerate(qualified, 1):
        metrics = candidate["metrics"]
        ranking.append({
            "rank": rank,
            "candidate_id": candidate["candidate_id"],
            "objective": objective,
            "objective_value": metrics[objective_key],
            "geomean_gflops": metrics["geomean_gflops"],
            "total_latency_seconds": metrics["total_latency_seconds"],
            "library": candidate["library"],
        })
    return ranking


def publish_best_library(source: Path, best_dir: Path) -> Path:
    """Copy a self-contained final .so only after a winner has been selected."""
    if not source.is_file():
        raise ValueError("winner library does not exist: %s" % source)
    best_dir.mkdir(parents=True, exist_ok=True)
    target = best_dir / LIBRARY_NAME
    temporary = best_dir / (".%s.%d.tmp" % (LIBRARY_NAME, os.getpid()))
    try:
        shutil.copy2(source, temporary)
        if sha256(source) != sha256(temporary):
            raise RuntimeError("copied winner library hash does not match source")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return "run-%s-p%d" % (stamp, os.getpid())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="int8gemm-kblas reference/source directory")
    parser.add_argument("--kernel-objects", type=Path, nargs="+", required=True,
                        help="four validated SME kernel/packer .o files")
    parser.add_argument("--test-bin", type=Path, required=True,
                        help="compiled int8/test_unigemm executable")
    parser.add_argument("--output", type=Path, required=True,
                        help="one search output root, containing candidates/runs/best")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="baseline backend JSON containing all target shapes")
    parser.add_argument("--p-values", default=None,
                        help="optional comma-separated P list; default is 64,128,256")
    parser.add_argument("--r-values", default=None,
                        help="optional comma-separated R list; default is 2048,4096,8192")
    parser.add_argument("--mode", default="kblas",
                        help="test_unigemm mode; default is kblas")
    parser.add_argument("--sme-cc", "--compiler", dest="sme_cc", default=None,
                        help="optional BiSheng-clang path passed to generated Makefiles")
    parser.add_argument("--build-timeout", type=float, default=3600.0,
                        help="maximum seconds for one candidate make invocation")
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="maximum seconds for one test process / shape")
    parser.add_argument("--objective", choices=("geomean_gflops", "total_latency"),
                        default="geomean_gflops",
                        help="global ranking objective; default is 9-shape GFLOPS geometric mean")
    parser.add_argument("--run-id", default=None,
                        help="optional simple name for this run under output/runs")
    parser.add_argument("--force", action="store_true",
                        help="allow reusing a non-empty output root; prior runs remain intact")
    parser.add_argument("--skip-generate", action="store_true",
                        help="debug/resume only: reuse output/candidates instead of regenerating")
    parser.add_argument("--skip-build", action="store_true",
                        help="debug/resume only: reuse existing candidate .so files")
    parser.add_argument("--skip-verify", action="store_true",
                        help="debug only: do not verify correctness and never publish a best .so")
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path, List[Path], List[Tuple[int, int, int]]]:
    reference_root = args.reference_root.resolve()
    test_bin = args.test_bin.resolve()
    output_root = args.output.resolve()
    config_path = args.config.resolve()
    kernel_objects = [path.resolve() for path in args.kernel_objects]

    if args.build_timeout <= 0 or args.timeout <= 0:
        raise ValueError("--build-timeout and --timeout must be greater than zero")
    if not reference_root.is_dir():
        raise ValueError("reference root is not a directory: %s" % reference_root)
    if not test_bin.is_file() or not os.access(test_bin, os.X_OK):
        raise ValueError("test binary is missing or not executable: %s" % test_bin)
    missing_objects = [str(path) for path in kernel_objects if not path.is_file()]
    if missing_objects:
        raise ValueError("kernel object does not exist: %s" % ", ".join(missing_objects))
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("output exists and is not a directory: %s" % output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.force:
        raise ValueError("output is not empty; pass --force to reuse it: %s" % output_root)
    if not config_path.is_file():
        raise ValueError("config does not exist: %s" % config_path)

    config = read_json(config_path)
    validate_config(config)
    shapes = [as_shape(value) for value in config["shapes"]]
    if len(set(shapes)) != len(shapes):
        raise ValueError("config shapes must not contain duplicates")
    return reference_root, test_bin, output_root, config_path, kernel_objects, shapes


def classify_evaluation_failure(rejections: Sequence[Dict[str, Any]], verify: bool) -> str:
    if verify and any(record.get("phase") == "correctness" for record in rejections):
        return "correctness_failed"
    return "evaluation_failed"


def run_search(args: argparse.Namespace) -> int:
    (
        reference_root,
        test_bin,
        output_root,
        config_path,
        kernel_objects,
        target_shapes,
    ) = validate_inputs(args)

    run_id = args.run_id or default_run_id()
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("--run-id may contain only letters, digits, '.', '_' and '-'")
    output_root.mkdir(parents=True, exist_ok=True)
    candidates_root = output_root / "candidates"
    run_dir = output_root / "runs" / run_id
    if run_dir.exists():
        raise ValueError("run directory already exists; use a new --run-id: %s" % run_dir)
    logs_dir = run_dir / "logs"
    records_dir = run_dir / "records"
    run_dir.mkdir(parents=True)

    inputs = {
        "reference_root": str(reference_root),
        "kernel_objects": [str(path) for path in kernel_objects],
        "test_bin": str(test_bin),
        "output_root": str(output_root),
        "candidates_root": str(candidates_root),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "target_shapes": [list(shape) for shape in target_shapes],
        "mode": args.mode,
        "sme_cc": args.sme_cc,
        "build_timeout": args.build_timeout,
        "measurement_timeout": args.timeout,
        "objective": args.objective,
        "skip_generate": args.skip_generate,
        "skip_build": args.skip_build,
        "skip_verify": args.skip_verify,
    }
    summary: Dict[str, Any] = {
        "schema_version": 1,
        "workflow": "sme_int8_full_correctness_gated_search",
        "run_id": run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "inputs": inputs,
        "generation": None,
        "candidates": [],
        "ranking": [],
        "winner": None,
    }

    def checkpoint() -> None:
        summary["updated_at_utc"] = utc_now()
        atomic_write_json(run_dir / "summary.json", summary)
        atomic_write_json(output_root / "summary.json", summary)

    checkpoint()

    if args.skip_generate:
        summary["generation"] = {
            "status": "skipped",
            "candidates_index": str(candidates_root / "candidates.json"),
        }
    else:
        generator_command = [
            sys.executable,
            str(GENERATOR),
            "--config", str(config_path),
            "--output", str(candidates_root),
            "--reference-root", str(reference_root),
        ]
        if args.p_values is not None:
            generator_command.extend(["--p-values", args.p_values])
        if args.r_values is not None:
            generator_command.extend(["--r-values", args.r_values])
        if args.force:
            generator_command.append("--force")
        generation = run_command(generator_command, os.environ.copy(), args.build_timeout)
        generation_log = logs_dir / "generate_candidates.log"
        write_log(generation_log, generation)
        summary["generation"] = {
            "status": "passed" if generation.get("returncode") == 0 else "failed",
            "command": command_summary(generation),
            "log": str(generation_log),
            "candidates_index": str(candidates_root / "candidates.json"),
        }
        if generation.get("returncode") != 0:
            summary["status"] = "generation_failed"
            summary["finished_at_utc"] = utc_now()
            checkpoint()
            print("候选生成失败；详见 %s" % generation_log, file=sys.stderr)
            return 1

    index_path = candidates_root / "candidates.json"
    try:
        index = read_json(index_path)
        candidates = candidate_entries(index)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        summary["status"] = "candidate_index_failed"
        summary["candidate_index_error"] = str(error)
        summary["finished_at_utc"] = utc_now()
        checkpoint()
        print("候选索引无效：%s" % error, file=sys.stderr)
        return 1

    summary["candidate_count"] = len(candidates)
    checkpoint()

    for candidate_position, candidate in enumerate(candidates, 1):
        identifier = str(candidate["candidate_id"])
        bundle = candidates_root / str(candidate.get("bundle_dir", identifier))
        config = bundle / "config.json"
        library = bundle / LIBRARY_NAME
        entry: Dict[str, Any] = {
            "candidate_id": identifier,
            "bundle": str(bundle),
            "config": str(config),
            "library": str(library),
            "status": "pending",
        }
        print(
            "[%d/%d] 处理候选 %s" % (candidate_position, len(candidates), identifier),
            flush=True,
        )

        if not path_inside(candidates_root, bundle):
            entry["status"] = "invalid_bundle_path"
            summary["candidates"].append(entry)
            checkpoint()
            continue
        if not config.is_file():
            entry["status"] = "missing_config"
            summary["candidates"].append(entry)
            checkpoint()
            continue
        try:
            candidate_config = read_json(config)
            validate_config(candidate_config)
            candidate_shapes = [as_shape(value) for value in candidate_config["shapes"]]
            if candidate_shapes != target_shapes:
                raise ValueError("candidate target shapes do not match the requested config")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            entry["status"] = "invalid_config"
            entry["config_error"] = str(error)
            summary["candidates"].append(entry)
            checkpoint()
            continue

        if args.skip_build:
            entry["build"] = {"status": "skipped"}
            if not library.is_file():
                entry["status"] = "missing_library"
                summary["candidates"].append(entry)
                checkpoint()
                continue
        else:
            build_env = os.environ.copy()
            build_env["KERNEL_OBJECTS"] = " ".join(str(path) for path in kernel_objects)
            if args.sme_cc:
                build_env["SME_CC"] = args.sme_cc
            build_command = [
                "make", "-C", str(bundle), "REF_ROOT=%s" % reference_root, "all", "check",
            ]
            build = run_command(build_command, build_env, args.build_timeout)
            build_log = logs_dir / (identifier + ".build.log")
            write_log(build_log, build)
            entry["build"] = {
                "status": "passed" if build.get("returncode") == 0 else "failed",
                "command": command_summary(build),
                "log": str(build_log),
            }
            if build.get("returncode") != 0:
                entry["status"] = "build_failed"
                print("  构建失败，跳过。", flush=True)
                summary["candidates"].append(entry)
                checkpoint()
                continue

        abi, abi_result = dynamic_export_check(library)
        abi_log = logs_dir / (identifier + ".abi.log")
        write_log(abi_log, abi_result)
        entry["abi"] = dict(abi, log=str(abi_log))
        if not abi.get("passed"):
            entry["status"] = "abi_failed"
            print("  ABI 检查失败，跳过。", flush=True)
            summary["candidates"].append(entry)
            checkpoint()
            continue

        results_path = records_dir / identifier / "valid_results.jsonl"
        rejection_path = Path(str(results_path) + ".rejects.jsonl")
        evaluator_command = [
            sys.executable,
            str(EVALUATOR),
            "--library", str(library),
            "--test-bin", str(test_bin),
            "--config", str(config),
            "--results", str(results_path),
            "--rejections", str(rejection_path),
            "--mode", args.mode,
            "--timeout", str(args.timeout),
        ]
        if not args.skip_verify:
            evaluator_command.append("--verify")
        evaluator_timeout = args.timeout * len(target_shapes) * (1 if args.skip_verify else 2)
        evaluation = run_command(
            evaluator_command,
            os.environ.copy(),
            evaluator_timeout + PROCESS_OVERHEAD_SECONDS,
        )
        evaluation_log = logs_dir / (identifier + ".evaluation.log")
        write_log(evaluation_log, evaluation)
        measurements = summary_for_measurements(
            read_jsonl(results_path), target_shapes, not args.skip_verify
        )
        rejections = read_jsonl(rejection_path)
        entry["evaluation"] = {
            "command": command_summary(evaluation),
            "log": str(evaluation_log),
            "results": str(results_path),
            "rejections": str(rejection_path),
        }
        entry["measurements"] = measurements

        if evaluation.get("returncode") != 0:
            entry["status"] = classify_evaluation_failure(rejections, not args.skip_verify)
            entry["correctness"] = (
                "failed" if entry["status"] == "correctness_failed" else "unknown"
            )
        elif not measurements["complete"]:
            entry["status"] = "measurement_failed"
            entry["correctness"] = "passed" if not args.skip_verify else "not_checked"
        elif args.skip_verify:
            entry["status"] = "unverified_measured"
            entry["correctness"] = "not_checked"
        elif not measurements["correctness_passed"]:
            # This is defensive: evaluate_candidate.py should not reach the
            # measurement phase when a verified shape is not marked passed.
            entry["status"] = "correctness_failed"
            entry["correctness"] = "failed"
        else:
            try:
                entry["metrics"] = metrics_for_records(measurements["records"])
                entry["status"] = "qualified"
                entry["correctness"] = "passed"
            except (KeyError, TypeError, ValueError) as error:
                entry["status"] = "measurement_failed"
                entry["measurement_error"] = str(error)
                entry["correctness"] = "passed"

        if entry["status"] == "qualified":
            print(
                "  正确性通过；完整性能已记录（GFLOPS 几何平均 %.3f）。" %
                entry["metrics"]["geomean_gflops"],
                flush=True,
            )
        elif entry["status"] == "correctness_failed":
            print("  正确性失败，未记录性能。", flush=True)
        elif entry["status"] == "unverified_measured":
            print("  未验证调试测量完成；不会参与最终发布。", flush=True)
        else:
            print("  候选结束：%s。" % entry["status"], flush=True)

        summary["candidates"].append(entry)
        checkpoint()

    ranking = rank_qualified_candidates(summary["candidates"], args.objective)
    summary["ranking"] = ranking
    if args.skip_verify:
        summary["status"] = "completed_unverified"
        summary["finished_at_utc"] = utc_now()
        checkpoint()
        print("搜索完成，但 --skip-verify 模式不会发布 best .so。")
        return 0
    if not ranking:
        summary["status"] = "no_qualified_candidate"
        summary["finished_at_utc"] = utc_now()
        checkpoint()
        print("没有通过完整正确性与性能门槛的候选；未发布 best .so。", file=sys.stderr)
        return 1

    winner_rank = ranking[0]
    winner_entry = next(
        entry for entry in summary["candidates"]
        if entry["candidate_id"] == winner_rank["candidate_id"]
    )
    try:
        best_library = publish_best_library(Path(winner_entry["library"]), output_root / "best")
        winner = {
            "candidate_id": winner_entry["candidate_id"],
            "objective": args.objective,
            "objective_value": winner_rank["objective_value"],
            "metrics": winner_entry["metrics"],
            "source_library": winner_entry["library"],
            "source_library_sha256": sha256(Path(winner_entry["library"])),
            "best_library": str(best_library),
            "best_library_sha256": sha256(best_library),
            "run_id": run_id,
        }
        summary["winner"] = winner
        atomic_write_json(output_root / "best_candidate.json", winner)
    except (OSError, RuntimeError, ValueError) as error:
        summary["status"] = "best_publish_failed"
        summary["best_publish_error"] = str(error)
        summary["finished_at_utc"] = utc_now()
        checkpoint()
        print("已选出候选，但发布 best .so 失败：%s" % error, file=sys.stderr)
        return 1

    summary["status"] = "complete"
    summary["finished_at_utc"] = utc_now()
    checkpoint()
    print("搜索完成：%s" % winner["candidate_id"])
    print("最佳库：%s" % winner["best_library"])
    print("汇总：%s" % (output_root / "summary.json"))
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run_search(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print("总控搜索启动失败：%s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
