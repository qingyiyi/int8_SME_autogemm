#!/usr/bin/env python3
"""Measure an INT8 SME candidate, with optional correctness validation.

The test executable is the existing ``int8/test_unigemm.cpp`` binary (or a
compatible binary).  The candidate library is injected with ``LD_PRELOAD``;
the executable's input allocation and reference setup therefore stay outside
the library implementation.  Pass ``--verify`` when correctness should gate
performance; the default is measurement-only for fast parameter sweeps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from generate_driver import DEFAULT_CONFIG, as_shape, read_json, validate_config


PERF_RE = re.compile(
    r"average\s+int8gemm\s+time\s*=\s*"
    r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)"
)
MAX_CAPTURE_BYTES = 64 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tail(text: str) -> str:
    if len(text) <= MAX_CAPTURE_BYTES:
        return text
    return "...[truncated]...\n" + text[-MAX_CAPTURE_BYTES:]


def candidate_env(library: Path) -> Dict[str, str]:
    env = os.environ.copy()
    previous = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = str(library) + ((":" + previous) if previous else "")
    return env


def check_export(library: Path) -> None:
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
        if len(fields) >= 3 and fields[-1] == "cblas_gemm_s8s8s32" and fields[-2] in {"T", "t"}:
            return
    raise RuntimeError("候选库没有导出 T cblas_gemm_s8s8s32")


def parse_shape(value: str) -> Tuple[int, int, int]:
    fields = value.split(",")
    if len(fields) != 3:
        raise ValueError("shape must be M,N,K, got %r" % value)
    try:
        parsed = [int(field.strip()) for field in fields]
    except ValueError as error:
        raise ValueError("shape must be M,N,K integers, got %r" % value) from error
    return as_shape(parsed)


def run_case(
    test_bin: Path,
    shape: Tuple[int, int, int],
    mode: str,
    verify: bool,
    env: Dict[str, str],
    timeout: float,
) -> Dict[str, Any]:
    m, n, k = shape
    command = [str(test_bin), str(m), str(n), str(k), mode, "verify" if verify else "0"]
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            env=env,
            timeout=timeout,
        )
        return {
            "command": command,
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
            "command": command,
            "returncode": None,
            "stdout": tail(output),
            "stderr": tail(error_output),
            "wall_seconds": time.monotonic() - started,
            "timeout": timeout,
        }
    except OSError as error:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
            "wall_seconds": time.monotonic() - started,
        }


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def reject(
    path: Path,
    library: Path,
    config_path: Path,
    phase: str,
    reason: str,
    shape: Optional[Tuple[int, int, int]] = None,
    case: Optional[Dict[str, Any]] = None,
) -> None:
    # Deliberately omit latency/gflops fields.  Consumers can never mistake a
    # rejected candidate for a measured valid result.
    record: Dict[str, Any] = {
        "schema_version": 1,
        "status": "rejected",
        "phase": phase,
        "reason": reason,
        "library": str(library),
        "library_sha256": sha256(library) if library.is_file() else None,
        "config": str(config_path),
        "timestamp_unix": time.time(),
    }
    if shape is not None:
        record["shape"] = list(shape)
    if case is not None:
        record["case"] = case
    append_jsonl(path, record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True,
                        help="候选 .so；将通过 LD_PRELOAD 注入测试进程")
    parser.add_argument("--test-bin", type=Path, required=True,
                        help="已有的 int8/test_unigemm.cpp 可执行文件")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="包含目标 shape 列表的 backend JSON")
    parser.add_argument("--results", type=Path, required=True,
                        help="性能 JSONL 输出")
    parser.add_argument("--rejections", type=Path, default=None,
                        help="拒绝记录 JSONL；默认是 results 后缀 .rejects.jsonl")
    parser.add_argument("--mode", default="kblas",
                        help="测试程序模式，默认 kblas")
    parser.add_argument("--verify", action="store_true",
                        help="在性能测试前先对全部 shape 做正确性测试；默认关闭以加速搜索")
    parser.add_argument("--shape", action="append", default=None,
                        help="只测一个 shape（格式 M,N,K）；可重复指定，默认测配置中的全部 shape")
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="每个 shape 的最大秒数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    library = args.library.resolve()
    test_bin = args.test_bin.resolve()
    config_path = args.config.resolve()
    results_path = args.results.resolve()
    rejection_path = args.rejections.resolve() if args.rejections else Path(str(results_path) + ".rejects.jsonl")

    if not library.is_file():
        print("候选库不存在: %s" % library, file=sys.stderr)
        return 2
    if not test_bin.is_file() or not os.access(test_bin, os.X_OK):
        print("测试程序不存在或不可执行: %s" % test_bin, file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("--timeout 必须大于 0", file=sys.stderr)
        return 2

    try:
        config = read_json(config_path)
        validate_config(config)
        configured_shapes = [as_shape(value) for value in config["shapes"]]
        if args.shape:
            shapes = [parse_shape(value) for value in args.shape]
            if len(set(shapes)) != len(shapes):
                raise ValueError("--shape contains duplicates")
            unsupported = [shape for shape in shapes if shape not in configured_shapes]
            if unsupported:
                raise ValueError("requested shape is not in config: %r" % (unsupported[0],))
        else:
            shapes = configured_shapes
        check_export(library)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        reject(rejection_path, library, config_path, "preflight", str(error))
        print("候选拒绝: %s" % error, file=sys.stderr)
        return 1

    env = candidate_env(library)
    if args.verify:
        print("开始正确性检查，共 %d 个 shape" % len(shapes))
        for shape in shapes:
            case = run_case(test_bin, shape, args.mode, True, env, args.timeout)
            if case.get("returncode") != 0:
                reject(
                    rejection_path,
                    library,
                    config_path,
                    "correctness",
                    "正确性测试失败，未执行性能测试",
                    shape,
                    case,
                )
                print("拒绝 shape=%s：正确性失败，未记录性能" % (shape,), file=sys.stderr)
                return 1
            print("通过正确性 shape=%s" % (shape,))
    else:
        print("跳过正确性检查（搜索模式）；如需验收请加 --verify")

    print("开始性能测试")
    library_hash = sha256(library)
    for shape in shapes:
        case = run_case(test_bin, shape, args.mode, False, env, args.timeout)
        if case.get("returncode") != 0:
            reject(
                rejection_path,
                library,
                config_path,
                "performance",
                "性能测试进程失败，未记录该 shape 性能",
                shape,
                case,
            )
            print("性能失败 shape=%s，未记录性能" % (shape,), file=sys.stderr)
            continue

        match = PERF_RE.search(case.get("stdout", ""))
        if match is None:
            reject(
                rejection_path,
                library,
                config_path,
                "performance",
                "无法从测试输出解析 average int8gemm time，未记录性能",
                shape,
                case,
            )
            print("性能输出无法解析 shape=%s，未记录性能" % (shape,), file=sys.stderr)
            continue

        latency_seconds = float(match.group(1))
        if not latency_seconds > 0:
            reject(
                rejection_path,
                library,
                config_path,
                "performance",
                "性能时间不是正数，未记录性能",
                shape,
                case,
            )
            print("性能时间无效 shape=%s，未记录性能" % (shape,), file=sys.stderr)
            continue

        m, n, k = shape
        record = {
            "schema_version": 1,
            "status": "valid",
            "correctness": "passed" if args.verify else "not_checked",
            "shape": list(shape),
            "m": m,
            "n": n,
            "k": k,
            "latency_seconds": latency_seconds,
            "gflops": (2.0 * m * n * k * 1.0e-9) / latency_seconds,
            "library": str(library),
            "library_sha256": library_hash,
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "driver": config["driver"],
            "test_bin": str(test_bin),
            "timestamp_unix": time.time(),
        }
        append_jsonl(results_path, record)
        print(
            "记录性能 shape=%s latency=%.6f s gflops=%.3f" %
            (shape, latency_seconds, record["gflops"])
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
