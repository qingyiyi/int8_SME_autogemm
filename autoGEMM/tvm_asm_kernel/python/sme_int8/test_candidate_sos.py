#!/usr/bin/env python3
"""Run existing ``test_unigemm`` against generated INT8 SME candidate .so files.

The test executable remains entirely external to this tool: it keeps owning
huge-page/HBM allocation and its own correctness logic.  Before each process
launch this tool atomically changes the single library pathname resolved by
the executable, then restores the original directory entry at the end.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

from generate_driver import DEFAULT_CONFIG, as_shape, read_json, validate_config


DEFAULT_TEST_BIN = Path("/data1/cxz/int8/test_unigemm")
DEFAULT_INSTALL_LIBRARY = Path("/data1/cxz/int8/lib/libint8gemm.so")
DEFAULT_SHAPE = (8192, 8192, 2048)
EXPECTED_K = 2048
SHAPE_MULTIPLE = 2048
PERF_RE = re.compile(
    r"average\s+int8gemm\s+time\s*=\s*"
    r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)"
)
MAX_CAPTURE_BYTES = 128 * 1024


def tail(text: str) -> str:
    if len(text) <= MAX_CAPTURE_BYTES:
        return text
    return "...[truncated]...\n" + text[-MAX_CAPTURE_BYTES:]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_shape(value: str) -> Tuple[int, int, int]:
    fields = value.split(",")
    if len(fields) != 3:
        raise ValueError("shape must be M,N,K, got %r" % value)
    try:
        return tuple(int(field.strip()) for field in fields)  # type: ignore[return-value]
    except ValueError as error:
        raise ValueError("shape must contain three integers, got %r" % value) from error


def validate_target_shape(shape: Tuple[int, int, int]) -> None:
    m, n, k = shape
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("shape 必须为正数: %s" % (shape,))
    if k != EXPECTED_K or m % SHAPE_MULTIPLE or n % SHAPE_MULTIPLE:
        raise ValueError(
            "当前 SME 候选只支持 K=2048 且 M/N 为 2048 的倍数: %s" % (shape,)
        )


def configured_shapes(config: Path) -> List[Tuple[int, int, int]]:
    parsed = read_json(config)
    validate_config(parsed)
    return [as_shape(value) for value in parsed["shapes"]]


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
        if (
            len(fields) >= 3
            and fields[-1] == "cblas_gemm_s8s8s32"
            and fields[-2] in {"T", "t"}
        ):
            return
    raise RuntimeError("候选库没有导出 T cblas_gemm_s8s8s32: %s" % library)


def discover_libraries(directory: Path) -> List[Path]:
    if not directory.is_dir():
        raise ValueError("候选 .so 目录不存在: %s" % directory)
    # A refreshed build output can retain old source directories for diagnosis.
    # Prefer the current build summary when present, so a narrowed P/R rerun
    # never accidentally re-tests stale flat libraries from a prior sweep.
    build_summary = directory.parent / "build_summary.json"
    if build_summary.is_file():
        try:
            value = json.loads(build_summary.read_text(encoding="utf-8"))
            entries = value.get("candidates", [])
            if not isinstance(entries, list):
                raise ValueError("build_summary.json candidates 字段不是列表")
            libraries = sorted(
                (
                    Path(entry["library"]).resolve()
                    for entry in entries
                    if isinstance(entry, dict) and entry.get("status") == "built"
                ),
                key=lambda path: path.name,
            )
            if not libraries:
                raise ValueError("build_summary.json 中没有成功构建的候选")
            missing = [str(path) for path in libraries if not path.is_file()]
            if missing:
                raise ValueError("build_summary.json 引用的候选库不存在: %s" % ", ".join(missing))
            return libraries
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("无法使用 %s: %s" % (build_summary, error)) from error
    libraries = sorted(
        (path.resolve() for path in directory.glob("*.so") if path.is_file()),
        key=lambda path: path.name,
    )
    if not libraries:
        raise ValueError("候选 .so 目录中没有 *.so: %s" % directory)
    return libraries


def test_environment(
    base: Dict[str, str], proc_bind: str, places: str, threads: int
) -> Dict[str, str]:
    env = dict(base)
    # A pre-existing preload could put an unrelated libint8gemm ahead of the
    # path that this runner atomically switches.  This flow intentionally uses
    # the test binary's normal dynamic-link path, never LD_PRELOAD.
    env.pop("LD_PRELOAD", None)
    env.update({
        "OMP_PROC_BIND": proc_bind,
        "OMP_PLACES": places,
        "OMP_NUM_THREADS": str(threads),
    })
    return env


def make_test_command(
    numactl: str,
    cpu_bind: str,
    memory_nodes: str,
    test_bin: Path,
    shape: Tuple[int, int, int],
    mode: str,
    verify_arg: str,
) -> List[str]:
    m, n, k = shape
    return [
        numactl,
        "--all",
        "--physcpubind=%s" % cpu_bind,
        "-m",
        memory_nodes,
        str(test_bin),
        str(m),
        str(n),
        str(k),
        mode,
        verify_arg,
    ]


def run_command(
    command: Sequence[str], env: Dict[str, str], timeout: float
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


def parse_latency(stdout: str) -> float | None:
    match = PERF_RE.search(stdout)
    if match is None:
        return None
    latency = float(match.group(1))
    return latency if math.isfinite(latency) and latency > 0 else None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (".%s.tmp-%d" % (path.name, os.getpid()))
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def display_launch(command: Sequence[str], environment: Dict[str, str]) -> str:
    """Render the three controlled OMP variables plus the exact test command."""
    prefixes = [
        "%s=%s" % (name, shlex.quote(environment[name]))
        for name in ("OMP_PROC_BIND", "OMP_PLACES", "OMP_NUM_THREADS")
    ]
    return " ".join(prefixes + [shlex.join(command)])


def write_log(path: Path, result: Dict[str, Any], launch: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# launch\n%s\n\n# command\n%s\n\n# returncode\n%s\n\n# stdout\n%s\n\n# stderr\n%s\n" % (
            launch,
            shlex.join(result.get("command", [])),
            result.get("returncode"),
            result.get("stdout", ""),
            result.get("stderr", ""),
        ),
        encoding="utf-8",
    )


class InstallLock:
    """Prevent two sweep processes from racing on one loader pathname."""

    def __init__(self, install_library: Path) -> None:
        self.path = install_library.parent / (".%s.autogemm-sme.lock" % install_library.name)
        self.fd: int | None = None

    def __enter__(self) -> "InstallLock":
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError(
                "无法获取库替换锁（已有测试正在使用此路径？）: %s" % self.path
            ) from error
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


class LibrarySwap:
    """Atomically replace one dynamic-loader entry and restore it afterwards."""

    def __init__(self, install_library: Path, journal: Path) -> None:
        self.install_library = install_library
        self.journal = journal
        token = "%d-%s" % (os.getpid(), uuid.uuid4().hex)
        self.backup = install_library.parent / (
            ".%s.autogemm-sme-backup-%s" % (install_library.name, token)
        )
        self.active_temporary = install_library.parent / (
            ".%s.autogemm-sme-active-%s" % (install_library.name, token)
        )
        self._active = False

    @staticmethod
    def _lexists(path: Path) -> bool:
        return os.path.lexists(path)

    def _write_journal(self, state: str, candidate: Path | None = None) -> None:
        value: Dict[str, Any] = {
            "schema_version": 1,
            "state": state,
            "install_library": str(self.install_library),
            "backup": str(self.backup),
            "updated_at_unix": time.time(),
        }
        if candidate is not None:
            value["candidate"] = str(candidate)
        write_json(self.journal, value)

    def __enter__(self) -> "LibrarySwap":
        if not self._lexists(self.install_library):
            raise RuntimeError("待替换的动态库不存在: %s" % self.install_library)
        if self.install_library.is_symlink():
            os.symlink(os.readlink(self.install_library), self.backup)
        elif self.install_library.is_file():
            try:
                os.link(self.install_library, self.backup)
            except OSError:
                shutil.copy2(self.install_library, self.backup)
        else:
            raise RuntimeError("待替换路径不是普通文件或符号链接: %s" % self.install_library)
        self._write_journal("backup_created")
        return self

    def activate(self, candidate: Path) -> None:
        if not candidate.is_file():
            raise RuntimeError("候选库不存在: %s" % candidate)
        if self._lexists(self.active_temporary):
            self.active_temporary.unlink()
        os.symlink(str(candidate), self.active_temporary)
        os.replace(self.active_temporary, self.install_library)
        self._active = True
        self._write_journal("candidate_active", candidate)

    def restore(self) -> None:
        if self._lexists(self.backup):
            os.replace(self.backup, self.install_library)
        if self._lexists(self.active_temporary):
            self.active_temporary.unlink()
        self._active = False
        self._write_journal("restored")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


def recover(journal: Path) -> int:
    try:
        value = json.loads(journal.read_text(encoding="utf-8"))
        install_library = Path(value["install_library"])
        backup = Path(value["backup"])
        if value.get("state") == "restored":
            print("恢复日志表明原库已经恢复：%s" % install_library)
            return 0
        if not os.path.lexists(backup):
            raise RuntimeError("找不到备份文件: %s" % backup)
        os.replace(backup, install_library)
        value["state"] = "restored"
        value["recovered_at_unix"] = time.time()
        write_json(journal, value)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print("恢复失败：%s" % error, file=sys.stderr)
        return 1
    print("已恢复原库：%s" % install_library)
    return 0


@contextmanager
def interrupt_as_keyboard_interrupt() -> Iterator[None]:
    """Let Ctrl-C, HUP, and TERM reach LibrarySwap.__exit__ before termination."""
    signals = [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]
    previous = {item: signal.getsignal(item) for item in signals}

    def handler(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt("收到终止信号 %d" % signum)

    try:
        for item in signals:
            signal.signal(item, handler)
        yield
    finally:
        for item, handler_value in previous.items():
            signal.signal(item, handler_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libraries-dir", type=Path,
                        help="build_candidate_sos.py 输出的 libraries 目录")
    parser.add_argument("--test-bin", type=Path, default=DEFAULT_TEST_BIN,
                        help="已有的 /data1/cxz/int8/test_unigemm")
    parser.add_argument("--install-library", type=Path, default=DEFAULT_INSTALL_LIBRARY,
                        help="test_unigemm RPATH 中的 libint8gemm.so")
    parser.add_argument("--results-dir", type=Path,
                        help="日志和 JSON 结果目录；默认建在 libraries 旁")
    parser.add_argument("--shape", action="append", default=[],
                        help="M,N,K；可重复指定，默认 8192,8192,2048")
    parser.add_argument("--all-shapes", action="store_true",
                        help="遍历 backend JSON 中的全部 9 个 shape")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="--all-shapes 使用的 backend JSON")
    parser.add_argument("--mode", default="kblas", choices=("kblas", "unigemm", "both"),
                        help="传给 test_unigemm 的实现选择")
    parser.add_argument("--verify", default="verify", choices=("verify", "false"),
                        help="传给 test_unigemm 的校验参数")
    parser.add_argument("--numactl", default="numactl", help="numactl 可执行文件")
    parser.add_argument("--physcpubind", default="570-601",
                        help="numactl --physcpubind 值")
    parser.add_argument("--membind", default="15,31", help="numactl -m 值")
    parser.add_argument("--omp-proc-bind", default="false", help="OMP_PROC_BIND 值")
    parser.add_argument("--omp-places", default="cores", help="OMP_PLACES 值")
    parser.add_argument("--omp-num-threads", type=int, default=32,
                        help="OMP_NUM_THREADS 值")
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="每个候选/shape 最长运行秒数")
    parser.add_argument("--recover", type=Path,
                        help="只根据上一次的 install_state.json 恢复原库")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.recover is not None:
        return recover(args.recover.resolve())

    try:
        if args.libraries_dir is None:
            raise ValueError("必须提供 --libraries-dir")
        if args.timeout <= 0:
            raise ValueError("--timeout 必须大于零")
        if args.omp_num_threads <= 0:
            raise ValueError("--omp-num-threads 必须大于零")
        if args.shape and args.all_shapes:
            raise ValueError("--shape 与 --all-shapes 不能同时使用")
        libraries_dir = args.libraries_dir.resolve()
        libraries = discover_libraries(libraries_dir)
        test_bin = args.test_bin.resolve()
        # Do not call Path.resolve() here.  The filename itself is the one in
        # test_unigemm's RPATH, and it may already be a symbolic link.  We
        # need to replace that directory entry rather than its resolved target.
        install_library = Path(os.path.abspath(args.install_library))
        if not test_bin.is_file() or not os.access(test_bin, os.X_OK):
            raise ValueError("测试程序不存在或不可执行: %s" % test_bin)
        if not install_library.parent.is_dir():
            raise ValueError("安装库目录不存在: %s" % install_library.parent)
        if args.shape:
            shapes = [parse_shape(value) for value in args.shape]
        elif args.all_shapes:
            shapes = configured_shapes(args.config.resolve())
        else:
            shapes = [DEFAULT_SHAPE]
        if len(set(shapes)) != len(shapes):
            raise ValueError("shape 不能重复")
        for shape in shapes:
            validate_target_shape(shape)
        for library in libraries:
            check_export(library)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print("候选测试启动失败：%s" % error, file=sys.stderr)
        return 2

    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else libraries_dir.parent / ("test-results-%s" % time.strftime("%Y%m%d-%H%M%S"))
    )
    if results_dir.exists() and any(results_dir.iterdir()):
        print("结果目录非空，拒绝混入本次数据：%s" % results_dir, file=sys.stderr)
        return 2
    results_dir.mkdir(parents=True, exist_ok=True)
    journal = results_dir / "install_state.json"
    environment = test_environment(
        os.environ, args.omp_proc_bind, args.omp_places, args.omp_num_threads
    )
    summary: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "sme_int8_direct_link_test",
        "library_injection": "atomic_replace_rpath_library",
        "ld_preload_used": False,
        "test_bin": str(test_bin),
        "install_library": str(install_library),
        "verify_argument": args.verify,
        "shapes": [list(shape) for shape in shapes],
        "environment": {
            "OMP_PROC_BIND": environment["OMP_PROC_BIND"],
            "OMP_PLACES": environment["OMP_PLACES"],
            "OMP_NUM_THREADS": environment["OMP_NUM_THREADS"],
        },
        "numactl": {
            "path": args.numactl,
            "physcpubind": args.physcpubind,
            "membind": args.membind,
        },
        "candidates": [],
    }
    failed_cases = 0
    print("将按固定动态库路径测试 %d 个候选；结束后会恢复原库。" % len(libraries))
    try:
        with interrupt_as_keyboard_interrupt(), InstallLock(install_library), LibrarySwap(
            install_library, journal
        ) as swap:
            for library in libraries:
                identifier = library.stem
                print("测试候选 %s" % identifier, flush=True)
                swap.activate(library)
                entry: Dict[str, Any] = {
                    "candidate_id": identifier,
                    "library": str(library),
                    "library_sha256": sha256_file(library),
                    "cases": [],
                }
                for shape in shapes:
                    command = make_test_command(
                        args.numactl,
                        args.physcpubind,
                        args.membind,
                        test_bin,
                        shape,
                        args.mode,
                        args.verify,
                    )
                    result = run_command(command, environment, args.timeout)
                    launch = display_launch(command, environment)
                    shape_id = "%dx%dx%d" % shape
                    log = results_dir / "logs" / identifier / (shape_id + ".log")
                    write_log(log, result, launch)
                    case: Dict[str, Any] = {
                        "shape": list(shape),
                        "command": result["command"],
                        "launch": launch,
                        "returncode": result["returncode"],
                        "wall_seconds": result["wall_seconds"],
                        "log": str(log),
                    }
                    if result.get("returncode") != 0:
                        case["status"] = "failed"
                        failed_cases += 1
                        print("  失败 shape=%s，日志：%s" % (shape, log), file=sys.stderr)
                    elif args.verify == "verify":
                        case["status"] = "correctness_passed"
                        # The test source intentionally uses one iteration in
                        # verify mode, so its average timing line is invalid.
                        case["performance_recorded"] = False
                        print("  正确性通过 shape=%s" % (shape,))
                    else:
                        latency = parse_latency(result.get("stdout", ""))
                        if latency is None:
                            case["status"] = "performance_unparsed"
                            failed_cases += 1
                            print("  未解析到性能 shape=%s，日志：%s" % (shape, log), file=sys.stderr)
                        else:
                            m, n, k = shape
                            case.update({
                                "status": "performance_recorded",
                                "latency_seconds": latency,
                                "gflops": (2.0 * m * n * k * 1.0e-9) / latency,
                            })
                            print(
                                "  性能 shape=%s time=%.6f s gflops=%.3f" % (
                                    shape, latency, case["gflops"]
                                )
                            )
                    entry["cases"].append(case)
                summary["candidates"].append(entry)
                write_json(results_dir / "summary.json", summary)
    except KeyboardInterrupt as error:
        summary["status"] = "interrupted"
        summary["interruption"] = str(error)
        write_json(results_dir / "summary.json", summary)
        print("测试被中断；已尝试恢复原库。可检查：%s" % journal, file=sys.stderr)
        return 130
    except (OSError, RuntimeError) as error:
        summary["status"] = "runner_failed"
        summary["error"] = str(error)
        write_json(results_dir / "summary.json", summary)
        print("测试运行失败；已尝试恢复原库：%s" % error, file=sys.stderr)
        return 1

    summary["failed_cases"] = failed_cases
    summary["status"] = "complete" if not failed_cases else "completed_with_failures"
    summary["finished_at_unix"] = time.time()
    write_json(results_dir / "summary.json", summary)
    print("测试汇总：%s" % (results_dir / "summary.json"))
    return 0 if not failed_cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
