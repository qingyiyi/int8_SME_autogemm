#!/usr/bin/env bash
set -euo pipefail

# Select the fastest correctness-qualified candidate.  The script is
# intentionally self-contained so it can be copied to a target machine
# without another helper file.
exec python3 - "$@" <<'PY'
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn


BEST_LIBRARY_NAME = "libint8gemm_autogemm.so"
GOOD_SUMMARY_STATUSES = {"complete", "completed_with_failures"}
CANDIDATE_PARAMS_RE = re.compile(r"(?:^|-)p(?P<p>\d+)-r(?P<r>\d+)(?:-|$)")
TABLE_FIELDS = (
    "shape",
    "candidate_id",
    "p",
    "r",
    "latency_seconds",
    "gflops",
    "library",
    "library_sha256",
    "status",
    "reason",
)


def fail(message: str) -> NoReturn:
    raise SystemExit(f"选择最佳候选失败：{message}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"文件不存在：{path}")
    except (OSError, json.JSONDecodeError) as error:
        fail(f"无法读取 JSON {path}：{error}")
    if not isinstance(value, dict):
        fail(f"JSON 顶层不是对象：{path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shape_key(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    if not all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value):
        return None
    return (value[0], value[1], value[2])


def shape_text(shape: tuple[int, int, int]) -> str:
    return "%dx%dx%d" % shape


def shape_list(value: Any, description: str) -> list[tuple[int, int, int]]:
    if not isinstance(value, list) or not value:
        fail(f"{description} 缺少非空 shapes 列表")
    result: list[tuple[int, int, int]] = []
    for item in value:
        shape = shape_key(item)
        if shape is None:
            fail(f"{description} 含有非法 shape：{item!r}")
        if shape in result:
            fail(f"{description} 含有重复 shape：{shape}")
        result.append(shape)
    return result


def summary_shapes(summary: dict[str, Any], description: str) -> list[tuple[int, int, int]]:
    """Read the tested shape list, with a case-based fallback for old summaries."""
    raw_shapes = summary.get("shapes")
    if raw_shapes is not None:
        return shape_list(raw_shapes, description)

    discovered: list[tuple[int, int, int]] = []
    candidates = summary.get("candidates")
    if isinstance(candidates, list):
        for entry in candidates:
            if not isinstance(entry, dict) or not isinstance(entry.get("cases"), list):
                continue
            for case in entry["cases"]:
                if not isinstance(case, dict):
                    continue
                shape = shape_key(case.get("shape"))
                if shape is not None and shape not in discovered:
                    discovered.append(shape)
    if not discovered:
        fail(f"{description} 缺少 shapes，且无法从 cases 推导")
    return discovered


def candidate_cases(entry: Any) -> list[dict[str, Any]] | None:
    if not isinstance(entry, dict) or not isinstance(entry.get("cases"), list):
        return None
    cases = entry["cases"]
    if not cases or not all(isinstance(case, dict) for case in cases):
        return None
    return cases


def case_map(entry: Any) -> dict[tuple[int, int, int], dict[str, Any]]:
    cases = candidate_cases(entry)
    if cases is None:
        return {}
    result: dict[tuple[int, int, int], dict[str, Any]] = {}
    for case in cases:
        shape = shape_key(case.get("shape"))
        if shape is not None:
            # A fresh summary should contain one case per shape.  Keeping the
            # last one is deterministic and matches the runner's checkpoint
            # semantics if an old summary was resumed.
            result[shape] = case
    return result


def candidate_entry_map(summary: dict[str, Any], description: str) -> dict[str, dict[str, Any]]:
    candidates = summary.get("candidates")
    if not isinstance(candidates, list):
        fail(f"{description} 缺少 candidates 列表")
    result: dict[str, dict[str, Any]] = {}
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        candidate_id = entry.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            continue
        if candidate_id in result:
            fail(f"{description} 中 candidate_id 重复：{candidate_id}")
        result[candidate_id] = entry
    return result


def verified_candidates(
    summary: dict[str, Any], expected_shapes: list[tuple[int, int, int]] | None = None
) -> dict[str, set[tuple[int, int, int]]]:
    """Return candidates whose complete expected shape set passed correctness."""
    candidates = summary.get("candidates")
    if not isinstance(candidates, list):
        fail("正确性 summary 缺少 candidates 列表")

    result: dict[str, set[tuple[int, int, int]]] = {}
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        candidate_id = entry.get("candidate_id")
        cases = candidate_cases(entry)
        if not isinstance(candidate_id, str) or cases is None:
            continue

        passed_shapes: set[tuple[int, int, int]] = set()
        passed = True
        for case in cases:
            shape = shape_key(case.get("shape"))
            if case.get("status") != "correctness_passed" or shape is None:
                passed = False
                break
            passed_shapes.add(shape)
        expected = set(expected_shapes) if expected_shapes is not None else None
        complete_shape_set = expected is None or (
            len(cases) == len(expected) and passed_shapes == expected
        )
        if passed and passed_shapes and complete_shape_set:
            result[candidate_id] = passed_shapes
    return result


def performance_rows(
    summary: dict[str, Any],
    verified: dict[str, set[tuple[int, int, int]]],
    libraries_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Build rows for the legacy all-shape (geometric-mean) selection."""
    candidates = summary.get("candidates")
    if not isinstance(candidates, list):
        fail("性能 summary 缺少 candidates 列表")

    rows: list[dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        candidate_id = entry.get("candidate_id")
        library_text = entry.get("library")
        cases = candidate_cases(entry)
        if not isinstance(candidate_id, str) or not isinstance(library_text, str):
            continue
        if cases is None or candidate_id not in verified:
            continue

        perf_shapes: set[tuple[int, int, int]] = set()
        gflops: list[float] = []
        latencies: list[float] = []
        valid = True
        for case in cases:
            shape = shape_key(case.get("shape"))
            latency = case.get("latency_seconds")
            case_gflops = case.get("gflops")
            if (
                case.get("status") != "performance_recorded"
                or shape is None
                or not isinstance(latency, (int, float))
                or not math.isfinite(float(latency))
                or float(latency) <= 0
                or not isinstance(case_gflops, (int, float))
                or not math.isfinite(float(case_gflops))
                or float(case_gflops) <= 0
            ):
                valid = False
                break
            perf_shapes.add(shape)
            latencies.append(float(latency))
            gflops.append(float(case_gflops))

        # Do not select a candidate measured on a shape that was not verified.
        if not valid or not perf_shapes or perf_shapes != verified[candidate_id]:
            continue

        library = resolve_library(candidate_id, library_text, libraries_dir)
        # For one shape this is exactly the measured GFLOPS.  For multiple
        # shapes, use the geometric mean so large shapes do not dominate merely
        # because they contain more operations.
        geomean_gflops = math.exp(sum(math.log(value) for value in gflops) / len(gflops))
        rows.append(
            {
                "candidate_id": candidate_id,
                "library": str(library),
                "shapes": [list(shape) for shape in sorted(perf_shapes)],
                "latencies_seconds": latencies,
                "gflops": gflops,
                "total_latency_seconds": sum(latencies),
                "geomean_gflops": geomean_gflops,
                "library_sha256_recorded": entry.get("library_sha256"),
            }
        )
    return rows


def resolve_library(
    candidate_id: str, library_text: str, libraries_dir: Path | None
) -> Path:
    """Resolve a library, optionally relocating summaries copied from another host."""
    if libraries_dir is None:
        return Path(library_text).expanduser().resolve()

    candidates = [
        libraries_dir / (candidate_id + ".so"),
        libraries_dir / Path(library_text).name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def candidate_parameters(candidate_id: str) -> tuple[int | None, int | None]:
    match = CANDIDATE_PARAMS_RE.search(candidate_id)
    if match is None:
        return None, None
    return int(match.group("p")), int(match.group("r"))


def valid_performance_case(case: Any) -> tuple[float, float] | None:
    if not isinstance(case, dict) or case.get("status") != "performance_recorded":
        return None
    latency = case.get("latency_seconds")
    gflops = case.get("gflops")
    if (
        not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or float(latency) <= 0
        or not isinstance(gflops, (int, float))
        or not math.isfinite(float(gflops))
        or float(gflops) <= 0
    ):
        return None
    return float(latency), float(gflops)


def select_by_shape(
    verify: dict[str, Any],
    perf: dict[str, Any],
    shapes: list[tuple[int, int, int]],
    libraries_dir: Path | None,
) -> list[dict[str, Any]]:
    """Select independently for each shape; a candidate may fail another shape."""
    verify_entries = candidate_entry_map(verify, "正确性 summary")
    perf_entries = candidate_entry_map(perf, "性能 summary")
    hash_cache: dict[Path, str] = {}
    rows: list[dict[str, Any]] = []

    for shape in shapes:
        eligible: list[dict[str, Any]] = []
        rejection_reasons: Counter[str] = Counter()
        for candidate_id, perf_entry in perf_entries.items():
            verify_entry = verify_entries.get(candidate_id)
            if verify_entry is None:
                rejection_reasons["correctness_candidate_missing"] += 1
                continue
            verify_case = case_map(verify_entry).get(shape)
            if verify_case is None or verify_case.get("status") != "correctness_passed":
                rejection_reasons["correctness_failed_or_missing"] += 1
                continue
            perf_case = case_map(perf_entry).get(shape)
            metrics = valid_performance_case(perf_case)
            if metrics is None:
                rejection_reasons["performance_missing_or_invalid"] += 1
                continue

            library_text = perf_entry.get("library")
            if not isinstance(library_text, str) or not library_text:
                rejection_reasons["library_path_missing"] += 1
                continue
            library = resolve_library(candidate_id, library_text, libraries_dir)
            if not library.is_file():
                rejection_reasons["library_missing"] += 1
                continue
            if library not in hash_cache:
                hash_cache[library] = sha256_file(library)
            actual_sha256 = hash_cache[library]
            recorded_sha256 = perf_entry.get("library_sha256")
            if (
                isinstance(recorded_sha256, str)
                and recorded_sha256
                and actual_sha256 != recorded_sha256
            ):
                fail(
                    "候选库 SHA256 与性能 summary 不一致："
                    f"candidate={candidate_id} summary={recorded_sha256} "
                    f"actual={actual_sha256}"
                )

            latency, gflops = metrics
            p_value, r_value = candidate_parameters(candidate_id)
            eligible.append(
                {
                    "shape": list(shape),
                    "candidate_id": candidate_id,
                    "p": p_value,
                    "r": r_value,
                    "latency_seconds": latency,
                    "gflops": gflops,
                    "library": str(library),
                    "library_sha256": actual_sha256,
                    "status": "selected_candidate",
                    "reason": "correctness_passed_and_performance_recorded",
                }
            )

        eligible.sort(
            key=lambda row: (
                -float(row["gflops"]),
                float(row["latency_seconds"]),
                str(row["candidate_id"]),
            )
        )
        if eligible:
            rows.append(eligible[0])
        else:
            reason = ", ".join(
                "%s=%d" % item for item in sorted(rejection_reasons.items())
            ) or "no_candidate_records"
            rows.append(
                {
                    "shape": list(shape),
                    "candidate_id": "",
                    "p": "",
                    "r": "",
                    "latency_seconds": "",
                    "gflops": "",
                    "library": "",
                    "library_sha256": "",
                    "status": "no_qualified_candidate",
                    "reason": reason,
                }
            )
    return rows


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    try:
        temporary.write_text(text, encoding="utf-8", newline="")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def csv_text(rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=TABLE_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def markdown_text(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# INT8 SME 每个 shape 的最快正确候选",
        "",
        "| M | N | K | candidate_id | P | R | time (s) | GFLOPS | .so | status |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        m, n, k = row["shape"]
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                m,
                n,
                k,
                markdown_cell(row["candidate_id"]),
                markdown_cell(row["p"]),
                markdown_cell(row["r"]),
                (
                    "%.6f" % float(row["latency_seconds"])
                    if row["status"] == "selected_candidate"
                    else ""
                ),
                (
                    "%.3f" % float(row["gflops"])
                    if row["status"] == "selected_candidate"
                    else ""
                ),
                markdown_cell(row["library"]),
                markdown_cell(row["status"]),
            )
        )
    return "\n".join(lines) + "\n"


def print_shape_table(rows: list[dict[str, Any]]) -> None:
    print("=== 每个 shape 的最快正确候选 ===")
    print("shape(MxNxK) | candidate_id | P | R | time(s) | GFLOPS | .so")
    for row in rows:
        shape = "x".join(str(value) for value in row["shape"])
        if row["status"] == "selected_candidate":
            print(
                "%s | %s | %s | %s | %.6f | %.3f | %s"
                % (
                    shape,
                    row["candidate_id"],
                    row["p"],
                    row["r"],
                    float(row["latency_seconds"]),
                    float(row["gflops"]),
                    row["library"],
                )
            )
        else:
            print("%s | - | - | - | - | - | [%s]" % (shape, row["reason"]))


def write_per_shape_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    verify_path: Path,
    perf_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "kind": "sme_int8_best_candidate_by_shape",
        "selection_objective": "per_shape_gflops",
        "rows": rows,
        "verify_summary": str(verify_path),
        "performance_summary": str(perf_path),
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "只生成映射表，不复制 .so；library 字段指向 libraries/ 中的原候选库。",
    }
    atomic_write_json(output_dir / "best_by_shape.json", metadata)
    atomic_write_text(output_dir / "best_by_shape.csv", csv_text(rows))
    atomic_write_text(output_dir / "best_by_shape.md", markdown_text(rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从正确性和性能 summary 中选择全局或每个 shape 的最快 INT8 SME 候选库"
    )
    parser.add_argument(
        "--verify-summary",
        type=Path,
        required=True,
        help="test_candidate_sos.py --verify verify 生成的 summary.json",
    )
    parser.add_argument(
        "--perf-summary",
        type=Path,
        required=True,
        help="test_candidate_sos.py --verify false 生成的 summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="选择结果输出目录；默认是日期目录下的 autogemm-sme-candidates/best（--per-shape 时为 best-by-shape）",
    )
    parser.add_argument(
        "--per-shape",
        action="store_true",
        help="按每个 shape 独立选择最快候选，并输出 best_by_shape.{md,csv,json}；不复制 .so",
    )
    parser.add_argument(
        "--libraries-dir",
        type=Path,
        default=None,
        help="可选：summary 从另一台机器复制过来时，在此目录按 candidate_id 重新定位 .so",
    )
    return parser.parse_args()


def validate_summary_headers(
    verify: dict[str, Any], perf: dict[str, Any]
) -> list[tuple[int, int, int]]:
    if verify.get("verify_argument") != "verify":
        fail(f"正确性 summary 的 verify_argument 不是 verify：{verify.get('verify_argument')!r}")
    if perf.get("verify_argument") != "false":
        fail(f"性能 summary 的 verify_argument 不是 false：{perf.get('verify_argument')!r}")
    if verify.get("status") not in GOOD_SUMMARY_STATUSES:
        fail(f"正确性 summary 尚未正常完成，status={verify.get('status')!r}")
    if perf.get("status") not in GOOD_SUMMARY_STATUSES:
        fail(f"性能 summary 尚未正常完成，status={perf.get('status')!r}")

    verify_shapes = summary_shapes(verify, "正确性 summary")
    perf_shapes = summary_shapes(perf, "性能 summary")
    if set(verify_shapes) != set(perf_shapes):
        fail(
            "正确性和性能 summary 的 shape 集合不一致："
            f"verify={[list(shape) for shape in verify_shapes]} "
            f"perf={[list(shape) for shape in perf_shapes]}"
        )
    # Keep the correctness-run order in the output table; the selector itself
    # is independent of order.
    return verify_shapes


def run_per_shape(
    args: argparse.Namespace,
    verify: dict[str, Any],
    perf: dict[str, Any],
    verify_path: Path,
    perf_path: Path,
    output_dir: Path,
) -> int:
    shapes = validate_summary_headers(verify, perf)
    libraries_dir = args.libraries_dir.expanduser().resolve() if args.libraries_dir else None
    if libraries_dir is not None and not libraries_dir.is_dir():
        fail(f"libraries 目录不存在：{libraries_dir}")
    rows = select_by_shape(verify, perf, shapes, libraries_dir)
    write_per_shape_outputs(output_dir, rows, verify_path, perf_path)
    print_shape_table(rows)
    print("\n=== 每个 shape 的选择表 ===")
    print(f"markdown = {output_dir / 'best_by_shape.md'}")
    print(f"csv      = {output_dir / 'best_by_shape.csv'}")
    print(f"json     = {output_dir / 'best_by_shape.json'}")
    missing = [row for row in rows if row["status"] != "selected_candidate"]
    if missing:
        print("有 shape 没有可用候选；已写出表格，详情见 reason。", file=sys.stderr)
        return 1
    return 0


def run_overall(
    args: argparse.Namespace,
    verify: dict[str, Any],
    perf: dict[str, Any],
    verify_path: Path,
    perf_path: Path,
    output_dir: Path,
    shapes: list[tuple[int, int, int]],
) -> int:
    # Legacy behavior: only candidates that passed every shape participate, and
    # one aggregate best .so is copied to best/libint8gemm_autogemm.so.
    verified = verified_candidates(verify, shapes)
    libraries_dir = args.libraries_dir.expanduser().resolve() if args.libraries_dir else None
    if libraries_dir is not None and not libraries_dir.is_dir():
        fail(f"libraries 目录不存在：{libraries_dir}")
    rows = performance_rows(perf, verified, libraries_dir)
    if not rows:
        fail("没有同时通过正确性、并且在相同 shape 上记录性能的候选")

    rows.sort(
        key=lambda row: (
            -row["geomean_gflops"],
            row["total_latency_seconds"],
            row["candidate_id"],
        )
    )
    best = rows[0]
    source = Path(best["library"])
    if not source.is_file():
        fail(f"最佳候选库不存在：{source}")

    actual_sha256 = sha256_file(source)
    recorded_sha256 = best["library_sha256_recorded"]
    if isinstance(recorded_sha256, str) and recorded_sha256 and actual_sha256 != recorded_sha256:
        fail(
            "最佳候选库 SHA256 与性能 summary 不一致："
            f"summary={recorded_sha256} actual={actual_sha256}"
        )

    destination = output_dir / BEST_LIBRARY_NAME
    atomic_copy(source, destination)
    best_sha256 = sha256_file(destination)

    result = {
        "schema_version": 1,
        "kind": "sme_int8_best_candidate",
        "selection_objective": "geomean_gflops",
        "candidate_id": best["candidate_id"],
        "source_library": str(source),
        "source_library_sha256": actual_sha256,
        "best_library": str(destination),
        "best_library_sha256": best_sha256,
        "metrics": {
            "shapes": best["shapes"],
            "latencies_seconds": best["latencies_seconds"],
            "gflops": best["gflops"],
            "total_latency_seconds": best["total_latency_seconds"],
            "geomean_gflops": best["geomean_gflops"],
        },
        "verify_summary": str(verify_path),
        "performance_summary": str(perf_path),
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output_dir / "best_candidate.json", result)

    print("=== 通过正确性后的性能排名 ===")
    for rank, row in enumerate(rows, 1):
        print(
            f"#{rank:2d}  {row['candidate_id']}  "
            f"geomean_GFLOPS={row['geomean_gflops']:.3f}  "
            f"total_time={row['total_latency_seconds']:.6f}s"
        )
    print("\n=== 最佳候选 ===")
    print(f"candidate_id = {best['candidate_id']}")
    print(f"source_so    = {source}")
    print(f"best_so      = {destination}")
    print(f"sha256       = {best_sha256}")
    print(f"metadata     = {output_dir / 'best_candidate.json'}")
    return 0


def main() -> int:
    args = parse_args()
    verify_path = args.verify_summary.expanduser().resolve()
    perf_path = args.perf_summary.expanduser().resolve()
    verify = read_json(verify_path)
    perf = read_json(perf_path)
    shapes = validate_summary_headers(verify, perf)

    if args.output_dir is None:
        output_name = "best-by-shape" if args.per_shape else "best"
        output_dir = perf_path.parent.parent / "autogemm-sme-candidates" / output_name
    else:
        output_dir = args.output_dir.expanduser().resolve()

    if args.per_shape:
        # Avoid parsing the summaries twice while retaining the same validation
        # and output order as the common header check above.
        return run_per_shape(args, verify, perf, verify_path, perf_path, output_dir)

    # Overall mode uses the same shape-set validation; its detailed candidate
    # filter additionally requires every candidate case to be correct.
    _ = shapes
    return run_overall(args, verify, perf, verify_path, perf_path, output_dir, shapes)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(1)
PY
