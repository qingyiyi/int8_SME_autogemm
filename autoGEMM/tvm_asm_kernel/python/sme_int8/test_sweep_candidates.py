#!/usr/bin/env python3
"""Host-only tests for the remote sweep orchestration helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest


MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from sweep_candidates import (
    candidate_entries,
    main,
    parse_shape,
    rank_quick_results,
    valid_records,
    write_json,
)


class SweepHelpersTest(unittest.TestCase):
    def test_parse_shape(self) -> None:
        self.assertEqual(parse_shape("2048, 4096,2048"), (2048, 4096, 2048))
        with self.assertRaises(ValueError):
            parse_shape("2048,4096")
        with self.assertRaises(ValueError):
            parse_shape("2048,0,2048")

    def test_rank_quick_results_ignores_unmeasured(self) -> None:
        results = [
            {"candidate_id": "slow", "quick": {"gflops": 10.0}},
            {"candidate_id": "fast", "quick": {"gflops": 20.0}},
            {"candidate_id": "failed", "status": "quick_measurement_failed"},
        ]
        ranked = rank_quick_results(results, 2)
        self.assertEqual([item["candidate_id"] for item in ranked], ["fast", "slow"])

    def test_candidate_entries_rejects_empty_or_malformed_index(self) -> None:
        with self.assertRaises(ValueError):
            candidate_entries({"candidates": []})
        with self.assertRaises(ValueError):
            candidate_entries({"candidates": [{"bundle_dir": "missing-id"}]})

    def test_main_refuses_nonempty_results_directory_before_running_target_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            (results / "existing.jsonl").write_text("old\\n", encoding="ascii")
            candidates = root / "candidates"
            candidates.mkdir()
            (candidates / "candidates.json").write_text('{"candidates": [{"candidate_id": "x"}]}', encoding="ascii")
            reference = root / "reference"
            reference.mkdir()
            kernel = root / "kernel.o"
            kernel.write_bytes(b"")
            test_bin = root / "test"
            test_bin.write_text("#!/bin/sh\\n", encoding="ascii")
            test_bin.chmod(0o755)
            original = sys.argv
            sys.argv = [
                "sweep_candidates.py",
                "--candidates-root", str(candidates),
                "--reference-root", str(reference),
                "--kernel-objects", str(kernel),
                "--test-bin", str(test_bin),
                "--results-dir", str(results),
            ]
            try:
                with self.assertRaisesRegex(SystemExit, "results directory must be empty"):
                    main()
            finally:
                sys.argv = original

    def test_valid_records_ignores_rejected_and_malformed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            path.write_text(
                "not json\n"
                '{"status": "rejected"}\n'
                '{"status": "valid", "shape": [2048, 2048, 2048]}\n',
                encoding="ascii",
            )
            self.assertEqual(valid_records(path), [{"status": "valid", "shape": [2048, 2048, 2048]}])

    def test_write_json_emits_one_parseable_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            value = {"candidate_count": 2}
            write_json(path, value)
            self.assertEqual(json.loads(path.read_text(encoding="ascii")), value)

    def test_tail_uses_real_line_break_when_truncating(self) -> None:
        from sweep_candidates import tail

        output = tail("x" * 200000)
        self.assertTrue(output.startswith("...[truncated]\n"))


if __name__ == "__main__":
    unittest.main()
