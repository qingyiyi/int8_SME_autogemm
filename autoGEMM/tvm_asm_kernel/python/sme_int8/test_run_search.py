#!/usr/bin/env python3
"""Host-only tests for the correctness-gated SME total-search helpers."""

from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock


MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

import generate_driver as driver
import run_search as workflow

from run_search import (
    classify_evaluation_failure,
    metrics_for_records,
    publish_best_library,
    rank_qualified_candidates,
    summary_for_measurements,
)


SHAPES = ((2048, 2048, 2048), (2048, 4096, 2048))


def record(shape, latency, gflops, correctness="passed"):
    return {
        "status": "valid",
        "shape": list(shape),
        "latency_seconds": latency,
        "gflops": gflops,
        "correctness": correctness,
    }


class FullSearchHelpersTest(unittest.TestCase):
    def test_only_all_verified_shapes_are_qualified(self) -> None:
        records = [
            record(SHAPES[0], 1.0, 10.0),
            record(SHAPES[1], 2.0, 20.0),
        ]
        result = summary_for_measurements(records, SHAPES, True)
        self.assertTrue(result["complete"])
        self.assertTrue(result["correctness_passed"])
        self.assertEqual(result["missing_shapes"], [])

        unverified = summary_for_measurements(
            [record(SHAPES[0], 1.0, 10.0, "not_checked"), record(SHAPES[1], 2.0, 20.0)],
            SHAPES,
            True,
        )
        self.assertTrue(unverified["complete"])
        self.assertFalse(unverified["correctness_passed"])
        self.assertEqual(unverified["wrong_correctness_shapes"], [list(SHAPES[0])])

    def test_missing_or_invalid_measurement_cannot_rank(self) -> None:
        missing = summary_for_measurements([record(SHAPES[0], 1.0, 10.0)], SHAPES, True)
        self.assertFalse(missing["complete"])
        self.assertEqual(missing["missing_shapes"], [list(SHAPES[1])])

        invalid = summary_for_measurements(
            [record(SHAPES[0], 1.0, 10.0), record(SHAPES[1], 0.0, 20.0)], SHAPES, True
        )
        self.assertFalse(invalid["complete"])
        self.assertEqual(invalid["invalid_metric_shapes"], [list(SHAPES[1])])

    def test_rank_supports_both_global_objectives(self) -> None:
        candidates = [
            {
                "candidate_id": "higher-geomean",
                "library": "/tmp/a.so",
                "status": "qualified",
                "metrics": {"geomean_gflops": 200.0, "total_latency_seconds": 3.0},
            },
            {
                "candidate_id": "lower-latency",
                "library": "/tmp/b.so",
                "status": "qualified",
                "metrics": {"geomean_gflops": 190.0, "total_latency_seconds": 2.0},
            },
            {
                "candidate_id": "unverified",
                "library": "/tmp/c.so",
                "status": "unverified_measured",
                "metrics": {"geomean_gflops": 999.0, "total_latency_seconds": 0.1},
            },
        ]
        by_gflops = rank_qualified_candidates(candidates, "geomean_gflops")
        self.assertEqual([item["candidate_id"] for item in by_gflops], ["higher-geomean", "lower-latency"])
        by_latency = rank_qualified_candidates(candidates, "total_latency")
        self.assertEqual([item["candidate_id"] for item in by_latency], ["lower-latency", "higher-geomean"])

    def test_metrics_use_geometric_mean_and_total_latency(self) -> None:
        metrics = metrics_for_records([
            record(SHAPES[0], 1.0, 4.0),
            record(SHAPES[1], 2.0, 9.0),
        ])
        self.assertAlmostEqual(metrics["total_latency_seconds"], 3.0)
        self.assertAlmostEqual(metrics["geomean_gflops"], 6.0)

    def test_best_is_a_real_copied_library_not_a_link(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate.so"
            source.write_bytes(b"verified candidate library")
            target = publish_best_library(source, root / "best")
            self.assertTrue(target.is_file())
            self.assertFalse(target.is_symlink())
            self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_correctness_rejection_has_priority_over_generic_evaluation_failure(self) -> None:
        self.assertEqual(
            classify_evaluation_failure([{"phase": "correctness"}], True),
            "correctness_failed",
        )
        self.assertEqual(
            classify_evaluation_failure([{"phase": "performance"}], True),
            "evaluation_failed",
        )

    def test_full_run_publishes_only_verified_complete_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_root = root / "reference"
            reference_root.mkdir()
            test_bin = root / "test_unigemm"
            test_bin.write_text("#!/bin/sh\n", encoding="ascii")
            test_bin.chmod(0o755)
            output = root / "output"
            candidates_root = output / "candidates"
            candidates_root.mkdir(parents=True)

            base = driver.read_json(driver.DEFAULT_CONFIG)
            candidate_index = []
            for identifier, p, r in (("good", 64, 2048), ("bad", 128, 4096)):
                config = deepcopy(base)
                config["driver"]["p"] = p
                config["driver"]["r"] = r
                bundle = candidates_root / identifier
                driver.write_bundle(config, bundle, str(reference_root), False)
                (bundle / "libint8gemm_autogemm.so").write_bytes(identifier.encode("ascii"))
                candidate_index.append({"candidate_id": identifier, "bundle_dir": identifier})
            (candidates_root / "candidates.json").write_text(
                json.dumps({"candidates": candidate_index}), encoding="ascii"
            )

            def fake_run(command, env, timeout):
                del env, timeout
                if command[1] != str(workflow.EVALUATOR):
                    self.fail("unexpected command: %r" % (command,))
                library = Path(command[command.index("--library") + 1])
                results = Path(command[command.index("--results") + 1])
                rejections = Path(command[command.index("--rejections") + 1])
                if library.read_bytes() == b"bad":
                    rejections.parent.mkdir(parents=True, exist_ok=True)
                    rejections.write_text(
                        json.dumps({"phase": "correctness", "status": "rejected"}) + "\n",
                        encoding="ascii",
                    )
                    return {"command": list(command), "returncode": 1, "stdout": "", "stderr": "bad"}

                results.parent.mkdir(parents=True, exist_ok=True)
                lines = [
                    json.dumps(record(shape, index + 1.0, 100.0 + index))
                    for index, shape in enumerate(tuple(tuple(item) for item in base["shapes"]))
                ]
                results.write_text("\n".join(lines) + "\n", encoding="ascii")
                return {"command": list(command), "returncode": 0, "stdout": "ok", "stderr": ""}

            args = Namespace(
                reference_root=reference_root,
                test_bin=test_bin,
                output=output,
                config=driver.DEFAULT_CONFIG,
                p_values=None,
                r_values=None,
                mode="kblas",
                sme_cc=None,
                build_timeout=10.0,
                timeout=10.0,
                objective="geomean_gflops",
                run_id="integration",
                force=True,
                skip_generate=True,
                skip_build=True,
                skip_verify=False,
            )
            with mock.patch.object(workflow, "run_command", side_effect=fake_run), mock.patch.object(
                workflow,
                "dynamic_export_check",
                return_value=({"passed": True, "command": {}}, {"command": [], "stdout": "", "stderr": ""}),
            ):
                self.assertEqual(workflow.run_search(args), 0)

            summary = json.loads((output / "summary.json").read_text(encoding="ascii"))
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["winner"]["candidate_id"], "good")
            self.assertEqual([item["candidate_id"] for item in summary["ranking"]], ["good"])
            statuses = {entry["candidate_id"]: entry["status"] for entry in summary["candidates"]}
            self.assertEqual(statuses, {"good": "qualified", "bad": "correctness_failed"})
            self.assertEqual((output / "best" / "libint8gemm_autogemm.so").read_bytes(), b"good")

    def test_full_run_without_qualified_candidate_does_not_publish_best(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_root = root / "reference"
            reference_root.mkdir()
            test_bin = root / "test_unigemm"
            test_bin.write_text("#!/bin/sh\n", encoding="ascii")
            test_bin.chmod(0o755)
            output = root / "output"
            bundle = output / "candidates" / "bad"
            bundle.parent.mkdir(parents=True)
            driver.write_bundle(driver.read_json(driver.DEFAULT_CONFIG), bundle, str(reference_root), False)
            (bundle / "libint8gemm_autogemm.so").write_bytes(b"bad")
            (bundle.parent / "candidates.json").write_text(
                json.dumps({"candidates": [{"candidate_id": "bad", "bundle_dir": "bad"}]}),
                encoding="ascii",
            )

            def fake_run(command, env, timeout):
                del env, timeout
                rejections = Path(command[command.index("--rejections") + 1])
                rejections.parent.mkdir(parents=True, exist_ok=True)
                rejections.write_text(
                    json.dumps({"phase": "correctness", "status": "rejected"}) + "\n",
                    encoding="ascii",
                )
                return {"command": list(command), "returncode": 1, "stdout": "", "stderr": "bad"}

            args = Namespace(
                reference_root=reference_root,
                test_bin=test_bin,
                output=output,
                config=driver.DEFAULT_CONFIG,
                p_values=None,
                r_values=None,
                mode="kblas",
                sme_cc=None,
                build_timeout=10.0,
                timeout=10.0,
                objective="geomean_gflops",
                run_id="no-winner",
                force=True,
                skip_generate=True,
                skip_build=True,
                skip_verify=False,
            )
            with mock.patch.object(workflow, "run_command", side_effect=fake_run), mock.patch.object(
                workflow,
                "dynamic_export_check",
                return_value=({"passed": True, "command": {}}, {"command": [], "stdout": "", "stderr": ""}),
            ):
                self.assertEqual(workflow.run_search(args), 1)

            summary = json.loads((output / "summary.json").read_text(encoding="ascii"))
            self.assertEqual(summary["status"], "no_qualified_candidate")
            self.assertFalse((output / "best" / "libint8gemm_autogemm.so").exists())


if __name__ == "__main__":
    unittest.main()
