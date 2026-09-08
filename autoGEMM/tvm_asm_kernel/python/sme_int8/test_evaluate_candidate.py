#!/usr/bin/env python3
"""Host-only contract tests for the SME candidate evaluator."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

import evaluate_candidate as evaluator


class CandidateEvaluatorContractTest(unittest.TestCase):
    def test_current_test_harness_performance_line_is_parseable(self) -> None:
        """Keep the parser aligned with int8/test_unigemm.cpp's printed line."""
        output = (
            "M=2048 N=4096 K=2048ldc=2048  int8_gemm_time: 0.123456 seconds\n"
            "average int8gemm time = 0.012345, average int8gemm effi = 88.00\n"
        )

        match = evaluator.PERF_RE.search(output)

        self.assertIsNotNone(match)
        assert match is not None
        self.assertAlmostEqual(float(match.group(1)), 0.012345)

    def test_candidate_library_is_prepended_to_existing_preload(self) -> None:
        with mock.patch.dict(
            evaluator.os.environ,
            {"LD_PRELOAD": "/existing/libdependency.so", "KEEP": "yes"},
            clear=True,
        ):
            env = evaluator.candidate_env(Path("/candidate/libint8gemm_autogemm.so"))

        self.assertEqual(
            env["LD_PRELOAD"],
            "/candidate/libint8gemm_autogemm.so:/existing/libdependency.so",
        )
        self.assertEqual(env["KEEP"], "yes")

    def test_test_harness_command_uses_verify_or_zero_flag(self) -> None:
        completed = mock.Mock(returncode=0, stdout="ok", stderr="")
        shape = (2048, 4096, 2048)
        test_bin = Path("/tmp/test_unigemm")
        env = {"LD_PRELOAD": "/tmp/libint8gemm_autogemm.so"}

        for verify, expected_flag in ((True, "verify"), (False, "0")):
            with self.subTest(verify=verify), mock.patch.object(
                evaluator.subprocess, "run", return_value=completed
            ) as run:
                result = evaluator.run_case(test_bin, shape, "kblas", verify, env, 5.0)

            self.assertEqual(result["returncode"], 0)
            self.assertEqual(
                run.call_args.args[0],
                [str(test_bin), "2048", "4096", "2048", "kblas", expected_flag],
            )
            self.assertEqual(run.call_args.kwargs["env"], env)


if __name__ == "__main__":
    unittest.main()
