#!/usr/bin/env python3
"""Host-only contract tests for direct-link candidate testing."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
import json


MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

import test_candidate_sos as runner


class CandidateTestContractTest(unittest.TestCase):
    def test_command_matches_the_requested_numactl_and_omp_launch_shape(self) -> None:
        command = runner.make_test_command(
            "numactl",
            "570-601",
            "15,31",
            Path("/data1/cxz/int8/test_unigemm"),
            (8192, 8192, 2048),
            "kblas",
            "verify",
        )
        self.assertEqual(
            command,
            [
                "numactl",
                "--all",
                "--physcpubind=570-601",
                "-m",
                "15,31",
                "/data1/cxz/int8/test_unigemm",
                "8192",
                "8192",
                "2048",
                "kblas",
                "verify",
            ],
        )
        env = runner.test_environment({"LD_PRELOAD": "/other/library.so"}, "false", "cores", 32)
        self.assertEqual(
            env,
            {"OMP_PROC_BIND": "false", "OMP_PLACES": "cores", "OMP_NUM_THREADS": "32"},
        )
        self.assertEqual(
            runner.display_launch(command, env),
            "OMP_PROC_BIND=false OMP_PLACES=cores OMP_NUM_THREADS=32 "
            "numactl --all --physcpubind=570-601 -m 15,31 "
            "/data1/cxz/int8/test_unigemm 8192 8192 2048 kblas verify",
        )

    def test_library_swap_restores_a_regular_original_library(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            install = root / "libint8gemm.so"
            candidate = root / "candidate.so"
            journal = root / "install_state.json"
            install.write_bytes(b"original")
            candidate.write_bytes(b"candidate")

            with runner.LibrarySwap(install, journal) as swap:
                swap.activate(candidate)
                self.assertTrue(install.is_symlink())
                self.assertEqual(Path(os.readlink(install)), candidate)

            self.assertFalse(install.is_symlink())
            self.assertEqual(install.read_bytes(), b"original")
            state = journal.read_text(encoding="utf-8")
            self.assertIn('"state": "restored"', state)

    def test_library_swap_restores_an_original_symlink_without_resolving_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "reference-library.so"
            install = root / "libint8gemm.so"
            candidate = root / "candidate.so"
            journal = root / "install_state.json"
            original.write_bytes(b"original")
            candidate.write_bytes(b"candidate")
            install.symlink_to(original.name)

            with runner.LibrarySwap(install, journal) as swap:
                swap.activate(candidate)
                self.assertEqual(Path(os.readlink(install)), candidate)

            self.assertTrue(install.is_symlink())
            self.assertEqual(os.readlink(install), original.name)

    def test_verify_output_is_not_accepted_as_a_latency(self) -> None:
        self.assertIsNone(runner.parse_latency("average int8gemm time = -nan"))
        self.assertAlmostEqual(
            runner.parse_latency("average int8gemm time = 0.012345, average int8gemm effi = 1") or 0,
            0.012345,
        )

    def test_runner_rejects_shapes_outside_the_fixed_backend_contract(self) -> None:
        runner.validate_target_shape((8192, 8192, 2048))
        with self.assertRaisesRegex(ValueError, "K=2048"):
            runner.validate_target_shape((8192, 8192, 1024))
        with self.assertRaisesRegex(ValueError, "M/N"):
            runner.validate_target_shape((1024, 8192, 2048))

    def test_current_build_summary_filters_out_stale_flat_libraries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            libraries = root / "libraries"
            libraries.mkdir()
            current = libraries / "current.so"
            stale = libraries / "stale.so"
            current.write_bytes(b"current")
            stale.write_bytes(b"stale")
            (root / "build_summary.json").write_text(
                json.dumps({
                    "candidates": [{"status": "built", "library": str(current)}]
                }),
                encoding="utf-8",
            )

            self.assertEqual(runner.discover_libraries(libraries), [current.resolve()])


if __name__ == "__main__":
    unittest.main()
