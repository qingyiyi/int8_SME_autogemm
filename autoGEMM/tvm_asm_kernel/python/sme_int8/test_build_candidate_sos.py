#!/usr/bin/env python3
"""Host-only contract tests for the candidate-library build step."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

import build_candidate_sos as builder
import generate_driver as driver


class CandidateBuildContractTest(unittest.TestCase):
    def test_make_command_builds_bundled_assembly_sources(self) -> None:
        command = builder.make_build_command(
            Path("/tmp/out/sme-int8-k2048-p256-r8192-j32-t32x1"),
            Path("/data1/cxz/int8gemm-kblas"),
            "/toolchain/BiSheng-clang",
        )

        self.assertEqual(command[:3], ["make", "-C", "/tmp/out/sme-int8-k2048-p256-r8192-j32-t32x1"])
        self.assertIn("REF_ROOT=/data1/cxz/int8gemm-kblas", command)
        self.assertIn("SME_CC=/toolchain/BiSheng-clang", command)
        self.assertEqual(command[-2:], ["all", "check"])
        self.assertNotIn("KERNEL_OBJECTS", " ".join(command))
        self.assertNotIn("test_unigemm", " ".join(command))

    def test_build_index_uses_paired_thread_groups_from_json(self) -> None:
        base = driver.read_json(driver.DEFAULT_CONFIG)
        configured = driver.configured_search_space(base)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidates"
            output.mkdir()
            index = builder.build_index(
                driver.DEFAULT_CONFIG,
                base,
                Path("/reference/int8gemm-kblas"),
                configured["p"],
                configured["r"],
                output,
                False,
            )
            self.assertEqual(index["candidate_count"], 18)
            self.assertEqual(index["search_space"], driver.search_space_json(base))
            self.assertEqual(
                index["fixed_driver"],
                {
                    "q": 2048,
                    "jblock": 32,
                    "region_align": 4,
                    "b_packing": "reference_full_panel_barrier",
                },
            )
            candidate_ids = {candidate["candidate_id"] for candidate in index["candidates"]}
            self.assertIn("sme-int8-k2048-p64-r2048-j32-t32x1", candidate_ids)
            self.assertIn("sme-int8-k2048-p64-r2048-j32-t16x2", candidate_ids)

    def test_tail_keeps_compiler_tail_and_real_newline(self) -> None:
        output = builder.tail("x" * (builder.MAX_CAPTURE_BYTES + 10))
        self.assertTrue(output.startswith("...[truncated]...\n"))
        self.assertEqual(len(output), builder.MAX_CAPTURE_BYTES + len("...[truncated]...\n"))


if __name__ == "__main__":
    unittest.main()
