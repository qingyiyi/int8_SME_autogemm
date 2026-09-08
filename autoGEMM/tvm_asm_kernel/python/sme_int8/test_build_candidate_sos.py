#!/usr/bin/env python3
"""Host-only contract tests for the candidate-library build step."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

import build_candidate_sos as builder


class CandidateBuildContractTest(unittest.TestCase):
    def test_make_command_only_builds_bundle_and_reference_objects(self) -> None:
        command = builder.make_build_command(
            Path("/tmp/out/sme-int8-k2048-p256-r8192-j32-t32x1"),
            Path("/data1/cxz/int8gemm-kblas"),
            [Path("/obj/kernel.o"), Path("/obj/pack_a.o"), Path("/obj/pack_b.o")],
            "/toolchain/BiSheng-clang",
        )

        self.assertEqual(command[:3], ["make", "-C", "/tmp/out/sme-int8-k2048-p256-r8192-j32-t32x1"])
        self.assertIn("REF_ROOT=/data1/cxz/int8gemm-kblas", command)
        self.assertIn("KERNEL_OBJECTS=/obj/kernel.o /obj/pack_a.o /obj/pack_b.o", command)
        self.assertIn("SME_CC=/toolchain/BiSheng-clang", command)
        self.assertEqual(command[-2:], ["all", "check"])
        self.assertNotIn("test_unigemm", " ".join(command))

    def test_tail_keeps_compiler_tail_and_real_newline(self) -> None:
        output = builder.tail("x" * (builder.MAX_CAPTURE_BYTES + 10))
        self.assertTrue(output.startswith("...[truncated]...\n"))
        self.assertEqual(len(output), builder.MAX_CAPTURE_BYTES + len("...[truncated]...\n"))


if __name__ == "__main__":
    unittest.main()
