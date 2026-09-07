#!/usr/bin/env python3
"""Host-only tests for constrained INT8 SME candidate generation."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

import generate_driver as driver


class CandidateGenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = driver.read_json(driver.DEFAULT_CONFIG)

    def test_baseline_is_an_allowed_candidate(self) -> None:
        driver.validate_config(self.baseline)
        self.assertEqual(
            driver.candidate_id(self.baseline),
            "sme-int8-k2048-p256-r8192-j32-t32x1",
        )

    def test_rejects_unsafe_p_and_r(self) -> None:
        invalid_p = deepcopy(self.baseline)
        invalid_p["driver"]["p"] = 512
        with self.assertRaisesRegex(ValueError, "driver.p"):
            driver.validate_config(invalid_p)

        invalid_r = deepcopy(self.baseline)
        invalid_r["driver"]["r"] = 1024
        with self.assertRaisesRegex(ValueError, "driver.r"):
            driver.validate_config(invalid_r)

    def test_bundle_manifest_matches_external_buffer_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "baseline"
            manifest = driver.write_bundle(
                self.baseline,
                bundle,
                "/reference/int8gemm-kblas",
                False,
            )
            self.assertEqual(manifest["candidate_id"], "sme-int8-k2048-p256-r8192-j32-t32x1")
            self.assertEqual(manifest["buffer_contract"]["sa"]["required_bytes"], 16777216)
            self.assertEqual(manifest["buffer_contract"]["sb"]["required_bytes"], 16777216)
            self.assertEqual(
                manifest["driver_synchronization"]["protocol"],
                "reference_full_panel_barrier",
            )
            self.assertTrue((bundle / "cblas_gemm_s8s8s32_autogemm.c").is_file())
            self.assertTrue((bundle / "Makefile").is_file())
            stored_manifest = json.loads((bundle / "manifest.json").read_text(encoding="ascii"))
            self.assertEqual(stored_manifest["driver"], self.baseline["driver"])

    def test_bulk_generator_renders_nine_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidates"
            result = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_DIR / "generate_candidates.py"),
                    "--output",
                    str(output),
                    "--reference-root",
                    "/reference/int8gemm-kblas",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            index = json.loads((output / "candidates.json").read_text(encoding="ascii"))
            self.assertEqual(index["candidate_count"], 9)
            self.assertEqual(index["search_space"], {"p": [64, 128, 256], "r": [2048, 4096, 8192]})
            for candidate in index["candidates"]:
                bundle = output / candidate["bundle_dir"]
                self.assertTrue((bundle / "config.json").is_file())
                self.assertTrue((bundle / "manifest.json").is_file())

    def test_render_matches_reference_full_panel_barrier_protocol(self) -> None:
        source = driver.render(self.baseline)
        self.assertNotIn("bufferB[mypos] = NULL;", source)
        self.assertEqual(source.count("#pragma omp barrier"), 2)
        self.assertNotIn("#pragma omp flush", source)
        self.assertNotIn("while (flag)", source)
        self.assertIn("int myJ = (minJ + threads - 1) / threads;", source)
        self.assertIn("const int start_off = (int)js;", source)
        self.assertIn(
            "sa + (LEVEL3_GEMM_Q * LEVEL3_GEMM_P * thread_id)",
            source,
        )

    def test_reduced_r_uses_the_same_per_panel_barrier_protocol(self) -> None:
        config = deepcopy(self.baseline)
        config["driver"]["r"] = 2048
        source = driver.render(config)
        self.assertEqual(source.count("#pragma omp barrier"), 2)
        self.assertNotIn("bufferB[mypos] = NULL;", source)
        self.assertNotIn("#pragma omp flush", source)
        self.assertNotIn("while (flag)", source)


if __name__ == "__main__":
    unittest.main()
