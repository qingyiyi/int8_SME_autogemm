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
            self.assertEqual(manifest["buffer_contract"]["sa"]["candidate_bytes"], 33554432)
            self.assertEqual(manifest["buffer_contract"]["sb"]["candidate_reference_allocation_bytes"], 67108864)
            self.assertEqual(
                manifest["driver_synchronization"]["b_ready_state"],
                "reference_single_nk_block",
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

    def test_baseline_keeps_reference_single_block_synchronization(self) -> None:
        source = driver.render(self.baseline)
        self.assertNotIn("bufferB[mypos] = NULL;", source)
        self.assertEqual(source.count("#pragma omp barrier"), 1)
        self.assertNotIn("#pragma omp flush", source)

    def test_render_resets_b_ready_state_for_multiple_n_blocks(self) -> None:
        config = deepcopy(self.baseline)
        config["driver"]["r"] = 2048
        source = driver.render(config)
        reset = "bufferB[mypos] = NULL;"
        publish = "bufferB[mypos] = bufbb;"
        self.assertEqual(source.count(reset), 1)
        self.assertEqual(source.count(publish), 1)
        reset_index = source.index(reset)
        self.assertGreaterEqual(source[:reset_index].count("#pragma omp barrier"), 1)
        self.assertGreaterEqual(source[reset_index:].count("#pragma omp barrier"), 1)
        self.assertIn("#pragma omp flush", source[source.index(publish):])


if __name__ == "__main__":
    unittest.main()
