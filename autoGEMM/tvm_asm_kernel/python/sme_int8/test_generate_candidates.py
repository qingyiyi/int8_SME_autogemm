#!/usr/bin/env python3
"""Host-only tests for JSON-configured INT8 SME candidate generation."""

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

import generate_candidates as candidates
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
        self.assertEqual(
            driver.search_space_json(self.baseline),
            {
                "p": [64, 128, 256],
                "r": [2048, 4096, 8192],
                "thread_groups": [
                    {"threads_m": 32, "threads_n": 1},
                    {"threads_m": 16, "threads_n": 2},
                ],
            },
        )

    def test_p_and_r_must_be_declared_in_json_search_space(self) -> None:
        invalid_p = deepcopy(self.baseline)
        invalid_p["driver"]["p"] = 512
        with self.assertRaisesRegex(ValueError, "driver.p"):
            driver.validate_config(invalid_p)

        invalid_r = deepcopy(self.baseline)
        invalid_r["driver"]["r"] = 1024
        with self.assertRaisesRegex(ValueError, "driver.r"):
            driver.validate_config(invalid_r)

    def test_p_and_r_can_be_changed_by_editing_json_search_space(self) -> None:
        config = deepcopy(self.baseline)
        config["search_space"]["p"] = [320]
        config["search_space"]["r"] = [3072]
        config["search_space"]["thread_groups"] = [
            {"threads_m": 16, "threads_n": 2},
        ]
        config["driver"].update({
            "p": 320,
            "r": 3072,
            "threads_m": 16,
            "threads_n": 2,
        })
        driver.validate_config(config)
        self.assertEqual(
            driver.candidate_id(config),
            "sme-int8-k2048-p320-r3072-j32-t16x2",
        )

    def test_thread_groups_are_indivisible_pairs_with_32_workers(self) -> None:
        config = candidates.candidate_config(self.baseline, 256, 8192, 16, 2)
        driver.validate_config(config)
        self.assertEqual(
            driver.candidate_id(config),
            "sme-int8-k2048-p256-r8192-j32-t16x2",
        )

        wrong_total = deepcopy(self.baseline)
        wrong_total["search_space"]["thread_groups"] = [
            {"threads_m": 32, "threads_n": 2},
        ]
        wrong_total["driver"]["threads_m"] = 32
        wrong_total["driver"]["threads_n"] = 2
        with self.assertRaisesRegex(ValueError, "exactly 32 worker threads"):
            driver.validate_config(wrong_total)

        unsupported_pair = deepcopy(self.baseline)
        unsupported_pair["search_space"]["thread_groups"] = [
            {"threads_m": 8, "threads_n": 4},
        ]
        unsupported_pair["driver"]["threads_m"] = 8
        unsupported_pair["driver"]["threads_n"] = 4
        with self.assertRaisesRegex(ValueError, "must be one of"):
            driver.validate_config(unsupported_pair)

        missing_pair = deepcopy(self.baseline)
        missing_pair["search_space"]["thread_groups"] = [
            {"threads_m": 32, "threads_n": 1},
        ]
        missing_pair["driver"]["threads_m"] = 16
        missing_pair["driver"]["threads_n"] = 2
        with self.assertRaisesRegex(ValueError, "not present in search_space.thread_groups"):
            driver.validate_config(missing_pair)

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
                manifest["buffer_contract"]["sb"]["reference_capacity_bytes"],
                33554432,
            )
            self.assertEqual(
                manifest["driver_synchronization"]["protocol"],
                "reference_full_panel_barrier",
            )
            self.assertTrue((bundle / "cblas_gemm_s8s8s32_autogemm.c").is_file())
            self.assertTrue((bundle / "Makefile").is_file())
            bundled_assembly = bundle / "assembly"
            self.assertEqual(
                sorted(path.name for path in bundled_assembly.iterdir()),
                sorted(driver.EMBEDDED_ASSEMBLY_FILES),
            )
            for name in driver.EMBEDDED_ASSEMBLY_FILES:
                self.assertEqual(
                    driver.sha256(bundled_assembly / name),
                    driver.sha256(driver.EMBEDDED_ASSEMBLY_DIR / name),
                )
            self.assertEqual(
                sorted(manifest["assembly_sources"]),
                sorted(driver.EMBEDDED_ASSEMBLY_FILES),
            )
            self.assertEqual(
                manifest["assembly_source_sha256"],
                {
                    name: driver.sha256(driver.EMBEDDED_ASSEMBLY_DIR / name)
                    for name in driver.EMBEDDED_ASSEMBLY_FILES
                },
            )
            stored_manifest = json.loads((bundle / "manifest.json").read_text(encoding="ascii"))
            self.assertEqual(stored_manifest["driver"], self.baseline["driver"])

    def test_generated_makefile_compiles_bundled_assembly_and_tracks_includes(self) -> None:
        makefile = driver.MAKEFILE_TEMPLATE.read_text(encoding="ascii")
        self.assertIn("ASM_DIR := assembly", makefile)
        self.assertNotIn("KERNEL_OBJECTS", makefile)
        self.assertIn("$(ASM_DIR)/gemm_sme_packing.S", makefile)
        self.assertIn("$(ASM_DIR)/gemm_sme_base.S", makefile)
        self.assertIn("$(ASM_DIR)/int8_gemm_common.S", makefile)
        self.assertIn("$(ASM_DIR)/gemm_sme_reg_defs.h", makefile)

    def test_16x2_candidate_has_separate_b_panels(self) -> None:
        config = candidates.candidate_config(self.baseline, 256, 8192, 16, 2)
        contract = driver.buffer_contract(config)
        self.assertEqual(contract["sa"]["required_bytes"], 16777216)
        self.assertEqual(contract["sb"]["required_bytes"], 33554432)
        source = driver.render(config)
        self.assertIn("int nthreadsM = 16;", source)
        self.assertIn("int nthreadsN = 2;", source)

    def test_bulk_generator_renders_eighteen_bundles(self) -> None:
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
            self.assertEqual(index["candidate_count"], 18)
            self.assertEqual(index["search_space"], driver.search_space_json(self.baseline))
            candidate_ids = {candidate["candidate_id"] for candidate in index["candidates"]}
            self.assertIn("sme-int8-k2048-p256-r8192-j32-t32x1", candidate_ids)
            self.assertIn("sme-int8-k2048-p256-r8192-j32-t16x2", candidate_ids)
            for candidate in index["candidates"]:
                bundle = output / candidate["bundle_dir"]
                self.assertTrue((bundle / "config.json").is_file())
                self.assertTrue((bundle / "manifest.json").is_file())
                self.assertTrue((bundle / "assembly" / "gemm_sme_base.S").is_file())

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
        driver.validate_config(config)
        source = driver.render(config)
        self.assertEqual(source.count("#pragma omp barrier"), 2)
        self.assertNotIn("bufferB[mypos] = NULL;", source)
        self.assertNotIn("#pragma omp flush", source)


if __name__ == "__main__":
    unittest.main()
