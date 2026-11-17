# -*- coding: utf-8 -*-
"""Verification test suite for v2 and v2.1 deterministic possession pipeline (Stages 2B, 3, and 4A).

Tests:
    1. GT firewall: Greps src/v2/ for forbidden strings.
    2. Determinism: Two successive runs yield bit-for-bit identical SHA-256 for both v2.1 and apriori.
    3. Label vocabulary: All labels in {'H', 'A', 'D', 'X'}.
    4. Graded block boundaries: 'X' strictly outside graded blocks, 0 'X' inside.
    5. Cache firewall: Confirms no read of probability cache or v1 CSV files.
    6. Official format validator: validate_submission.py passes with 'OK' for both outputs.
    7. Clean-clone test: Starts with NO cache files, runs from raw tracking, reproduces both hashes.
    8. No legacy imports: v2 imports nothing from src.mtah_* or src.jev_*.
    9. No v1 feature cache read: v2 code never references cache_bar_ath_*.
"""

import glob
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from src.v2.constants import (
    HALF_1_END,
    HALF_1_START,
    HALF_2_END,
    HALF_2_START,
    N_VIDEO_FRAMES,
)
from src.v2.run_v2 import execute_pipeline

HASH_V2_1 = "828baabcd1344a81fb57ed0c41007cb647c1dd6b40507e4beddef419c65d3525"
HASH_APRIORI = "8853814bd01d753ac9390249df6171aca8d367ad2d5eff460ef20c8524f3e545"


class TestV2Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir_obj = tempfile.TemporaryDirectory()
        cls.tmp_dir = cls.tmp_dir_obj.name

        cls.v2_1_csv = os.path.join(cls.tmp_dir, "submission_v2_1_external.csv")
        cls.apriori_csv = os.path.join(cls.tmp_dir, "submission_v2.csv")

        execute_pipeline(output_path=cls.v2_1_csv, config="v2_1", overwrite=True)
        execute_pipeline(output_path=cls.apriori_csv, config="apriori", overwrite=True)

        with open(cls.v2_1_csv, "rb") as f:
            cls.v2_1_hash = hashlib.sha256(f.read()).hexdigest()
        with open(cls.apriori_csv, "rb") as f:
            cls.apriori_hash = hashlib.sha256(f.read()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir_obj.cleanup()

    def test_01_gt_firewall(self):
        """Test 1: GT firewall - grep src/v2/ for forbidden strings."""
        forbidden = [
            "evaluation/gt",
            "possession_gt",
            "public.csv",
            "PUBLIC_SLICE",
            "possession_metrics",
        ]
        v2_files = glob.glob(os.path.join("src", "v2", "**", "*.py"), recursive=True)
        self.assertGreater(len(v2_files), 0, "No v2 Python files found")

        violations = []
        for fpath in v2_files:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f, start=1):
                    for token in forbidden:
                        if token in line:
                            violations.append(f"{fpath}:{line_idx} contains '{token}'")

        self.assertEqual(
            violations,
            [],
            f"GT firewall violated by forbidden strings:\n" + "\n".join(violations),
        )

    def test_02_determinism(self):
        """Test 2: Determinism - verify baseline hashes and run-to-run determinism."""
        self.assertEqual(
            self.v2_1_hash,
            HASH_V2_1,
            f"v2.1 hash mismatch: {self.v2_1_hash} != {HASH_V2_1}",
        )
        self.assertEqual(
            self.apriori_hash,
            HASH_APRIORI,
            f"apriori baseline hash mismatch: {self.apriori_hash} != {HASH_APRIORI}",
        )

        with tempfile.TemporaryDirectory() as tmp_run2:
            run2_v2_1 = os.path.join(tmp_run2, "sub_v2_1_run2.csv")
            run2_apriori = os.path.join(tmp_run2, "sub_apriori_run2.csv")

            execute_pipeline(output_path=run2_v2_1, config="v2_1", overwrite=True)
            execute_pipeline(output_path=run2_apriori, config="apriori", overwrite=True)

            with open(run2_v2_1, "rb") as f:
                hash2_v2_1 = hashlib.sha256(f.read()).hexdigest()
            with open(run2_apriori, "rb") as f:
                hash2_apriori = hashlib.sha256(f.read()).hexdigest()

            self.assertEqual(self.v2_1_hash, hash2_v2_1, "Non-deterministic v2.1 execution")
            self.assertEqual(self.apriori_hash, hash2_apriori, "Non-deterministic apriori execution")

    def test_03_label_vocabulary(self):
        """Test 3: Labels strictly in {H, A, D, X} for both configurations."""
        valid_labels = {"H", "A", "D", "X"}

        for csv_path, cfg_name in [(self.v2_1_csv, "v2.1"), (self.apriori_csv, "apriori")]:
            seen_labels = set()
            count = 0
            with open(csv_path, "r", encoding="utf-8") as f:
                header = f.readline().strip()
                self.assertEqual(header, "frame,label", f"{cfg_name}: Malformed CSV header")
                for line in f:
                    parts = line.strip().split(",")
                    self.assertEqual(len(parts), 2, f"{cfg_name}: Malformed line: {line.strip()}")
                    seen_labels.add(parts[1])
                    count += 1

            self.assertEqual(count, N_VIDEO_FRAMES, f"{cfg_name}: Expected {N_VIDEO_FRAMES} rows, got {count}")
            self.assertTrue(
                seen_labels.issubset(valid_labels),
                f"{cfg_name}: Invalid labels found: {seen_labels - valid_labels}",
            )

    def test_04_block_boundaries(self):
        """Test 4: X exactly outside graded blocks, 0 X inside graded blocks."""
        graded_ranges = [
            (HALF_1_START, HALF_1_END),
            (HALF_2_START, HALF_2_END),
        ]
        graded_mask = [False] * N_VIDEO_FRAMES
        for s, e in graded_ranges:
            for i in range(s, e + 1):
                graded_mask[i] = True

        for csv_path, cfg_name in [(self.v2_1_csv, "v2.1"), (self.apriori_csv, "apriori")]:
            labels = []
            with open(csv_path, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    labels.append(line.strip().split(",")[1])

            x_inside_graded = sum(1 for i in range(N_VIDEO_FRAMES) if graded_mask[i] and labels[i] == "X")
            non_x_outside = sum(1 for i in range(N_VIDEO_FRAMES) if not graded_mask[i] and labels[i] != "X")
            total_x = sum(1 for l in labels if l == "X")

            self.assertEqual(x_inside_graded, 0, f"{cfg_name}: Found {x_inside_graded} 'X' labels inside graded blocks")
            self.assertEqual(non_x_outside, 0, f"{cfg_name}: Found {non_x_outside} non-'X' labels outside graded blocks")
            self.assertEqual(total_x, 40462, f"{cfg_name}: Total 'X' frames must be exactly 40,462, got {total_x}")

    def test_05_cache_firewall(self):
        """Test 5: v2 code never imports, loads, or references v1 cache or probability artifacts."""
        forbidden_files = [
            "cache_bar_ath_probs.npz",
            "submission_v1",
            "submission_pub",
            "BAR-ATH_probs",
        ]
        v2_files = glob.glob(os.path.join("src", "v2", "**", "*.py"), recursive=True)
        violations = []
        for fpath in v2_files:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                for target in forbidden_files:
                    if target in content:
                        violations.append(f"{fpath} references {target}")

        self.assertEqual(
            violations,
            [],
            f"Cache firewall violated by references to v1 artifacts:\n" + "\n".join(violations),
        )

    def test_06_official_validator(self):
        """Test 6: Format validator passes with 'OK' on both configurations."""
        for csv_path in (self.v2_1_csv, self.apriori_csv):
            cmd = [
                sys.executable,
                "-B",
                os.path.join("evaluation", "validate_submission.py"),
                csv_path,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                proc.returncode,
                0,
                f"validate_submission.py failed on {csv_path}:\n{proc.stdout}\n{proc.stderr}",
            )
            self.assertIn("Valid.", proc.stdout)

    def test_07_clean_clone_no_cache_reproducibility(self):
        """Test 7: Clean-clone scratch copy starts with NO cache files and reproduces both hashes."""
        with tempfile.TemporaryDirectory() as scratch_dir:
            shutil.copytree("src", os.path.join(scratch_dir, "src"))

            # Provide ONLY raw tracking files (NO cache files under data/)
            scratch_data = os.path.join(scratch_dir, "data")
            scratch_tactical = os.path.join(scratch_data, "tactical")
            os.makedirs(scratch_tactical, exist_ok=True)
            shutil.copy(
                os.path.join("data", "tactical", "BAR-ATH_tactical_atd.txt"),
                os.path.join(scratch_tactical, "BAR-ATH_tactical_atd.txt"),
            )
            shutil.copy(
                os.path.join("data", "tactical", "BAR-ATH_tactical_atd.xml"),
                os.path.join(scratch_tactical, "BAR-ATH_tactical_atd.xml"),
            )
            shutil.copy(
                os.path.join("data", "tactical", "BAR-ATH_tactical_smart_tagging.xml"),
                os.path.join(scratch_tactical, "BAR-ATH_tactical_smart_tagging.xml"),
            )

            # Confirm no cache files exist in scratch_dir
            found_caches = glob.glob(os.path.join(scratch_dir, "**", "*.npz"), recursive=True)
            self.assertEqual(found_caches, [], "Scratch clone must start with zero .npz caches")

            # 7a. Test default invocation (v2_1)
            cmd_v2_1 = [
                sys.executable,
                "-B",
                "-m",
                "src.v2.run_v2",
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = scratch_dir
            proc_v2_1 = subprocess.run(
                cmd_v2_1,
                cwd=scratch_dir,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(
                proc_v2_1.returncode,
                0,
                f"Clone default run failed:\n{proc_v2_1.stdout}\n{proc_v2_1.stderr}",
            )

            clone_v2_1_csv = os.path.join(scratch_dir, "outputs_v2", "submission_v2_1_external.csv")
            self.assertTrue(os.path.isfile(clone_v2_1_csv), "Clone run did not produce default v2_1 CSV")
            with open(clone_v2_1_csv, "rb") as f:
                clone_v2_1_hash = hashlib.sha256(f.read()).hexdigest()

            self.assertEqual(
                HASH_V2_1,
                clone_v2_1_hash,
                f"Clean-clone default hash mismatch:\nExpected: {HASH_V2_1}\nClone:    {clone_v2_1_hash}",
            )

            # 7b. Test apriori configuration invocation
            cmd_apriori = [
                sys.executable,
                "-B",
                "-m",
                "src.v2.run_v2",
                "--config",
                "apriori",
            ]
            proc_apriori = subprocess.run(
                cmd_apriori,
                cwd=scratch_dir,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(
                proc_apriori.returncode,
                0,
                f"Clone apriori run failed:\n{proc_apriori.stdout}\n{proc_apriori.stderr}",
            )

            clone_apriori_csv = os.path.join(scratch_dir, "outputs_v2", "submission_v2.csv")
            self.assertTrue(os.path.isfile(clone_apriori_csv), "Clone run did not produce apriori CSV")
            with open(clone_apriori_csv, "rb") as f:
                clone_apriori_hash = hashlib.sha256(f.read()).hexdigest()

            self.assertEqual(
                HASH_APRIORI,
                clone_apriori_hash,
                f"Clean-clone apriori hash mismatch:\nExpected: {HASH_APRIORI}\nClone:    {clone_apriori_hash}",
            )

    def test_08_no_legacy_imports(self):
        """Test 8: v2 files import nothing from src.mtah_* or src.jev_*."""
        v2_files = glob.glob(os.path.join("src", "v2", "**", "*.py"), recursive=True)
        violations = []
        forbidden_modules = ["mtah_", "jev_", "starter", "evaluation"]
        for fpath in v2_files:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f, start=1):
                    if "import" in line:
                        for mod in forbidden_modules:
                            if mod in line:
                                violations.append(f"{fpath}:{idx}: {line.strip()}")

        self.assertEqual(violations, [], f"Forbidden legacy imports in v2:\n" + "\n".join(violations))

    def test_09_no_v1_cache_read(self):
        """Test 9: v2 files never reference cache_bar_ath_*."""
        v2_files = glob.glob(os.path.join("src", "v2", "**", "*.py"), recursive=True)
        violations = []
        for fpath in v2_files:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f, start=1):
                    if "cache_bar_ath" in line:
                        violations.append(f"{fpath}:{idx}: {line.strip()}")

        self.assertEqual(violations, [], f"References to cache_bar_ath in v2:\n" + "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
