# -*- coding: utf-8 -*-
"""Verification test suite for v2 deterministic possession pipeline (Stage 2B).

Tests:
    1. GT firewall: Greps src/v2/ for forbidden strings.
    2. Determinism: Two successive runs yield bit-for-bit identical SHA-256.
    3. Label vocabulary: All labels in {'H', 'A', 'D', 'X'}.
    4. Graded block boundaries: 'X' strictly outside graded blocks, 0 'X' inside.
    5. Cache firewall: Confirms no read of probability cache or v1 CSV files.
    6. Official format validator: validate_submission.py passes with 'OK'.
    7. Clean-clone test: Starts with NO cache files, runs from raw tracking, reproduces SHA-256.
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

BASELINE_HASH = "89ff92a62a6e571430c0ff06716f6cf1136c8367cbf3eb1c3be4dfcfb04ef6f2"


class TestV2Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primary_csv = os.path.join("outputs_v2", "submission_v2.csv")
        execute_pipeline(output_path=cls.primary_csv)
        with open(cls.primary_csv, "rb") as f:
            cls.primary_hash = hashlib.sha256(f.read()).hexdigest()

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
        """Test 2: Determinism - two runs produce bit-for-bit identical SHA-256."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            run2_path = os.path.join(tmp_dir, "submission_v2_run2.csv")
            execute_pipeline(output_path=run2_path)
            with open(run2_path, "rb") as f:
                hash2 = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                self.primary_hash,
                hash2,
                f"Non-deterministic pipeline output: {self.primary_hash} != {hash2}",
            )
            self.assertEqual(
                self.primary_hash,
                BASELINE_HASH,
                f"Hash deviated from baseline: {self.primary_hash} != {BASELINE_HASH}",
            )

    def test_03_label_vocabulary(self):
        """Test 3: Labels strictly in {H, A, D, X}."""
        valid_labels = {"H", "A", "D", "X"}
        seen_labels = set()
        count = 0
        with open(self.primary_csv, "r", encoding="utf-8") as f:
            header = f.readline().strip()
            self.assertEqual(header, "frame,label", "Malformed CSV header")
            for line in f:
                parts = line.strip().split(",")
                self.assertEqual(len(parts), 2, f"Malformed line: {line.strip()}")
                seen_labels.add(parts[1])
                count += 1

        self.assertEqual(count, N_VIDEO_FRAMES, f"Expected {N_VIDEO_FRAMES} rows, got {count}")
        self.assertTrue(
            seen_labels.issubset(valid_labels),
            f"Invalid labels found: {seen_labels - valid_labels}",
        )

    def test_04_block_boundaries(self):
        """Test 4: X exactly outside graded blocks, 0 X inside graded blocks."""
        labels = []
        with open(self.primary_csv, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                labels.append(line.strip().split(",")[1])

        graded_ranges = [
            (HALF_1_START, HALF_1_END),
            (HALF_2_START, HALF_2_END),
        ]
        graded_mask = [False] * N_VIDEO_FRAMES
        for s, e in graded_ranges:
            for i in range(s, e + 1):
                graded_mask[i] = True

        x_inside_graded = sum(1 for i in range(N_VIDEO_FRAMES) if graded_mask[i] and labels[i] == "X")
        non_x_outside = sum(1 for i in range(N_VIDEO_FRAMES) if not graded_mask[i] and labels[i] != "X")
        total_x = sum(1 for l in labels if l == "X")

        self.assertEqual(x_inside_graded, 0, f"Found {x_inside_graded} 'X' labels inside graded blocks")
        self.assertEqual(non_x_outside, 0, f"Found {non_x_outside} non-'X' labels outside graded blocks")
        self.assertEqual(total_x, 40462, f"Expected 40462 total 'X' frames, got {total_x}")

    def test_05_cache_firewall(self):
        """Test 5: Confirms no read of probability cache or v1 CSVs."""
        forbidden_files = [
            "cache_bar_ath_probs.npz",
            "submission.csv",
            "The_Outliers.csv",
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
        """Test 6: Format validator passes with 'OK'."""
        cmd = [
            sys.executable,
            "-B",
            os.path.join("evaluation", "validate_submission.py"),
            self.primary_csv,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(
            proc.returncode,
            0,
            f"validate_submission.py failed:\n{proc.stdout}\n{proc.stderr}",
        )
        self.assertIn("OK    submission_v2.csv", proc.stdout)
        self.assertIn("Valid.", proc.stdout)

    def test_07_clean_clone_no_cache_reproducibility(self):
        """Test 7: Clean-clone scratch copy starts with NO cache files and reproduces baseline hash."""
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

            # Confirm no cache files exist in scratch_dir
            found_caches = glob.glob(os.path.join(scratch_dir, "**", "*.npz"), recursive=True)
            self.assertEqual(found_caches, [], "Scratch clone must start with zero .npz caches")

            cmd = [
                sys.executable,
                "-B",
                "-m",
                "src.v2.run_v2",
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = scratch_dir
            proc = subprocess.run(
                cmd,
                cwd=scratch_dir,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"Clone run failed:\n{proc.stdout}\n{proc.stderr}",
            )

            clone_csv = os.path.join(scratch_dir, "outputs_v2", "submission_v2.csv")
            self.assertTrue(os.path.isfile(clone_csv), "Clone run did not produce CSV")
            with open(clone_csv, "rb") as f:
                clone_hash = hashlib.sha256(f.read()).hexdigest()

            self.assertEqual(
                BASELINE_HASH,
                clone_hash,
                f"Clean-clone hash mismatch:\nBaseline: {BASELINE_HASH}\nClone:    {clone_hash}",
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
