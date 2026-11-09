"""Unit tests verifying the external data tuning firewall.

Rule S1 from STAGE 3:
All tuning code lives in tools/tuning/. It must not read, import, or reference:
- BAR-ATH data or fixtures
- data/tactical/
- data/cache_*
- outputs_v2/
- evaluation/gt/
- or any v1 file (mtah_*, jev_*)
evaluation/possession_metrics.py may be imported ONLY from tools/tuning/ and
called ONLY on arrays derived from external data.
"""

import os
import glob
import unittest


class TestTuningFirewall(unittest.TestCase):
    """Firewall tests for tools/tuning/."""

    def test_tuning_firewall_forbidden_strings(self):
        """Grep tools/tuning/ for forbidden BAR-ATH and v1 strings."""
        forbidden = [
            "evaluation/gt",
            "possession_gt",
            "public.csv",
            "PUBLIC_SLICE",
            "data/tactical",
            "data/cache",
            "outputs_v2",
            "BAR-ATH_",
            "mtah_",
            "jev_",
        ]
        
        tuning_py_files = glob.glob(os.path.join("tools", "tuning", "**", "*.py"), recursive=True)
        violations = []
        for fpath in tuning_py_files:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f, start=1):
                    # Strip comments or test statements if any, but strictly check tokens
                    for token in forbidden:
                        if token in line:
                            violations.append(f"{fpath}:{line_idx} contains '{token}'")

        self.assertEqual(
            violations,
            [],
            "Tuning firewall violated by forbidden strings:\n" + "\n".join(violations),
        )

    def test_tuning_firewall_isolated_from_src_v2(self):
        """Confirm src/v2/ does not import tools/tuning/."""
        v2_files = glob.glob(os.path.join("src", "v2", "**", "*.py"), recursive=True)
        violations = []
        for fpath in v2_files:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f, start=1):
                    if "tools.tuning" in line or "tools/tuning" in line:
                        violations.append(f"{fpath}:{line_idx} imports tuning")

        self.assertEqual(
            violations,
            [],
            "src/v2/ must not import tuning tools:\n" + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
