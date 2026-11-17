"""Data-free regression tests for repository hardening behavior."""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.v2.extractor import CACHE_SCHEMA_VERSION, extract_v2_primitives
from src.v2.run_v2 import _ensure_output_parent
from tools.docs.check_quotes import verify_quote


class TestExternalCitationIsolation(unittest.TestCase):
    def _quote_item(self, quote: str = "External source quotation"):
        return {"citation": (True, "README.md", 1, 1), "quote": quote}

    def test_external_citation_never_falls_back_to_repo_readme(self):
        with tempfile.TemporaryDirectory() as repo_root:
            with open(os.path.join(repo_root, "README.md"), "w", encoding="utf-8") as f:
                f.write("External source quotation\n")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SKILLCORNER_DATA_DIR", None)
                status, _ = verify_quote(self._quote_item(), repo_root=repo_root)
            self.assertEqual(status, "SKIPPED")

    def test_external_citation_uses_explicit_external_root(self):
        with tempfile.TemporaryDirectory() as repo_root, tempfile.TemporaryDirectory() as ext_root:
            with open(os.path.join(ext_root, "README.md"), "w", encoding="utf-8") as f:
                f.write("External source quotation\n")
            with mock.patch.dict(os.environ, {"SKILLCORNER_DATA_DIR": ext_root}, clear=False):
                status, _ = verify_quote(self._quote_item(), repo_root=repo_root)
            self.assertEqual(status, "PASS")


class TestCacheProvenance(unittest.TestCase):
    def _write_inputs(self, root: str, ball_x: float = 0.10):
        txt_path = os.path.join(root, "tracking.txt")
        xml_path = os.path.join(root, "tracking.xml")
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(
                '<Root><Team id="home"/><Team id="away"/>'
                '<Player teamId="home"/><Player teamId="away"/></Root>'
            )
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"1:0.1,0.1,1;0.9,0.9,1:{ball_x},0.1,1\n")
            f.write(f"2:0.1,0.1,1;0.9,0.9,1:{ball_x},0.1,1\n")
        return txt_path, xml_path

    def test_cache_records_provenance_and_invalidates_on_input_change(self):
        with tempfile.TemporaryDirectory() as root:
            txt_path, xml_path = self._write_inputs(root, ball_x=0.10)
            cache_path = os.path.join(root, "primitives.npz")
            first = extract_v2_primitives(
                raw_txt_path=txt_path, raw_xml_path=xml_path,
                cache_path=cache_path, n_frames=2,
            )
            with np.load(cache_path, allow_pickle=False) as cached:
                self.assertEqual(int(cached["__cache_schema_version"].item()), CACHE_SCHEMA_VERSION)
                self.assertIn("__raw_txt_sha256", cached.files)
                self.assertIn("__raw_xml_sha256", cached.files)

            self._write_inputs(root, ball_x=0.20)
            second = extract_v2_primitives(
                raw_txt_path=txt_path, raw_xml_path=xml_path,
                cache_path=cache_path, n_frames=2,
            )
            self.assertNotEqual(float(first["ball_x_m"][0]), float(second["ball_x_m"][0]))

    def test_legacy_and_corrupt_caches_are_rebuilt(self):
        with tempfile.TemporaryDirectory() as root:
            txt_path, xml_path = self._write_inputs(root)
            cache_path = os.path.join(root, "primitives.npz")
            legacy = {
                key: np.zeros(2, dtype=np.float32)
                for key in (
                    "ball_valid", "ball_x_m", "ball_y_m", "d_h1", "d_a1",
                    "p75_player_speed",
                )
            }
            np.savez_compressed(cache_path, **legacy)
            extract_v2_primitives(
                raw_txt_path=txt_path, raw_xml_path=xml_path,
                cache_path=cache_path, n_frames=2,
            )
            with np.load(cache_path, allow_pickle=False) as rebuilt:
                self.assertIn("__cache_schema_version", rebuilt.files)

            with open(cache_path, "wb") as f:
                f.write(b"not-an-npz")
            result = extract_v2_primitives(
                raw_txt_path=txt_path, raw_xml_path=xml_path,
                cache_path=cache_path, n_frames=2,
            )
            self.assertEqual(result["ball_valid"].shape, (2,))


class TestOutputPathHandling(unittest.TestCase):
    def test_bare_and_nested_output_parents(self):
        with tempfile.TemporaryDirectory() as root:
            previous = os.getcwd()
            try:
                os.chdir(root)
                bare_parent = _ensure_output_parent("submission.csv")
                self.assertEqual(os.path.realpath(bare_parent), os.path.realpath(root))
                nested_path = os.path.join("nested", "results", "submission.csv")
                nested_parent = _ensure_output_parent(nested_path)
                self.assertTrue(os.path.isdir(nested_parent))
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
