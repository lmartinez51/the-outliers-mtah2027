import os
import tempfile
import unittest

from src.v2.smart_tagging import load_set_piece_support


class TestSmartTaggingSupport(unittest.TestCase):
    def test_only_set_piece_windows_are_support(self):
        xml = """<root>
        <instance><start>1</start><end>2</end><code>SET PIECES</code></instance>
        <instance><start>3</start><end>4</end><code>BUILD UP</code></instance>
        </root>"""
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "tags.xml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(xml)
            mask = load_set_piece_support(path, n_frames=100, fps=10)
        self.assertTrue(mask[10:20].all())
        self.assertFalse(mask[:10].any())
        self.assertFalse(mask[20:].any())

    def test_missing_optional_file_is_neutral(self):
        mask = load_set_piece_support(None, n_frames=5, fps=25)
        self.assertFalse(mask.any())

    def test_configured_missing_file_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            load_set_piece_support("does-not-exist.xml", n_frames=5, fps=25)
