"""Unit test for project documentation quote verification."""

import os
import unittest
from tools.docs.check_quotes import run_quote_check


class TestDocsQuotes(unittest.TestCase):
    """Verifies that all quotes in project documentation match their cited source lines."""

    def test_documentation_quotes_valid(self):
        docs = [
            "README.md",
            "README_v2.md",
            "CONSTANTS.md",
            "TECHNICAL_DOCUMENT.md",
            os.path.join("tools", "tuning", "README.md"),
        ]
        target_docs = [d for d in docs if os.path.exists(d)]
        self.assertGreater(len(target_docs), 0, "No target documentation files found to verify.")

        all_passed, results = run_quote_check(target_docs)
        self.assertGreater(len(results), 0, "No quotes were detected in documentation.")

        failures = [
            f"{r['doc']}:{r['doc_line']} [{r['citation']}]: {r['reason']} (Quote: {r['quote'][:50]}...)"
            for r in results
            if r["status"] == "FAIL"
        ]
        self.assertEqual(len(failures), 0, f"Found {len(failures)} quote verification failure(s):\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
