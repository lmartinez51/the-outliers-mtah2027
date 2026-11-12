"""Unit test for project documentation quote verification."""

import os
import unittest
from tools.docs.check_quotes import run_quote_check, DEFAULT_DOCS


class TestDocsQuotes(unittest.TestCase):
    """Verifies that all quotes in project documentation match their cited source lines and have no uncited quotes."""

    def test_documentation_quotes_valid(self):
        target_docs = [d for d in DEFAULT_DOCS if os.path.exists(d)]
        self.assertGreater(len(target_docs), 0, "No target documentation files found to verify.")

        all_passed, results = run_quote_check(target_docs)
        self.assertGreater(len(results), 0, "No quotes were detected in documentation.")

        failures = [
            f"{r['doc']}:{r['doc_line']} [{r['citation']}]: status={r['status']} reason={r['reason']} (Quote: {r['quote'][:50]}...)"
            for r in results
            if r["status"] in ("FAIL", "UNCITED")
        ]
        self.assertEqual(
            len(failures), 0, f"Found {len(failures)} quote verification failure(s) or uncited quote(s):\n" + "\n".join(failures)
        )
        self.assertTrue(all_passed, "Coverage run_quote_check reported failure status.")


if __name__ == "__main__":
    unittest.main()
