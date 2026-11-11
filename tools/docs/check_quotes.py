#!/usr/bin/env python3
"""Automated Quote Verification Tool for Project Documentation.

Verifies that quoted text citing (FILE:LINE) or (FILE:START-END)
actually appears in the cited source lines.

Convention:
  A quote is a blockquote ('> ...') or inline quoted string ('"..."')
  associated with a citation: (FILE:LINE), (FILE:START-END),
  (ext:FILE:LINE), or (ext:FILE:START-END).
"""

import os
import re
import sys
from typing import Dict, List, Optional, Tuple


def normalize_text(text: str) -> str:
    """Normalize text by stripping markdown formatting, escapes, and whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[`*_#\>\$\\\"\']", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def parse_citation(cit_str: str) -> Optional[Tuple[bool, str, int, int]]:
    """Parse a citation string like 'FILE:LINE', 'FILE:START-END', 'ext:FILE:LINE'."""
    cit_str = cit_str.strip()
    cit_str = re.sub(r"^[\(\[\s]+", "", cit_str)
    cit_str = re.sub(r"[\)\]\s]+$", "", cit_str)

    is_ext = False
    if cit_str.startswith("ext:"):
        is_ext = True
        cit_str = cit_str[4:]

    m = re.match(r"^(.+?):(\d+)(?:-(\d+))?$", cit_str)
    if not m:
        return None

    path = m.group(1).strip()
    start_l = int(m.group(2))
    end_l = int(m.group(3)) if m.group(3) else start_l
    return (is_ext, path, start_l, end_l)


def extract_quotes_from_doc(doc_path: str) -> List[Dict[str, any]]:
    """Extract all quotes and their associated citations from a markdown document."""
    if not os.path.exists(doc_path):
        return []

    with open(doc_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    quotes = []
    num_lines = len(lines)
    i = 0

    citation_pattern = re.compile(
        r"\(((?:ext:)?[a-zA-Z0-9_\-\./\\]+:\d+(?:-\d+)?)\)|"
        r"\[((?:ext:)?[a-zA-Z0-9_\-\./\\]+:\d+(?:-\d+)?)\]\([^\)]+\)"
    )

    while i < num_lines:
        line = lines[i]
        line_num = i + 1
        stripped = line.strip()

        # Case 1: Blockquote
        if stripped.startswith(">"):
            block_lines = []
            start_line_num = line_num
            while i < num_lines and lines[i].strip().startswith(">"):
                raw_b = lines[i].strip()
                content_b = re.sub(r"^>\s*", "", raw_b)
                block_lines.append(content_b)
                i += 1

            quote_text = " ".join(block_lines)

            # Look for citation on line preceding block, inside block, or line after block
            cit_match = None
            if start_line_num > 1:
                prev_line = lines[start_line_num - 2]
                m = citation_pattern.search(prev_line)
                if m:
                    cit_match = m.group(1) or m.group(2)

            if not cit_match:
                m = citation_pattern.search(quote_text)
                if m:
                    cit_match = m.group(1) or m.group(2)

            if not cit_match and i < num_lines:
                next_line = lines[i]
                m = citation_pattern.search(next_line)
                if m:
                    cit_match = m.group(1) or m.group(2)

            if cit_match:
                parsed = parse_citation(cit_match)
                if parsed:
                    quotes.append({
                        "doc": doc_path,
                        "doc_line": start_line_num,
                        "quote": quote_text,
                        "citation": parsed,
                        "raw_cit": cit_match,
                    })
            continue

        # Case 2: Inline quote directly associated with citation
        # Pattern 2A: citation followed by ("...") or (*"..."*)
        for m in re.finditer(r'(?:\[((?:ext:)?[a-zA-Z0-9_\-\./\\]+:\d+(?:-\d+)?)\]\([^\)]+\)|\(((?:ext:)?[a-zA-Z0-9_\-\./\\]+:\d+(?:-\d+)?)\))\s*(?:\:\s*|\s+)\(?[\*\_]?\"([^\"]+)\"[\*\_]?\)?', line):
            cit_str = m.group(1) or m.group(2)
            quote_text = m.group(3)
            parsed = parse_citation(cit_str)
            if parsed and len(quote_text.strip()) > 3:
                quotes.append({
                    "doc": doc_path,
                    "doc_line": line_num,
                    "quote": quote_text.strip(),
                    "citation": parsed,
                    "raw_cit": cit_str,
                })

        # Pattern 2B: quote followed by citation: "..." (FILE:LINE) or *"..."* (FILE:LINE)
        for m in re.finditer(r'[\*\_]?\"([^\"]+)\"[\*\_]?\s*(?:\:\s*|\s+)?(?:\(((?:ext:)?[a-zA-Z0-9_\-\./\\]+:\d+(?:-\d+)?)\)|\[((?:ext:)?[a-zA-Z0-9_\-\./\\]+:\d+(?:-\d+)?)\]\([^\)]+\))', line):
            quote_text = m.group(1)
            cit_str = m.group(2) or m.group(3)
            parsed = parse_citation(cit_str)
            if parsed and len(quote_text.strip()) > 3:
                # Avoid duplicate if already matched by 2A
                already = any(q["doc_line"] == line_num and q["raw_cit"] == cit_str for q in quotes)
                if not already:
                    quotes.append({
                        "doc": doc_path,
                        "doc_line": line_num,
                        "quote": quote_text.strip(),
                        "citation": parsed,
                        "raw_cit": cit_str,
                    })

        i += 1

    return quotes


def verify_quote(q_item: Dict[str, any], repo_root: str = ".") -> Tuple[str, str]:
    """Verify a single quote item. Returns (status, reason)."""
    is_ext, path, start_l, end_l = q_item["citation"]

    # Resolve path
    if is_ext:
        candidates = [
            os.path.join(repo_root, "..", "ext_data", "skillcorner", path),
            os.path.join(repo_root, "..", "ext_data", path),
            os.path.expanduser(f"~/ext_data/skillcorner/{path}"),
            os.path.expanduser(f"~/ext_data/{path}"),
            os.path.join(os.environ.get("SKILLCORNER_DATA_DIR", ""), path),
            path,
        ]
        target_path = None
        for c in candidates:
            if c and os.path.exists(c):
                target_path = c
                break
        if target_path is None:
            return ("SKIPPED", f"External file not found on host: {path}")
    else:
        target_path = os.path.join(repo_root, path)
        if not os.path.exists(target_path):
            doc_dir = os.path.dirname(q_item["doc"])
            alt = os.path.join(doc_dir, path)
            if os.path.exists(alt):
                target_path = alt
            else:
                return ("FAIL", f"Source file does not exist: {path}")

    # Read cited lines
    try:
        with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
    except Exception as e:
        return ("FAIL", f"Could not read source file: {e}")

    if start_l < 1 or start_l > len(all_lines):
        return ("FAIL", f"Cited line {start_l} out of bounds (file has {len(all_lines)} lines)")
    if end_l < start_l or end_l > len(all_lines):
        return ("FAIL", f"Cited end line {end_l} out of bounds (file has {len(all_lines)} lines)")

    cited_text = "".join(all_lines[start_l - 1 : end_l])

    norm_quote = normalize_text(q_item["quote"])
    norm_source = normalize_text(cited_text)

    # Check match: quote in source or source in quote
    if norm_quote in norm_source or norm_source in norm_quote:
        return ("PASS", "Exact/normalized match")

    # Check substring tokens (ignoring punctuation/formatting)
    words_q = norm_quote.split()
    words_s = norm_source.split()
    if len(words_q) >= 4 and " ".join(words_q[:4]) in norm_source:
        return ("PASS", "Prefix match in cited range")

    return ("FAIL", f"Quote text not found in {path}:{start_l}-{end_l}")


def run_quote_check(doc_paths: List[str], repo_root: str = ".") -> Tuple[bool, List[Dict[str, any]]]:
    """Run quote check on multiple documents."""
    results = []
    all_passed = True

    for doc in doc_paths:
        quotes = extract_quotes_from_doc(doc)
        for q in quotes:
            status, reason = verify_quote(q, repo_root=repo_root)
            results.append({
                "doc": doc,
                "doc_line": q["doc_line"],
                "quote": q["quote"],
                "citation": q["raw_cit"],
                "status": status,
                "reason": reason,
            })
            if status == "FAIL":
                all_passed = False

    return all_passed, results


if __name__ == "__main__":
    docs = [
        "README.md",
        "README_v2.md",
        "CONSTANTS.md",
        "TECHNICAL_DOCUMENT.md",
        os.path.join("tools", "tuning", "README.md"),
    ]
    target_docs = [d for d in docs if os.path.exists(d)]
    print(f"Running quote checker on {len(target_docs)} files: {target_docs}")
    passed, results = run_quote_check(target_docs)

    print("\n" + "=" * 105)
    print(f"{'Doc Location':<25} {'Citation':<30} {'Status':<10} {'Quote Snippet'}")
    print("-" * 105)
    for r in results:
        q_disp = r['quote'].replace('\n', ' ')
        if len(q_disp) > 36:
            q_disp = q_disp[:33] + "..."
        doc_loc = f"{r['doc']}:{r['doc_line']}"
        print(f"{doc_loc:<25} {r['citation']:<30} {r['status']:<10} {q_disp}")
    print("=" * 105)
    print(f"VERDICT: {'ALL PASS' if passed else 'FAILURES DETECTED'} (Total quotes: {len(results)})")

    sys.exit(0 if passed else 1)
