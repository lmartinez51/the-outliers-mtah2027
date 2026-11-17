#!/usr/bin/env python3
"""Automated Quote Verification and Coverage Tool for Project Documentation.

Verifies that all quoted text (blockquotes, italic quotes, inline quotes) in
project documentation carries a valid citation to upstream documentation or
external data files, and verifies that the quoted content matches the cited source lines.

Convention:
  A citation is (FILE:LINE), (FILE:START-END), (ext:FILE:LINE), or (ext:FILE:START-END),
  or markdown link form [`FILE:LINE`](...).
  It may appear on the line preceding the quote, inside the quote, on the same line
  (preceding or following), or on the line immediately following the quote.

Coverage Mode:
  Scans all target documents, lists EVERY blockquote, italic quote, and inline quote,
  and marks each as:
    - PASS: Cited and text verified against source
    - SKIPPED: Cited to an external file not present on current host
    - FAIL: Cited but source text does not match or source file missing
    - UNCITED: Quote has no recognized citation
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
    cit_str = re.sub(r"^[\(\[\s`]+", "", cit_str)
    cit_str = re.sub(r"[\)\]\s`]+$", "", cit_str)
    cit_str = cit_str.strip("`")

    is_ext = False
    if cit_str.startswith("ext:"):
        is_ext = True
        cit_str = cit_str[4:]

    m = re.match(r"^([a-zA-Z0-9_\-\./\\]+):(\d+)(?:-(\d+))?$", cit_str)
    if not m:
        return None

    path = m.group(1).strip()
    start_l = int(m.group(2))
    end_l = int(m.group(3)) if m.group(3) else start_l
    return (is_ext, path, start_l, end_l)


CITATION_RE = re.compile(
    r"\(((?:ext:)?[a-zA-Z0-9_\-\./\\]+:\d+(?:-\d+)?)\)|"
    r"\[`?((?:ext:)?[a-zA-Z0-9_\-\./\\]+:\d+(?:-\d+)?)`?\](?:\([^\)]*\))?|"
    r"(?:\b)((?:ext:)[a-zA-Z0-9_\-\./\\]+:\d+(?:-\d+)?)(?=:|\s|$)|"
    r"(?:\b)([a-zA-Z0-9_\-\./\\]+\.(?:md|py|xml|csv|txt):\d+(?:-\d+)?)(?=:|\s|$)"
)


def find_citations_in_string(s: str) -> List[Tuple[str, Tuple[bool, str, int, int]]]:
    """Find all valid citations in a string."""
    found = []
    for m in CITATION_RE.finditer(s):
        c_str = m.group(1) or m.group(2) or m.group(3) or m.group(4)
        if c_str:
            p = parse_citation(c_str)
            if p:
                found.append((c_str, p))
    return found


def extract_quotes_from_doc(doc_path: str) -> List[Dict[str, any]]:
    """Extract all quotes and evaluate citations from a markdown document."""
    if not os.path.exists(doc_path):
        return []

    with open(doc_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    quotes = []
    num_lines = len(lines)
    i = 0
    in_code_block = False

    while i < num_lines:
        line = lines[i]
        line_num = i + 1
        stripped = line.strip()

        # Fenced code block tracking
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            i += 1
            continue

        if in_code_block:
            i += 1
            continue

        # 1. Blockquote
        if stripped.startswith(">"):
            start_idx = i
            block_lines = []
            while i < num_lines and lines[i].strip().startswith(">"):
                raw_b = lines[i].strip()
                content_b = re.sub(r"^>\s*", "", raw_b)
                block_lines.append(content_b)
                i += 1
            block_end_idx = i - 1
            block_text = " ".join(block_lines)

            # Search citations: inside block, preceding non-empty line, following non-empty line
            cits = find_citations_in_string(block_text)
            if not cits and start_idx > 0:
                p_idx = start_idx - 1
                while p_idx >= 0 and not lines[p_idx].strip():
                    p_idx -= 1
                if p_idx >= 0:
                    cits = find_citations_in_string(lines[p_idx])
            if not cits and block_end_idx + 1 < num_lines:
                n_idx = block_end_idx + 1
                while n_idx < num_lines and not lines[n_idx].strip():
                    n_idx += 1
                if n_idx < num_lines:
                    cits = find_citations_in_string(lines[n_idx])

            cit_info = cits[0] if cits else None
            quotes.append({
                "doc": doc_path,
                "doc_line": start_idx + 1,
                "type": "BLOCKQUOTE",
                "quote": block_text,
                "citation": cit_info[1] if cit_info else None,
                "raw_cit": cit_info[0] if cit_info else None,
            })
            continue

        # 2. Italic quotes directly on line
        line_masked = list(line)
        for m in re.finditer(r'\*"([^"]+)"\*', line):
            q_text = m.group(1).strip()
            pre_text = line[:m.start()]
            cits_pre = find_citations_in_string(pre_text)
            post_text = line[m.end():]
            cits_post = find_citations_in_string(post_text)

            cit_info = None
            if cits_pre:
                cit_info = cits_pre[-1]
            elif cits_post:
                cit_info = cits_post[0]
            else:
                p_idx = i - 1
                while p_idx >= 0 and not lines[p_idx].strip():
                    p_idx -= 1
                if p_idx >= 0:
                    cits_p = find_citations_in_string(lines[p_idx])
                    if cits_p:
                        cit_info = cits_p[-1]

            quotes.append({
                "doc": doc_path,
                "doc_line": line_num,
                "type": "ITALIC-QUOTE",
                "quote": q_text,
                "citation": cit_info[1] if cit_info else None,
                "raw_cit": cit_info[0] if cit_info else None,
            })

            # Mask out this italic quote so inline quote finder doesn't cross it
            for idx in range(m.start(), m.end()):
                line_masked[idx] = ' '

        line_no_italic = "".join(line_masked)

        # 3. Inline quotes directly on line (with italic quotes masked out)
        for m in re.finditer(r'"([^"]{4,})"', line_no_italic):
            # check not inside code span
            pre_str = line[:m.start()]
            if pre_str.count('`') % 2 != 0:
                continue
            # check not inside markdown link url
            if re.match(r'\]\([^)]*$', pre_str):
                continue

            q_text = m.group(1).strip()
            if any(q["doc_line"] == line_num and q["quote"] == q_text for q in quotes):
                continue

            pre_text = line[:m.start()]
            cits_pre = find_citations_in_string(pre_text)
            post_text = line[m.end():]
            cits_post = find_citations_in_string(post_text)

            cit_info = None
            if cits_pre:
                cit_info = cits_pre[-1]
            elif cits_post:
                cit_info = cits_post[0]
            else:
                p_idx = i - 1
                while p_idx >= 0 and not lines[p_idx].strip():
                    p_idx -= 1
                if p_idx >= 0:
                    cits_p = find_citations_in_string(lines[p_idx])
                    if cits_p:
                        cit_info = cits_p[-1]

            quotes.append({
                "doc": doc_path,
                "doc_line": line_num,
                "type": "INLINE-QUOTE",
                "quote": q_text,
                "citation": cit_info[1] if cit_info else None,
                "raw_cit": cit_info[0] if cit_info else None,
            })

        i += 1

    return quotes


def verify_quote(q_item: Dict[str, any], repo_root: str = ".") -> Tuple[str, str]:
    """Verify a single quote item. Returns (status, reason).

    Status is one of: PASS, SKIPPED, FAIL, UNCITED.
    """
    if not q_item["citation"]:
        return ("UNCITED", "No recognized citation format found for quote")

    is_ext, path, start_l, end_l = q_item["citation"]

    # Resolve file path
    if is_ext:
        external_root = os.environ.get("SKILLCORNER_DATA_DIR")
        candidates = [
            os.path.join(repo_root, "..", "ext_data", "skillcorner", path),
            os.path.join(repo_root, "..", "ext_data", path),
            os.path.expanduser(f"~/ext_data/skillcorner/{path}"),
            os.path.expanduser(f"~/ext_data/{path}"),
        ]
        if external_root:
            candidates.insert(0, os.path.join(external_root, path))

        # Never fall back to a bare relative path for ext: citations. Doing so
        # can silently resolve ext:README.md to this repository's README.md.
        target_path = None
        for c in candidates:
            if c and os.path.exists(c):
                target_path = os.path.realpath(c)
                break
        if target_path is None:
            return ("SKIPPED", f"External file not found on host: {path}")
    else:
        target_path = os.path.join(repo_root, path)
        if not os.path.exists(target_path):
            doc_dir = os.path.dirname(q_item["doc"])
            alt = os.path.join(doc_dir, path)
            if os.path.exists(alt):
                target_path = os.path.realpath(alt)
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

    # 1. Exact / normalized match
    if norm_quote in norm_source or norm_source in norm_quote:
        return ("PASS", "Exact/normalized match in cited range")

    # 2. Token overlap match (at least 4 matching consecutive words)
    words_q = norm_quote.split()
    words_s = norm_source.split()
    if len(words_q) >= 4 and " ".join(words_q[:4]) in norm_source:
        return ("PASS", "Prefix match in cited range")

    return ("FAIL", f"Quote text not found in {path}:{start_l}-{end_l}")


def run_quote_check(doc_paths: List[str], repo_root: str = ".") -> Tuple[bool, List[Dict[str, any]]]:
    """Run quote check across documentation files with full coverage analysis."""
    results = []
    all_passed = True

    for doc in doc_paths:
        quotes = extract_quotes_from_doc(doc)
        for q in quotes:
            status, reason = verify_quote(q, repo_root=repo_root)
            results.append({
                "doc": doc,
                "doc_line": q["doc_line"],
                "type": q["type"],
                "quote": q["quote"],
                "citation": q["raw_cit"] if q["raw_cit"] else "UNCITED",
                "status": status,
                "reason": reason,
            })
            if status in ("FAIL", "UNCITED"):
                all_passed = False

    return all_passed, results


DEFAULT_DOCS = [
    "README.md",
    "CONSTANTS.md",
    "TECHNICAL_DOCUMENT.md",
]


if __name__ == "__main__":
    target_docs = [d for d in DEFAULT_DOCS if os.path.exists(d)]
    print(f"Running quote checker (coverage mode) on {len(target_docs)} files: {target_docs}")
    passed, results = run_quote_check(target_docs)

    print("\n" + "=" * 115)
    print(f"{'Doc Location':<28} {'Type':<14} {'Citation':<30} {'Status':<10} {'Quote Snippet'}")
    print("-" * 115)
    for r in results:
        q_disp = r['quote'].replace('\n', ' ')
        if len(q_disp) > 30:
            q_disp = q_disp[:27] + "..."
        doc_loc = f"{r['doc']}:{r['doc_line']}"
        print(f"{doc_loc:<28} {r['type']:<14} {r['citation']:<30} {r['status']:<10} {q_disp}")
    print("=" * 115)
    
    uncited_count = sum(1 for r in results if r['status'] == 'UNCITED')
    failed_count = sum(1 for r in results if r['status'] == 'FAIL')
    passed_count = sum(1 for r in results if r['status'] == 'PASS')
    skipped_count = sum(1 for r in results if r['status'] == 'SKIPPED')

    print(f"SUMMARY: Total={len(results)} | PASS={passed_count} | SKIPPED={skipped_count} | UNCITED={uncited_count} | FAIL={failed_count}")
    verdict = "ALL PASS" if passed else "FAILURES DETECTED"
    print(f"VERDICT: {verdict}")

    sys.exit(0 if passed else 1)
