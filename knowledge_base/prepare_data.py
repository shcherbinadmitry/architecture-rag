#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

try:
    from bs4 import BeautifulSoup
except ImportError:  # Lightweight fallback if bs4 isn't present
    BeautifulSoup = None  # type: ignore


def ensure_bs4_available() -> None:
    if BeautifulSoup is None:
        raise SystemExit(
            "BeautifulSoup4 is required. Install with: pip install beautifulsoup4 html5lib"
        )


def strip_html_content(html: str, parser: str = "html5lib") -> str:
    bs_parser = parser
    if parser == "html5lib":
        try:
            import html5lib  # noqa: F401
        except Exception:
            bs_parser = "html.parser"

    soup = BeautifulSoup(html, bs_parser)

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    ref_like_ids = {
        "references",
        "reference",
        "bibliography",
        "sources",
        "citations",
        "footnotes",
        "notes",
    }
    for el in list(soup.find_all(True)):
        try:
            el_id = (el.get("id") or "").lower()
            el_class = " ".join([c.lower() for c in (el.get("class") or [])])
        except Exception:
            el_id, el_class = "", ""
        if any(token in el_id for token in ref_like_ids) or any(
            token in el_class for token in ref_like_ids
        ):
            el.decompose()

    text = soup.get_text("\n")

    text = re.sub(r"\[(?:\s*(?:citation needed|note\s*\d+|n\.?d\.|ibid\.|cf\.)|\s*\d+[a-z]?\s*)\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\((?:[A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?(?:,?\s*&\s*[A-Z][A-Za-z\-]+)*)\s*,?\s*\d{4}[a-z]?\)", "", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    cat_markers = [
        re.compile(r"^categories:?$", re.IGNORECASE),
        re.compile(r"^hidden\s+category:?$", re.IGNORECASE),
    ]
    cut_index = None
    for i, line in enumerate(lines):
        if any(pat.match(line) for pat in cat_markers):
            cut_index = i
            break
    if cut_index is not None:
        lines = lines[:cut_index]

    normalized = "\n".join(lines).strip()
    return normalized


def process_file(src: Path, dst: Path) -> None:
    html = src.read_text(encoding="utf-8", errors="ignore")
    text = strip_html_content(html)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text + "\n", encoding="utf-8")


def main(input_dir: Optional[str] = None, output_dir: Optional[str] = None) -> None:
    ensure_bs4_available()

    repo_root = Path(__file__).resolve().parents[1]
    in_dir = Path(input_dir or repo_root / "knowledge_base" / "originals").resolve()
    out_dir = Path(output_dir or repo_root / "knowledge_base" / "clean").resolve()

    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"Input directory not found or is not a directory: {in_dir}")

    html_exts = {".html", ".htm"}
    files = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in html_exts]

    if not files:
        print(f"No HTML files found in {in_dir}")
        return

    for src in files:
        dst_name = src.with_suffix(".txt").name
        dst = out_dir / dst_name
        process_file(src, dst)
        try:
            rel_out = dst.relative_to(repo_root)
        except Exception:
            rel_out = dst
        print(f"Processed: {src.name} -> {rel_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip HTML from files into plain text")
    parser.add_argument("--input", dest="input_dir", default=None, help="Input directory path")
    parser.add_argument("--output", dest="output_dir", default=None, help="Output directory path")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
