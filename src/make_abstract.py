"""Generate report/abstract.txt from the abstract inside report/report.md.

    python -m src.make_abstract

The abstract otherwise exists in two places, which is two places to drift. It is
written once in the report and rendered here, so `abstract.txt` cannot disagree
with the paper (D-37). A test asserts the file on disk equals what this produces.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "report.md"
OUT = ROOT / "report" / "abstract.txt"

# Unicode the plain-text file should not carry.
PLAIN = {
    "—": "--", "–": "-", "×": "x", "ρ": "rho",
    "Δ": "Delta", "−": "-", "≤": "<=", "≥": ">=",
    "’": "'", "α": "alpha",
}


def title() -> str:
    for line in REPORT.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    raise SystemExit("no '# ' title found in report.md")


def abstract_body() -> str:
    s = REPORT.read_text(encoding="utf-8")
    if "## Abstract" not in s:
        raise SystemExit("no '## Abstract' section in report.md")
    return s.split("## Abstract", 1)[1].split("*(", 1)[0].strip()


def to_plain(text: str) -> str:
    for a, b in PLAIN.items():
        text = text.replace(a, b)
    return text.replace("**", "")


def render() -> str:
    body = to_plain(abstract_body())
    n = len(body.split())
    head = to_plain(title())
    return head + "\n\n" + body + "\n\n[" + str(n) + " words]\n"


def word_count() -> int:
    return len(to_plain(abstract_body()).split())


def main() -> None:
    out = render()
    OUT.write_text(out, encoding="utf-8")
    n = word_count()
    flag = "  <-- OVER THE 150-WORD LIMIT" if n > 150 else ""
    print(f"wrote {OUT} ({n} words){flag}")


if __name__ == "__main__":
    main()
