"""Render report/report.md to a PDF.

    python -m src.make_pdf

NOTE ON TEMPLATES. This uses a plain academic-looking stylesheet, **not** any
competition or venue template — none was supplied with this repository. If an
official template exists, the PDF produced here is a readable preview, not a
submission-formatted document, and the styling below should be replaced.

Pure Python (markdown + xhtml2pdf), so it needs no LaTeX or system libraries.
"""

from __future__ import annotations

import re
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "report" / "report.md"
OUT = ROOT / "report" / "report.pdf"

CSS = """
@page { size: A4; margin: 2.0cm 1.9cm 2.2cm 1.9cm;
        @frame footer { -pdf-frame-content: footer; bottom: 1.1cm;
                        margin-left: 1.9cm; margin-right: 1.9cm; height: 1cm; } }
body { font-family: Helvetica, Arial, sans-serif; font-size: 9.2pt;
       line-height: 1.42; color: #111; }
h1 { font-size: 16pt; margin: 0 0 2pt 0; }
h2 { font-size: 12pt; margin: 14pt 0 4pt 0; border-bottom: 0.6pt solid #bbb;
     padding-bottom: 2pt; }
h3 { font-size: 10.2pt; margin: 10pt 0 3pt 0; }
h4 { font-size: 9.4pt; margin: 8pt 0 2pt 0; }
p  { margin: 0 0 5pt 0; text-align: justify; }
ul, ol { margin: 0 0 5pt 14pt; }
li { margin-bottom: 2pt; }
code { font-family: Courier, monospace; font-size: 8.2pt; background: #f2f2f2; }
pre  { font-family: Courier, monospace; font-size: 7.8pt; background: #f6f6f6;
       border: 0.4pt solid #ddd; padding: 4pt; margin: 0 0 6pt 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 8pt 0;
        font-size: 7.9pt; }
th { background: #ececec; border: 0.4pt solid #999; padding: 2.5pt 3.5pt;
     text-align: left; font-weight: bold; }
td { border: 0.4pt solid #bbb; padding: 2.5pt 3.5pt; vertical-align: top; }
blockquote { margin: 0 0 6pt 10pt; padding-left: 7pt; border-left: 2pt solid #ccc;
             color: #333; }
a { color: #14458c; text-decoration: none; }
.subtitle { font-size: 10pt; color: #444; margin: 0 0 8pt 0; }
#footer { text-align: center; font-size: 7pt; color: #777; }
"""


def build_html() -> str:
    text = SRC.read_text(encoding="utf-8")
    # The "### subtitle" directly under the H1 should read as a subtitle.
    text = re.sub(r"^(# .+)\n### (.+)$",
                  r"\1\n\n<p class='subtitle'>\2</p>", text, count=1, flags=re.M)
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    return f"""<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>
<div id="footer">PrefLens — page <pdf:pagenumber> of <pdf:pagecount></div>
{body}
</body></html>"""


def main() -> None:
    html = build_html()
    with OUT.open("wb") as fh:
        result = pisa.CreatePDF(html, dest=fh, encoding="utf-8")
    if result.err:
        raise SystemExit(f"PDF generation failed with {result.err} error(s)")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({kb:.0f} KB)")
    print("NOTE: plain stylesheet, not an official venue template (none supplied).")


if __name__ == "__main__":
    main()
