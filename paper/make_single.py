"""Build a single self-contained main_single.tex: numbers, tables and bibliography inlined.

    python paper/make_tables.py && (cd paper && pdflatex main && bibtex main)   # refresh inputs
    python paper/make_single.py                                                 # -> paper/main_single.tex

Only the figure PDFs stay external (flat names: F1_pipeline.pdf, ...).
"""
from __future__ import annotations

import re
from pathlib import Path

P = Path(__file__).resolve().parent
GEN = P / "generated"

AUTHORS = r"""\author{
\IEEEauthorblockN{Tanishq Singh Sisodiya}
\IEEEauthorblockA{\textit{Symbiosis Institute of Technology}\\
\textit{Hyderabad Campus}\\
\textit{Symbiosis International}\\
\textit{(Deemed University)}\\
Pune, India\\
sisodiya.tanishq.singh@gmail.com}
\and
\IEEEauthorblockN{Mallellu Sai Prashanth*}
\IEEEauthorblockA{\textit{Symbiosis Institute of Technology}\\
\textit{Hyderabad Campus}\\
\textit{Symbiosis International}\\
\textit{(Deemed University)}\\
Pune, India\\
saiprashanth08@ieee.org}
\and
\IEEEauthorblockN{Rajanikanth Aluvalu}
\IEEEauthorblockA{\textit{Symbiosis Institute of Technology}\\
\textit{Hyderabad Campus}\\
\textit{Symbiosis International}\\
\textit{(Deemed University)}\\
Pune, India\\
Rajanikanth.aluvalu@ieee.org}
}"""


def main():
    tex = (P / "main.tex").read_text()
    # numbers: \newcommand{\name}{value}
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (GEN / "numbers.tex").read_text()))
    tex = tex.replace("\\input{generated/numbers.tex}\n", "")
    for name in sorted(macros, key=len, reverse=True):
        val = macros[name]
        tex = re.sub(r"\\" + name + r"\\ ", lambda m: val + " ", tex)
        tex = re.sub(r"\\" + name + r"(?![A-Za-z])", lambda m: val, tex)
    # tables
    tex = re.sub(r"\\input\{generated/(\w+)\.tex\}", lambda m: (GEN / f"{m.group(1)}.tex").read_text().strip(), tex)
    # figures: flat file names
    tex = tex.replace("{figures/", "{")
    # authors
    tex = re.sub(r"\\author\{.*?\}\}\n", lambda m: AUTHORS + "\n", tex, count=1, flags=re.S)
    # bibliography
    bbl = (P / "main.bbl").read_text().strip()
    tex = tex.replace("\\bibliographystyle{IEEEtran}\n\\bibliography{refs}", bbl)
    leftover = re.findall(r"\\(" + "|".join(map(re.escape, macros)) + r")(?![A-Za-z])", tex)
    assert not leftover, leftover
    assert "\\input{" not in tex and "\\bibliography{" not in tex
    (P / "main_single.tex").write_text(tex)
    print("wrote", P / "main_single.tex", len(tex.splitlines()), "lines")


if __name__ == "__main__":
    main()
