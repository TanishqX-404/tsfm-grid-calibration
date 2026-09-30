# Paper draft (IEEE conference template)

    python paper/make_tables.py          # tables, numbers and figures from results/
    cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main

Every number in `main.tex` is a macro from `generated/numbers.tex`, produced from the frozen test
results in `results/`; edit the script, not the numbers.

## Before submission (TODO)
- Author affiliation and e-mail (title block).
- `refs.bib`: entries marked VERIFY/TODO (the five prior-work arXiv papers have no author lists;
  Chronos-2, Moirai 2.0, TiRex, GIFT-Eval authors/titles need checking).
- Check the IEEE PES template version, page limit and anonymity rules for the venue.
- Optional: confirm in the Chronos-2 technical report whether the `ercot` dataset was excluded
  from pretraining (the leakage sentence is hedged either way).
