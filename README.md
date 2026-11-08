# A Parallel Paradigm: The Coming Economics of Model Ownership

Thatayotlhe Tsenang · Independent researcher, Gaborone, Botswana · November 2026

**Paper:** [`paper/a-parallel-paradigm.pdf`](paper/a-parallel-paradigm.pdf)

Most AI products in 2026 rent intelligence through APIs. This paper asks whether organizations
will increasingly own and adapt their models instead, and answers from public prices and surveys,
a cost model and three experiments. Ownership is growing, but as an owned layer of narrow, adapted
models beneath a rented frontier rather than as a replacement for it. For one narrow task moved
from a frontier API to a fine-tuned open 8B model, owning becomes more likely than not to be
cheaper at a monthly API bill of about $22,000, or about $37,000 at US labor rates. Ignoring the
falling price of rented capability puts that point about seven times too low. Labor, not compute,
sets the threshold. The experiments show that training-data defects cost money mainly through
silent harm rather than crashed runs.

This repository holds the paper's LaTeX source, every script behind its original numbers, the
raw outputs, and the research notes with source URLs.

## Disclosure

The author develops [Parallelogram](https://github.com/Thatayotlhe04/Parallelogram), an
open-source linter for fine-tuning data that is one of the tools evaluated in sections 5, 6
and 9. The experiments include defect types it was never designed to catch, and its failures are
reported alongside its successes.

## Layout

```
paper/
  a-parallel-paradigm.pdf     compiled paper
  main.tex                    front matter, layout and bibliography setup
  body.tex, summary.tex       sections 1-9 and appendices (generated, see below)
  refs.bib                    bibliography, every entry with its URL
  figures/                    Figures 1-3 (from code/make_figures.py)
  build/essay.md              the essay text the LaTeX is generated from
  build/build_tex.py          essay.md -> body.tex, summary.tex, refs.bib (pandoc 3)
  build/refs_map.py           URL -> bibliography entry map
code/
  econ_model.py               section 4: dynamic break-even, Monte Carlo even-odds points,
                              sensitivity, profiles and named scenarios
  e1_detection.py             E1: inject 23 defect types, measure five detectors
  e1b_followup.py             E1: MinHash threshold trade-off
  e1c_fix_fate.py             E1: per-defect outcome of `parallelogram check --fix`
  e4_trl_audit.py             E4: TRL 1.17.1 behaviour on each defect (local tiny model)
  e4b_defaults_truncation.py  E4: TRL defaults and silent truncation
  e2_harm.py                  E2: harm dose-response, small model, 5 seeds
  e2_analyze.py               E2: paired differences and 95% intervals
  e2_real_model.py            E2 at 0.5B scale with LoRA (needs a GPU; not run for the paper)
  make_figures.py             Figures 1-3 from the result files
results/                      raw outputs of every script above
notes/                        research notes with source URLs, and the adversarial review
                              of the draft whose fixes are applied in the paper
```

## Reproducing the results

Environment used: Python 3.13; torch 2.15.0 (CPU); transformers 5.22.2; datasets 5.1.0;
trl 1.17.1; peft 0.21.2; parallelogram 1.6.1; datasketch 1.18.1; numpy 2.5.4; pandas 3.1.1;
scipy 1.18.1; matplotlib 3.11. All runs used a two-core CPU, 7 GB of RAM, no GPU and no
Hugging Face Hub access.

Data (check both hashes before running; keep downloads in their own directory):

| File | Source | SHA-256 |
|---|---|---|
| alpaca_data_cleaned.json | https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json | bd844b8247a0f543804b6ce0882b0aaec4bbf5e8d66167df6213a0f1e4fe878b |
| helpful-base/train.jsonl.gz | https://media.githubusercontent.com/media/anthropics/hh-rlhf/master/helpful-base/train.jsonl.gz | 518a5bf288456fc9f3b7c980c54116fba0c52f274d3f4d344675d83e4058f6f4 |

```bash
python -I code/econ_model.py
python -I code/e1_detection.py data/alpaca_data_cleaned.json data/train.jsonl.gz results/e1
python -I code/e1b_followup.py results/e1
python -I code/e1c_fix_fate.py results/e1
HF_HUB_OFFLINE=1 python -I code/e4_trl_audit.py data/alpaca_data_cleaned.json results/e4
HF_HUB_OFFLINE=1 python -I code/e4b_defaults_truncation.py data/alpaca_data_cleaned.json results/e4
python -I code/e2_harm.py results/e2 0,1,2,3,4 0.10,0.30
python -I code/e2_analyze.py results/e2
python -I code/make_figures.py
```

Seeds are fixed inside each script: E1 20261108; E1 repair fates 7; E4 0 and 1; E2 0 to 4;
Monte Carlo 20261108. E2 takes about 50 minutes on two CPU cores. The E1 working files (about
276 MB) and the E4 tiny-model checkpoints are not committed; the scripts regenerate them.

One figure in the paper is not produced by these scripts: the 350 identical UltraFeedback
pairs in section 5.3 come from Parallelogram's public-datasets documentation.

The first E2 run used 600 fine-tuning steps at learning rate 1e-3, where the clean baseline did
not converge reliably (99.7% on one seed, about 17% on two). It was stopped and is kept as
`results/e2/e2_runs_pilot_600steps.jsonl`. The reported run uses 1,000 steps at 2e-3.

## Building the paper

Needs TeX Live with pdflatex, latexmk, natbib, mathpazo and seqsplit.

```bash
make paper      # figures, then paper/a-parallel-paradigm.pdf
make body       # regenerate body.tex, summary.tex and refs.bib from paper/build/essay.md
```

## Citation

```bibtex
@misc{tsenang2026parallel,
  author = {Tsenang, Thatayotlhe},
  title  = {A Parallel Paradigm: The Coming Economics of Model Ownership},
  year   = {2026},
  month  = nov,
  url    = {https://github.com/Thatayotlhe04/a-parallel-paradigm}
}
```

## Licence

No licence has been chosen yet. Until one is added, all rights are reserved by the author.
