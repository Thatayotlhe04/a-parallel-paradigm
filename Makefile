# Build targets for "A Parallel Paradigm". Run from the repository root.
PY ?= python

.PHONY: paper body figures econ clean

paper: figures
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex && cp main.pdf a-parallel-paradigm.pdf

body:            # regenerate body.tex, summary.tex and refs.bib from the essay text (needs pandoc 3)
	cd paper/build && python3 build_tex.py essay.md ..

figures:
	$(PY) -I code/make_figures.py

econ:
	$(PY) -I code/econ_model.py

clean:
	cd paper && latexmk -C main.tex
