"""Convert the essay's markdown export into LaTeX body files plus a BibTeX file.

Usage: python3 build_tex.py paper_export.md ../paper
Writes summary.tex, body.tex and refs.bib into the output directory.
main.tex (written by hand) includes them.
"""
import re
import subprocess
import sys
from pathlib import Path

from refs_map import R, AUTHOR_TEXTS, bibtex

src, outdir = Path(sys.argv[1]), Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)
md = src.read_text()


def raw(tex):
    """Inline raw LaTeX for pandoc."""
    return "`" + tex + "`{=latex}"


# ---- split: drop title/byline, take Summary, body, drop References ----------
summary = md[md.index("## Summary"):md.index("## 1. The question")]
summary = summary.replace("## Summary\n", "")
body = md[md.index("## 1. The question"):md.index("## References")]

# ---- headings: LaTeX numbers them ------------------------------------------
body = re.sub(r"^(##+) (\d+\.)+\d* ?", r"\1 ", body, flags=re.M)
body = re.sub(r"^(##+) \d+\. ", r"\1 ", body, flags=re.M)
body = body.replace("## Appendix A: Notation and model details",
                    "```{=latex}\n\\appendix\n```\n\n## Notation and model details")
body = body.replace("## Appendix B: Reproducibility and open questions",
                    "## Reproducibility and open questions")

# ---- display equations -------------------------------------------------------
EQ_LABELS = ["eq:taskcost", "eq:breakeven", "eq:halflife", "eq:cap", "eq:dpo", "eq:waste", "eq:contam"]
eqs = iter(EQ_LABELS)


def eq(m):
    return ("```{=latex}\n\\begin{equation}\\label{%s}\n%s\n\\end{equation}\n```"
            % (next(eqs), m.group(1).strip()))


body = re.sub(r"```latex\n(.*?)```", eq, body, flags=re.S)

# ---- figures -----------------------------------------------------------------
FIGS = {
    "Figure 1": ("fig1_ownership_curves", "fig:curves",
                 "Share of simulated draws in which owning a fine-tuned open 8B model is cheaper than "
                 "renting Claude Sonnet 5.5 over 36 months, by today's monthly API bill for one narrow task. "
                 "The shares describe the model's assumed input ranges (section~\\ref{sec:math}), not the "
                 "odds facing any particular organization. 20,000 draws per bill level; even-odds points "
                 "interpolated on a log scale. "
                 "Source: \\texttt{results/econ/montecarlo\\_curves.csv}."),
    "Figure 2": ("fig2_sensitivity", "fig:tornado",
                 "Sensitivity of the break-even bill to one input at a time, at US labor rates with "
                 "every other input at its median (deterministic model, solved by bisection). The share "
                 "of the price decline owners capture has no effect at this scale and is omitted. "
                 "Source: \\texttt{results/econ/tornado.csv}."),
    "Figure 3": ("fig3_e2_harm", "fig:e2",
                 "E2: paired change in exact-match accuracy against the clean fine-tune of the same "
                 "seed; mean and 95\\% t-interval over five seeds. Systematic errors affected 12.1\\% "
                 "and 31.9\\% of rows. Source: \\texttt{results/e2/e2\\_summary.json}."),
}


def fig(m):
    name = re.search(r"Figure \d", m.group(0)).group(0)
    f, lab, cap = FIGS[name]
    return ("```{=latex}\n\\begin{figure}[t]\n\\centering\n"
            "\\includegraphics[width=\\linewidth]{figures/%s.pdf}\n"
            "\\caption{%s}\\label{%s}\n\\end{figure}\n```" % (f, cap, lab))


body = re.sub(r"&#91;embedded content: Figure \d[^\n]*", fig, body)

# ---- epistemic labels ----------------------------------------------------------
for both in (body, summary):
    pass
LBL = r"\\\[(Observed|Calculated|Hypothesis|Speculation)\\\]"
body = re.sub(LBL, lambda m: raw(r"\lbl{%s}" % m.group(1)), body)
summary = re.sub(LBL, lambda m: raw(r"\lbl{%s}" % m.group(1)), summary)
# a label in its own bold run reads better unbolded
body = re.sub(r"\*\*(`\\lbl\{\w+\}`\{=latex\})\*\*", r"\1", body)
summary = re.sub(r"\*\*(`\\lbl\{\w+\}`\{=latex\})\*\*", r"\1", summary)

# ---- variables named in prose -> math (exact phrases from the essay) -------------------
M = lambda t: raw("$" + t + "$")
PRE = [
    ("inference cost c per task and success rate s, where a failure costs c\\_err",
     "inference cost " + M("c") + " per task and success rate " + M("s") + ", where a failure costs " + M(r"c_{\mathrm{err}}")),
    ("Owning option O beats renting option R per task", "Owning option " + M("O") + " beats renting option " + M("R") + " per task"),
    ("where F is the one-off adaptation cost, T the model's useful life, M monthly upkeep including idle serving capacity, and V monthly volume",
     "where " + M("F") + " is the one-off adaptation cost, " + M("T") + " the model's useful life, " + M("M") +
     " monthly upkeep including idle serving capacity, and " + M("V") + " monthly volume"),
    ("Suppose owning starts R times cheaper", "Suppose owning starts " + M("R") + " times cheaper"),
    ("growing at yearly rate g, with t in years", "growing at yearly rate " + M("g") + ", with " + M("t") + " in years"),
    ("a preferred answer y\\_w and a rejected answer y\\_l to the same prompt x",
     "a preferred answer " + M("y_w") + " and a rejected answer " + M("y_l") + " to the same prompt " + M("x")),
    ("In a batch of B pairs of which k are identical", "In a batch of " + M("B") + " pairs of which " + M("k") + " are identical"),
    ("Suppose k crash-causing records", "Suppose " + M("k") + " crash-causing records"),
    ("the expected minimum of k uniform draws", "the expected minimum of " + M("k") + " uniform draws"),
    ("the sum of i/(k + 1) for i from 1 to k, which is k/2 epochs",
     "the sum of " + M("i/(k+1)") + " for " + M("i") + " from 1 to " + M("k") + ", which is " + M("k/2") + " epochs"),
    ("If a fraction c of the test set", "If a fraction " + M("c") + " of the test set"),
    ("The bias can never exceed c.", "The bias can never exceed " + M("c") + "."),
    ("| c, c\\_R, c\\_O |", "| " + M("c, c_R, c_O") + " |"),
    ("| s, s\\_R, s\\_O |", "| " + M("s, s_R, s_O") + " |"),
    ("| c\\_err |", "| " + M(r"c_{\mathrm{err}}") + " |"),
    ("| F |", "| " + M("F") + " |"), ("| T |", "| " + M("T") + " |"), ("| M |", "| " + M("M") + " |"),
    ("| V |", "| " + M("V") + " |"), ("| g |", "| " + M("g") + " |"), ("| h |", "| " + M("h") + " |"),
    ("| u |", "| " + M("u") + " |"),
    ("at utilization u.", "at utilization " + M("u") + "."),
    ("| `e2_real_model.py` |",
     "| `make_figures.py` | Figures 1 to 3 from the result files | none | `paper/figures/` |\n| `e2_real_model.py` |"),
]
for a, b in PRE:
    if a not in body:
        raise SystemExit("PRE phrase not found: " + a[:60])
    body = body.replace(a, b)

# ---- symbols and inline math --------------------------------------------------
SUBS = [
    ("exp(−λ\\_O T\\_b)", raw(r"$\exp(-\lambda_O T_b)$")),
    ("A₀ exp((g − λ\\_A) t)", raw(r"$A_0 \exp\big((g-\lambda_A)\,t\big)$")),
    ("φ × λ\\_A", raw(r"$\varphi\lambda_A$")),
    ("(B − k)/B", raw(r"$(B-k)/B$")),
    ("aᵤ < θ ≤ â", raw(r"$a_u < \theta \le \hat{a}$")),
    ("2 × 10⁻⁷", raw(r"$2\times10^{-7}$")),
    ("2 × 10⁻³", raw(r"$2\times10^{-3}$")),
    ("10⁻³", raw(r"$10^{-3}$")),
    ("λ\\_chip", raw(r"$\lambda_{\mathrm{chip}}$")),
    ("λ\\_A", raw(r"$\lambda_A$")),
    ("λ\\_O", raw(r"$\lambda_O$")),
    ("T\\_b", raw(r"$T_b$")),
    ("A₀", raw(r"$A_0$")),
    ("aₛ", raw(r"$a_s$")),
    ("aᵤ", raw(r"$a_u$")),
    ("φ", raw(r"$\varphi$")),
    ("κ", raw(r"$\kappa$")),
    ("δ", raw(r"$\delta$")),
    ("θ", raw(r"$\theta$")),
    ("÷", raw(r"$\div$")),
    ("c\\_err", raw(r"$c_{\mathrm{err}}$")),
    ("c\\_R", raw(r"$c_R$")), ("c\\_O", raw(r"$c_O$")),
    ("s\\_R", raw(r"$s_R$")), ("s\\_O", raw(r"$s_O$")),
]
for a, b in SUBS:
    body = body.replace(a, b)
    summary = summary.replace(a, b)
# "1,234×" multipliers in tables
body = re.sub(r"(\d)×", lambda m: m.group(1) + raw(r"$\times$"), body)

# ---- links -> citations ----------------------------------------------------------
# merge links split by earlier edits: [A](u)[B](u) -> [AB](u)
body = re.sub(r"\[([^\]]+)\]\((\S+?)\)\[([^\]]+)\]\(\2\)", r"[\1\3](\2)", body)
links = []


def tok(m):
    links.append((m.group(1), m.group(2)))
    return f"\x00{len(links)-1}\x00"


body = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", tok, body)
GITHUB = {u for u, (k, *_rest) in R.items() if u.startswith("https://github.com/")}
missing = sorted({u for _, u in links if u not in R})
if missing:
    raise SystemExit("no bib entry for: " + ", ".join(missing))


def key(i):
    return R[links[i][1]][0]


def running(i):
    text, url = links[i]
    if url in GITHUB:
        return raw(r"\href{%s}{\texttt{%s}}" % (url, text.replace("_", r"\_")))
    if text in AUTHOR_TEXTS:
        return raw(r"\citet{%s}" % key(i))
    return text + raw(r"~\citep{%s}" % key(i))


def process_line(line):
    if "\x00" not in line:
        return line
    in_table = line.lstrip().startswith("|")

    # parenthetical groups made only of citations -> one \citep
    def group(m):
        inner = m.group(1)
        ids = [int(x) for x in re.findall(r"\x00(\d+)\x00", inner)]
        rest = re.sub(r"\x00\d+\x00", "", inner)
        if ids and re.fullmatch(r"[\s,;]*(and)?[\s,;]*", rest):
            if any(links[i][1] in GITHUB for i in ids):
                return "(" + ", ".join(running(i) for i in ids) + ")"
            return raw(r"\citep{%s}" % ",".join(dict.fromkeys(key(i) for i in ids)))
        # mixed prose: keep the parentheses, cite inside
        return "(" + re.sub(r"\x00(\d+)\x00",
                            lambda t: (running(int(t.group(1))) if links[int(t.group(1))][1] in GITHUB
                                       else raw(r"\citealp{%s}" % key(int(t.group(1))))), inner) + ")"

    line = re.sub(r"\(([^()]*\x00\d+\x00[^()]*)\)", group, line)
    if in_table:
        line = re.sub(r"\x00(\d+)\x00", lambda t: (running(int(t.group(1)))
                      if links[int(t.group(1))][1] in GITHUB else raw(r"\citealp{%s}" % key(int(t.group(1))))), line)
    line = re.sub(r"\x00(\d+)\x00", lambda t: running(int(t.group(1))), line)
    return line


body = "\n".join(process_line(l) for l in body.split("\n"))
assert "\x00" not in body

# ---- tables: captions and relative column widths ----------------------------------
CAPTIONS = [
    ("Model", "tab:prices", "Cost of one narrow task (1,500 input and 200 output tokens) at list prices on 8 November 2026."),
    ("Level", "tab:ladder", "The ownership ladder: what an organization owns and rents at each level."),
    ("Price decline of rented capability", "tab:halflife",
     "How long an initial cost advantage lasts for an owner who never re-bases, and the cap on lifetime savings with flat volume."),
    ("Channel", "tab:channels", "Five channels through which data defects cost an adaptation project."),
    ("Layer", "tab:detectors", "Detectors by data-quality layer, with speed and coverage."),
    ("Defect | Parallelogram", "tab:e1",
     "E1: share of 100 injected rows flagged, by defect type and detector. For Parallelogram, E marks an error and W a warning."),
    ("Defect | TRL", "tab:e4", "E4: what TRL 1.17.1 does with each defect."),
    ("Test items leaked into training", "tab:contam",
     "E2, scarce-data regime: leaked test items inflate reported accuracy."),
    ("Profile", "tab:profiles", "Six cases on price: three-year discounted cost of renting and owning, at US labor rates."),
    ("Signpost", "tab:signposts", "Signposts for the coming decade, with their latest readings."),
    ("Symbol", "tab:notation", "Notation and the values or ranges used in the model."),
    ("File", "tab:data", "Data files and their SHA-256 hashes."),
    ("Script", "tab:scripts", "Scripts, seeds and outputs."),
]


def visible_text(s):
    def r(m):
        t = m.group(1)
        if "cite" in t:
            return "Edwards and Emberson, 2026"[:24]
        if "href" in t:
            return re.search(r"texttt\{([^}]*)\}", t).group(1) if "texttt" in t else "link"
        return re.sub(r"\\\w+|[{}$]", "", t)[:12]
    return re.sub(r"`([^`]*)`\{=latex\}", r, s).strip()


def visible_len(s):
    return len(visible_text(s))


def longest_word(s):
    return max((len(w) for w in visible_text(s).split()), default=0)


def fix_table(block):
    lines = block.strip("\n").split("\n")
    header = lines[0]
    cells = [[c for c in re.split(r"(?<!\\)\|", l)[1:-1]] for l in lines if not re.match(r"^\|\s*-", l)]
    ncol = len(cells[0])
    PT_PER_CHAR, AVAIL = 5.4, 446 - 12 * ncol      # \\small Palatino, textwidth minus padding
    mins, prefs = [], []
    for j in range(ncol):
        w = max(visible_len(r[j]) for r in cells if j < len(r))
        lw = max(longest_word(r[j]) for r in cells if j < len(r))
        if len(cells) > 1 and re.fullmatch(r"[0-9a-f]{64}", visible_text(cells[1][j])):
            lw, w = 24, 30
        mins.append((lw + 1) * PT_PER_CHAR)
        prefs.append(min(w, 60))
    rest = max(AVAIL - sum(mins), 0)
    widths = [m + rest * p_ / sum(prefs) for m, p_ in zip(mins, prefs)]
    widths = [max(3, round(x / 4)) for x in widths]
    sep = "|" + "|".join("-" * w for w in widths) + "|"
    lines[1] = sep
    cap = ""
    hdr = " | ".join(c.strip() for c in cells[0])
    for start, lab, text in CAPTIONS:
        if hdr.startswith(start):
            cap = f"\n\nTable: {text} " + raw("\\label{" + lab + "}")
            break
    if not cap:
        print("WARNING: no caption for table", hdr[:60])
    return "\n".join(lines) + cap


body = re.sub(r"(?:^\|.*\|[ \t]*\n)+", lambda m: fix_table(m.group(0)) + "\n", body, flags=re.M)

# ---- pandoc ---------------------------------------------------------------------
def pandoc(text):
    r = subprocess.run(
        ["pandoc", "-f", "markdown-tex_math_dollars-auto_identifiers+raw_attribute",
         "-t", "latex", "--columns=60", "--shift-heading-level-by=-1", "--wrap=preserve"],
        input=text, capture_output=True, text=True, check=True)
    return r.stdout


body_tex = pandoc(body)
summary_tex = pandoc(summary)

# pandoc table captions carry the label inside; make longtables small, captions left
body_tex = body_tex.replace("\\begin{longtable}", "{\\small\n\\begin{longtable}")
body_tex = body_tex.replace("\\end{longtable}", "\\end{longtable}\n}")
parts = body_tex.split("{\\small\n\\begin{longtable}")
for i in range(1, len(parts)):
    head = parts[i][:parts[i].find("\\end{longtable}")]
    if "tab:e1}" in head or "tab:profiles}" in head or "tab:e4}" in head:
        parts[i] = "\x01" + parts[i]
body_tex = "{\\small\n\\begin{longtable}".join(parts).replace(
    "{\\small\n\\begin{longtable}\x01", "{\\footnotesize\n\\begin{longtable}")
body_tex = re.sub(r"(\\end\{equation\}\n```?\n*|\\end\{equation\}\n\n)(where )", r"\1\\noindent \2", body_tex)
# 64-character hashes must be able to break inside table cells
body_tex = re.sub(r"\b([0-9a-f]{64})\b", r"{\\ttfamily\\footnotesize\\seqsplit{\1}}", body_tex)
for title, lab in [("The question", "sec:question"), ("The data bottleneck", "sec:data"),
                   ("An experimental investigation", "sec:experiments"),
                   ("The mathematics of ownership", "sec:math"), ("Synthesis", "sec:synthesis")]:
    body_tex = body_tex.replace("\\section{%s}" % title, "\\section{%s}\\label{%s}" % (title, lab), 1)
# pandoc's tightlist macro is defined in main.tex
(outdir / "body.tex").write_text(body_tex)
(outdir / "summary.tex").write_text(summary_tex)
(outdir / "refs.bib").write_text(bibtex())
print("ok: body.tex", len(body_tex), "summary.tex", len(summary_tex), "links", len(links))
