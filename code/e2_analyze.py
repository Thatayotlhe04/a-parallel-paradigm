"""Analyse E2 runs: paired differences vs the clean run of the same seed (and
regime), mean and 95% t-interval across seeds, for each condition and rate.
Usage: python e2_analyze.py results/e2
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

d = Path(sys.argv[1])
rows = [json.loads(l) for l in open(d / "e2_runs.jsonl")]
by = defaultdict(dict)
for r in rows:
    by[r["seed"]][(r["condition"], r["rate"])] = r

METRICS = ["exact_match", "exact_match_sampled", "empty_rate_sampled", "gold_nll"]


def ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan"), float("nan"), n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return m, m - h, m + h, n


conds = sorted({(r["condition"], r["rate"]) for r in rows if r["rate"] > 0})
out = []
for cond, rate in conds:
    ref = "lowdata_clean" if cond.startswith("lowdata") else "clean"
    rec = {"condition": cond, "rate": rate}
    for met in METRICS:
        diffs, levels = [], []
        for s in by:
            if (cond, rate) in by[s] and (ref, 0.0) in by[s]:
                diffs.append(by[s][(cond, rate)][met] - by[s][(ref, 0.0)][met])
                levels.append(by[s][(cond, rate)][met])
        m, lo, hi, n = ci(diffs)
        rec[f"{met}_mean"] = sum(levels) / len(levels)
        rec[f"{met}_diff"] = m
        rec[f"{met}_lo"] = lo
        rec[f"{met}_hi"] = hi
        rec["n_seeds"] = n
        if n >= 2:
            sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1)) or 1e-12
            rec[f"{met}_p"] = 2 * stats.t.sf(abs(m) / (sd / math.sqrt(n)), n - 1)
    if cond.endswith("contamination"):
        infl = []
        for s in by:
            if (cond, rate) in by[s]:
                r = by[s][(cond, rate)]
                infl.append(r["exact_match"] - r["acc_unseen"])
        m, lo, hi, n = ci(infl)
        rec.update({"inflation_mean": m, "inflation_lo": lo, "inflation_hi": hi,
                    "acc_seen_mean": sum(by[s][(cond, rate)]["acc_seen"] for s in by if (cond, rate) in by[s]) / n,
                    "acc_unseen_mean": sum(by[s][(cond, rate)]["acc_unseen"] for s in by if (cond, rate) in by[s]) / n})
    if cond == "label_noise_systematic":
        vals = [by[s][(cond, rate)] for s in by if (cond, rate) in by[s]]
        rec["acc_on_affected_mean"] = sum(v["acc_on_affected_cities"] for v in vals) / len(vals)
        rec["rows_affected_share_mean"] = sum(v["rows_affected_share"] for v in vals) / len(vals)
    out.append(rec)

base = {}
for ref in ["clean", "lowdata_clean"]:
    for met in METRICS:
        xs = [by[s][(ref, 0.0)][met] for s in by if (ref, 0.0) in by[s]]
        base[f"{ref}_{met}"] = ci(xs)

keys = sorted({k for r in out for k in r})
with open(d / "e2_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(out)
(d / "e2_summary.json").write_text(json.dumps({"baselines": base, "conditions": out}, indent=2))
print("baselines", {k: tuple(round(x, 4) if isinstance(x, float) else x for x in v) for k, v in base.items()})
for r in out:
    line = f"{r['condition']:<26}{r['rate']:<6}n={r['n_seeds']} EM {100*r['exact_match_mean']:.1f} d={100*r['exact_match_diff']:+.1f} [{100*r['exact_match_lo']:+.1f},{100*r['exact_match_hi']:+.1f}] p={r.get('exact_match_p', float('nan')):.3g} | sampEM d={100*r['exact_match_sampled_diff']:+.1f} | empty_s {100*r['empty_rate_sampled_mean']:.1f} | nll d={r['gold_nll_diff']:+.3f}"
    if "inflation_mean" in r:
        line += f" | infl {100*r['inflation_mean']:+.1f} [{100*r['inflation_lo']:+.1f},{100*r['inflation_hi']:+.1f}] seen {100*r['acc_seen_mean']:.1f} unseen {100*r['acc_unseen_mean']:.1f}"
    if "acc_on_affected_mean" in r:
        line += f" | affected acc {100*r['acc_on_affected_mean']:.1f} rows {100*r['rows_affected_share_mean']:.1f}%"
    print(line)
