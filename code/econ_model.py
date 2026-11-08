"""Ownership economics model (Stages III and VI).

Compares renting a capable API for one narrow task against owning a LoRA-tuned
open-weight 8B model served on rented GPUs, at matched task quality (the
"replace the frontier API with a fine-tuned small model" case), over a
36-month horizon in monthly steps.

Rent:   R(t) = A0 * exp((g - lam_A) t)            monthly API bill
Own:    F at t=0 (labor + data + compute)
        + F_r at each re-base every T_b months      re-adaptation
        + serving: max(GPU floor, V(t) * h / (C_h * u_eff))
        + maintenance m hours/month at labor rate w
        serving cost per task falls at lam_O = phi * lam_A after each re-base
Discount rate delta. Own beats rent when its discounted cost is lower.

Inputs are sourced in notes/pricing.md (Nov 2026); ranges are stated below.
Outputs go to results/econ/.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results" / "econ"
OUT.mkdir(parents=True, exist_ok=True)

# ── sourced point inputs (notes/pricing.md) ───────────────────────────────
TAU_IN, TAU_OUT = 1500, 200                 # tokens per narrow task (assumption)
SONNET = (2.00, 10.00)                      # USD per 1M tokens, Claude Sonnet 5.5
HAIKU = (0.10, 0.50)                        # Claude Haiku 5.5 (prompt <=100K)
OPEN8B_HOSTED = (0.02, 0.04)                # DeepInfra Llama 3.1 8B Turbo
H100_INDEX = 3.00                           # USD/GPU-hr, Silicon Data, 2026-11-08
TOGETHER_DEDICATED = 5.49                   # USD/GPU-hr dedicated H100 endpoint
BENCH_OUT_TPS = 14992                       # TensorRT-LLM, Llama 3.1 8B FP8, H100, 1000/1000
US_ML_RATE = 251250 / 2080                  # USD/hr from Levels.fyi median TC
HOURS_PER_MONTH = 730


def per_task(price):
    return (TAU_IN * price[0] + TAU_OUT * price[1]) / 1e6


def tasks_per_gpu_hour(kappa, eff):
    """Capacity from the 1000-in/1000-out benchmark, converting input tokens to
    output-equivalents with weight kappa (prefill is cheaper than decode)."""
    units_per_sec = BENCH_OUT_TPS / 1000 * (1000 + kappa * 1000)
    units_per_task = TAU_OUT + kappa * TAU_IN
    return units_per_sec / units_per_task * 3600 * eff


def simulate(A0, p):
    """Discounted 36-month cost of renting and owning. p: dict of arrays (MC)
    or floats. Returns (rent_cost, own_cost)."""
    months = np.arange(36)
    t = months / 12.0
    disc = np.exp(-p["delta"][..., None] * t)
    shape = np.broadcast(A0, p["lam_A"]).shape
    A0 = np.broadcast_to(A0, shape)
    lam_A, g = p["lam_A"], p["g"]
    rent = (A0[..., None] * np.exp((g[..., None] - lam_A[..., None]) * t) * disc).sum(-1)
    if p.get("switch_hours") is not None:
        # renter migrates to a cheaper equal-capability model every 6 months,
        # paying engineering time to re-validate prompts and evaluations
        sw = ((months % 6 == 0) & (months > 0)).astype(float)
        rent = rent + ((p["switch_hours"] * p["w"])[..., None] * sw[None, :] * disc).sum(-1)

    V0 = A0 / per_task(SONNET)                                  # tasks per month today
    V = V0[..., None] * np.exp(g[..., None] * t)
    C_h = tasks_per_gpu_hour(p["kappa"], p["eff"])
    # owned unit cost falls only at re-bases, in steps of exp(-lam_O * T_b)
    lam_O = p["phi"] * lam_A
    n_rebases = np.floor(months[None, :] / p["T_b"][..., None])
    step_factor = np.exp(-lam_O[..., None] * n_rebases * (p["T_b"][..., None] / 12.0))
    gpu_hours = V / (C_h[..., None] * p["u"][..., None]) * step_factor
    serving = np.maximum(HOURS_PER_MONTH * p["h"][..., None], gpu_hours * p["h"][..., None])
    if p.get("pooled") is not None:
        # level 3: adapter served on shared multi-tenant infrastructure, billed
        # per token at the hosted base-model price; no idle-GPU floor
        pooled = V * per_task(OPEN8B_HOSTED) * p["pooled"][..., None] * step_factor
        serving = np.where(p["pooled"][..., None] > 0, pooled, serving)
    maint = (p["m_hours"] * p["w"])[..., None]
    F = p["F_hours"] * p["w"] + p["n_examples"] * per_task(SONNET) * 3 + p["review_hours"] * p["w"] + 50.0
    F_r = p["rho"] * p["F_hours"] * p["w"]
    rebase_month = (months[None, :] % p["T_b"][..., None] == 0) & (months[None, :] > 0)
    rebases = (rebase_month * F_r[..., None] * disc).sum(-1)
    own = F + rebases + ((serving + maint) * disc).sum(-1)
    return rent, own


def draw(n, rng, labor_rate=None):
    lu = lambda a, b: np.exp(rng.uniform(math.log(a), math.log(b), n))  # noqa: E731
    return {
        "lam_A": rng.uniform(math.log(4.7), math.log(13.0), n),   # Epoch: mature vs average capability
        "g": rng.uniform(0.0, math.log(2.0), n),                   # volume growth 0-100%/yr
        "delta": np.full(n, 0.10),
        "kappa": rng.uniform(0.1, 0.4, n),
        "eff": rng.uniform(0.25, 0.7, n),                          # real vs offline max-load throughput
        "u": rng.uniform(0.15, 0.6, n),                            # utilization of provisioned capacity
        "h": rng.uniform(2.0, TOGETHER_DEDICATED, n),
        "phi": rng.uniform(0.3, 1.0, n),                           # share of price decline owners capture
        "T_b": rng.integers(4, 13, n),                             # months between re-bases
        "rho": rng.uniform(0.15, 0.5, n),                          # re-adaptation share of initial labor
        "F_hours": lu(160, 480),                                   # 1-3 engineer-months
        "review_hours": rng.uniform(40, 120, n),
        "n_examples": rng.uniform(2000, 10000, n),
        "m_hours": rng.uniform(10, 40, n),
        "w": np.full(n, labor_rate) if labor_rate else lu(20, US_ML_RATE),
    }


def median_params(labor_rate=US_ML_RATE):
    return {"lam_A": np.array([math.log(math.sqrt(4.7 * 13.0))]), "g": np.array([math.log(2.0) / 2]),
            "delta": np.array([0.10]), "kappa": np.array([0.25]), "eff": np.array([0.475]),
            "u": np.array([0.375]), "h": np.array([(2.0 + TOGETHER_DEDICATED) / 2]), "phi": np.array([0.65]),
            "T_b": np.array([8]), "rho": np.array([0.325]), "F_hours": np.array([math.sqrt(160 * 480)]),
            "review_hours": np.array([80.0]), "n_examples": np.array([6000.0]), "m_hours": np.array([25.0]),
            "w": np.array([labor_rate])}


def breakeven(p, lo=10.0, hi=1e8):
    """Smallest monthly API bill A0 at which owning is cheaper (bisection on log A0)."""
    def diff(a):
        r, o = simulate(np.array([a]), p)
        return float(o[0] - r[0])
    if diff(hi) > 0:
        return float("inf")
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if diff(mid) > 0:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    rng = np.random.default_rng(20261108)
    res = {"inputs": {"per_task_sonnet": per_task(SONNET), "per_task_haiku": per_task(HAIKU),
                      "per_task_open8b_hosted": per_task(OPEN8B_HOSTED),
                      "tasks_per_gpu_hour_median": tasks_per_gpu_hour(0.25, 0.475),
                      "gpu_cost_per_task_at_median": (H100_INDEX / tasks_per_gpu_hour(0.25, 0.475) / 0.375),
                      "us_ml_rate_per_hour": US_ML_RATE}}

    # Monte Carlo: probability owning is cheaper vs current monthly API bill.
    # Primary measure everywhere in the essay: the bill at which P(own cheaper)
    # crosses 50%, interpolated on a log scale ("even-odds point").
    grid = np.exp(np.linspace(math.log(100), math.log(3e6), 40))
    n = 20000

    def scen_base(labor=None):
        return draw(n, rng, labor)

    def scen_static():
        p = draw(n, rng, None); p["lam_A"] = np.zeros(n); p["g"] = np.zeros(n); p["phi"] = np.zeros(n); return p

    def scen_static_growth():
        p = draw(n, rng, None); p["lam_A"] = np.zeros(n); p["phi"] = np.zeros(n); return p

    def scen_pooled():
        p = draw(n, rng, None); p["pooled"] = rng.uniform(1.0, 3.0, n); return p

    def scen_mature():
        p = draw(n, rng, None); p["lam_A"] = np.full(n, math.log(4.7)); return p

    def scen_switch():
        p = draw(n, rng, None); p["switch_hours"] = rng.uniform(10, 40, n); return p

    def scen_labor10(pooled=False):
        p = draw(n, rng, None)
        for k_ in ("F_hours", "review_hours", "m_hours"):
            p[k_] = p[k_] / 10
        if pooled:
            p["pooled"] = rng.uniform(1.0, 3.0, n)
        return p

    scenarios = [
        ("labor uncertain $20-$121/hr", lambda: scen_base(None)),
        ("US labor $121/hr", lambda: scen_base(US_ML_RATE)),
        ("lower-cost labor $25/hr", lambda: scen_base(25.0)),
        ("pooled adapter serving (level 3)", scen_pooled),
        ("static prices (counterfactual)", scen_static),
        ("static prices, growth kept", scen_static_growth),
        ("mature capability only, 4.7x/yr", scen_mature),
        ("renter pays to switch models", scen_switch),
        ("labor hours / 10", lambda: scen_labor10(False)),
        ("labor hours / 10 + pooled serving", lambda: scen_labor10(True)),
    ]
    curves = []
    for label, make in scenarios:
        p = make()
        for a in grid:
            r, o = simulate(np.full(n, a), p)
            curves.append({"scenario": label, "monthly_api_bill": round(float(a), 2),
                           "p_own_cheaper": float((o < r).mean()),
                           "median_own_over_rent": float(np.median(o / r))})
    with open(OUT / "montecarlo_curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0].keys())); w.writeheader(); w.writerows(curves)

    def even_odds(label):
        rows_ = [c for c in curves if c["scenario"] == label]
        for i, c in enumerate(rows_):
            if c["p_own_cheaper"] >= 0.5:
                if i == 0:
                    return c["monthly_api_bill"]
                a_, b_ = rows_[i - 1], c
                f_ = (0.5 - a_["p_own_cheaper"]) / (b_["p_own_cheaper"] - a_["p_own_cheaper"])
                return math.exp(math.log(a_["monthly_api_bill"]) + f_ * (math.log(b_["monthly_api_bill"]) - math.log(a_["monthly_api_bill"])))
        return None
    res["even_odds_point_usd_per_month"] = {lab: round(even_odds(lab)) for lab, _ in scenarios}
    res["labor_uncertain_median_rate"] = math.sqrt(20 * US_ML_RATE)

    # deterministic break-even at median parameters, and a tornado
    base = median_params()
    be0 = breakeven(base)
    res["breakeven_median_params_usd_per_month"] = be0
    static = median_params(); static["lam_A"][:] = 0; static["g"][:] = 0; static["phi"][:] = 0
    res["breakeven_static_prices_usd_per_month"] = breakeven(static)
    ranges = {
        "lam_A": (math.log(4.7), math.log(13.0), "API price decline at fixed capability (4.7x to 13x per year)"),
        "g": (0.0, math.log(2.0), "workload growth (0% to 100% per year)"),
        "u": (0.6, 0.15, "utilization of provisioned GPUs (60% to 15%)"),
        "h": (2.0, TOGETHER_DEDICATED, "GPU price ($2.00 to $5.49 per hour)"),
        "eff": (0.7, 0.25, "achieved vs benchmark throughput (70% to 25%)"),
        "F_hours": (160, 480, "initial engineering (1 to 3 engineer-months)"),
        "m_hours": (10, 40, "maintenance (10 to 40 hours per month)"),
        "T_b": (12, 4, "months between re-bases (12 to 4)"),
        "rho": (0.15, 0.5, "re-adaptation cost (15% to 50% of initial labor)"),
        "w": (25.0, US_ML_RATE, "labor rate ($25 to $121 per hour)"),
        "phi": (1.0, 0.3, "share of price decline owners capture (100% to 30%)"),
    }
    tornado = []
    for k, (lo_v, hi_v, desc) in ranges.items():
        vals = []
        for v in (lo_v, hi_v):
            q = {kk: vv.copy() for kk, vv in base.items()}
            q[k] = np.array([v], dtype=q[k].dtype if k != "T_b" else int)
            vals.append(breakeven(q))
        tornado.append({"parameter": desc, "breakeven_low_setting": vals[0], "breakeven_high_setting": vals[1],
                        "base": be0})
    tornado.sort(key=lambda r: -abs(math.log(max(r["breakeven_high_setting"], 1)) - math.log(max(r["breakeven_low_setting"], 1))))
    with open(OUT / "tornado.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tornado[0].keys())); w.writeheader(); w.writerows(tornado)
    res["tornado"] = tornado

    # half-life and savings cap (Stage 0 derivation, recomputed)
    hl = []
    for name, lamA in [("mature capability, 4.7x/yr", math.log(4.7)), ("average, 13x/yr", math.log(13)),
                       ("just-debuted, 75x/yr", math.log(75))]:
        for R in (3, 10, 30):
            t_star = math.log(R) / (lamA - math.log(1.49))
            hl.append({"price_decline": name, "initial_cost_advantage": R, "years_until_gone": round(t_star, 2)})
        hl.append({"price_decline": name, "initial_cost_advantage": "cap", "years_until_gone": None,
                   "savings_cap_months_of_todays_bill": round(12 / lamA, 1)})
    res["half_life"] = hl

    # personas (median parameters, explicit volumes and comparator)
    personas = [
        ("Individual developer", 50_000, "Haiku 5.5", per_task(HAIKU)),
        ("Individual developer (frontier quality)", 50_000, "Sonnet 5.5", per_task(SONNET)),
        ("Small software company", 2_000_000, "Sonnet 5.5", per_task(SONNET)),
        ("Medium-sized business", 20_000_000, "Sonnet 5.5", per_task(SONNET)),
        ("Large enterprise (one task)", 200_000_000, "Sonnet 5.5", per_task(SONNET)),
        ("Small company that only needs small-model quality", 2_000_000, "Haiku 5.5", per_task(HAIKU)),
    ]
    rows = []
    for name, V, comp, ppt in personas:
        A0 = V * ppt
        # convert A0 into the Sonnet-equivalent volume the simulator expects
        p = median_params()
        scale = per_task(SONNET) / ppt
        r, o = simulate(np.array([A0 * scale]), p)
        r = r * (1 / scale)  # rent cost at the actual comparator price
        rows.append({"persona": name, "tasks_per_month": V, "comparator": comp,
                     "monthly_api_bill": round(A0, 2), "rent_36mo_discounted": round(float(r[0])),
                     "own_36mo_discounted": round(float(o[0])), "own_over_rent": round(float(o[0] / r[0]), 2)})
    with open(OUT / "personas.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    res["personas"] = rows
    # section 4.2 worked example: owning vs Haiku at 2M tasks/month, a
    # 5-point accuracy gain; error cost at which owning pays
    own_36 = [r_ for r_ in rows if r_["persona"].startswith("Small company that only")][0]["own_36mo_discounted"]
    own_month_disc = own_36 / 36
    p_ = median_params(); r_u, o_u = simulate(np.array([2_000_000 * per_task(SONNET)]), {**p_, "delta": np.array([1e-9])})
    own_month_undisc = float(o_u[0]) / 36
    haiku_bill = 2_000_000 * per_task(HAIKU)
    res["worked_example_4_2"] = {
        "own_monthly_discounted_avg": own_month_disc, "own_monthly_undiscounted_avg": own_month_undisc,
        "haiku_monthly_bill": haiku_bill,
        "error_cost_threshold_discounted": (own_month_disc - haiku_bill) / (0.05 * 2_000_000),
        "error_cost_threshold_undiscounted": (own_month_undisc - haiku_bill) / (0.05 * 2_000_000)}
    (OUT / "econ_results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "tornado"}, indent=1, default=float))
    for t_ in tornado:
        print(t_)


if __name__ == "__main__":
    main()
