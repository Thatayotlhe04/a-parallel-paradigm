"""E1 follow-up: (a) per-defect fate under `parallelogram check --fix`,
(b) MinHash threshold trade-off (recall on near-duplicates vs flags on the
clean base), (c) natural 13-gram overlap inside a random split.

Usage: python e1b_followup.py <results/e1 dir>
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e1_detection import tool_minhash, score  # noqa: E402


def first(rec, role):
    for m in rec.get("messages", []):
        if isinstance(m, dict) and m.get("role") == role:
            return m.get("content")
    return None


def main(d):
    d = Path(d)
    w = d / "e1_files"
    mixed_lines = (w / "sft_mixed.jsonl").read_text().splitlines()
    fixed = [json.loads(l) for l in (w / "sft_mixed_fixed.jsonl").read_text().splitlines() if l.strip()]
    base = [json.loads(l) for l in (w / "base_clean.jsonl").read_text().splitlines()]
    base_c = Counter(json.dumps(r, sort_keys=True) for r in base)

    by_user = defaultdict(list)
    by_asst = defaultdict(list)
    for r in fixed:
        by_user[first(r, "user")].append(r)
        by_asst[first(r, "assistant")].append(r)
    fixed_c = Counter(json.dumps(r, sort_keys=True) for r in fixed)

    # rebuild truth: slots whose line differs from base_clean at same index
    fate = defaultdict(Counter)
    meta_truth = {}
    for i, line in enumerate(mixed_lines):
        orig = json.dumps(base[i], sort_keys=True)
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            fate["invalid_json"]["dropped"] += 1
            continue
        c = json.dumps(rec, sort_keys=True)
        if c == orig:
            continue
        meta_truth[i] = rec
    # identify defect type for each changed slot by re-running the generator is
    # costly; instead classify by observable signature
    def classify(rec):
        msgs = rec.get("messages", [])
        roles = [m.get("role") for m in msgs]
        a = first(rec, "assistant") or ""
        u = first(rec, "user") or ""
        if "assistant" not in roles:
            return "missing_assistant"
        if roles.count("assistant") >= 2:
            return "role_order"
        if a == "":
            return "empty_response"
        if a.strip() == "":
            return "whitespace_response"
        if "AKIA" in a:
            return "secret_aws"
        if "ghp_" in a:
            return "secret_github"
        if "@gmail.com" in a:
            return "pii_email"
        if "â€™" in a:
            return "mojibake"
        if u.startswith("Background notes:"):
            return "overlong_user"
        if len(a) > 8000:
            return "overlong_response"
        if u.startswith("  ") and u.endswith(" \n"):
            return "whitespace_duplicate"
        return "other"

    for i, rec in meta_truth.items():
        t = classify(rec)
        c = json.dumps(rec, sort_keys=True)
        if t == "other":
            # duplicates, label swaps, truncations, eval copies, near-dups:
            # identical survivors mean "kept"; duplicates are kept once if deduped
            n_out = fixed_c[c]
            n_in = sum(1 for l in mixed_lines if l.strip() and l == json.dumps(rec, ensure_ascii=False))
            t = "other_" + ("exact_dup" if base_c[c] else "nondup")
            fate[t]["kept"] += 1 if n_out else 0
            fate[t]["dropped"] += 0 if n_out else 1
            if base_c[c]:
                fate[t]["copies_in_output"] += n_out
            continue
        u, a = first(rec, "user"), first(rec, "assistant")
        cands = by_user.get(u, []) + by_asst.get(a, [])
        if fixed_c[json.dumps(rec, sort_keys=True)]:
            fate[t]["kept_unchanged"] += 1
        elif cands:
            fate[t]["repaired"] += 1
        else:
            fate[t]["dropped"] += 1

    # (b) MinHash thresholds
    trade = []
    nd = [json.loads(l) for l in (w / "sft_near_duplicate.jsonl").read_text().splitlines()]
    ep = [json.loads(l) for l in (w / "sft_eval_paraphrase.jsonl").read_text().splitlines()]
    res = json.loads((d / "e1_results.json").read_text())
    for thr in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        clean_flags = len(tool_minhash(base, thr)["any"])
        # injected near-dup slots: rows that differ from base at same index
        idx = [i for i, r in enumerate(nd) if json.dumps(r, sort_keys=True) != json.dumps(base[i], sort_keys=True)]
        fl = tool_minhash(nd, thr)["any"]
        # pair-level: either the copy or (unknown) partner flagged; partner
        # flags show up as collateral, so count copy-or-any-new-flag
        hit = sum(1 for i in idx if i in fl)
        new_flags = len(fl) - clean_flags if len(fl) >= clean_flags else 0
        trade.append({"threshold": thr, "clean_base_flags": clean_flags,
                      "near_dup_copy_flagged": round(hit / len(idx), 3), "flags_in_file": len(fl),
                      "approx_pair_recall": round(min(1.0, max(0, len(fl) - clean_flags) / len(idx)), 3)})
        print(trade[-1], flush=True)

    out = {"fix_fate": {k: dict(v) for k, v in fate.items()}, "minhash_tradeoff": trade}
    (d / "e1b_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["fix_fate"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
