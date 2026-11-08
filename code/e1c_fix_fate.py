"""E1c: per-defect outcome of `parallelogram check --fix`, using the tool's own
per-line `outcomes`. Builds a mixed file: the clean base plus 10 injected rows
of each SFT defect type placed at random slots (seeded), runs --fix, and maps
each line's disposition back to its known defect. Also checks whether clean
rows survive unchanged.
Usage: python e1c_fix_fate.py <results/e1 dir>
"""
import json, random, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

d = Path(sys.argv[1]); w = d / "e1_files"
base_lines = (w / "base_clean.jsonl").read_text().splitlines()
types = ["invalid_json","missing_assistant","role_order","empty_response","whitespace_response",
         "overlong_response","overlong_user","secret_aws","secret_github","pii_email","mojibake",
         "label_swap","truncated_response","exact_duplicate","whitespace_duplicate","near_duplicate",
         "eval_exact","eval_prompt_only","eval_paraphrase"]
rng = random.Random(7)
mixed = list(base_lines); truth = {}
slots = list(range(len(mixed))); rng.shuffle(slots); ptr = 0
for t in types:
    lines = (w / f"sft_{t}.jsonl").read_text().splitlines()
    inj = [i for i, l in enumerate(lines) if l != base_lines[i]]
    for i in inj[:10]:
        s = slots[ptr]; ptr += 1
        mixed[s] = lines[i]; truth[s] = t
mp = w / "sft_mixed_c.jsonl"; mp.write_text("\n".join(mixed) + "\n")
p = subprocess.run(["parallelogram","check",str(mp),"--fix","--json","--max-seq-len","2048",
                    "--eval",str(w/"eval.jsonl"),"--output",str(w/"sft_mixed_c_fixed.jsonl")],
                   capture_output=True, text=True)
rep = json.loads(p.stdout)
outc = {o["line"]-1: o for o in rep["outcomes"]}
fate = defaultdict(Counter); clean = Counter(); rules = defaultdict(Counter)
for i in range(len(mixed)):
    o = outc.get(i, {"disposition": "missing"})
    if i in truth:
        fate[truth[i]][o["disposition"]] += 1
        for r in o.get("rules_fixed", []) + o.get("rules_unfixable", []): rules[truth[i]][r] += 1
    else:
        clean[o["disposition"]] += 1
# which clean rows were dropped, and why
clean_dropped = [ (i, outc[i].get("rules_fixed"), outc[i].get("rules_unfixable")) for i in range(len(mixed)) if i not in truth and outc.get(i,{}).get("disposition") != "unchanged"]
res = {"summary": rep["summary"], "fixes_by_rule": rep.get("fixes_by_rule"),
       "fate": {k: dict(v) for k, v in fate.items()}, "rules": {k: dict(v) for k, v in rules.items()},
       "clean": dict(clean), "clean_not_unchanged": clean_dropped}
(d / "e1c_results.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=1))
