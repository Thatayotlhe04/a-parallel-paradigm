"""E1: defect-detection benchmark.

Injects known defects into clean public fine-tuning data at known rows, then
measures which tools flag them. Ground truth is known by construction.

Datasets (downloaded from GitHub, SHA-256 recorded in results):
  - gururise/AlpacaDataCleaned alpaca_data_cleaned.json  -> chat SFT records
  - anthropics/hh-rlhf helpful-base train.jsonl.gz        -> preference pairs

Tools compared:
  P  parallelogram 1.6.1 (CLI, --json), all rules, --max-seq-len 2048, --eval
  C  OpenAI cookbook format checks (faithful re-implementation of the
     notebook's loop) plus its >16,385-token warning (approximate count,
     4 chars/token, because tiktoken's ranks cannot be downloaded here)
  X  naive exact dedup on the canonical JSON of each record
  M  MinHash LSH near-duplicate detection (datasketch), word 5-gram
     shingles, Jaccard threshold 0.8, 128 permutations
  N  13-gram train/eval overlap (GPT-3-paper style), lower-cased words

Usage: python e1_detection.py <alpaca_json> <hh_train_jsonl_gz> <outdir>
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import string
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from datasketch import MinHash, MinHashLSH

SEED = 20261108
N_BASE = 10_000          # SFT rows in each benchmark file
N_PREF = 5_000           # preference rows
N_EVAL = 1_000           # eval rows for overlap checks
K = 100                  # injected rows per defect type (1% of SFT file)
MAX_SEQ = 2048           # tokens; approx 4 chars/token without a tokenizer


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── loading ────────────────────────────────────────────────────────────────

def load_alpaca(path):
    rows = json.loads(Path(path).read_text())
    out = []
    for r in rows:
        user = r["instruction"].strip()
        if r.get("input", "").strip():
            user += "\n\n" + r["input"].strip()
        out.append({"messages": [{"role": "user", "content": user},
                                 {"role": "assistant", "content": r["output"].strip()}]})
    return out


HH_SPLIT = re.compile(r"\n\n(Human|Assistant): ")


def parse_hh(text):
    parts = HH_SPLIT.split("\n\n" + text.strip() if not text.startswith("\n\n") else text)
    msgs = []
    for i in range(1, len(parts) - 1, 2):
        role = "user" if parts[i] == "Human" else "assistant"
        msgs.append({"role": role, "content": parts[i + 1].strip()})
    return msgs


def load_hh(path):
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            c, rj = parse_hh(r["chosen"]), parse_hh(r["rejected"])
            if len(c) < 2 or len(rj) < 2 or c[:-1] != rj[:-1]:
                continue
            if c[-1]["role"] != "assistant" or rj[-1]["role"] != "assistant":
                continue
            out.append({"prompt": c[:-1], "chosen": [c[-1]], "rejected": [rj[-1]]})
    return out


def text_of(rec):
    if "messages" in rec:
        return "\n".join(m.get("content") or "" for m in rec["messages"] if isinstance(m, dict))
    msgs = rec.get("prompt", []) + rec.get("chosen", []) + rec.get("rejected", [])
    return "\n".join(m.get("content") or "" for m in msgs if isinstance(m, dict))


def write_jsonl(path, records, raw_lines=None):
    """records: list of dicts; raw_lines: {index: raw_string} overrides."""
    raw_lines = raw_lines or {}
    with open(path, "w", encoding="utf-8") as f:
        for i, r in enumerate(records):
            f.write(raw_lines.get(i, json.dumps(r, ensure_ascii=False)) + "\n")


# ── tools ──────────────────────────────────────────────────────────────────

def tool_parallelogram(path, eval_path=None):
    cmd = ["parallelogram", "check", str(path), "--json", "--max-seq-len", str(MAX_SEQ)]
    if eval_path:
        cmd += ["--eval", str(eval_path)]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    rep = json.loads(p.stdout)
    err, anyl = set(), set()
    rules = defaultdict(set)
    for iss in rep["issues"]:
        if iss["line"] is None or iss["severity"] == "info":
            continue
        idx = iss["line"] - 1
        anyl.add(idx)
        rules[iss["rule"]].add(idx)
        if iss["severity"] == "error":
            err.add(idx)
    return {"error": err, "any": anyl, "rules": rules, "seconds": dt, "exit": p.returncode}


def tool_cookbook(path):
    """Faithful re-implementation of the OpenAI cookbook checks, per example.
    The notebook parses with json.loads in a list comprehension, so one bad
    line aborts the whole script; we record that as a file-level crash and
    flag the offending line, which is the most charitable reading."""
    flagged, crashed = set(), False
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        try:
            ex = json.loads(line)
        except json.JSONDecodeError:
            crashed = True
            flagged.add(i)
            continue
        bad = False
        if not isinstance(ex, dict):
            flagged.add(i); continue
        messages = ex.get("messages", None)
        if not messages:
            flagged.add(i); continue
        for message in messages:
            if "role" not in message or "content" not in message:
                bad = True
            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                bad = True
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                bad = True
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            if (not content and not function_call) or not isinstance(content, str):
                bad = True
        if not any(message.get("role", None) == "assistant" for message in messages):
            bad = True
        n_tok = -(-len(text_of(ex)) // 4)
        if n_tok > 16385:
            bad = True
        if bad:
            flagged.add(i)
    return {"any": flagged, "crashed": crashed}


def canon(rec):
    return json.dumps(rec, sort_keys=True, ensure_ascii=False)


def tool_exact(records):
    seen, flagged = {}, set()
    for i, r in enumerate(records):
        if r is None:
            continue
        h = hashlib.sha256(canon(r).encode()).hexdigest()
        if h in seen:
            flagged.add(i)
        else:
            seen[h] = i
    return {"any": flagged}


def shingles(text, n=5):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def tool_minhash(records, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    flagged = set()
    for i, r in enumerate(records):
        if r is None:
            continue
        m = MinHash(num_perm=128, seed=1)
        for s in shingles(text_of(r)):
            m.update(s.encode())
        if lsh.query(m):
            flagged.add(i)
        lsh.insert(str(i), m)
    return {"any": flagged}


def ngrams13(text):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + 13]) for i in range(len(w) - 12)}


def tool_13gram(records, eval_records):
    ev = set()
    for r in eval_records:
        ev |= ngrams13(text_of(r))
    flagged = {i for i, r in enumerate(records) if r is not None and ngrams13(text_of(r)) & ev}
    return {"any": flagged}


# ── defects ────────────────────────────────────────────────────────────────

def rand_token(rng, n, alphabet=string.ascii_uppercase + string.digits):
    return "".join(rng.choice(alphabet) for _ in range(n))


def perturb_words(text, rng, frac=0.06):
    w = text.split(" ")
    for _ in range(max(1, int(len(w) * frac))):
        j = rng.randrange(len(w))
        op = rng.random()
        if op < 0.4 and len(w) > 3:
            w.pop(j)
        elif op < 0.7:
            w.insert(j, rng.choice(["really", "also", "just", "very", "the"]))
        else:
            w[j] = w[j].upper() if w[j].islower() else w[j].lower()
    return " ".join(w)


def asst(r):
    return [m for m in r["messages"] if m["role"] == "assistant"][-1]


def make_sft_defects(rng, base, pool_other, evalset):
    """Returns {defect: (records, raw_overrides, injected_idx, partner_idx)}.
    Each defect gets its own copy of the base with K injected rows."""
    out = {}

    def fresh():
        return [json.loads(json.dumps(r)) for r in base]

    def positions():
        return sorted(rng.sample(range(len(base)), K))

    # single-record defects: structural and secret/PII types, plus two semantic
    # ones (label_swap, truncated_response) that no deterministic rule targets
    for name in ["invalid_json", "missing_assistant", "role_order", "empty_response",
                 "whitespace_response", "overlong_response", "overlong_user",
                 "secret_aws", "secret_github", "pii_email", "mojibake",
                 "label_swap", "truncated_response"]:
        recs, raw, idx = fresh(), {}, positions()
        for i in idx:
            r = recs[i]
            a = asst(r)
            if name == "invalid_json":
                raw[i] = json.dumps(r, ensure_ascii=False)[:-7]
            elif name == "missing_assistant":
                r["messages"] = [m for m in r["messages"] if m["role"] != "assistant"]
            elif name == "role_order":
                c = a["content"]
                h = max(1, len(c) // 2)
                r["messages"] = [r["messages"][0], {"role": "assistant", "content": c[:h]},
                                 {"role": "assistant", "content": c[h:] or c}]
            elif name == "empty_response":
                a["content"] = ""
            elif name == "whitespace_response":
                a["content"] = "  \n "
            elif name == "overlong_response":
                a["content"] = (a["content"] + " ") * (MAX_SEQ * 5 // max(20, len(a["content"])) + 2)
            elif name == "overlong_user":
                filler = " ".join(rng.choice(base)["messages"][1]["content"] for _ in range(60))
                r["messages"][0]["content"] = "Background notes: " + filler[:MAX_SEQ * 6] + "\n\n" + r["messages"][0]["content"]
            elif name == "secret_aws":
                a["content"] += f"\n\nUse access key AKIA{rand_token(rng, 16)} for the bucket."
            elif name == "secret_github":
                a["content"] += f"\n\nexport GITHUB_TOKEN=ghp_{rand_token(rng, 36, string.ascii_letters + string.digits)}"
            elif name == "pii_email":
                user = rand_token(rng, 8, string.ascii_lowercase)
                a["content"] += f"\n\nContact {user}.{rand_token(rng, 5, string.ascii_lowercase)}@gmail.com or call +1 412 {rng.randint(200, 999)} {rng.randint(1000, 9999)}."
            elif name == "mojibake":
                a["content"] = "Itâ€™s worth noting: " + a["content"].replace("'", "â€™") + " CafÃ© rÃ©sumÃ©."
            elif name == "label_swap":
                a["content"] = asst(rng.choice(pool_other))["content"]
            elif name == "truncated_response":
                c = a["content"]
                a["content"] = c[: max(1, int(len(c) * 0.4))]
        out[name] = (recs, raw, idx, {})

    # duplicates: copy a random other row into the injected slot (pair-level truth)
    for name in ["exact_duplicate", "whitespace_duplicate", "near_duplicate"]:
        recs, idx = fresh(), positions()
        partners = {}
        others = [j for j in range(len(base)) if j not in set(idx)]
        for i in idx:
            j = rng.choice(others)
            partners[i] = j
            r = json.loads(json.dumps(base[j]))
            if name == "whitespace_duplicate":
                for m in r["messages"]:
                    m["content"] = "  " + m["content"] + " \n"
            elif name == "near_duplicate":
                for m in r["messages"]:
                    m["content"] = perturb_words(m["content"], rng)
            recs[i] = r
        out[name] = (recs, {}, idx, partners)

    # train/eval overlap
    for name in ["eval_exact", "eval_prompt_only", "eval_paraphrase"]:
        recs, idx = fresh(), positions()
        evs = rng.sample(evalset, K)
        for i, e in zip(idx, evs):
            r = json.loads(json.dumps(e))
            if name == "eval_prompt_only":
                asst(r)["content"] = asst(rng.choice(pool_other))["content"]
            elif name == "eval_paraphrase":
                for m in r["messages"]:
                    m["content"] = perturb_words(m["content"], rng)
            recs[i] = r
        out[name] = (recs, {}, idx, {})
    return out


def make_pref_defects(rng, base, pool_other):
    out = {}
    for name in ["identical_pair", "empty_chosen", "flipped_pair", "duplicate_pair"]:
        recs = [json.loads(json.dumps(r)) for r in base]
        idx = sorted(rng.sample(range(len(base)), K))
        partners = {}
        for i in idx:
            r = recs[i]
            if name == "identical_pair":
                r["rejected"] = json.loads(json.dumps(r["chosen"]))
            elif name == "empty_chosen":
                r["chosen"][-1]["content"] = ""
            elif name == "flipped_pair":
                r["chosen"], r["rejected"] = r["rejected"], r["chosen"]
            elif name == "duplicate_pair":
                j = rng.choice([x for x in range(len(base)) if x not in idx])
                partners[i] = j
                recs[i] = json.loads(json.dumps(base[j]))
        out[name] = (recs, {}, idx, partners)
    return out


# ── scoring ────────────────────────────────────────────────────────────────

def score(flagged, idx, partners):
    """Recall: injected rows detected (for pairs: either member flagged).
    Collateral: flags on rows that are neither injected nor a partner."""
    idx_set, part_set = set(idx), set(partners.values())
    hits = 0
    for i in idx:
        if i in flagged or (i in partners and partners[i] in flagged):
            hits += 1
    collateral = len(flagged - idx_set - part_set)
    return hits / len(idx), collateral


def main(alpaca, hh, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "e1_files"
    work.mkdir(exist_ok=True)
    rng = random.Random(SEED)

    sft_all = load_alpaca(alpaca)
    pref_all = load_hh(hh)
    meta = {"alpaca_sha256": sha256(alpaca), "hh_sha256": sha256(hh),
            "alpaca_rows": len(sft_all), "hh_pairs_parsed": len(pref_all), "seed": SEED}

    # natural defects in the full Alpaca-cleaned set
    write_jsonl(work / "alpaca_full.jsonl", sft_all)
    nat = tool_parallelogram(work / "alpaca_full.jsonl")
    meta["natural_full_alpaca"] = {
        "seconds": round(nat["seconds"], 2), "rows_per_sec": round(len(sft_all) / nat["seconds"]),
        "rows_with_error": len(nat["error"]), "rows_with_any": len(nat["any"]),
        "by_rule": {k: len(v) for k, v in nat["rules"].items()}}

    order = list(range(len(sft_all)))
    rng.shuffle(order)
    cand = [sft_all[i] for i in order[: N_BASE + N_EVAL + 4000]]
    evalset = cand[N_BASE: N_BASE + N_EVAL]
    pool_other = cand[N_BASE + N_EVAL:]
    base_raw = cand[:N_BASE]

    # clean the base: drop rows any tool flags (union), report how many
    write_jsonl(work / "base_raw.jsonl", base_raw)
    write_jsonl(work / "eval.jsonl", evalset)
    flags = {
        "parallelogram": tool_parallelogram(work / "base_raw.jsonl", work / "eval.jsonl")["any"],
        "cookbook": tool_cookbook(work / "base_raw.jsonl")["any"],
        "exact": tool_exact(base_raw)["any"],
        "minhash": tool_minhash(base_raw)["any"],
        "ngram13": tool_13gram(base_raw, evalset)["any"],
    }
    meta["base_raw_flags"] = {k: len(v) for k, v in flags.items()}
    union = set().union(*flags.values())
    base = [r for i, r in enumerate(base_raw) if i not in union]
    meta["base_clean_rows"] = len(base)
    write_jsonl(work / "base_clean.jsonl", base)
    # sanity: clean base is flagged by nothing
    b_flags = {
        "parallelogram": len(tool_parallelogram(work / "base_clean.jsonl", work / "eval.jsonl")["any"]),
        "cookbook": len(tool_cookbook(work / "base_clean.jsonl")["any"]),
        "exact": len(tool_exact(base)["any"]),
        "minhash": len(tool_minhash(base)["any"]),
        "ngram13": len(tool_13gram(base, evalset)["any"]),
    }
    meta["base_clean_flags"] = b_flags

    pref_order = list(range(len(pref_all)))
    rng.shuffle(pref_order)
    pref_raw = [pref_all[i] for i in pref_order[:N_PREF]]
    write_jsonl(work / "pref_raw.jsonl", pref_raw)
    pflags = tool_parallelogram(work / "pref_raw.jsonl")
    pex = tool_exact(pref_raw)["any"]
    pmh = tool_minhash(pref_raw)["any"]
    punion = pflags["any"] | pex | pmh
    meta["pref_raw_flags"] = {"parallelogram": len(pflags["any"]), "exact": len(pex), "minhash": len(pmh),
                              "parallelogram_by_rule": {k: len(v) for k, v in pflags["rules"].items()}}
    pref = [r for i, r in enumerate(pref_raw) if i not in punion]
    meta["pref_clean_rows"] = len(pref)

    results = []
    sft_defects = make_sft_defects(rng, base, pool_other, evalset)
    for name, (recs, raw, idx, partners) in sft_defects.items():
        path = work / f"sft_{name}.jsonl"
        write_jsonl(path, recs, raw)
        parsed = [None if i in raw else r for i, r in enumerate(recs)]
        P = tool_parallelogram(path, work / "eval.jsonl")
        C = tool_cookbook(path)
        X = tool_exact(parsed)
        M = tool_minhash(parsed)
        N = tool_13gram(parsed, evalset)
        row = {"file": "sft", "defect": name, "n_injected": len(idx), "rows": len(recs)}
        for tname, fl in [("P_error", P["error"]), ("P_any", P["any"]), ("cookbook", C["any"]),
                          ("exact", X["any"]), ("minhash", M["any"]), ("ngram13", N["any"])]:
            rec_, coll = score(fl, idx, partners)
            row[f"{tname}_recall"] = round(rec_, 3)
            row[f"{tname}_collateral"] = coll
        row["P_rules"] = sorted({r for r, s in P["rules"].items() if s & set(idx)})
        row["cookbook_crashed"] = C["crashed"]
        row["P_seconds"] = round(P["seconds"], 2)
        results.append(row)
        print(name, {k: v for k, v in row.items() if k.endswith("recall")}, flush=True)

    pref_defects = make_pref_defects(rng, pref, pool_other)
    for name, (recs, raw, idx, partners) in pref_defects.items():
        path = work / f"pref_{name}.jsonl"
        write_jsonl(path, recs, raw)
        P = tool_parallelogram(path)
        X = tool_exact(recs)
        M = tool_minhash(recs)
        row = {"file": "pref", "defect": name, "n_injected": len(idx), "rows": len(recs)}
        for tname, fl in [("P_error", P["error"]), ("P_any", P["any"]), ("exact", X["any"]),
                          ("minhash", M["any"])]:
            rec_, coll = score(fl, idx, partners)
            row[f"{tname}_recall"] = round(rec_, 3)
            row[f"{tname}_collateral"] = coll
        row["P_rules"] = sorted({r for r, s in P["rules"].items() if s & set(idx)})
        results.append(row)
        print(name, {k: v for k, v in row.items() if k.endswith("recall")}, flush=True)

    # repair quality: --fix on a mixed file (10 rows of each fixable SFT defect)
    mixed, raw_m, truth = [json.loads(json.dumps(r)) for r in base], {}, {}
    free = list(range(len(mixed)))
    rng.shuffle(free)
    ptr = 0
    for name, (recs, raw, idx, partners) in sft_defects.items():
        for i in idx[:10]:
            slot = free[ptr]; ptr += 1
            if i in raw:
                raw_m[slot] = raw[i]
            mixed[slot] = recs[i]
            truth[slot] = name
    mpath = work / "sft_mixed.jsonl"
    write_jsonl(mpath, mixed, raw_m)
    fixed_path = work / "sft_mixed_fixed.jsonl"
    p = subprocess.run(["parallelogram", "check", str(mpath), "--fix", "--json", "--max-seq-len", str(MAX_SEQ),
                        "--eval", str(work / "eval.jsonl"), "--output", str(fixed_path)],
                       capture_output=True, text=True)
    fix_rep = json.loads(p.stdout)
    kept = [json.loads(l) for l in fixed_path.read_text().splitlines() if l.strip()]
    kept_c = {canon(r) for r in kept}
    clean_slots = [i for i in range(len(mixed)) if i not in truth]
    clean_kept = sum(1 for i in clean_slots if canon(mixed[i]) in kept_c)
    fate = defaultdict(lambda: {"kept_unchanged": 0, "removed_or_changed": 0})
    for slot, name in truth.items():
        if slot in raw_m:
            fate[name]["removed_or_changed"] += 1
            continue
        key = "kept_unchanged" if canon(mixed[slot]) in kept_c else "removed_or_changed"
        fate[name][key] += 1
    meta["fix"] = {"summary": fix_rep.get("summary"), "clean_rows": len(clean_slots),
                   "clean_rows_kept_unchanged": clean_kept, "injected_fate": fate}

    # performance: cookbook timing on the 10k mixed file
    t0 = time.perf_counter(); tool_cookbook(mpath); meta["cookbook_seconds_10k"] = round(time.perf_counter() - t0, 3)
    t0 = time.perf_counter(); tool_minhash([json.loads(l) if i not in raw_m else None for i, l in enumerate(mpath.read_text().splitlines())]); meta["minhash_seconds_10k"] = round(time.perf_counter() - t0, 2)

    (out / "e1_results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2, default=dict))
    print(json.dumps(meta, indent=2, default=dict))


if __name__ == "__main__":
    main(*sys.argv[1:4])
