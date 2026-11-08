"""E2: harm dose-response for training-data defects, small-scale and controlled.

A small character-level GPT (about 0.8M parameters) is pretrained on unlabeled
domain sentences, then supervised fine-tuned (assistant-only loss) on a narrow
extraction task: read an order sentence, emit
    qty=<n>;item=<x>;color=<c>;city=<t>;day=<d>
Ground truth is exact, so harm is measured as exact-match accuracy on held-out
sentences with unseen field combinations.

Conditions (rate r = share of the fine-tuning set affected):
  clean            no defects
  subsample        train on N(1-r) unique rows (control for lost diversity)
  dup_concentrated r*N rows replaced by copies of a 20-row subset
  empty_target     r*N answers empty (model sees only the end token)
  truncated_target r*N answers cut mid-way with no end token (as keep_start
                   truncation does when a row exceeds max_length)
  label_noise      r*N answers have one field silently wrong (semantic)
  contamination    r of the test set copied into training (replacing rows)
DPO stage (from a deliberately under-trained SFT model):
  dpo_clean / dpo_identical (r of pairs have chosen == rejected) /
  dpo_flipped (r of pairs have chosen and rejected swapped)

Usage: python e2_harm.py <outdir> [seeds] [rates]
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)

ITEMS = ["widget", "lantern", "bucket", "kettle", "blanket", "drill", "ladder", "basket", "helmet", "mirror",
         "shovel", "teapot", "jacket", "pillow", "candle", "hammer", "bottle", "carpet", "saddle", "wrench",
         "pencil", "camera", "anchor", "bridle", "funnel", "goblet", "hinge", "magnet", "pulley", "trowel"]
COLORS = ["red", "blue", "green", "black", "white", "orange", "purple", "grey", "yellow", "brown", "pink", "teal"]
CITIES = ["Gaborone", "Francistown", "Maun", "Kasane", "Serowe", "Palapye", "Lobatse", "Molepolole", "Kanye",
          "Mahalapye", "Jwaneng", "Ghanzi", "Tsabong", "Letlhakane", "Mochudi", "Ramotswa", "Tlokweng",
          "Bobonong", "Shakawe", "Orapa", "Nata", "Gumare", "Tutume", "Thamaga", "Mogoditshane"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TEMPLATES = [
    "please send {q} {c} {i} to {t} on {d} .",
    "{t} needs {q} {i} in {c} by {d} , ref {r} .",
    "order {r} : {q} x {c} {i} , deliver {d} to {t} .",
    "can you get {q} {c} {i} over to {t} before {d} ?",
    "deliver to {t} on {d} : {i} ( {c} ) , quantity {q} .",
    "ref {r} - {d} - {t} - {q} {i} , colour {c} .",
    "we want {q} of the {c} {i} shipped to {t} ; {d} works .",
    "for {d} , {t} branch requests {q} {i} , all {c} .",
    "ship {i} , {c} , count {q} , destination {t} , day {d} .",
    "hi , {q} {c} {i} for {t} please . arrive {d} . ticket {r} .",
]
U, A, E = "<u>", "<a>", "<e>"   # user start, assistant start, end of answer


def sample_fields(rng):
    return dict(q=rng.randint(1, 20), i=rng.choice(ITEMS), c=rng.choice(COLORS), t=rng.choice(CITIES),
                d=rng.choice(DAYS), r=rng.randint(1, 99))


def _fmt(f):
    g = dict(f); g["q"] = f"n{f['q']}"; g["r"] = f"n{f['r']}"; g["t"] = f["t"].lower(); g["d"] = f["d"].lower()
    return g


def render(f, rng):
    return rng.choice(TEMPLATES).format(**_fmt(f))


def answer(f):
    g = _fmt(f)
    return f"qty {g['q']} item {g['i']} color {g['c']} city {g['t']} day {g['d']}"


def corrupt(f, rng):
    g = dict(f)
    k = rng.choice(["q", "i", "c", "t", "d"])
    pools = {"q": list(range(1, 21)), "i": ITEMS, "c": COLORS, "t": CITIES, "d": DAYS}
    g[k] = rng.choice([v for v in pools[k] if v != f[k]])
    return g


_WORDS = set()
for _t in TEMPLATES:
    _WORDS |= set(_t.replace("{q}", "").replace("{c}", "").replace("{i}", "").replace("{t}", "")
                  .replace("{d}", "").replace("{r}", "").split())
CHARS = sorted(_WORDS | set(ITEMS) | set(COLORS) | {c.lower() for c in CITIES} | {d.lower() for d in DAYS}
               | {f"n{k}" for k in range(1, 100)} | {"qty", "item", "color", "city", "day", U, A, E})
STOI = {c: i for i, c in enumerate(CHARS)}
PAD = len(CHARS)
VOCAB = len(CHARS) + 1
CTX = 48


def enc(s):
    return [STOI[w] for w in s.split()]


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        y = self.ln1(x)
        x = x + self.attn(y, y, y, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, d=96, h=4, L=3):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(CTX, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(CTX, CTX, dtype=torch.bool), 1), persistent=False)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))
        m = self.mask[:T, :T]
        for b in self.blocks:
            x = b(x, m)
        return self.head(self.ln(x))


def batchify(seqs, loss_from):
    """seqs: list of token lists; loss_from: index where loss starts (answer)."""
    T = min(CTX, max(len(s) for s in seqs))
    x = torch.full((len(seqs), T), PAD, dtype=torch.long)
    y = torch.full((len(seqs), T), -100, dtype=torch.long)
    for k, (s, lf) in enumerate(zip(seqs, loss_from)):
        s = s[:T + 1]
        n = len(s) - 1
        x[k, :n] = torch.tensor(s[:-1])
        tgt = torch.tensor(s[1:])
        pos = torch.arange(n)
        tgt[pos < lf - 1] = -100
        y[k, :n] = tgt
    return x, y


def pretrain(seed, steps=1500, bs=48):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    m = GPT()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    corpus = [enc(U + " " + render(sample_fields(rng), rng)) for _ in range(30000)]
    for step in range(steps):
        lr = 2e-3 * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr
        b = rng.sample(corpus, bs)
        x, y = batchify(b, [0] * bs)
        loss = F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


def make_example(f, rng, ans=None, end=True, cut=None):
    p = U + " " + render(f, rng) + " " + A
    a = (answer(f) if ans is None else ans).split()
    if cut is not None:
        a = a[:cut]
    s = enc(p) + [STOI[w] for w in a] + ([STOI[E]] if end else [])
    return s, len(enc(p))


def sft(model, data, seed, steps=600, bs=32, lr=1e-3):
    rng = random.Random(seed + 7)
    torch.manual_seed(seed + 7)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    order = []
    for step in range(steps):
        if len(order) < bs:
            order = list(range(len(data))); rng.shuffle(order)
        idx = [order.pop() for _ in range(bs)]
        lr_t = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr_t
        x, y = batchify([data[i][0] for i in idx], [data[i][1] for i in idx])
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def generate(model, prompts, max_new=14, sample=False, gen_seed=0):
    """Greedy decoding. Prompts are bucketed by length so no padding enters the
    context and absolute positions match training."""
    model.eval()
    g = torch.Generator().manual_seed(gen_seed)
    seqs = [enc(p) for p in prompts]
    outs = [None] * len(seqs)
    buckets = {}
    for j, s_ in enumerate(seqs):
        buckets.setdefault(len(s_), []).append(j)
    for L, ids in buckets.items():
        x = torch.tensor([seqs[j] for j in ids], dtype=torch.long)
        done = torch.zeros(len(ids), dtype=torch.bool)
        gen = [[] for _ in ids]
        for _ in range(max_new):
            logits = model(x[:, -CTX:])[:, -1, :]
            logits[:, PAD] = -1e9
            if sample:
                nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g)[:, 0]
            else:
                nxt = logits.argmax(-1)
            for k in range(len(ids)):
                if not done[k]:
                    if nxt[k].item() == STOI[E]:
                        done[k] = True
                    else:
                        gen[k].append(CHARS[nxt[k].item()])
            if done.all():
                break
            x = torch.cat([x, nxt[:, None]], 1)
        for k, j in enumerate(ids):
            outs[j] = (" ".join(gen[k]), bool(done[k]))
    model.train()
    return outs


def evaluate(model, test):
    prompts = [U + " " + s_ + " " + A for s_, _ in test]
    outs = generate(model, prompts)
    em = sum(o == gold for (o, _), (_, gold) in zip(outs, test)) / len(test)
    empty = sum(o == "" for o, _ in outs) / len(test)
    nostop = sum(not d for _, d in outs) / len(test)
    samp = generate(model, prompts, sample=True, gen_seed=123)
    em_s = sum(o == gold for (o, _), (_, gold) in zip(samp, test)) / len(test)
    empty_s = sum(o == "" for o, _ in samp) / len(test)
    with torch.no_grad():
        model.eval()
        seqs = [enc(p + " " + gold + " " + E) for p, (_, gold) in zip(prompts, test)]
        lfs = [len(enc(p)) for p in prompts]
        x, y = batchify(seqs, lfs)
        nll = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100).item()
        model.train()
    return {"exact_match": em, "empty_rate": empty, "no_stop_rate": nostop,
            "exact_match_sampled": em_s, "empty_rate_sampled": empty_s, "gold_nll": nll}, outs


def seqlogp(model, seqs, lfs):
    x, y = batchify(seqs, lfs)
    lp = torch.log_softmax(model(x), -1)
    mask = y != -100
    tok = lp.gather(-1, y.clamp(min=0)[..., None])[..., 0]
    return (tok * mask).sum(-1)


def dpo(model, ref, pairs, seed, steps=200, bs=32, lr=3e-4, beta=0.1):
    rng = random.Random(seed + 11)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for step in range(steps):
        b = rng.sample(pairs, bs)
        c = seqlogp(model, [p[0] for p in b], [p[2] for p in b])
        r = seqlogp(model, [p[1] for p in b], [p[2] for p in b])
        with torch.no_grad():
            c0 = seqlogp(ref, [p[0] for p in b], [p[2] for p in b])
            r0 = seqlogp(ref, [p[1] for p in b], [p[2] for p in b])
        loss = -F.logsigmoid(beta * ((c - c0) - (r - r0))).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


def clone(m):
    c = GPT(); c.load_state_dict(m.state_dict()); return c


def main(outdir, seeds="0,1,2,3,4", rates="0.10,0.30"):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in seeds.split(",")]
    rates = [float(x) for x in rates.split(",")]
    N, NLOW, NTEST, STEPS, LR = 2000, 200, 600, 1000, 2e-3
    resf = out / "e2_runs.jsonl"
    # resume support: skip (seed, condition, rate) rows already recorded
    existing = set()
    if resf.exists():
        for line in open(resf):
            r_ = json.loads(line)
            existing.add((r_["seed"], r_["condition"], r_["rate"]))
    t0 = time.time()
    for seed in seeds:
        rng = random.Random(1000 + seed)
        base = pretrain(seed)
        seen = set()
        def fresh():
            while True:
                f = sample_fields(rng)
                key = (f["q"], f["i"], f["c"], f["t"], f["d"])
                if key not in seen:
                    seen.add(key); return f
        train_f = [fresh() for _ in range(N)]
        test_f = [fresh() for _ in range(NTEST)]
        test = [(render(f, rng), answer(f)) for f in test_f]
        clean = [make_example(f, rng) for f in train_f]

        def record(cond, rate, metrics, extra=None):
            row = {"seed": seed, "condition": cond, "rate": rate, **metrics, **(extra or {}),
                   "elapsed_min": round((time.time() - t0) / 60, 1)}
            with open(resf, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(row, flush=True)

        def need(cond, rate):
            return (seed, cond, rate) not in existing

        def run(data, steps=STEPS):
            return sft(clone(base), data, seed, steps=steps, lr=LR)

        def contamination(cond, data_clean, r):
            if not need(cond, r):
                return
            kc = int(r * NTEST)
            drng = random.Random(seed * 7919 + int(r * 1000) + len(data_clean))
            leak_idx = set(drng.sample(range(NTEST), kc))
            leaked = []
            for i in sorted(leak_idx):
                p = U + " " + test[i][0] + " " + A
                leaked.append((enc(p + " " + test[i][1] + " " + E), len(enc(p))))
            d = data_clean[: len(data_clean) - kc] + leaked
            met, outs = evaluate(run(d), test)
            seen_acc = sum(outs[i][0] == test[i][1] for i in leak_idx) / max(1, kc)
            unseen = [i for i in range(NTEST) if i not in leak_idx]
            unseen_acc = sum(outs[i][0] == test[i][1] for i in unseen) / len(unseen)
            record(cond, r, met, {"acc_seen": seen_acc, "acc_unseen": unseen_acc})

        if need("clean", 0.0):
            record("clean", 0.0, evaluate(run(clean), test)[0])
        low = clean[:NLOW]
        if need("lowdata_clean", 0.0):
            record("lowdata_clean", 0.0, evaluate(run(low), test)[0])

        for r in rates:
            k = int(r * N)
            drng = random.Random(seed * 100 + int(r * 100))
            hit = set(drng.sample(range(N), k))
            keep = [clean[i] for i in range(N) if i not in hit]
            pool = [clean[i] for i in drng.sample([j for j in range(N) if j not in hit], 20)]
            if need("dup_concentrated", r):
                record("dup_concentrated", r, evaluate(run(keep + [pool[j % 20] for j in range(k)]), test)[0])
            d = [make_example(train_f[i], random.Random(i), ans="") if i in hit else clean[i] for i in range(N)]
            if need("empty_target", r):
                record("empty_target", r, evaluate(run(d), test)[0])
            d = []
            for i in range(N):
                if i in hit:
                    full = answer(train_f[i])
                    d.append(make_example(train_f[i], random.Random(i), end=False,
                                          cut=drng.randint(1, len(full.split()) - 1)))
                else:
                    d.append(clean[i])
            if need("truncated_target", r):
                record("truncated_target", r, evaluate(run(d), test)[0])
            d = [make_example(train_f[i], random.Random(i), ans=answer(corrupt(train_f[i], drng))) if i in hit else clean[i]
                 for i in range(N)]
            if need("label_noise_random", r):
                record("label_noise_random", r, evaluate(run(d), test)[0])
            # systematic noise: a fixed set of cities is always mislabelled as one
            # fixed wrong city; the set is grown until it covers ~r of the rows
            cities = CITIES[:]
            drng.shuffle(cities)
            affected, cover = set(), 0
            for c in cities:
                if cover >= k:
                    break
                affected.add(c); cover += sum(1 for f in train_f if f["t"] == c)
            wrong = {c: cities[(cities.index(c) + 1) % len(cities)] for c in affected}
            d = []
            for i, f in enumerate(train_f):
                if f["t"] in affected:
                    g_ = dict(f); g_["t"] = wrong[f["t"]]
                    d.append(make_example(f, random.Random(i), ans=answer(g_)))
                else:
                    d.append(clean[i])
            if need("label_noise_systematic", r):
                met, outs = evaluate(run(d), test)
                aff_test = [j for j, f in enumerate(test_f) if f["t"] in affected]
                acc_aff = sum(outs[j][0] == test[j][1] for j in aff_test) / max(1, len(aff_test))
                record("label_noise_systematic", r, met, {"rows_affected_share": cover / N, "acc_on_affected_cities": acc_aff})
            contamination("contamination", clean, r)
            # low-data regime (200 rows): the model does not generalize here,
            # which is the regime where contamination inflates scores. The
            # low-data duplicate and subsample conditions sat at the accuracy
            # floor in seeds 0-1 and were dropped as uninformative.
            contamination("lowdata_contamination", low, r)

        # Preference (DPO) conditions were piloted and dropped: from a partly
        # trained start, DPO degraded the model even with clean pairs
        # (likelihood displacement), which confounds any noise contrast.


DPO_START_STEPS = 450

if __name__ == "__main__":
    main(*sys.argv[1:])
