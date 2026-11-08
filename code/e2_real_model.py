"""E2 at real-model scale (NOT executed in the paper's workspace: it needs a GPU
and Hugging Face Hub access). Repeats the E2 defect conditions with LoRA on a
small open instruct model, e.g. on a free Kaggle or Colab T4.

  pip install torch transformers peft
  python e2_real_model.py --model Qwen/Qwen2.5-0.5B-Instruct --out results_real

Same synthetic order-extraction task as e2_harm.py, rendered in natural text
(digits, capitalised names). Loss on the answer tokens only. Each run: LoRA
r=16 on all linear layers, lr 2e-4, 1 epoch over 2,000 examples, batch 16.
Expect roughly 2-4 minutes per run on a T4; 3 seeds x 5 defects x 3 rates
+ clean = 48 runs.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ITEMS = ["widget", "lantern", "bucket", "kettle", "blanket", "drill", "ladder", "basket", "helmet", "mirror",
         "shovel", "teapot", "jacket", "pillow", "candle", "hammer", "bottle", "carpet", "saddle", "wrench"]
COLORS = ["red", "blue", "green", "black", "white", "orange", "purple", "grey", "yellow", "brown"]
CITIES = ["Gaborone", "Francistown", "Maun", "Kasane", "Serowe", "Palapye", "Lobatse", "Molepolole", "Kanye",
          "Mahalapye", "Jwaneng", "Ghanzi", "Tsabong", "Letlhakane", "Mochudi", "Ramotswa"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TEMPLATES = [
    "Please send {q} {c} {i}s to {t} on {d}.", "{t} needs {q} {i}s in {c} by {d}, ref {r}.",
    "Order {r}: {q} x {c} {i}, deliver {d} to {t}.", "Can you get {q} {c} {i}s over to {t} before {d}?",
    "Deliver to {t} on {d}: {i} ({c}), quantity {q}.", "Ref {r} - {d} - {t} - {q} {i}s, colour {c}.",
]
SYSTEM = "Extract the order as: qty=<n>;item=<item>;color=<color>;city=<city>;day=<day>"


def fields(rng):
    return dict(q=rng.randint(1, 20), i=rng.choice(ITEMS), c=rng.choice(COLORS), t=rng.choice(CITIES),
                d=rng.choice(DAYS), r=rng.randint(1, 99))


def answer(f):
    return f"qty={f['q']};item={f['i']};color={f['c']};city={f['t']};day={f['d']}"


def corrupt(f, rng):
    g = dict(f); k = rng.choice("qictd")
    pools = {"q": list(range(1, 21)), "i": ITEMS, "c": COLORS, "t": CITIES, "d": DAYS}
    g[k] = rng.choice([v for v in pools[k] if v != f[k]]); return g


def encode(tok, prompt, ans, end=True, max_len=128):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pid = tok(p, add_special_tokens=False)["input_ids"]
    aid = tok(ans + (tok.eos_token if end else ""), add_special_tokens=False)["input_ids"]
    ids = (pid + aid)[:max_len]
    labels = ([-100] * len(pid) + aid)[:max_len]
    return ids, labels


def train(base_name, data, seed, device):
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(base_name)
    model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    rng = random.Random(seed); order = list(range(len(data))); rng.shuffle(order)
    model.train()
    for k in range(0, len(order), 16):
        b = [data[i] for i in order[k:k + 16]]
        L = max(len(x[0]) for x in b)
        ids = torch.full((len(b), L), tok.pad_token_id or 0); lab = torch.full((len(b), L), -100); att = torch.zeros((len(b), L), dtype=torch.long)
        for j, (x, y) in enumerate(b):
            ids[j, :len(x)] = torch.tensor(x); lab[j, :len(y)] = torch.tensor(y); att[j, :len(x)] = 1
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            loss = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device)).loss
        opt.zero_grad(); loss.backward(); opt.step()
    return tok, model


@torch.no_grad()
def evaluate(tok, model, test, device):
    model.eval(); tok.padding_side = "left"; correct = empty = 0; outs = []
    for k in range(0, len(test), 32):
        chunk = test[k:k + 32]
        texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                                         add_generation_prompt=True, tokenize=False) for p, _ in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=40, do_sample=False)
        for (p, gold), g in zip(chunk, gen):
            o = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            outs.append(o); correct += o == gold; empty += o == ""
    return {"exact_match": correct / len(test), "empty_rate": empty / len(test)}, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="results_real")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--rates", default="0.05,0.15,0.30")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(exist_ok=True)
    tok0 = AutoTokenizer.from_pretrained(a.model)
    for seed in [int(s) for s in a.seeds.split(",")]:
        rng = random.Random(1000 + seed)
        tr = [fields(rng) for _ in range(2000)]; te = [fields(rng) for _ in range(600)]
        test = [(rng.choice(TEMPLATES).format(**f), answer(f)) for f in te]
        prompts = [rng.choice(TEMPLATES).format(**f) for f in tr]
        clean = [encode(tok0, p, answer(f)) for p, f in zip(prompts, tr)]
        def run(cond, rate, data, extra=None):
            t0 = time.time(); tok, m = train(a.model, data, seed, device)
            met, outs = evaluate(tok, m, test, device)
            row = {"seed": seed, "condition": cond, "rate": rate, **met, **(extra(outs) if extra else {}), "minutes": round((time.time() - t0) / 60, 1)}
            print(row, flush=True); open(out / "runs.jsonl", "a").write(json.dumps(row) + "\n")
        run("clean", 0.0, clean)
        for r in [float(x) for x in a.rates.split(",")]:
            drng = random.Random(seed * 100 + int(r * 100)); hit = set(drng.sample(range(2000), int(r * 2000)))
            run("empty_target", r, [encode(tok0, prompts[i], "") if i in hit else clean[i] for i in range(2000)])
            run("truncated_target", r, [encode(tok0, prompts[i], answer(tr[i])[:drng.randint(5, 30)], end=False) if i in hit else clean[i] for i in range(2000)])
            run("label_noise_random", r, [encode(tok0, prompts[i], answer(corrupt(tr[i], drng))) if i in hit else clean[i] for i in range(2000)])
            cities = CITIES[:]; drng.shuffle(cities); affected, cover = set(), 0
            for c in cities:
                if cover >= len(hit): break
                affected.add(c); cover += sum(1 for f in tr if f["t"] == c)
            wrong = {c: cities[(cities.index(c) + 1) % len(cities)] for c in affected}
            run("label_noise_systematic", r, [encode(tok0, prompts[i], answer({**f, "t": wrong[f["t"]]})) if f["t"] in affected else clean[i] for i, f in enumerate(tr)])
            kc = int(r * 600); leak = set(drng.sample(range(600), kc))
            leaked = [encode(tok0, test[i][0], test[i][1]) for i in sorted(leak)]
            run("contamination", r, clean[: 2000 - kc] + leaked,
                lambda outs: {"acc_seen": sum(outs[i] == test[i][1] for i in leak) / max(1, kc),
                              "acc_unseen": sum(outs[i] == test[i][1] for i in range(600) if i not in leak) / (600 - kc)})


if __name__ == "__main__":
    main()
