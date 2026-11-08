"""E4: what does Hugging Face TRL do with each training-data defect?

Builds a tiny GPT-2-style model and a BPE tokenizer locally (no Hub access
needed), gives it a ChatML chat template with {% generation %} markers so
assistant-only loss works, then feeds SFTTrainer and DPOTrainer small datasets
containing one defect type each. For every case we record:
  - stage at which anything happens: load, preprocess, train, or never
  - outcome: exception (loud), warning, or silent
  - for truncation: how many assistant tokens survive max_length
  - for DPO identical pairs: loss and gradient norm (theory: ln 2 and 0)

Usage: python e4_trl_audit.py <alpaca_json> <outdir>
"""
from __future__ import annotations

import io
import json
import logging
import math
import random
import sys
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

import trl
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

CHATML = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] + '<|im_end|>' }}{% endgeneration %}"
    "{% else %}{{ message['content'] + '<|im_end|>' }}{% endif %}{{ '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def build_tokenizer(texts, outdir):
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=2000, special_tokens=["<unk>", "<pad>", "<|im_start|>", "<|im_end|>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, tr)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>",
                                   bos_token="<|im_start|>", eos_token="<|im_end|>")
    fast.chat_template = CHATML
    fast.save_pretrained(outdir)
    return fast


def build_model(tok, outdir):
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=len(tok), n_positions=1024, n_embd=64, n_layer=2, n_head=2,
                     bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                     pad_token_id=tok.pad_token_id)
    m = GPT2LMHeadModel(cfg)
    m.save_pretrained(outdir)
    return m


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(f"{record.name}: {record.getMessage()[:300]}")


def run_case(fn):
    cap = Capture()
    root = logging.getLogger()
    root.addHandler(cap)
    buf_err, buf_out = io.StringIO(), io.StringIO()
    result = {"exception": None, "stage": None}
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            with redirect_stderr(buf_err), redirect_stdout(buf_out):
                extra = fn(result)
            if extra:
                result.update(extra)
        except Exception as e:  # noqa: BLE001
            result["exception"] = f"{type(e).__name__}: {str(e)[:300]}"
            result["traceback_tail"] = traceback.format_exc().splitlines()[-3:]
    root.removeHandler(cap)
    msgs = [f"{w.category.__name__}: {str(w.message)[:300]}" for w in wlist]
    noise = ("torch_dtype", "deprecat", "FutureWarning", "pin_memory", "tokenizers", "TRANSFORMERS")
    result["warnings"] = sorted({m for m in msgs + cap.records if not any(n.lower() in m.lower() for n in noise)})[:12]
    tail = [l for l in buf_err.getvalue().splitlines() if "warn" in l.lower() or "trunc" in l.lower()]
    result["stderr_warn_lines"] = tail[:8]
    return result


def sft_args(out, **kw):
    base = dict(output_dir=str(out), max_steps=2, per_device_train_batch_size=4, report_to=[],
                logging_steps=1, save_strategy="no", use_cpu=True, assistant_only_loss=True,
                max_length=256, dataset_num_proc=None, seed=0)
    base.update(kw)
    return SFTConfig(**base)


def main(alpaca, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1)
    rows = json.loads(Path(alpaca).read_text())
    rng.shuffle(rows)
    clean = []
    for r in rows[:400]:
        u = r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")
        clean.append({"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": r["output"]}]})
    tok = build_tokenizer([m["content"] for c in clean for m in c["messages"]], out / "tok")
    build_model(tok, out / "model")

    def model():
        return GPT2LMHeadModel.from_pretrained(out / "model")

    def ds_with(defect_rows, n_clean=12):
        return Dataset.from_list(clean[:n_clean] + defect_rows)

    a0 = clean[100]
    u, a = a0["messages"][0]["content"], a0["messages"][1]["content"]
    sft_cases = {
        "control_clean": [],
        "missing_assistant": [{"messages": [{"role": "user", "content": u}]}] * 4,
        "ends_on_user": [{"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a},
                                       {"role": "user", "content": "Thanks"}]}] * 4,
        "consecutive_assistant": [{"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a[:20]},
                                                {"role": "assistant", "content": a[20:] or a}]}] * 4,
        "assistant_first": [{"messages": [{"role": "assistant", "content": a}, {"role": "user", "content": u}]}] * 4,
        "empty_response": [{"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": ""}]}] * 4,
        "unknown_role": [{"messages": [{"role": "human", "content": u}, {"role": "bot", "content": a}]}] * 4,
        "exact_duplicates": [clean[0]] * 40,
    }
    results = {"versions": {"trl": trl.__version__, "torch": torch.__version__,
                            "transformers": __import__("transformers").__version__,
                            "datasets": __import__("datasets").__version__}}

    for name, rows_ in sft_cases.items():
        def fn(res, rows_=rows_, name=name):
            res["stage"] = "preprocess"
            tr = SFTTrainer(model=model(), args=sft_args(out / "run" / name), train_dataset=ds_with(rows_),
                            processing_class=tok)
            ds = tr.train_dataset
            masks = [sum(x) for x in ds["assistant_masks"]] if "assistant_masks" in ds.column_names else None
            res["stage"] = "train"
            st = tr.train()
            res["stage"] = "completed"
            hist = [h.get("loss") for h in tr.state.log_history if "loss" in h]
            return {"train_loss": st.training_loss, "loss_history": hist,
                    "rows_after_preprocess": len(ds),
                    "rows_with_zero_assistant_tokens": None if masks is None else sum(1 for m in masks if m == 0)}
        results[f"sft_{name}"] = run_case(fn)
        print(name, results[f"sft_{name}"].get("exception"), results[f"sft_{name}"].get("stage"), flush=True)

    # invalid JSON line at load time
    bad = out / "bad.jsonl"
    with open(bad, "w") as f:
        for c in clean[:10]:
            f.write(json.dumps(c) + "\n")
        f.write(json.dumps(clean[10])[:-6] + "\n")

    def fn_json(res):
        res["stage"] = "load"
        load_dataset("json", data_files=str(bad), split="train")
        res["stage"] = "completed"
    results["load_invalid_json"] = run_case(fn_json)

    # truncation: long user context pushes the answer past max_length
    long_rows = []
    for c in clean[20:36]:
        filler = " ".join(x["messages"][1]["content"] for x in clean[200:260])
        long_rows.append({"messages": [{"role": "user", "content": "Notes: " + filler + "\n\n" + c["messages"][0]["content"]},
                                       c["messages"][1]]})

    def fn_trunc(res, max_len=256):
        res["stage"] = "preprocess"
        full = SFTTrainer(model=model(), args=sft_args(out / "run" / "full", max_length=None),
                          train_dataset=Dataset.from_list(long_rows), processing_class=tok).train_dataset
        res["stage"] = "preprocess_truncated"
        tr = SFTTrainer(model=model(), args=sft_args(out / "run" / "trunc", max_length=max_len),
                        train_dataset=Dataset.from_list(long_rows), processing_class=tok)
        cut = tr.train_dataset
        before = [sum(x) for x in full["assistant_masks"]]
        lens = [len(x) for x in full["input_ids"]]
        after = [sum(x) for x in cut["assistant_masks"]] if "assistant_masks" in cut.column_names else None
        res["stage"] = "train"
        st = tr.train()
        res["stage"] = "completed"
        hist = [h.get("loss") for h in tr.state.log_history if "loss" in h]
        return {"max_length": max_len, "median_full_length": sorted(lens)[len(lens) // 2],
                "assistant_tokens_before": sum(before), "assistant_tokens_after": None if after is None else sum(after),
                "rows_with_all_assistant_tokens_cut": None if after is None else sum(1 for x in after if x == 0),
                "rows": len(long_rows), "train_loss": st.training_loss, "loss_history": hist}
    results["sft_truncation_long_context"] = run_case(fn_trunc)
    print("trunc", results["sft_truncation_long_context"], flush=True)

    # DPO: identical pairs give loss ln 2 and zero gradient, by theory
    def pref(c, rej):
        return {"prompt": [c["messages"][0]], "chosen": [c["messages"][1]], "rejected": [{"role": "assistant", "content": rej}]}

    ident = [pref(c, c["messages"][1]["content"]) for c in clean[40:56]]
    normal = [pref(c, clean[300 + i]["messages"][1]["content"]) for i, c in enumerate(clean[40:56])]

    def dpo_case(rows_, label):
        def fn(res):
            res["stage"] = "preprocess"
            m = model()
            args = DPOConfig(output_dir=str(out / "run" / label), max_steps=1, per_device_train_batch_size=16,
                             report_to=[], logging_steps=1, save_strategy="no", use_cpu=True, max_length=256,
                             learning_rate=1e-3, seed=0)
            tr = DPOTrainer(model=m, args=args, train_dataset=Dataset.from_list(rows_), processing_class=tok)
            before = [p.detach().clone() for p in m.parameters()]
            res["stage"] = "train"
            tr.train()
            res["stage"] = "completed"
            delta = math.sqrt(sum(float(((p.detach() - b) ** 2).sum()) for p, b in zip(m.parameters(), before)))
            logs = [h for h in tr.state.log_history if "loss" in h]
            return {"loss": logs[0].get("loss") if logs else None, "grad_norm": logs[0].get("grad_norm") if logs else None,
                    "param_change_l2": delta, "ln2": math.log(2)}
        return fn
    results["dpo_identical_pairs"] = run_case(dpo_case(ident, "dpo_ident"))
    results["dpo_normal_pairs"] = run_case(dpo_case(normal, "dpo_norm"))
    print("dpo", results["dpo_identical_pairs"], results["dpo_normal_pairs"], flush=True)

    (out / "e4_results.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:3])
