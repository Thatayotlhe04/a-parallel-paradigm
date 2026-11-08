"""E4b: TRL 1.17 defaults (assistant_only_loss=False, max_length=1024,
truncation_mode=keep_start) on defects that raised errors only under
assistant_only_loss=True, plus a direct measurement of how much of the
assistant answer survives keep_start truncation.
Usage: python e4b_defaults_truncation.py <alpaca_json> <results/e4 dir>
"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from e4_trl_audit import run_case, sft_args  # noqa: E402
from datasets import Dataset
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

alpaca, out = sys.argv[1], Path(sys.argv[2])
tok = PreTrainedTokenizerFast.from_pretrained(out / "tok")
rows = json.loads(Path(alpaca).read_text()); random.Random(1).shuffle(rows)
clean = [{"messages": [{"role": "user", "content": r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")},
                       {"role": "assistant", "content": r["output"]}]} for r in rows[:400]]
u, a = clean[100]["messages"][0]["content"], clean[100]["messages"][1]["content"]
cases = {
  "missing_assistant": [{"messages": [{"role": "user", "content": u}]}] * 4,
  "unknown_role": [{"messages": [{"role": "human", "content": u}, {"role": "bot", "content": a}]}] * 4,
  "empty_response": [{"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": ""}]}] * 4,
}
res = {}
for name, defect in cases.items():
    def fn(r, defect=defect, name=name):
        r["stage"] = "preprocess"
        args = SFTConfig(output_dir=str(out / "runb" / name), max_steps=2, per_device_train_batch_size=4,
                         report_to=[], save_strategy="no", use_cpu=True, logging_steps=1, seed=0)
        tr = SFTTrainer(model=GPT2LMHeadModel.from_pretrained(out / "model"), args=args,
                        train_dataset=Dataset.from_list(clean[:12] + defect), processing_class=tok)
        r["stage"] = "train"; st = tr.train(); r["stage"] = "completed"
        return {"train_loss": st.training_loss, "config": "defaults (assistant_only_loss=False)"}
    res[f"defaults_{name}"] = run_case(fn)
    print(name, res[f"defaults_{name}"]["exception"], res[f"defaults_{name}"]["stage"], res[f"defaults_{name}"]["warnings"])

# truncation arithmetic with keep_start at several max_length values
long_rows = []
filler = " ".join(x["messages"][1]["content"] for x in clean[200:230])
for c in clean[20:60]:
    long_rows.append({"messages": [{"role": "user", "content": "Notes: " + filler + "\n\n" + c["messages"][0]["content"]}, c["messages"][1]]})
stats = []
for L in [512, 1024, 2048]:
    lost_all = partial = 0; asst_total = asst_kept = 0; over = 0
    for r in long_rows:
        enc = tok.apply_chat_template(r["messages"], tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
        mask = enc["assistant_masks"]; n = len(mask)
        if n > L: over += 1
        kept = sum(mask[:L]); tot = sum(mask)
        asst_total += tot; asst_kept += kept
        if tot and kept == 0: lost_all += 1
        elif kept < tot: partial += 1
    stats.append({"max_length": L, "rows": len(long_rows), "rows_over_limit": over,
                  "rows_answer_fully_cut": lost_all, "rows_answer_partly_cut": partial,
                  "assistant_tokens_kept_share": round(asst_kept / asst_total, 3)})
    print(stats[-1])
# confirm TRL actually truncates to max_length with the default keep_start, silently
def fn_t(r):
    r["stage"] = "preprocess"
    args = SFTConfig(output_dir=str(out / "runb" / "trunc"), max_steps=2, per_device_train_batch_size=4, report_to=[],
                     save_strategy="no", use_cpu=True, max_length=1024, seed=0)
    tr = SFTTrainer(model=GPT2LMHeadModel.from_pretrained(out / "model"), args=args,
                    train_dataset=Dataset.from_list(long_rows), processing_class=tok)
    lens = [len(x) for x in tr.train_dataset["input_ids"]]
    r["stage"] = "train"; tr.train(); r["stage"] = "completed"
    return {"max_len_after": max(lens), "rows_at_cap": sum(1 for x in lens if x == 1024)}
res["trl_truncation_default"] = run_case(fn_t)
res["truncation_arithmetic"] = stats
print(res["trl_truncation_default"])
(out / "e4b_results.json").write_text(json.dumps(res, indent=2, default=str))
