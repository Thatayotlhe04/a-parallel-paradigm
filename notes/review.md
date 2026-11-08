# Adversarial review: "A Parallel Paradigm: The Coming Economics of Model Ownership"

Reviewed 2026-11-08. Draft read in full at doc rev 21, plus the rev 25 delta (title, new section 1 paragraph, section 9 Synthesis, Appendix B bundle name). In scope: sections 1–5, 7, 8, 9, Appendix A, Appendix B, References. Section 6.3 and the Summary were skipped. Sections 6.1, 6.2 and 6.4 are cited only where sections 5, 9 or the Parallelogram disclosure depend on them.

How it was checked:
- Every table and worked number was recomputed.
- `code/econ_model.py` was re-run in a scratch copy. The project's `results/` folder was not touched.
- The E1 and E4 JSON outputs and scripts were read.
- 20 cited pages were spot-checked with WebFetch: Epoch ×5, Menlo ×3, OpenRouter paper ×2, LoRA Land ×2, Thinking Machines, DataCamp, Li et al. 2023 ×2, Belcak, Anthropic Fable statement, a16z.

"Verified" below means I opened the page. "Notes" means the claim rests on `notes/*.md`, whose authors opened the page.

Arithmetic that checks out (no action needed):
- All 12 per-task costs in section 2.
- The 2M-task Sonnet bill and the 7-cent threshold (6.7 cents).
- $43k up front, $3k/month upkeep, $11k per re-base, $2,734 GPU floor, 56k tasks per GPU-hour, 7.3M tasks and 350 GPU-hours at break-even.
- Labor share of 0.671.
- Every profile-table row.
- The savings cap 12/λ_A.
- The half-life formula and table (against the current code).
- The 1/(k+1) expectation.
- Labonne scaling, the Fireworks $13 run, and 1.67× growth to 17% and 28%.
- The 19-day count.
- Monte Carlo crossings, interpolated on a log scale: $22,402, $36,873, $16,863, $10,383 and $5,136; ratio 4.36.
- Section 8.3: labor ÷10 gives $14,477; adding pooled serving gives $2,559.

---

## Critical

### 1. Section 4.6 gets the direction of its own biases backwards
- **Quote:** "The model favors renting in three ways. It ignores the 50% batch discount…"
- **Problem:** Ignoring batch and enterprise discounts overstates what renting costs. Assuming the owned 8B matches frontier quality flatters owning. All three biases therefore favor *owning* and push the threshold too low. The second list (renter migration costs, non-price rights, quality gains) is the group that favors *renting*. The draft's own next sentence, "Where those apply, the threshold falls", only makes sense for that second group.
- **Evidence:** In `simulate()`, rent = A0·e^{(g−λ_A)t}, priced at list. A 50% discount halves rent at a fixed volume, and the draft itself says that "would double the threshold".
- **Fix:** Swap the two labels. Then say outright that the net direction of the bias is unknown, or quantify it with a batch-discount run.

### 2. E1's "no false positives" is true only because the clean base was built from rows no tool flagged
- **Quote (6.1):** "No detector flagged a single untouched row at these settings, so every flag in these files was a true positive."
- **Problem:** The clean base was made by deleting every row that *any* tool flagged. `e1_detection.py` takes the union of all flags, and `base_clean_flags` is all zeros by construction. So zero collateral flags is a sanity check, not a finding.
- **Further problem:** On natural data, the one place precision could be measured, Parallelogram's PII rule flagged 57 rows in full Alpaca-cleaned. The draft itself calls them "nearly all invented examples", which makes them practical false positives.
- **Evidence:** `e1_results.json`: `base_raw_flags` (13 Parallelogram, 24 13-gram), `base_clean_rows: 9963`, `natural_full_alpaca.by_rule.pii: 57`.
- **Fix:** Delete the sentence, or reword it as "by construction, no untouched row was flagged". Report precision on the raw 10k sample and on the full 51,760 rows, giving the PII rule's 57 flags as warnings and noting that most were not real PII.

### 3. Section 9 and 6.1 overstate what E1 shows about Parallelogram
- **Quotes:**
  - Section 9: "Deterministic validators catch structural defects completely, at about 8,000 rows a second."
  - Section 6.1: "17 of the 18 structural defect types it was built for."
- **Problems:**
  1. **One tool, not "deterministic validators".** E1 tested one tool. The OpenAI cookbook checks are also deterministic, and they missed 16 of 19 SFT defect types.
  2. **Scope chosen after the fact.** "Built for" is defined as the 18 types it caught. The script's own comment, "structural, in Parallelogram's design scope", lists `label_swap` and `truncated_response`, which it caught 0% of.
  3. **Textbook injections.** The injected secrets, PII and mojibake follow textbook regex patterns (`AKIA`+16, `ghp_`+36, `+1 412 …`, `â€™`). 100% recall on them is close to guaranteed by construction.
  4. **Over-long answers are not structural-complete.** Parallelogram caught them at 98%, as warnings only.
- **Evidence:** `e1_detection.py` lines in `make_sft_defects`; `e1_results.json`.
- **Fixes:**
  - Section 9: "Parallelogram, the author's tool, flagged all injected rows for 17 of 18 structural defect types we constructed (98% for the 18th, as warnings)."
  - Section 6.1: drop "it was built for", or cite the README's rule list from before the experiment.
  - Section 6.4: add that injected secrets and PII use canonical formats, so Parallelogram's recall is optimistic too, not only MinHash's and the n-gram check's.
  - Disclosure: add section 9 to the sections where Parallelogram appears. Per the project brief, it must not dominate.

---

## Major

### 4. The Synthesis link between data validation and economics is contradicted by the essay's own model
- **Quotes:**
  - Section 9: "that pipeline's reliability sets the cost of each re-adaptation. That cost, in turn, sets how much of the falling price of intelligence an owner can capture."
  - Section 4.4: "the cost of re-adaptation, not GPU hours, governs how much of the decline an owner can capture."
- **Problem:** In the model, the captured share φ only scales serving GPU-hours, and serving sits on the one-GPU floor at break-even. Section 4.5 says capture "has no effect at all at this scale". Re-adaptation cost moves the threshold only from $33.7k to $39.3k, eighth of eleven inputs. Nothing in the essay measures validation's effect on re-adaptation cost.
- **Evidence:** `tornado.csv` (phi row is flat; rho row runs $33,748–$39,341); `simulate()`, where `step_factor` multiplies only `gpu_hours`.
- **Fix:**
  - Label 4.4's sentence [Hypothesis] and state that the model does not support it at break-even scale.
  - Rewrite the section 9 subsection without the causal chain. One option: "a reliable pipeline lowers the labor of each re-base, which the model shows matters modestly".
  - Shorten the subsection or merge it into "What this means for developers". As written it is the essay's only piece of product advocacy.

### 5. λ_O is defined two ways, and the shipped result file is stale
- **Quote:** Section 4.4 says "owned cost falls only with chip price-performance (1.49-fold a year)". Appendix A says "λ_O … φ × λ_A at each re-base, φ from 0.3 to 1".
- **Problem:** The half-life table assumes no re-basing (λ_O = ln 1.49). The simulation assumes re-basing captures 30–100% of the API decline. A reader cannot tell which owner the essay models.
- **Stale file:** `results/econ/econ_results.json` gives half-lives of 0.89/1.87/2.76 (4.7×) and 0.49/1.02/1.51 (13×). The draft and the current code give 0.96/2.00/2.96 and 0.51/1.06/1.57. The code was modified at 19:54, after the results were written at 19:50. Re-running reproduces the draft's numbers, and every other output is unchanged.
- **Fix:** Use two symbols, for example λ_chip = ln 1.49 for the 4.4 table and λ_O = φλ_A in the simulation. Say the 4.4 table is a no-rebase bound. Re-run `econ_model.py` before bundling.

### 6. "Break-even" means different things in different places, and the static comparison changes two variables
- **Quotes:**
  - Section 4.5: "about $5,100 … about $22,000 … That is a 4.4-fold shift from one omitted variable. At US labor rates it is about $37,000."
  - Section 8.3: "from about $36,500."
  - Section 7.4: "roughly $20,000."
  - Section 9: "about $22,000 … and about $37,000 at US labor rates."
- **Problem A, mixed definitions.** Monte Carlo crossings at P = 0.5 ($22.4k, $36.9k, $5.1k, $10.4k) are mixed with deterministic median-parameter bisection ($36.5k, $17.1k, $14.5k, $2.5k). Appendix A claims all break-even points "come from bisection at median parameters", which is wrong for the headline figures. The two methods disagree sharply:
  - Static: $5,136 (Monte Carlo) vs $8,322 (bisection).
  - Pooled: $10,383 (Monte Carlo, uncertain labor) vs $25,407 (bisection, US labor).
- **Problem B, hidden labor rate.** The headline $22k uses log-uniform labor from $20 to $121, whose median is $49/hr. Bisection at $49/hr gives $22,002. The text never says this. Section 8.2C ($22k→$10k) and section 8.3 ($36.5k→$2.5k) sit side by side on different labor bases.
- **Problem C, two variables, not one.** The static counterfactual sets both λ_A = 0 and g = 0. With static prices but growth kept, bisection gives $4,926, so price decline alone is a 7.4-fold shift, not 4.4.
- **Problem D, decision rule.** Rule 3's $20k threshold sits beside section 7.1's profiles at US rates, where break-even is $36.5k.
- **Fix:**
  - Define one primary measure, say the Monte Carlo even-odds crossing, and use it everywhere. Report bisection numbers only in the tornado, labeled as such.
  - State the $49/hr median labor rate wherever $22k appears.
  - Change "one omitted variable" to "omitting price decline and workload growth". Better, add a static-prices-with-growth scenario.
  - Tie rule 3 to a labor rate: "$20k at about $50/hr; about $37k at US rates".
  - Fix the Appendix A sentence.

### 7. The self-hosting comparison in section 2 compares unlike quantities
- **Quote:** "that is about $0.056 per million output tokens at _full_ utilization. DeepInfra sells the same model's output at $0.04. An organization cannot undercut a host…"
- **Problem:** The TensorRT-LLM run is 1,000 input / 1,000 output, so the $0.056 already pays for 1M input tokens processed alongside. Like for like, DeepInfra charges $0.02 + $0.04 = $0.06 for the same traffic. At 100% utilization, self-hosting is about 7% *cheaper*. It loses only at realistic load: at the model's own 47.5% efficiency and 37.5% utilization it costs about $0.31/M, roughly five times DeepInfra.
- **Evidence:** `notes/pricing.md` §5 already says "input tokens are processed on top (1:1)". Arithmetic: 3 / (14,992 × 3600 / 1e6) = 0.0556.
- **Fix:** "At full benchmark utilization, self-hosting roughly matches DeepInfra's $0.06 for the same million input and million output tokens. At realistic throughput and utilization it costs about five times more." Change "cannot undercut" to "is unlikely to undercut".

### 8. The "rephrased" evaluation rows in E1 are 6% word edits, which contradicts section 5.6
- **Quotes:**
  - Section 5.6: "10-gram decontamination caught none of the rephrased items."
  - Section 5.8: "MinHash, n-gram overlap … Catches … rephrased overlap."
  - Section 6.1: "The 13-gram check caught 96% of rephrased evaluation rows."
- **Problem:** `eval_paraphrase` uses the same `perturb_words(frac=0.06)` as the near-duplicate defect: word drops, inserted filler words, case flips. It is not paraphrase in Yang et al.'s sense, which uses LLM rewrites. That is why n-grams catch it. Section 9 then says rephrased contamination is "beyond structural checks", while 5.8 says n-grams catch it.
- **Fix:** Rename the defect "lightly edited evaluation row (about 6% of words)". In 5.8, change "rephrased overlap" to "lightly edited overlap; true paraphrase needs embedding or model-based checks (Yang et al.)".

### 9. The open-weight token trend is a rebound, and open tokens are not ownership
- **Quotes:**
  - Section 9: "Yes … open models' share of enterprise tokens rose from 11% to 19%."
  - Section 2: "The second carries a fast-growing share of the work."
  - Section 8.1: "If both trends continue…"
- **Problem A, rebound.** Menlo's 2025 report (verified) says enterprise open-source share "fell from 19% last year to 11%". The series is therefore 19% → 11% → 19%, a rebound with no trend.
- **Problem B, category error.** Most open-model tokens are bought per token from hosts. On the essay's own ladder (4.1) that is level 0, renting. Using token share as evidence of ownership contradicts the ladder.
- **Fix:**
  - Give the three-point series and drop "fast-growing".
  - In section 9, rest "yes" on post-training (6%→10%) and adapter or Hub activity, and say token share measures open-model *use*, mostly rented.
  - In 8.1, note that the earlier year fell.

### 10. Survey limitations and the source's interest are not acknowledged
- **Quote (section 2):** "Menlo's respondents say capability, not price, decides which model they buy."
- **Problems:**
  - The capability point is Menlo's own conclusion ("buyers continue to prioritize capability over cost"), not a stated respondent answer.
  - Menlo marks Anthropic as a "Menlo Ventures portfolio company" on the 2026 page (verified). The essay cites Menlo for Anthropic-led market shares without saying so.
  - Both samples are US-only enterprise decision-makers (495, then "nearly 400"), drawn by Morning Consult with no probability sampling.
  - "Builders" is undefined, and the 6% and 10% come from different samples.
  - The a16z sample is drawn from a vendor database (notes).
  - Despite all this, the essay generalizes to "organizations", including Botswana.
- **Fix:**
  - Add one short limitations sentence in section 2: US-only, non-probability, small, Menlo has an Anthropic stake, "builders" undefined.
  - Change "respondents say" to "Menlo concludes".
  - Optionally add the 2026 three-lab share (55 + 31 + 9 = 95%), which is newer than the 88% figure.

### 11. The Fable/Mythos episode is mis-cited and overstated
- **Quote (7.2):** "to comply with US Department of Commerce export controls. The controls were lifted on 30 June and access was restored on 1 July ([Anthropic])… the most capable rented models in the market were unavailable to the customers who had built on them."
- **Problems:**
  - **Agency.** The cited statement (verified) says only "The US government, citing national security authorities". Commerce is named only by secondary sources.
  - **Wrong page.** The 30 June lift and 1 July restoration are on a different page, "Redeploying Fable 5".
  - **Partial restoration.** Fable 5 returned on 1 July on first-party surfaces, with cloud partners later. Mythos 5 returned only for approved US organizations.
  - **Brand-new model.** Fable 5 launched publicly on 9 June (eWeek, in notes), three days before the suspension, so few customers "had built on" it.
  - **Changed behavior.** It came back with a new classifier, more false positives and a fallback to Opus 4.8. That is a stronger point for the behavioral-stability argument and is omitted.
  - **"Most capable" is unsourced.**
- **Fix:** "the US government (reported as the Commerce Department)". Cite both Anthropic pages. Say "Fable 5 returned on 1 July with a new classifier; Mythos 5 only for approved US organizations". Add the three-day launch context. Drop "most capable", or source it. Apply the same corrections to 8.2D and the 8.4 signpost row.

### 12. OpenRouter's one-third share is misread
- **Quote (section 2):** "about one third of the 100 trillion tokens routed through OpenRouter in the year to late 2025 went to open-weight models."
- **Problem:** The paper (verified, HTML) says open models were "reaching approximately one-third of usage by late 2025". That is a point-in-time weekly share. Its whole-period averages are 13.0% for Chinese open models and 13.7% for other open models, about 27% combined.
- **Fix:** "open-weight models reached about a third of OpenRouter's weekly tokens by late 2025, averaging roughly a quarter over the year."

### 13. The price-decline input contradicts the essay's own argument, and the setup is asymmetric
- **Quote (4.4):** "A narrow task usually needs mature capability, so the slower rate is the relevant one."
- **Problem A, wrong range.** The simulation still draws 4.7× to 13× (median 7.8×). At 4.7×, US-labor break-even is $27.3k rather than $36.5k (tornado).
- **Problem B, asymmetric switching.** The renter migrates to each cheaper equal-quality model for free. The owner pays 15–50% of initial labor per re-base. A hostile economist will attack this asymmetry first: it is what makes price decline the dominant omitted variable.
- **Problem C, one task only.** The model ignores economies of scope. An owner running several tasks spreads labor and the GPU floor across them.
- **Fix:**
  - Either make 4.7× the central case, or justify the 4.7–13× range.
  - Add a sensitivity run with a renter switching cost, for example 10–40 engineer-hours per migration.
  - Add "single task, no economies of scope" to 4.6, on the favors-renting side.

### 14. The reproducibility claims are stronger than the bundle supports
- **Quotes:**
  - Section 1: "Every number links to the page it came from. The code, data hashes and random seeds for every original result are listed in Appendix B."
  - Appendix B: "Every original number in this essay comes from a script…"
- **Not reproducible from the bundle:**
  - The 350 identical UltraFeedback pairs in 5.3: no script, dataset version, hash or link.
  - Section 8.3's $14,500 and $2,500: not in `econ_model.py`. I reproduced them as $14,477 and $2,559 with my own scenario code.
  - The interpolated crossings: `econ_results.json` stores the first grid point at or above 0.5 ($25,749, $43,687, $5,272…). Interpolation happens only inside the Figure 1 widget.
  - The 7-cent example and "ten GPUs": computed nowhere.
- **Wrong file name:** Appendix B lists `results/e2/e2_runs.jsonl`, but the actual file is `e2_runs_pilot_600steps.jsonl`.
- **Unlinked numbers:** the A100 $1.59 index, "Fireworks' documentation is inconsistent", the 5.8 judge cost ("dollars per 10,000 rows"), and the H100 $3.00 in section 2 (linked only in section 3).
- **Fix:**
  - Add a `scenarios()` block to `econ_model.py` for the 8.3 cases, the interpolated crossings and the 4.2 example.
  - Add an UltraFeedback script with the dataset revision, or cut the claim.
  - Correct the E2 file name.
  - Soften section 1 to "Sourced numbers link to their page".

### 15. The E1 comparison table is tilted, and the E4 verdict goes beyond the tests
- **Quotes:**
  - 6.1: "The other tools missed most of them."
  - 6.2: "Two of Parallelogram's framework claims hold for TRL 1.17.1 at default settings: … duplicates are trained on silently."
- **E1 problems:**
  - The cookbook re-implementation flags length only above 16,385 tokens, while Parallelogram ran at 2,048. The cookbook's 0% on over-long rows reflects that setting, not a missed detection.
  - Secrets, PII and mojibake are outside the cookbook's purpose.
  - The cookbook's "Script crashes" on invalid JSON is recorded as recall 1.0 in the JSON, because a crash is a loud detection.
- **E4 problems:**
  - Duplicates were tested only with `assistant_only_loss=True` and max_length 256. At defaults the table itself says "Not tested".
  - The DPO identical-pair run sits in the "defaults" column but used lr 1e-3 and max_length 256.
  - The assistant-only truncation case was attempted and crashed (`ValueError: Column 'assistant_masks' doesn't exist`), yet it is reported as "Not tested".
- **Fixes:**
  - Footnote each 0% that falls outside a tool's scope ("out of scope / threshold 16,385").
  - Replace "missed most of them" with "were not designed for most of them".
  - Change the verdict to "…truncated without a warning at defaults; duplicates trained silently (tested with assistant-only loss)".
  - Note the non-default DPO settings, and report the crash.

### 16. The GPU price index is presented inconsistently
- **Quotes:**
  - Section 3: "about $3.00 per hour … against $2.13 to $2.26 in August and September 2025, a rise of roughly 35 to 40%."
  - Section 8.4: "up 35–40% in a year."
  - Section 4.3: "one always-on GPU at $3.75 an hour."
- **Problems:**
  - The notes flag that Silicon Data's own FAQ shows $2.53 and that its 7-day change contradicts its table ("Treat the $3.00 headline with caution").
  - The baseline is the September 2025 trough, 14–15 months earlier. Against May 2025 ($2.37) the rise is about 27%.
  - The model's median GPU price is $3.75, the midpoint of $2.00 to Together's $5.49 managed rate, while sections 2 and 3 use $3.00.
  - "TrendForce attributes much of the pressure to memory": TrendForce's release is about DRAM contract prices, not GPU rental. The rental-driver attribution is SemiAnalysis's.
- **Fix:** Give the headline with its caveat ("$3.00 headline; the page's FAQ shows $2.53"). Say "+27% since May 2025; about +39% from the September 2025 low". Justify $3.75 by citing the RunPod, Lambda and Nebius H100 prices in the notes ($3.99, $4.59, $4.50). Attribute the memory-driver claim to SemiAnalysis.

### 17. The Synthesis drops caveats and labels that the body carries
- **Quotes from section 9:**
  - "Lifetime savings are capped at 4.7 to 7.8 months of today's API bill."
  - "Static-price analyses put that point 4.4 times lower."
  - "…faster than that of any technology Epoch has measured."
  - "outside the largest economies, where in-country API processing does not exist."
- **Problems:**
  - The cap holds only for flat volume, and it assumes the renter captures the full decline. At the model's median growth the cap is 10.0 months (4.7×) or 5.4 months (13×). At 100% growth it is 14.0 or 6.4.
  - "Static-price analyses" means the author's own counterfactual, not published analyses, and it also zeroes growth (item 6).
  - Epoch adds its own caveat: "comparing apples and oranges".
  - The residency evidence covers southern Africa only.
  - Section 9 carries no [Observed]/[Calculated]/[Hypothesis] labels at all, despite section 1's promise.
- **Fix:** Add "with flat volume" and "in my model's static counterfactual". Add Epoch's caveat in a clause. Narrow the residency point to "in southern Africa and similar markets". Label the bullets.

---

## Minor

### 18. Savings-cap formula has a units mismatch and unstated assumptions (4.4)
- **Problem:** A₀ is a monthly bill while t and λ are per year. The integral then gives A₀ × years, so the table's 12/λ_A needs a factor of 12 in the displayed formula. The bound also assumes c_O ≥ 0, no discounting, and a renter bill that follows A₀e^{(g−λ_A)t}, meaning the renter migrates.
- **Fix:** Write ≤ 12A₀/(λ_A − g) with t in years, and list the three assumptions.

### 19. The break-even inequality is static and leaves out the GPU floor (4.2)
- **Problem:** The inequality ignores discounting and the price decline that section 4.4 calls decisive. M is defined in Appendix A as maintenance hours only, so the always-on GPU floor appears in neither M nor c_O. The worked example then mixes a discounted dynamic own cost ($259,915 / 36 = $7,220) with a static Haiku bill. Undiscounted, the threshold is 7.7 cents rather than 6.7. c_err also excludes the cost of *finding* errors, which usually means reviewing every output.
- **Fix:** Define M to include idle serving cost. Say the example uses the dynamic model's average monthly own cost. Note detection cost.

### 20. The late-failure expectation is right but incomplete (5.4)
- **Problem:** E[min of k uniforms] = 1/(k+1) assumes i.i.d. uniform positions, no checkpointing, and a restart from scratch, none of which is stated. More importantly, it covers only the *first* crash. If each crash reveals one bad record and the engineer fixes it and reruns, total expected waste is Σ i/(k+1) = k/2 epochs: five epochs for ten bad records. That reverses "the rarer the defect, the more compute it burns" for total waste, and it is a stronger argument for validating up front.
- **Fix:** State the assumptions, and add the fix-and-rerun result in one sentence.

### 21. The DPO null-gradient argument needs caveats (5.3)
- **Problems:**
  - The derivation holds for the standard sigmoid loss, and for IPO and label smoothing, because the bracket is identically zero. It fails if an NLL or RPO term is added (TRL's `rpo_alpha`).
  - (B−k)/B assumes mean reduction over the batch.
  - "Loss ln 2" does not distinguish identical pairs: at initialization real pairs also log 0.6931 (`e4_results.json`, `dpo_normal_pairs.loss`). Only the gradient norm does.
  - "They waste 0.7% of the compute and do nothing else" sits in tension with 6.2's Adam observation.
- **Fix:** Add "for the standard DPO loss without an added SFT term". Compare on gradient norm only. Soften to "and almost nothing else".

### 22. The contamination-bias wording needs tightening (5.6)
- **Problem:** The bias c(a_s − a_u) is at most c. "Large only when … the task is hard for it" needs "and c is non-trivial". The decision rule ignores sampling error in â. GSM1k gaps are observational, and the notes say memorization explains only part of them, so citing them as contamination effects is causal overreach.
- **Fix:** Reword, and add "consistent with" for GSM1k.

### 23. Utilization matters at about one GPU, not ten (4.3)
- **Quote:** "Burstiness and utilization start to matter only above roughly ten GPUs."
- **Problem:** Nothing derives this. The floor binds up to 15.3M tasks/month at 37.5% utilization and only up to 6.1M at 15%. At 15% utilization the 7.3M-task break-even workload needs 874 GPU-hours, more than one GPU's 730.
- **Fix:** "Utilization starts to matter once the workload exceeds about one GPU (6–15M tasks a month in this model)."

### 24. Several literature citations are imprecise (sections 3 and 5)
- **Li et al. 2025:** 21.8 points is the cost of *inserting* steps. Shuffling cost 10.2. Corrupting all digits collapsed the score to 2.7, which undercuts "content barely matters".
- **Zhu 2022:** compares 40% noise on AG-News with 50% noise on Hausa.
- **Luo 2024 (RobustFT):** has no clean-SFT baseline, and its noise type is unstated. That conflicts with the section title "unless they are random".
- **Belcak:** the abstract says "many invocations", not "most agent sub-tasks" (verified).
- **LoRA Land:** "8B or fewer" includes Gemma-7b at 8.54B, and the GPT-4 baseline is gpt-4-0613 with zero- or one-shot prompts (verified). That is weak evidence for matching a 2026 frontier model.
- **Thinking Machines:** "slightly more than ⅔ of the FLOPs" (verified).
- **AlpaGasus:** "better" is judged by GPT-4, with 1–2 point benchmark gains.
- **Pretraining results:** Hernandez, Kandpal and Muennighoff are pretraining-scale but are applied to fine-tuning without a caveat. For fine-tuning PII, cite Lukas et al. 2023 instead (in the notes).
- **Fix:** Correct each wording, and add a one-line transfer caveat in 5.5.

### 25. Some citations point to the wrong page or are incomplete
- **Epoch lag:** the "three months through October 2025" figure is not on the cited page, which gives only four months since January 2026. It comes from Emberson, 30 Oct 2025, "Open-weight models lag state-of-the-art by around 3 months on average" (verified).
- **Composer 2:** the primary source for the three-quarters claim is Lee Robinson's X post. Cursor's blog did not disclose the base model (verified via DataCamp).
- **Mozilla report:** add that Mozilla also reports a 3.3-point *average* gap (notes, Business Standard), so the Terminal-Bench gap is not cherry-picked.
- **Cho 2024:** the reference lacks a title ("Unveiling Imitation Learning: Exploring the Impact of Data Falsity to Large Language Model").
- **Epoch consumer-GPU post:** credit its authors (Somala and Emberson, Sep 2025).
- **OpenAI fine-tuning wind-down:** omits the 2 July 2026 stage.
- **No over-length quotations were found** (copyright OK).

### 26. Labor-rate inputs need sourcing and caveats (4.3, 4.5)
- **Problem:** $121/hr is unloaded total compensation (Levels.fyi ÷ 2,080, no benefits or overhead, skewed toward big tech, per the notes). The $20 floor and the $25 scenario are unsourced. The notes have South Africa and Botswana salaries but no USD conversion.
- **Fix:** Say "unloaded". Source the low end, for example South Africa's Levels.fyi median of R658k/yr, converted. State the log-uniform median of $49/hr.

### 27. The residency survey leaves out two major clouds (7.3)
- **Quote:** "The major APIs do not offer processing in southern Africa."
- **Problem:** Only OpenAI, Anthropic and Bedrock are named. The notes also checked Azure Foundry (no South Africa regions in its deployment tables; a secondary report of a gpt-4o PTU exception retiring 2026-12-09) and Vertex (Gemini not offered in africa-south1).
- **Fix:** Add both, with the PTU caveat. Label "the strongest single driver of model ownership outside the largest economies" as [Speculation].

### 28. Decision rules have overclaims and a formatting bug (7.4)
- **Rule 2:** "nothing an organization builds will be cheaper". Hosted open 8B costs $0.04 per thousand tasks, and at high volume self-serving costs about $0.05.
- **Rule 1:** Ovadia tested *unsupervised* fine-tuning only.
- **Rule 3:** renders "$2****0****,000" (broken bold markup).
- **Fix:** Qualify rules 1 and 2, and fix the markup.

### 29. Section 3 cost estimates are inconsistent
- **Problem:** The Labonne figure is one epoch, while the Fireworks example assumes three. The A100 $1.59 is unlinked (the Silicon Data A100 page; its FAQ says $1.29). Appendix B's E3 estimate of "about 10 GPU-hours on a T4" depends on model size: the essay's own Labonne source gives about 47 h per 100k samples on a T4 for 8B, so 9 runs × 5k samples is about 21 h.
- **Fix:** Align epochs, add the link, and state the model size for E3.

### 30. The personas exaggerate the individual developer's costs (7.1)
- **Problem:** The individual-developer persona pays an always-on H100 floor that no individual would choose. A consumer GPU or per-token hosting is the realistic option. "The most favorable case for owning" is also arguable, since a quality gain would be more favorable.
- **Fix:** Note this in the row, or drop the row.

### 31. E1 bookkeeping has small inconsistencies (section 6, referenced by section 9)
- **Seed dependence:** e1c (seed 7) gives 3 of 10 padded copies surviving, while the e1 run gives 5 of 10.
- **Unreconciled drops:** 4 exact plus 3 whitespace kept copies imply 7 dropped originals, but e1c reports 6.
- **MinHash timing:** "began flagging 7 untouched rows" at 0.5, but flags already start at 0.7 (2 rows).
- **13-gram overlap:** the 24 overlaps in a random split may be boilerplate rather than "leakage".
- **HH-RLHF summary:** omits the 6 PII and 1 encoding flags.
- **Fix:** Report both seeds, reconcile the drop count, and change the MinHash wording to "flags begin at 0.7".

### 32. Some claims carry no label (sections 1, 3, 4, 5, 7)
- **Problem:** Several important claims lack labels or support:
  - "most analyses leave out" (section 4 intro)
  - "the case most often proposed" (4.5)
  - "The binding input has moved from GPUs to people" (3)
  - "Most of the harm documented in the literature lives in the last two" (5.1)
  - "ownership becomes likely only once … tens of thousands" (4.6)
  - "the frontier still sits behind closed APIs" (2)
- **Fix:** Label them, or cite support.

### 33. Two wording slips (4.3 and 6.2)
- **4.3:** "frontier-model labeling of 6,000 examples, all at US rates of $121 an hour". The labeling is API cost, about $90, not hourly labor.
- **6.2:** E4's "Cut to 1,024 tokens from the end" reads as keeping the end. Say "keeps the first 1,024 tokens". The unknown-role error message blames the chat template, which is worth noting as misleading.

### 34. Some pricing caveats from the notes are missing (section 2)
- **Problem:** The notes record that Claude 4.7+ tokenizes about 30% more tokens for the same text, that US-only and regional processing carry 1.1× uplifts at Anthropic and OpenAI, and that the 50% OpenAI batch reading is the notes-writer's inference.
- **Fix:** Add one clause each where relevant. This is especially needed for residency-driven users.
