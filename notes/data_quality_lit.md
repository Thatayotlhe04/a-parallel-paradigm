# Data quality literature: which training-data defects harm LM training and fine-tuning

Compiled 2026-11-08 for the "data bottleneck" section.

Verification protocol: every number below was read from the primary source (arXiv abstract, arXiv PDF/HTML, ACL Anthology, PMLR/official docs, or library source code) on 2026-11-08 using WebFetch, which reads pages through an extraction model. "Not verified" means I could not open the source or the number was not in the parts I could read. Where the extraction gave inconsistent readings of a table, the entry says so. Venues marked "(listing)" were confirmed only from an official proceedings URL returned by search, and that page was not opened. Quotes are under 15 words; everything else is paraphrase.

---

## Key takeaways (ranked by strength of evidence of harm)

1. **Benchmark contamination: the largest and best-quantified effect, but it inflates scores rather than hurting the model.** In controlled runs, training on rephrased test items lifted Llama-2-13B MMLU from 54.8 to 85.9 ([Yang et al. 2023](https://arxiv.org/abs/2311.04850)). 144 exposures inflated accuracy by 43 to 54 points at 1x Chinchilla ([Bordt et al. 2025](https://arxiv.org/abs/2410.03249)). Leaked training sets plus test prompts lifted LLaMA-2-7B MMLU from 42.95 to 56.04 ([Zhou et al. 2023](https://arxiv.org/abs/2311.01964)). In the wild, GSM8k scores beat fresh-problem scores by up to 8% ([Zhang et al. 2024](https://arxiv.org/abs/2405.00332)). SFT corpora carry overlap themselves (CodeAlpaca vs HumanEval 12.8%; MATHInstruct vs MATH test 15.4%), and 10-gram decontamination scored F1 = 0 on rephrased items ([Yang et al.](https://arxiv.org/abs/2311.04850)).
2. **Factually wrong SFT responses: strong, dose-dependent harm, worst on knowledge tasks.** At 100% falsity, LLaMA-2-13B MMLU fell from 57.49 to 19.74 ([Cho 2024](https://arxiv.org/abs/2404.09717)). With 30%, 50% and 70% noisy responses, Llama-3.1-8B MMLU fell from 65.3 (no fine-tuning) to 59.5, 47.5 and 37.3 ([Luo et al. 2024, RobustFT](https://arxiv.org/abs/2412.14922)). Fine-tuning on facts the model does not know raises hallucination roughly linearly: 100% Unknown data scored 37.5 at early stop vs 25.8 at convergence ([Gekhman et al. 2024](https://arxiv.org/abs/2405.05904)).
3. **Skewed duplication: strong evidence (pretraining-scale).** Repeating 0.1% of the data 100 times made an 800M model perform like a 400M model ([Hernandez et al. 2022](https://arxiv.org/abs/2205.10487)). Deduplication cut memorized output about 10x ([Lee et al. 2022](https://aclanthology.org/2022.acl-long.577/)). A string seen 10 times is regenerated about 1,000x more often than a singleton ([Kandpal et al. 2022](https://arxiv.org/abs/2202.06539)). Uniform whole-dataset repetition up to 4 epochs is close to harmless ([Muennighoff et al. 2023](https://arxiv.org/abs/2305.16264)). The harm comes from uneven repetition, not from epochs.
4. **Secrets and PII in fine-tuning data leak, and leakage scales with duplicates and model size.** Fine-tuned GPT-2 leaked PII with up to 22.96% recall (ECHR, GPT-2 Large). DP at epsilon = 8 cut this to about 3%, and leakage grew roughly linearly with PII frequency ([Lukas et al. 2023](https://arxiv.org/abs/2302.00539)). Head-only fine-tuning had 81.6% membership-inference recall vs 19.2% for full fine-tuning and 15.2 to 16.8% for adapters ([Mireshghallah et al. 2022](https://aclanthology.org/2022.emnlp-main.119)). A 10x larger model memorizes 19 percentage points more ([Carlini et al. 2023](https://arxiv.org/abs/2202.07646)).
5. **Preference-label noise: moderate evidence, nonlinear.** DPO tolerates noise below about 30%, where 10 extra points of noise cost under 10 points of win rate. Above 40%, 10 extra points cost about 20 to 30 points of win rate, and at 50% alignment becomes harmful ([Gao et al. 2024](https://arxiv.org/abs/2404.09824)). At 40% flips, DPO's IMDb reward fell from 0.99 to 0.72-0.93 with high variance, while rDPO held 0.99 ([Chowdhury et al. 2024](https://arxiv.org/abs/2403.00409)). A widely used set (UltraFeedback) shipped "a few hundred" mislabeled completions ([dataset card](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)).
6. **Near-identical chosen/rejected pairs: moderate, mechanistic evidence.** Pairs with similar embeddings drive "likelihood displacement". In one safety run, this dropped Llama-3-8B-Instruct's refusal rate from 74.4% to 33.4%, and filtering high-similarity pairs mitigated it ([Razin et al. 2025](https://arxiv.org/abs/2410.08847)). DPO lowers the likelihood of the preferred response most when edit distance is low ([Pal et al. 2024](https://arxiv.org/abs/2402.13228)). Adding ties to standard DPO degrades performance; I did not verify the numbers ([Chen et al.](https://arxiv.org/abs/2409.17431)). Exactly identical pairs give exactly zero DPO gradient (my derivation, see 4.7).
7. **Truncation: moderate evidence for pretraining, thin for SFT.** Packing that avoids truncation improved reading comprehension by 4.7%, context following by 16.8% and code synthesis by 9.2%, and cut closed-domain hallucination by up to 58.3% ([Ding et al. 2024](https://arxiv.org/abs/2404.10830)). TRL 1.17.1 silently truncates SFT and DPO sequences at max_length = 1024 by default, keeping the start. It silently drops examples left with no trainable tokens and logs no count of truncated examples ([TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer), [TRL DPO docs](https://huggingface.co/docs/trl/dpo_trainer), source code inspected). I found no controlled study of SFT response truncation.
8. **Low-quality filler in SFT sets: removing it is neutral or positive, so volume is not the binding constraint.** LIMA: 1,000 examples, and going from 2K to 32K examples brought no gain ([Zhou et al. 2023](https://arxiv.org/abs/2305.11206)). AlpaGasus kept 9,229 of 52,002 examples and trained 5.7x faster ([Chen et al. 2024](https://arxiv.org/abs/2307.08701)). Superfiltering kept 5% and raised AlpacaEval from 27.75 to 33.04 on LLaMA2-7B ([Li et al. 2024](https://aclanthology.org/2024.acl-long.769)). Filtering gains were small on the cleaner Alpaca-GPT4 source (71.32 to 72.13), so the benefit comes from removing bad data, not from shrinking.
9. **Largely harmless defects.** In long chain-of-thought SFT, wrong final answers cost only 3.2 points (66.3 to 63.1), and corrupting 20 to 70% of digits cost at most 4.3 points. Shuffling, deleting or inserting steps cost up to 21.8 points ([Li et al. 2025](https://arxiv.org/abs/2502.07374)). Perturbing 100% of instructions (typos, shuffles, deletions) matched or beat clean training on MMLU ([Alajrami et al. 2025](https://arxiv.org/abs/2510.03528)). Uniform injected label noise in BERT classification is mostly absorbed with early stopping (94.40 to 92.40 at 40% noise), but realistic feature-dependent noise is not (Hausa 83.55 to 46.97) ([Zhu et al. 2022](https://arxiv.org/abs/2204.09371)).
10. **Failure statistics barely capture data defects.** Llama 3 405B had 419 unexpected interruptions in 54 days, about 78% attributed to hardware and none to data ([Llama 3 paper](https://arxiv.org/abs/2407.21783)). In 4,960 failed Philly jobs, the Data category was 2.0% ([Zhang et al. ICSE 2020](https://hongyujohn.github.io/icse20-main-199.pdf)). Script errors were about 73% of failures in an LLM datacenter but about 3% of GPU time ([Hu et al. NSDI 2024](https://arxiv.org/abs/2403.07648)). No source quantifies data-caused fine-tuning failures. Data defects mostly degrade quality silently instead of crashing jobs (my interpretation).

---

## 1. Duplication

### 1.1 Lee et al. 2022: Deduplicating Training Data Makes Language Models Better
- **Citation:** Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini. "Deduplicating Training Data Makes Language Models Better." Proceedings of ACL 2022 (Volume 1: Long Papers), pp. 8424-8445. DOI 10.18653/v1/2022.acl-long.577.
- **URL:** https://aclanthology.org/2022.acl-long.577/ ; https://arxiv.org/abs/2107.06499
- **Findings (PDF):**
  - Near-duplicate share of training examples: C4 3.04%, RealNews 13.63%, LM1B 4.86%, Wiki-40B 0.39%.
  - Share of training tokens inside an exact duplicated 50-token substring: C4 7.18%, RealNews 19.4%, LM1B 0.76%, Wiki-40B 2.76%.
  - One 61-word sentence appears 61,036 times in C4 train and 61 times in validation.
  - Memorized share of tokens in 100k unprompted generations (1.5B "XL" model, 1 epoch): 1.926% trained on original C4, 0.189% on NearDup-deduplicated C4, 0.138% on ExactSubstr-deduplicated C4. The paper summarizes this as about 10x less memorized output.
  - Train-test overlap: over 4% of validation examples in standard sets (abstract). For C4, 4.60% of validation examples have a near-duplicate in train.
  - Perplexity on validation examples that are duplicated in train is roughly half that on unique ones (Transformer-XL on LM1B: 10.11 vs 23.58). This inflates evaluation.
  - Deduplicated datasets are up to 19% smaller. XL training time was about 131 h on original C4 vs about 126 h on NearDup C4.
  - The paper claims perplexity reductions of up to 10% in some cases.
- **Setting:** Decoder LMs at 110M ("Base") and 1.5B ("XL"); C4, RealNews, LM1B, Wiki-40B.
- **Relevance to fine-tuning:** Deduplication is cheap: building the C4 suffix array takes under 12 h, and deduplicating afterwards under 1 h. It cuts memorized regurgitation about 10x and removes train/validation overlap that would otherwise inflate held-out metrics. The same logic applies to SFT sets built from scraped or synthetic data.
- **Caveats:** Pretraining scale. Perplexity effects are mixed: deduplicated models do worse on validation examples that were duplicated in train. The abstract says fewer training steps were needed but gives no number, so I report none.

### 1.2 Hernandez et al. 2022: Scaling Laws and Interpretability of Learning from Repeated Data
- **Citation:** Danny Hernandez, Tom Brown, Tom Conerly, Nova DasSarma, Dawn Drain, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Tom Henighan, Tristan Hume, Scott Johnston, Ben Mann, Chris Olah, Catherine Olsson, Dario Amodei, Nicholas Joseph, Jared Kaplan, Sam McCandlish. arXiv:2205.10487, 2022 (preprint).
- **URL:** https://arxiv.org/abs/2205.10487
- **Findings (abstract):**
  - Repeating 0.1% of the data 100 times, with the remaining 90% of tokens unique, dropped an 800M-parameter model to the performance of a 400M model (2x).
  - Repetition can make test loss rise midway through training (double descent).
  - Copying and induction heads are disproportionately damaged.
- **Findings (PDF, as extracted):**
  - Models span about 1.5M to 805M parameters, each trained on a fixed 100B tokens.
  - Damage peaks at mid-range repetition, where the repeated subset is memorizable and consumes capacity. The peak band moves toward fewer repeats as model size grows.
  - Pretraining on heavily repeated data hurt later Python fine-tuning.
- **Setting:** Pretraining of small to mid-size transformers.
- **Relevance:** This is the clearest causal evidence that a small, heavily repeated slice of a corpus (boilerplate, templated rows, upsampled sources) costs effective model capacity. Fine-tuning sets with many copies of the same template or system prompt are the analogous case, though that analogy is not tested.
- **Caveats:** Loss and probe metrics, not downstream tasks. No dropout or weight-decay study. The repeated data was a random subset. As extracted, the body's effective-size figure (about 340M) differs from the abstract's 400M, and the fine-tuning magnitudes differed between sections. Treat fine-tuning magnitudes as not verified.

### 1.3 Kandpal, Wallace, Raffel 2022: Deduplicating Training Data Mitigates Privacy Risks in Language Models
- **Citation:** Nikhil Kandpal, Eric Wallace, Colin Raffel. ICML 2022, PMLR 162.
- **URL:** https://arxiv.org/abs/2202.06539
- **Findings (PDF):**
  - Regeneration rate grows superlinearly with duplicate count (log-log slope above 1 for every model). A sequence seen 10 times is generated about 1,000x more often than one seen once.
  - Membership inference is near chance (AUROC about 0.5) for singletons and reaches AUROC up to 0.90 for heavily duplicated sequences (Mistral 345M).
  - Deduplicated C4 vs original, 1M generated samples: unique 400-character training sequences emitted fell from 1,427,212 to 68,090 (0.14% to 0.007%, about 20x).
- **Setting:** 117M to 1.5B transformers; OpenWebText and C4.
- **Relevance:** Duplicated strings are the ones that get memorized and extracted. In a fine-tuning set, a credential or PII string repeated across many rows is far more dangerous than one that appears once.
- **Caveats:** Deduplication did not neutralize every attack. Reference-model AUROC stayed at 0.88 vs 0.87.

### 1.4 Carlini et al. 2023: Quantifying Memorization Across Neural Language Models
- **Citation:** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, Chiyuan Zhang. ICLR 2023 (header confirmed in PDF).
- **URL:** https://arxiv.org/abs/2202.07646
- **Findings (PDF):**
  - Memorization grows log-linearly with model size, duplicate count and prompt context length.
  - GPT-J 6B memorizes at least 1% of the Pile (a lower bound).
  - A 10x increase in model size adds 19 percentage points of memorization (log-linear fit, R^2 = 99.8%). Larger models memorize 2-5x more within a family.
  - Context length: 33% of sequences are extractable with 50 tokens of context vs 65% with 450 tokens (6B model).
  - For sequences repeated fewer than 35 times, a deduplicated-data model memorizes 1.2% vs 3.6% without deduplication.
  - T5 memorizes about an order of magnitude less than comparable decoder models (T5-XL: 3.5% of sequences repeated 100 times vs 53.6% for GPT-Neo 2.7B). OPT shows much smaller effects.
- **Setting:** GPT-Neo/GPT-J (125M to 6B) on the Pile; also T5 and OPT.
- **Relevance:** Fine-tuning larger base models on data with repeated sensitive strings raises extraction risk.
- **Caveats:** "Extractable" means the model completes a training prefix verbatim. Results depend strongly on model family.

### 1.5 Muennighoff et al. 2023: Scaling Data-Constrained Language Models (the "harmless repetition" counterpoint)
- **Citation:** Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, Colin Raffel. NeurIPS 2023 (listing: proceedings.neurips.cc).
- **URL:** https://arxiv.org/abs/2305.16264
- **Findings (abstract):**
  - Up to 4 epochs of repeated data gives a loss change close to zero compared with unique data.
  - Beyond that, the value of extra compute decays to zero.
  - 400 runs, models up to 9B parameters, up to 900B tokens.
- **Relevance:** Multi-epoch training on a whole dataset is not the problem. Uneven repetition of a subset is (1.2). This matters for fine-tuning, where 2-15 epochs is common (LIMA used 15).
- **Caveats:** Pretraining loss, not fine-tuning or downstream tasks.

---

## 2. Small curated SFT data

### 2.1 Zhou et al. 2023: LIMA: Less Is More for Alignment
- **Citation:** Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, Omer Levy. NeurIPS 2023 (listing: proceedings.neurips.cc).
- **URL:** https://arxiv.org/abs/2305.11206
- **Findings:**
  - Abstract: LLaMA-65B fine-tuned on 1,000 curated pairs. Its responses were equivalent to or preferred over GPT-4 in 43% of cases, Bard in 58% and DaVinci003 in 65%.
  - PDF: on 50 test prompts, 50% of responses were rated excellent, 38% pass and 12% fail. Training ran 15 epochs, with checkpoints chosen manually between epochs 5 and 10.
  - Ablations (7B, 2,000 examples): quality-filtered Stack Exchange scored about 0.5 points higher on a 1-6 helpfulness scale than unfiltered (the text states the 0.5-point gap; the bar values 3.83 vs 3.33 were inferred from the figure).
  - Scaling quality-filtered data 16-fold (2K to 32K) did not improve quality; it plateaued.
- **Relevance:** Quality and diversity of SFT data matter more than volume past a small threshold.
- **Caveats:** Small human evaluation. Checkpoint picking was manual. Rests on a strong pretrained base ("superficial alignment" hypothesis).

### 2.2 Chen et al. 2024: AlpaGasus: Training A Better Alpaca with Fewer Data
- **Citation:** Lichang Chen, Shiyang Li, Jun Yan, Hai Wang, Kalpa Gunaratna, Vikas Yadav, Zheng Tang, Vijay Srinivasan, Tianyi Zhou, Heng Huang, Hongxia Jin. ICLR 2024 (listing: iclr.cc virtual poster).
- **URL:** https://arxiv.org/abs/2307.08701
- **Findings (abstract and PDF):**
  - ChatGPT graded each example 0-5; at a 4.5 threshold, 9,229 of 52,002 Alpaca examples were kept (about 17.75%).
  - 7B training time fell from 80 to 14 min (5.7x). AlpaGasus-13B reached over 90% of Text-Davinci-003 on test tasks.
  - Human evaluation (13B, 160 prompts): 63 wins, 64 ties, 33 losses vs Alpaca.
  - Filtered 9k vs random 9k on benchmarks: MMLU 38.78 vs 36.93 (7B) and 46.12 vs 44.98 (13B). BBH 33.76 vs 31.89 (7B).
  - Dolly: 2,996 of 15k examples kept. Mixed benchmark results; 7B MMLU favored the full set (36.25 vs 35.83).
  - The authors attribute weaker results with a looser 39k subset to the harm of low-quality data.
- **Relevance:** About 82% of a popular distilled SFT set could be discarded with equal or better results, for a 5.7x training-cost cut. This is direct evidence for the fine-tuning-economics argument.
- **Caveats:** Wins are mostly judged by GPT-4. Benchmark gaps are 1-2 points. "Quality" is itself judged by an LLM.

### 2.3 Liu et al. 2024: What Makes Good Data for Alignment? (Deita)
- **Citation:** Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, Junxian He. ICLR 2024 (listing: proceedings.iclr.cc).
- **URL:** https://arxiv.org/abs/2312.15685
- **Findings (abstract):**
  - 6K SFT samples, which the authors call more than 10x less than baselines.
  - Deita-Mistral-7B plus DPO (6K SFT and 10K DPO samples) reached MT-Bench 7.55 and AlpacaEval 90.06%. It matched or beat open models trained on more data.
- **Caveats:** I read only the abstract, so baseline numbers are not verified. LLM-judge benchmarks.

### 2.4 Li et al. 2024: Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning
- **Citation:** Ming Li, Yong Zhang, Shwai He, Zhitao Li, Hongyu Zhao, Jianzong Wang, Ning Cheng, Tianyi Zhou. ACL 2024 (listing: aclanthology 2024.acl-long.769).
- **URL:** https://arxiv.org/abs/2402.00530
- **Findings (PDF):**
  - A GPT-2 (124M) model computes IFD scores to select 5%, 10% or 15% of the 52k Alpaca data.
  - AlpacaEval win rate at 5%: LLaMA2-7B 27.75 to 33.04, LLaMA2-13B 35.00 to 45.71. Open LLM average at 5% (7B): 55.25 to 55.67.
  - On the cleaner Alpaca-GPT4 source the gains were small: 7B AlpacaEval 71.32 to 72.13.
  - Filtering cost: about 8 min with GPT-2, vs about 120 min with ChatGPT scoring and about 1,400 min with a reward model.
- **Relevance:** Data selection can be nearly free compute-wise. The gain is largest when the source data is noisy.
- **Caveats:** Alpaca-family data only. GPT-4 pairwise judging.

### 2.5 Cao et al. 2023: Instruction Mining
- **Citation:** Yihan Cao, Yanbin Kang, Chi Wang, Lichao Sun. "Instruction Mining: Instruction Data Selection for Tuning Large Language Models." arXiv:2307.06290 (v3, 2024). Venue not verified.
- **URL:** https://arxiv.org/abs/2307.06290
- **Findings (abstract):** Selected 2,532 of 100,000 examples (about 2.5%). Reports state-of-the-art results on LLM-as-a-judge and the Open LLM leaderboard. MT-Bench numbers not verified.

### 2.6 Zhao et al. 2024: Long Is More for Alignment (caveat for "quality" filtering)
- **Citation:** Hao Zhao, Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion. arXiv:2402.04833. Venue not verified.
- **URL:** https://arxiv.org/abs/2402.04833
- **Findings (abstract):** Picking the 1,000 examples with the longest responses beat LIMA and AlpaGasus under GPT-4 and PaLM-2 judges (Llama-2-7B/13B, Mistral-7B). Numbers not verified.
- **Relevance:** Some "quality" gains may be length proxies interacting with judges' length bias. This is worth flagging when citing 2.1-2.4.

---

## 3. Response noise and label noise in SFT

### 3.1 Cho 2024: Unveiling Imitation Learning: Exploring the Impact of Data Falsity to Large Language Model
- **Citation:** Hyunsoo Cho. arXiv:2404.09717, 2024. Venue not verified.
- **URL:** https://arxiv.org/abs/2404.09717
- **Findings (PDF):**
  - Setup: the FACO dataset (about 20K samples from 9 datasets, with ChatGPT reasoning chains), with falsity applied at 0%, 25%, 50%, 75% and 100% (cumulative).
  - LLaMA-2-13B, MMLU: 57.49% at 0% falsity to 19.74% at 100% falsity. Average across 16 benchmarks: 57.00 to 45.70. ARC: 52.30 to 43.94.
  - LLaMA-1-13B, MMLU: 53.00 to 26.45.
  - The correlation between corruption rate and score was often below -0.9. Knowledge tasks (MMLU, BigBench) were hit hardest; commonsense tasks barely moved.
  - The stronger base (LLaMA-2) was more sensitive.
  - At 100% falsity the model scored below chance on MMLU, which the author reads as learned deliberate false answering.
  - Retraining on clean data recovered most of the loss but not all. The gap size is shown only in a figure, so it is not verified.
- **Relevance:** Wrong answers in SFT data teach the model to give wrong answers even when it "knows" the right one, and the effect is dose-dependent.
- **Caveats:** Single author, 13B models only. Only the endpoints (0% and 100%) were extracted, not intermediate rates.

### 3.2 Luo et al. 2024: RobustFT: Robust Supervised Fine-tuning for Large Language Models under Noisy Response
- **Citation:** Junyu Luo, Xiao Luo, Kaize Ding, Jingyang Yuan, Zhiping Xiao, Ming Zhang. arXiv:2412.14922, 2024. Venue not verified.
- **URL:** https://arxiv.org/abs/2412.14922 (HTML v1 read)
- **Findings (HTML):**
  - Llama-3.1-8B, MMLU: 65.3 without fine-tuning. Plain SFT on noisy data scored 59.5 at 30% noise, 47.5 at 50% and 37.3 at 70%.
  - The paper's own headline: with 30% noise, performance falls 8.9%.
  - The proposed method held 67.6-68.2 across noise rates.
  - Also tested: Llama-3.2-3B and Gemma2-9B, on ARC, PubMedQA, DROP and FPB.
- **Caveats:** The baseline is the untuned model; the paper reports no clean-SFT baseline, so the true cost of noise vs clean SFT is larger than these drops. The noise-injection procedure was not described in the parts I read.

### 3.3 Li et al. 2025: LLMs Can Easily Learn to Reason from Demonstrations: Structure, not content, is what matters!
- **Citation:** Dacheng Li, Shiyi Cao, Tyler Griggs, Shu Liu, Xiangxi Mo, Eric Tang, Sumanth Hegde, Kourosh Hakhamaneshi, Shishir G. Patil, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica. arXiv:2502.07374, 2025. Venue not verified.
- **URL:** https://arxiv.org/abs/2502.07374 (HTML v2 read)
- **Findings (HTML Table 2):**
  - Setup: Qwen2.5-32B-Instruct trained on about 4.6k QwQ long-CoT math traces. Average over MATH500, AIME24, AMC23 and OlympiadBench.
  - Base model 56.7; correct data 66.3.
  - Content corruption costs little: 100% wrong-answer traces 63.1 (-3.2). Digits corrupted at 20/50/70%: 64.5 / 64.2 / 62.0. Keywords removed (20-100%): 62.3 to 65.9.
  - Corrupting 100% of digits collapsed the model to 2.7.
  - Structural corruption costs more: 100% steps shuffled 56.1, 100% deleted 49.5, 100% inserted 44.5.
  - Abstract: 17k samples lifted AIME 2024 to 56.7% (+40.0) and LiveCodeBench to 57.0% (+8.1).
- **Relevance:** For reasoning-trace SFT, a wrong final answer is a minor defect, while broken step structure is a major one. This is a counterweight to 3.1.
- **Caveats:** One base model and domain (math). The abstract mentions LoRA, but the extracted table describes full fine-tuning; check before citing the method.

### 3.4 Gekhman et al. 2024: Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?
- **Citation:** Zorik Gekhman, Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Roi Reichart, Jonathan Herzig. EMNLP 2024 (listing: aclanthology 2024.emnlp-main.444).
- **URL:** https://arxiv.org/abs/2405.05904
- **Findings (PDF):**
  - Setup: PaLM 2-S on closed-book QA from EntityQuestions, with a fixed 6,142-example fine-tuning set and a varied share of "Unknown" examples.
  - Unknown examples are learned much more slowly than Known ones. Once fitted, they increase hallucination roughly linearly (in-distribution fit beta_known = 7.3, beta_unknown = -8.3, R^2 = 0.86).
  - Test accuracy, early stop vs convergence: 100% Unknown 37.5 vs 25.8; 50% Unknown 43.0 vs 38.8; natural mix 43.5 vs 41.8.
  - Early stopping mostly hides the harm.
- **Relevance:** "Correct but new to the model" responses are a data defect for SFT. Early stopping is the cheap mitigation.
- **Caveats:** One model, one QA task.

### 3.5 Zhu et al. 2022: Is BERT Robust to Label Noise? (classification fine-tuning; noise type matters)
- **Citation:** Dawei Zhu, Michael A. Hedderich, Fangzhou Zhai, David Ifeoluwa Adelani, Dietrich Klakow. Workshop on Insights from Negative Results in NLP, ACL 2022.
- **URL:** https://arxiv.org/abs/2204.09371
- **Findings (PDF):**
  - Uniform injected noise, AG-News: 94.40 clean, 92.40 at 40% noise, 89.53 at 60%, 85.49 at 70%.
  - Training to convergence with no early stopping at 70% noise gave 43.78.
  - Realistic weak-supervision noise was much worse: Hausa 83.55 to 46.97 (50.37% noisy labels); Yorùbá 74.45 to 64.72.
  - Noise-handling methods gained under 4 points.
- **Relevance:** Random noise is mostly absorbed with early stopping; systematic, feature-correlated noise is not. Real annotation errors are usually the systematic kind.
- **Caveats:** Encoder classification, not generative SFT.

### 3.6 Alajrami, Tan, Aletras 2025: Fine-Tuning on Noisy Instructions (harmless defect)
- **Citation:** Ahmed Alajrami, Xingwei Tan, Nikolaos Aletras. arXiv:2510.03528, 2025. IJCNLP-AACL 2025 per an underline.io listing; not verified.
- **URL:** https://arxiv.org/abs/2510.03528 (HTML v2 read)
- **Findings (HTML):**
  - Setup: perturbations applied to the instruction only (stop-word deletion, 25% shuffles, deletions, replacements, insertions, typos), mixed at 0-100%.
  - MMLU: Llama-3.1-8B 65.8 clean vs 66.0 at 100% perturbed; Llama-70B 78.1 vs 78.6.
  - GSM8K: mostly flat (Llama-70B 82.3 vs 82.0).
- **Relevance:** Messy prompts are close to harmless and can act as augmentation. The harm sits in the responses.
- **Caveats:** LoRA/QLoRA. Gains are inconsistent across models.

---

## 4. Preference-data noise and degenerate pairs

### 4.1 Chowdhury, Kini, Natarajan 2024: Provably Robust DPO: Aligning Language Models with Noisy Feedback
- **Citation:** Sayak Ray Chowdhury, Anush Kini, Nagarajan Natarajan. ICML 2024, PMLR 235 (listing: proceedings.mlr.press/v235/ray-chowdhury24a).
- **URL:** https://arxiv.org/abs/2403.00409
- **Findings (PDF):**
  - Labels are flipped at rate epsilon. The suboptimality bound scales as 1/(1-2 epsilon) times sqrt(d/n), so it blows up as epsilon approaches 0.5.
  - IMDb, gpt2-large, epsilon = 0.4: clean DPO reward 0.99. Noisy DPO 0.72-0.93 (std up to 0.43), cDPO about 0.82-0.84, IPO about 0.82-0.85, rDPO 0.99.
  - PPO at epsilon = 0.4: 0.78 vs clean 0.99.
  - Anthropic HH win-rate gain over SFT: Llama-2-7b DPO 45.78, cDPO 39.16, rDPO 51.20.
- **Relevance:** At high flip rates, vanilla DPO becomes unstable across seeds. TRL 1.17.1 ships this loss as `loss_type="robust"`, with `label_smoothing` as epsilon (verified in dpo_config.py).
- **Caveats:** Synthetic random flips. Small policy models.

### 4.2 Gao, Alon, Metzler 2024: Impact of Preference Noise on the Alignment Performance of Generative Language Models
- **Citation:** Yang Gao, Dana Alon, Donald Metzler. arXiv:2404.09824, 2024. Venue not verified.
- **URL:** https://arxiv.org/abs/2404.09824
- **Findings (abstract and PDF):**
  - Setup: T5-Large policy with a T5-XXL gold reward model, on Reddit TL;DR and Anthropic-Helpful. Three noise types (random flip, stochastic, Gaussian reward error) at 5-50%.
  - The authors estimate real preference sets are about 20-40% noisy.
  - Below 30% noise, +10 points of noise costs under 10 points of win rate. Above 40%, +10 points costs about 20-30.
  - Win rate stays above 0.5 at 45% noise and falls below 0.5 at 50%.
  - Noise type matters less than noise rate.
  - A higher KL weight did not help, and higher dropout hurt.
  - Confidence filtering (threshold 0.8) helped against stochastic and Gaussian noise but not random flips, and it removed over 50% of pairs.
- **Caveats:** Per-rate win rates are only in figures, so not verified. Small models.

### 4.3 Razin et al. 2025: Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization
- **Citation:** Noam Razin, Sadhika Malladi, Adithya Bhaskar, Danqi Chen, Sanjeev Arora, Boris Hanin. ICLR 2025 (listing: iclr.cc poster 27968).
- **URL:** https://arxiv.org/abs/2410.08847
- **Findings (abstract):**
  - DPO often lowers the likelihood of the preferred response, and probability can shift to responses with the opposite meaning.
  - Refusal training of Llama-3-8B-Instruct dropped its refusal rate from 74.4% to 33.4%.
  - The cause is preference pairs with similar hidden embeddings, measured by a CHES score. Filtering high-CHES pairs mitigates it.
- **Relevance:** Near-duplicate chosen/rejected pairs are not just wasted compute; they can actively misalign the model.

### 4.4 Pal et al. 2024: Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive
- **Citation:** Arka Pal, Deep Karkhanis, Samuel Dooley, Manley Roberts, Siddartha Naidu, Colin White. arXiv:2402.13228, 2024. Venue not verified.
- **URL:** https://arxiv.org/abs/2402.13228
- **Findings (abstract):** Standard DPO can reduce the preferred response's likelihood, and this is most pronounced when chosen and rejected have low edit distance. Magnitudes are not verified.

### 4.5 Chen et al.: On Extending Direct Preference Optimization to Accommodate Ties
- **Citation:** Jinghong Chen, Guangyu Yang, Weizhe Lin, Jingbiao Mei, Bill Byrne. arXiv:2409.17431. NeurIPS 2025 (listing: proceedings.neurips.cc 2025).
- **URL:** https://arxiv.org/abs/2409.17431
- **Findings (abstract):** Adding tied pairs to standard DPO degrades task performance. Tie-aware variants (Rao-Kupper, Davidson) avoid this. Ties strengthen KL regularization toward the reference policy. Numbers not verified.

### 4.6 Real-world preference-label defects (dataset cards)
- **HuggingFaceH4/ultrafeedback_binarized** (https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized):
  - After Zephyr-7b-beta, the card says "a few hundred completions" had incorrect labels, and some prompts came from TruthfulQA (contamination). Both were fixed in a later revision.
  - Chosen is the highest overall_score completion; rejected is a random one of the other three.
  - The card says nothing about how ties are handled.
- **argilla/ultrafeedback-binarized-preferences-cleaned** (https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned): removed TruthfulQA-contaminated prompts and binarized on mean preference rating; 60,917 rows. No counts of removed rows or ties are given.

### 4.7 Identical chosen/rejected pairs: derivation, no empirical source found
- In DPO, the per-pair loss is -log sigmoid(beta * [(log pi(y_w|x) - log pi_ref(y_w|x)) - (log pi(y_l|x) - log pi_ref(y_l|x))]). If y_w = y_l, the bracket is 0 for all parameters, so the loss is the constant log 2 and the gradient is exactly zero. The pair costs two forward passes and dilutes the batch mean, but teaches nothing. This is my derivation, not from a paper.
- Near-identical pairs are worse than useless (4.3, 4.4).
- Interaction with TRL (inference from source, section 6.3): prompt+chosen and prompt+rejected are each truncated to max_length independently. Pairs that differ only after the cutoff become identical after truncation, and nothing warns about it.

---

## 5. Contamination (train/test overlap)

### 5.1 Zhang et al. 2024: A Careful Examination of Large Language Model Performance on Grade School Arithmetic (GSM1k)
- **Citation:** Hugh Zhang, Jeff Da, Dean Lee, Vaughn Robinson, Catherine Wu, Will Song, Tiffany Zhao, Pranav Raja, Charlotte Zhuang, Dylan Slack, Qin Lyu, Sean Hendryx, Russell Kaplan, Michele Lunati, Summer Yue. NeurIPS 2024 Datasets and Benchmarks (per arXiv comment).
- **URL:** https://arxiv.org/abs/2405.00332
- **Findings (abstract and HTML):**
  - GSM1k has 1,205 fresh problems. Accuracy drops of up to 8% vs GSM8k.
  - Phi and Mistral families showed systematic overfitting (Phi-2 about 6%); Mistral Large did not.
  - Frontier models (GPT, Claude, Gemini) showed little or no gap.
  - The correlation between a model's GSM8k log-likelihood and its gap was Spearman 0.36 (p = 0.03). The abstract labels this "r^2", an inconsistency in the paper.
- **Caveats:** Observational. Memorization explains only part of the gap.

### 5.2 Yang et al. 2023: Rethinking Benchmark and Contamination for Language Models with Rephrased Samples
- **Citation:** Shuo Yang, Wei-Lin Chiang, Lianmin Zheng, Joseph E. Gonzalez, Ion Stoica. arXiv:2311.04850, 2023. Venue not verified.
- **URL:** https://arxiv.org/abs/2311.04850
- **Findings (PDF):**
  - Training on rephrased test items: Llama-2-13B MMLU 54.8 (5-shot) to 85.9-89.9 (16 epochs); Llama-2-7B 45.3 to 82.0-88.5.
  - GSM8K: Llama-2-13B 28.7 to 95.3.
  - HumanEval: CodeLlama-13B 36.0 to 81.1 (GPT-4: 67.0).
  - 10-gram overlap detection had F1 = 0 on the rephrased samples.
  - Real overlap with HumanEval: The Stack subset 18.9%, StarCoder-Data subset 15.9%, CodeAlpaca 12.8%, RedPajama subset 8.5%, Evol-Instruct-Code 7.9%. MATHInstruct overlaps 15.4% of MATH test. FLAN CoT overlaps 0.5% of MMLU.
- **Relevance:** Public SFT sets are contaminated, and string-match decontamination misses paraphrases. Fine-tuning on them inflates the evaluations that justify the fine-tune.
- **Caveats:** The headline inflation is a deliberate worst case (direct training on rephrased test items).

### 5.3 Bordt et al. 2025: How Much Can We Forget about Data Contamination? (inflation per exposure)
- **Citation:** Sebastian Bordt, Suraj Srinivas, Valentyn Boreiko, Ulrike von Luxburg. ICML 2025, PMLR 267 (listing: proceedings.mlr.press/v267/bordt25a).
- **URL:** https://arxiv.org/abs/2410.03249 (HTML v4 read)
- **Findings (HTML Table 1), accuracy inflation in points at 1x Chinchilla:**

  | Model | 4x | 12x | 32x | 144x |
  |---|---|---|---|---|
  | 124M | +5.92 | +14.70 | +38.48 | +54.23 |
  | 350M | +10.97 | +25.18 | +44.48 | +50.78 |
  | 774M | +15.60 | +32.14 | +43.79 | +46.89 |
  | 1.6B | +15.55 | +30.26 | +39.79 | +43.34 |

  - Inflation fades with more subsequent data: 12x contamination of a 124M model gives about 18 points at 2x Chinchilla and is within noise at 15x. Even 144x contamination is forgotten beyond 5x Chinchilla.
  - OLMo-7B continual pretraining: 4 insertions gave about +17 points; about 25k more steps removed about 87% of that.
- **Relevance to fine-tuning:** Forgetting needs many more tokens after exposure. A small SFT set has no such dilution, so contamination introduced at fine-tuning time should persist (inference).
- **Caveats:** Pretraining. The setup text says 36x while the tables say 32x (inconsistency in the source).

### 5.4 Zhou et al. 2023: Don't Make Your LLM an Evaluation Benchmark Cheater (controlled leakage via continual training)
- **Citation:** Kun Zhou, Yutao Zhu, Zhipeng Chen, Wentong Chen, Wayne Xin Zhao, Xu Chen, Yankai Lin, Ji-Rong Wen, Jiawei Han. arXiv:2311.01964, 2023. Venue not verified.
- **URL:** https://arxiv.org/abs/2311.01964
- **Findings (PDF):**
  - LLaMA-2-7B MMLU 42.95: 51.61 with the MMLU train set added, 56.04 with all train sets plus test prompts, 96.34 with test sets (reference only).
  - LLaMA-2-7B BoolQ: 71.68 to 88.72.
  - Harm to non-leaked tasks: LLaMA-2-7B HumanEval 26.83 to 8.54, XSum ROUGE-L 8.67 to 0.25. These persisted after instruction tuning (HumanEval 28.66 vs 20.73).
- **Relevance:** Benchmark-adjacent fine-tuning data both inflates target scores and degrades other abilities.
- **Caveats:** Extreme simulation. Continual training on leaked data only, with no data mixing.

### 5.5 Jiang et al. 2024: Investigating Data Contamination for Pre-training Language Models
- **Citation:** Minhao Jiang, Ken Ziyu Liu, Ming Zhong, Rylan Schaeffer, Siru Ouyang, Jiawei Han, Sanmi Koyejo. arXiv:2401.06059, 2024. Also listed as an ICLR 2024 workshop paper (mlanthology); not opened.
- **URL:** https://arxiv.org/abs/2401.06059
- **Findings (PDF):**
  - GPT-2-small trained from scratch on 3.3B tokens. Clean vs text contamination vs ground-truth contamination: SST-2 48.34 / 54.89 / 51.02; SQuAD F1 9.07 / 9.78 / 11.45; CNN/DM ROUGE-1 24.76 / 26.84 / 28.80.
  - The effect of repeated contamination is U-shaped: it peaks around 10 repetitions and some settings fall below clean at 20.
  - n-gram contamination definitions (as in Llama 2 reports) flag 3% to over 20% of tokens and poorly identify true contamination.
- **Caveats:** Small models, low absolute scores.

### 5.6 Magar & Schwartz 2022: Data Contamination: From Memorization to Exploitation
- **Citation:** Inbal Magar, Roy Schwartz. ACL 2022 (Short Papers) (listing: aclanthology 2022.acl-short.18).
- **URL:** https://arxiv.org/abs/2203.08242
- **Findings (PDF):**
  - BERT pretrained on Wikipedia plus labeled downstream data. Memorization does not imply exploitation.
  - SST-5 (BERT-base): exploitation about 40% at 200 copies. With 600M Wikipedia tokens, memorization was about 15% but exploitation under 1%.
  - Early insertion gives exploitation up to 17%.
  - Larger models memorize more.

### 5.7 Deng et al. 2024: Investigating Data Contamination in Modern Benchmarks for Large Language Models
- **Citation:** Chunyuan Deng, Yilun Zhao, Xiangru Tang, Mark Gerstein, Arman Cohan. NAACL 2024 (listing: aclanthology 2024.naacl-long.482).
- **URL:** https://arxiv.org/abs/2311.09783
- **Findings (abstract):** ChatGPT and GPT-4 guessed masked MMLU answer options with 52% and 57% exact match. Detection evidence, not an inflation measure.

### 5.8 Golchin & Surdeanu 2024: Time Travel in LLMs: Tracing Data Contamination in Large Language Models
- **Citation:** Shahriar Golchin, Mihai Surdeanu. ICLR 2024 (listing: iclr.cc poster 19550).
- **URL:** https://arxiv.org/abs/2308.08493
- **Findings (abstract):** Detection accuracy of 92-100% across seven datasets vs human experts. Reports GPT-4 contaminated with AG News, WNLI and XSum. Detection, not inflation.

### 5.9 Sainz et al. 2023: NLP Evaluation in trouble
- **Citation:** Oscar Sainz, Jon Ander Campos, Iker García-Ferrero, Julen Etxaniz, Oier Lopez de Lacalle, Eneko Agirre. Findings of EMNLP 2023 (listing: aclanthology 2023.findings-emnlp.722).
- **URL:** https://arxiv.org/abs/2310.18018
- **Findings:** Position paper with no measurements. Defines contamination levels and calls for per-benchmark contamination reporting.

---

## 6. Truncation and sequence length

### 6.1 Ding et al. 2024: Fewer Truncations Improve Language Modeling
- **Citation:** Hantian Ding, Zijian Wang, Giovanni Paolini, Varun Kumar, Anoop Deoras, Dan Roth, Stefano Soatto. ICML 2024, PMLR 235 (listing: proceedings.mlr.press/v235/ding24f).
- **URL:** https://arxiv.org/abs/2404.10830
- **Findings (abstract and PDF):**
  - Best-fit Packing vs concatenate-and-chunk: reading comprehension +4.7%, context following +16.8%, program synthesis +9.2% (relative), closed-domain hallucination down up to 58.3%.
  - Examples at 2k context: reading-comprehension average 46.71 to 48.92; NQ-Swap 45.62 to 51.03; HumanEval pass@100 35.28 to 40.57.
  - Undefined-name errors: HumanEval 5.10% to 2.41%, MBPP 9.52% to 3.97%.
  - Best-fit Packing adds at most about 0.01% more sequences. Commonsense QA was unchanged.
- **Setting:** 7B (code, 300B tokens) and 13B (natural language, 500B tokens); RefinedWeb and The Stack; 2k and 8k context.
- **Relevance:** Cutting documents away from their context measurably hurts grounding. For SFT, the analogue is cutting responses, which is untested.
- **Caveats:** Pretraining. The causal story rests on benchmarks plus a toy model.

### 6.2 Hugging Face TRL SFTTrainer: current behavior (TRL 1.17.1, latest on PyPI, released 2026-10-30)
- **Sources:** Docs at https://huggingface.co/docs/trl/sft_trainer (main branch; API links point to v1.17.1). Source: the trl-1.17.1 wheel, `trl/trainer/sft_trainer.py`, `sft_config.py` and `data_utils.py`, read directly.
- **Truncation defaults:**
  - `max_length` defaults to 1024. `None` disables truncation.
  - `truncation_mode` defaults to "keep_start", which slices `[:max_length]`, so the end of the sequence (usually the assistant response and its end-of-turn token) is cut.
  - Docs say "keep_end" is deprecated and will be removed in v2.0.0. The docs still say "from the left or right", which contradicts that.
- **No warning on truncation:** truncation happens in a `dataset.map` during preparation. No `logger.warning` reports how many examples were truncated (I grepped every warning call in sft_trainer.py). The only visible trace is a progress-bar label, "Truncating ... dataset".
- **Silent dropping:** after truncation, examples whose labels are all masked (for example a prompt that fills max_length) are removed by `dataset.filter`. Again the only trace is a progress-bar label.
- **Packing:**
  - Default strategy "bfd" (best-fit decreasing) preserves boundaries but truncates sequences longer than max_length and discards the overflow tokens (data_utils.py docstring).
  - "bfd_split" splits overflow instead. "wrapped" cuts mid-sequence.
- **Hard errors (not silent):**
  - `padding_free=True` without packing and with max_length set raises ValueError.
  - VLM image-token truncation raises an error.
- **Related warning:** the trainer warns when the chat template leaves the end-of-turn token outside the loss mask. The message says the model may not learn to stop, while training loss still looks healthy. That warning is about template masking, but truncated examples lose their end-of-turn supervision in the same way (inference).

### 6.3 Hugging Face TRL DPOTrainer: current behavior (TRL 1.17.1)
- **Sources:** Docs at https://huggingface.co/docs/trl/dpo_trainer. Source: `trl/trainer/dpo_trainer.py` and `dpo_config.py` from the trl-1.17.1 wheel.
- **Parameters:**
  - `max_length` defaults to 1024 and `truncation_mode` to "keep_start" ("keep_end" is deprecated).
  - `max_prompt_length` and `max_completion_length` no longer exist in 1.17.1.
  - Historical (v0.19.0 docs, https://huggingface.co/docs/trl/v0.19.0/en/dpo_trainer): max_length 1024, max_prompt_length 512, max_completion_length None, truncation_mode "keep_end". Tutorials written against older versions describe different behavior.
- **How truncation works:**
  - The collator (`DataCollatorForPreference`) truncates prompt+chosen and prompt+rejected each to max_length independently, at batch time.
  - Docs: custom collators must truncate themselves, because the trainer does not truncate after collation.
  - No warning is logged when truncation happens.
- **Silent dropping:** with keep_start, examples whose prompt alone is at least max_length are silently filtered out during preprocessing (progress-bar label only).
- **No pair checks:** the trainer does not check for identical or near-identical chosen/rejected pairs (grep found none). See 4.7 for why this matters.
- `padding_free=True` is temporarily unavailable; it warns and falls back to standard padding.
- `loss_type="robust"` implements rDPO (4.1).

### 6.4 Gap
I found no controlled study measuring how truncating SFT responses or preference pairs affects fine-tuned model quality. The case rests on the pretraining result (6.1), the TRL defaults (6.2, 6.3) and the mechanism (lost end-of-turn tokens, possibly identical truncated pairs).

---

## 7. Memorization of secrets and PII in fine-tuning

### 7.1 Mireshghallah et al. 2022: An Empirical Analysis of Memorization in Fine-tuned Autoregressive Language Models
- **Citation:** Fatemehsadat Mireshghallah, Archit Uniyal, Tianhao Wang, David Evans, Taylor Berg-Kirkpatrick. Proceedings of EMNLP 2022, pp. 1816-1826. DOI 10.18653/v1/2022.emnlp-main.119. The arXiv title is "Memorization in NLP Fine-tuning Methods" (arXiv:2205.12506).
- **URL:** https://aclanthology.org/2022.emnlp-main.119 ; https://arxiv.org/abs/2205.12506
- **Findings:**
  - Abstract: head-only fine-tuning is most susceptible to membership inference and extraction. Smaller adapters appear less vulnerable to extraction.
  - PDF Table 2 (Wikipedia, GPT-2): MIA recall was full FT 19.2%, head FT 81.6%, adapters (reduction factor 2) 16.8%, adapters (reduction factor 16) 15.2%, at comparable validation perplexity (21.75-23.93).
  - Parameters trained: full FT 124.4M, head 38.6M, adapters rf2 7.09M, rf16 0.90M.
  - Full FT and adapters sit on the Pareto frontier of attack recall vs perplexity.
- **Exposure (canary) table, Table 1:** two automated readings agreed on the numbers but disagreed on which dataset each row belongs to, so per-dataset values are not verified. Both readings agree that:
  - adapters (rf16) had the lowest exposure on every dataset or nearly every one;
  - every exposure value was below 20, the caption's threshold for reliably extractable canaries (all values were 0.44-14.54).
- **Relevance:** This bears directly on LoRA/adapter economics. Parameter-efficient methods do not increase memorization risk relative to full fine-tuning, and small adapters reduce it. Head-only tuning is the outlier.
- **Caveats:** GPT-2 124M only. Canaries were inserted at controlled rates.

### 7.2 Lukas et al. 2023: Analyzing Leakage of Personally Identifiable Information in Language Models
- **Citation:** Nils Lukas, Ahmed Salem, Robert Sim, Shruti Tople, Lukas Wutschitz, Santiago Zanella-Béguelin. IEEE Symposium on Security and Privacy (S&P) 2023 (header in arXiv PDF).
- **URL:** https://arxiv.org/abs/2302.00539
- **Findings (PDF):**
  - Setup: GPT-2 Small to XL fine-tuned on ECHR, Enron and Yelp-Health.
  - Undefended PII extraction on ECHR: precision 24.91-29.56%, recall 9.44% (Small) to 22.96% (Large). Enron: precision 27-35%, recall 6.3-7.2%.
  - With DP (epsilon = 8), ECHR precision and recall were about 3%. The abstract says DP still leaks about 3% of PII sequences.
  - Scrubbing ECHR (GPT-2 Large) gave 0% extraction in their attack. The abstract nonetheless says scrubbing reduces but does not prevent leakage, and costs utility.
  - Leakage grows roughly linearly with how often a PII string appears.
  - The new attacks extract up to 10x more PII than prior attacks.
- **Relevance:** Fine-tuning on raw logs, emails or tickets leaks PII at measurable rates. Leakage is proportional to duplication, and scrubbing is the cheapest defense.
- **Caveats:** The paper's text and table disagree on the direction of size scaling; use the table (recall rises with size).

### 7.3 Carlini et al. 2019: The Secret Sharer
- **Citation:** Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos, Dawn Song. "The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks." arXiv:1802.08232 (v3 2019). Venue not verified on the pages I opened.
- **URL:** https://arxiv.org/abs/1802.08232
- **Findings (abstract):** Introduces canary exposure testing. Shows that unique secret sequences such as credit card numbers can be extracted from models trained without memorization safeguards. Applied to Gmail Smart Compose. Numbers not verified.

### 7.4 Zeng et al. 2023: Exploring Memorization in Fine-tuned Language Models
- **Citation:** Shenglai Zeng, Yaxin Li, Jie Ren, Yiding Liu, Han Xu, Pengfei He, Yue Xing, Shuaiqiang Wang, Jiliang Tang, Dawei Yin. arXiv:2310.06714. Venue not verified.
- **URL:** https://arxiv.org/abs/2310.06714
- **Findings (abstract):** Memorization differs strongly across fine-tuning tasks and correlates with attention-score distribution. Numbers not verified.

See also 1.3 (Kandpal: duplicates drive extraction) and 1.4 (Carlini 2023: size and duplicates).

---

## 8. Training and fine-tuning job failures

### 8.1 Llama 3 (Meta, 2024)
- **Citation:** Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. "The Llama 3 Herd of Models." arXiv:2407.21783, 2024.
- **URL:** https://arxiv.org/abs/2407.21783 (HTML v3, section 3.3.4, Table 5)
- **Findings:**
  - Over a 54-day 405B pretraining snapshot there were 466 interruptions: 47 planned and 419 unexpected.
  - About 78% of unexpected interruptions were attributed to confirmed or suspected hardware, and effective training time was above 90%.
  - Top causes: faulty GPU 148, GPU HBM3 72, software bug 54, network switch/cable 35, unplanned host maintenance 32, silent data corruption (a hardware fault, not training data) 6.
  - No category refers to training data. "Dataset updates" appear only as an example of planned operator interruptions.
- **Caveat:** Table 5's printed percentages are internally inconsistent. 72/419 gives the printed 17.2% for HBM3, but 148/419 is 35.3%, not the printed 30.1% for faulty GPU. The paper's "GPU issues 58.7%" sums the printed values. Re-check the PDF before quoting percentages; the raw counts sum to 419.

### 8.2 OPT-175B (Meta, 2022)
- **Citation:** Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, et al. "OPT: Open Pre-trained Transformer Language Models." arXiv:2205.01068, 2022.
- **URL:** https://arxiv.org/abs/2205.01068 ; logbook index https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles. The GitHub page was blocked to the fetcher, so the logbook PDFs were not read.
- **Findings (PDF, section 2.5):**
  - At least 35 manual restarts and an estimated 70+ automatic restarts due to hardware. Over 100 hosts were cycled out over about 2 months.
  - Loss divergences were handled by lowering the learning rate and restarting from earlier checkpoints.
  - No failure is attributed to data in that section. Section 2.3 notes that some Pile subsets were excluded because they raised instability risk at 1.3B scale. This is the only data-instability link, and it is qualitative.

### 8.3 Hu et al. 2024: Characterization of Large Language Model Development in the Datacenter (Acme, Shanghai AI Lab)
- **Citation:** Qinghao Hu, Zhisheng Ye, Zerui Wang, Guoteng Wang, Meng Zhang, Qiaoling Chen, Peng Sun, Dahua Lin, Xiaolin Wang, Yingwei Luo, Yonggang Wen, Tianwei Zhang. NSDI 2024 (listing: usenix.org/conference/nsdi24/presentation/hu).
- **URL:** https://arxiv.org/abs/2403.07648
- **Findings (PDF):**
  - About 40% of jobs fail.
  - Infrastructure failures are about 11% of failed jobs but over 82% of the GPU time lost. Script errors are the bulk of failures (about 73%) but about 3% of GPU time (category totals derived from Table 3).
  - Data-adjacent rows: "Dataset Loading Error" 5 occurrences (0.00% of GPU time), "Dataloader Killed" 6 (4.38%), "File Not Found" 568 (2.83%; not necessarily data).
- **Caveat:** Row values come from a single extraction of a long table and were not independently re-checked.

### 8.4 Zhang et al. 2020: An Empirical Study on Program Failures of Deep Learning Jobs (Microsoft Philly)
- **Citation:** Ru Zhang, Wencong Xiao, Hongyu Zhang, Yu Liu, Haoxiang Lin, Mao Yang. ICSE 2020.
- **URL:** https://hongyujohn.github.io/icse20-main-199.pdf (author copy)
- **Findings:**
  - 4,960 failed jobs: execution environment 48.0%, general code error 36.5%, DL-specific 13.5%, data 2.0% (corrupt data 1.8%, unexpected encoding 0.2%).
  - Path-not-found alone is 39.4%.
  - In a 400-job subset, model-evaluation-phase failures were 24% of failures but 81% of GPU service time.
- **Caveat:** Pre-LLM era, general DL jobs.

### 8.5 Other sources (abstract only; data share not reported)
- Kokolis et al. 2024, "Revisiting Reliability in Large-Scale Machine Learning Research Clusters," arXiv:2410.21680 (https://arxiv.org/abs/2410.21680). 11 months, about 4M jobs, over 150M A100 GPU-hours. Taxonomy shares not verified.
- Jiang et al. 2025, "L4: Diagnosing Large-scale LLM Training Failures via Automated Log Analysis," arXiv:2503.20263 (https://arxiv.org/abs/2503.20263). 428 failures; hardware and user faults predominate. No shares or data category in the abstract.
- Gao, Lu, Lin, Chen 2025, "An Empirical Study of Issues in Large Language Model Training Systems," FSE 2025 Industry track, DOI 10.1145/3696630.3728538 (https://conf.researchr.org/details/fse-2025/fse-2025-industry-papers/10/An-Empirical-Study-of-Issues-in-Large-Language-Model-Training-Systems). 300 issues in DeepSpeed, Megatron-LM and Transformers: API misuse 19.67%, configuration error 18.33%, general code error 16.33%. No data category reported.

### 8.6 Conclusion for section 8
No source I found quantifies data-caused failures of LLM fine-tuning jobs, and no practitioner survey with such numbers turned up. Large-run reports attribute interruptions to hardware. Older cluster studies put explicit data failures at about 2%. Data defects mainly cause silent quality loss (sections 1-7), not crashes. That is an argument for validating data before training, since failure telemetry will not reveal these defects (interpretation).

---

## Not verified / open items
- Exact per-dataset exposure values in Mireshghallah Table 1 (row labels disputed between readings).
- Hernandez et al.: the effective-size numbers for subsequent fine-tuning (inconsistent between sections as extracted).
- Gao et al.: per-noise-rate win rates (figures only).
- Cho 2024: intermediate falsity rates (25-75%) and the size of the recovery gap.
- RobustFT: noise-injection procedure; no clean-SFT baseline exists in the paper.
- Deita baselines, Instruction Mining MT-Bench scores, Long-Is-More win rates, Smaug magnitudes, the ties paper's numbers, Zeng et al. numbers, Secret Sharer numbers and venue.
- OPT logbook PDFs (blocked); Kokolis et al. and L4 taxonomy shares.
- Venues marked "(listing)" were taken from official proceedings URLs in search results without opening those pages.
- Llama 3 Table 5 percentages are internally inconsistent in the source; quote counts, not percentages.
