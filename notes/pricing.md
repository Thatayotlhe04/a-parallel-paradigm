# Pricing and economic inputs: rent (API) vs own/adapt (open weights)

Compiled 2026-11-08. Every value comes from a page fetched that day with WebFetch. "Access date" means the page gave no date of its own. Pages were read through a fetch tool that converts them to text, and raw `curl` was blocked by the sandbox proxy, so a few numbers carry some risk of transcription error. Where it mattered I re-fetched and asked for verbatim cell text (DeepInfra, Together, Gemini, Fireworks fine-tuning). Rows marked **[secondary]** come from a non-official or third-party page. Rows marked **[derived]** are my own arithmetic on sourced numbers. All other rows are primary sources.

---

## Key takeaways (for the break-even model)

1. **API list prices, Nov 2026 (input / output, USD per 1M tokens):**
   - Claude Opus 5.5: $4 / $20
   - Claude Sonnet 5.5: $2 / $10
   - Claude Haiku 5.5: $0.10 / $0.50 for prompts ≤100K tokens
   - Claude Fable 5.1: $10 / $50
   - OpenAI gpt-6-astra (flagship): $10 / $50
   - OpenAI gpt-6-luna (cheapest): $0.10 / $0.50
   - Gemini 3.1 Pro Preview: $2 / $12
   - Gemini 3.5 Flash: $1.50 / $9
   - Gemini 3.1 Flash-Lite: $0.25 / $1.50

   All three vendors give 50% off for batch. Sources: https://platform.claude.com/docs/en/about-claude/pricing, https://developers.openai.com/api/docs/pricing, https://ai.google.dev/gemini-api/docs/pricing
2. **Hosted open-weight floor prices (DeepInfra, USD per 1M tokens, input / output):**
   - Llama 3.1 8B Turbo: $0.02 / $0.04
   - Qwen3-30B-A3B: $0.12 / $0.50
   - Llama 3.3 70B Turbo: $0.10 / $0.32
   - DeepSeek-V4.1-Flash: $0.20 / $0.60
   - Kimi-K3: $2.85 / $14.25

   Together and Fireworks no longer serve Llama 3.1 8B or 3.3 70B serverless. Source: https://deepinfra.com/pricing
3. **H100 rental price index:**
   - Latest value: Silicon Data SDH100RT was **$3.00/GPU-hr on 2026-11-08** (neocloud and marketplace tier).
   - Trend: about $2.13–2.26 in Aug–Sep 2025, then $2.57 (May 2026), $2.75 (Jul 2026) and $3.00 (Nov 2026). That is roughly a **+35–40% rise over the year** [derived].
   - Other GPUs on 2026-11-08: B200 $6.81, H200 $3.55, A100 $1.59.
   - Hyperscaler on-demand H100 was about $7.40–7.52 in Mar–Apr 2026, roughly 3x the neocloud tier.

   Sources: https://www.silicondata.com/products/silicon-index/h100, https://www.silicondata.com/blog/h100-rental-market-update-september-2025, https://www.silicondata.com/blog/h100-hyperscaler-index-april-2026
4. **Self-hosted serving throughput (TensorRT-LLM, FP8, H100 SXM, max-load offline benchmark):**
   - Llama 3.1 8B on 1 GPU: 14,992 output tok/s (ISL/OSL 1000/1000). At $3.00/hr that is **about $0.056 per 1M output tokens at 100% utilisation** [derived].
   - Llama 3.3 70B on 2 GPUs (TP2): 2,209 output tok/s/GPU. At $3.00/hr that is **about $0.38 per 1M output tokens** [derived], on par with DeepInfra's $0.32.

   Sources: https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html (Sep 2025), https://nvidia.github.io/TensorRT-LLM/developer-guide/perf-overview.html (updated 2026-11-03)
5. **LoRA fine-tuning price per 1M training tokens:**
   - Fireworks: $0.50 for models ≤16B, $3.00 for 16.1–80B, $6.00 for 80–300B, $10.00 above 300B.
   - Together: $0.34 for the ~8B tier and $2.03 for 70B (first table; probably the LoRA tab).
   - Example: 25M training tokens on an 8B model costs **about $8.50–$12.50** [derived].
   - OpenAI fine-tuning is being wound down. gpt-4.1-mini is $5.00 to train, and $0.80 / $3.20 per 1M tokens to run once tuned.

   Sources: https://fireworks.ai/pricing, https://www.together.ai/pricing, https://developers.openai.com/api/docs/pricing
6. **Serving a fine-tuned adapter has become a dedicated-GPU cost, not a per-token cost:**
   - Together has discontinued serverless LoRA and Serverless Multi-LoRA. Adapters now run on dedicated endpoints at $5.49/GPU-hr for H100.
   - Fireworks' fine-tuning doc calls on-demand deployment "the only supported method for serving trained models". Its LoRA doc, however, says LoRAs on select base models can be served serverless per token, so the two docs conflict.
   - Fireworks' GPU rate is $8.00/hr for H100.

   Sources: https://docs.together.ai/docs/lora-inference, https://docs.fireworks.ai/fine-tuning/fine-tuning-models, https://docs.fireworks.ai/fine-tuning/deploying-loras
7. **LoRA compute is small:**
   - QLoRA on Llama 3.1 8B over 100k samples (1 epoch, seq length 2048) took **4h45m on one A100 40GB**. Scaled linearly, 10k–50k samples is about 0.5–2.4 GPU-hours [derived].
   - QLoRA paper: a 65B model fits on one 48GB GPU, and Guanaco trained in about 24 h on one GPU.

   Sources: https://huggingface.co/blog/mlabonne/sft-llama3, https://arxiv.org/abs/2305.14314
8. **Labor and request size:**
   - Labor: US ML/AI software engineer median total compensation is **$251,250/yr** (Levels.fyi). That is about $121/hr at 2,080 hours [derived]. https://www.levels.fyi/t/software-engineer/focus/ml-ai
   - Request size: OpenRouter's average request grew to **about 6K prompt tokens and about 400 completion tokens** by late 2025. https://arxiv.org/html/2601.10088
   - Cost per average request [derived]: Sonnet 5.5 about $0.016, Haiku 5.5 about $0.0008, DeepInfra Llama 3.3 70B about $0.0007.

---

## 1. Proprietary API list prices (USD per 1M tokens)

### 1a. Anthropic

| Item | Value | Unit | As-of date | Source URL | Notes/uncertainty |
|---|---|---|---|---|---|
| Claude Opus 5.5 input | 4.00 | USD/MTok | access 2026-11-08 | https://platform.claude.com/docs/en/about-claude/pricing | Same on https://claude.com/pricing |
| Claude Opus 5.5 output | 20.00 | USD/MTok | access 2026-11-08 | same | |
| Claude Opus 5.5 cache write 5m / 1h | 5.00 / 8.00 | USD/MTok | access 2026-11-08 | same | 1.25x / 2x input |
| Claude Opus 5.5 cache hit | 0.20 | USD/MTok | access 2026-11-08 | same | 0.05x input |
| Claude Sonnet 5.5 input / output | 2.00 / 10.00 | USD/MTok | access 2026-11-08 | same | |
| Claude Sonnet 5.5 cache write 5m / 1h / hit | 2.50 / 4.00 / 0.10 | USD/MTok | access 2026-11-08 | same | hit = 0.05x input |
| Claude Haiku 5.5 input / output (prompt ≤100K) | 0.10 / 0.50 | USD/MTok | access 2026-11-08 | same | Haiku 5.5 is priced by prompt length |
| Claude Haiku 5.5 input / output (prompt >100K) | 0.50 / 2.50 | USD/MTok | access 2026-11-08 | same | Threshold counts all input tokens, including cache reads and writes |
| Claude Haiku 5.5 cache write 5m / 1h / hit (≤100K) | 0.125 / 0.20 / 0.01 | USD/MTok | access 2026-11-08 | same | |
| Claude Fable 5.1 input / output | 10.00 / 50.00 | USD/MTok | access 2026-11-08 | same | Cache hit $0.25 (0.025x). Batch $5 / $25 |
| Claude Mythos 5.1 input / output | 10.00 / 50.00 | USD/MTok | access 2026-11-08 | same | Listed in docs only; not on claude.com/pricing |
| Anthropic Batch API discount | 50 | % off input and output | access 2026-11-08 | same | Opus 5.5 batch $2 / $10; Sonnet 5.5 $1 / $5; Haiku 5.5 $0.05 / $0.25. Fast mode not available in batch |
| Opus 5.5 fast mode | 2x | multiplier | access 2026-11-08 | https://claude.com/pricing | |
| US-only inference | 1.1x | multiplier | access 2026-11-08 | https://claude.com/pricing | |
| Long context | Full 1M context at standard price | n/a | access 2026-11-08 | https://platform.claude.com/docs/en/about-claude/pricing | Applies to Claude 4.6+ except Haiku 5.5 |
| Tokenizer caveat | about 30% more tokens for the same text | n/a | access 2026-11-08 | same | Applies to Claude 4.7 and later. Matters when comparing per-token prices across vendors |

### 1b. OpenAI

| Item | Value | Unit | As-of date | Source URL | Notes/uncertainty |
|---|---|---|---|---|---|
| gpt-6-astra (flagship) input / cached / output | 10.00 / 1.00 / 50.00 | USD/MTok | access 2026-11-08 | https://developers.openai.com/api/docs/pricing | Models page calls Astra the flagship model: https://developers.openai.com/api/docs/models |
| gpt-6.1-sol input / cached / output | 2.00 / 0.10 / 10.00 | USD/MTok | access 2026-11-08 | same | Described as near-Astra performance at a lower cost |
| gpt-6.1-terra input / cached / output | 1.00 / 0.05 / 5.00 | USD/MTok | access 2026-11-08 | same | |
| gpt-6-luna (cheapest) input / cached / output | 0.10 / 0.01 / 0.50 | USD/MTok | access 2026-11-08 | same | "Our most efficient model for focused, high-volume tasks." No nano/mini text models are currently listed |
| Batch prices (Astra / Sol / Terra / Luna), input/output | 5/25; 1/5; 0.50/2.50; 0.05/0.25 | USD/MTok | access 2026-11-08 | same | The table sits under a "Flex/Batch" tab and is unlabelled; reading it as batch is my inference. It is exactly 50% of standard |
| Priority tier | renamed "Fast mode" on 2026-07-30 | n/a | access 2026-11-08 | same | |
| Regional processing uplift | 10 | % | access 2026-11-08 | same | For models released on or after 2026-03-05 |

### 1c. Google Gemini (paid tier, Standard)

| Item | Value | Unit | As-of date | Source URL | Notes/uncertainty |
|---|---|---|---|---|---|
| Gemini 3.1 Pro Preview input / output (≤200K prompt) | 2.00 / 12.00 | USD/MTok | page "Last updated 2026-06-02 UTC"; access 2026-11-08 | https://ai.google.dev/gemini-api/docs/pricing | >200K prompt: $4 / $18. No GA 3.x Pro or 3.5 Pro is listed |
| Gemini 3.1 Pro Preview context caching | 0.20 (+ $4.50/MTok/hr storage) | USD/MTok | same | same | |
| Gemini 3.1 Pro Preview batch input / output | 1.00 / 6.00 | USD/MTok | same | same | |
| Gemini 3.5 Flash input / output | 1.50 / 9.00 | USD/MTok | same | same | Newest Flash and much dearer than older Flash. Cache $0.15 + $1/MTok/hr. Batch $0.75 / $4.50. Has a free tier |
| Gemini 3 Flash Preview input / output | 0.50 / 3.00 | USD/MTok | same | same | Text, image and video input |
| Gemini 3.1 Flash-Lite input / output | 0.25 / 1.50 | USD/MTok | same | same | Cache $0.025. Batch $0.125 / $0.75 |
| Gemini 2.5 Flash-Lite input / output | 0.10 / 0.40 | USD/MTok | same | same | Older, cheapest text model |
| Gemini Batch API discount | 50 | % | same | same | Page wording: "Batch API (50% cost reduction)" |

---

## 2. Hosted open-weight inference (serverless, USD per 1M tokens, input / output)

| Item | Value | Unit | As-of date | Source URL | Notes/uncertainty |
|---|---|---|---|---|---|
| **~8B dense** DeepInfra Meta-Llama-3.1-8B-Instruct-Turbo | 0.02 / 0.04 | USD/MTok | access 2026-11-08 | https://deepinfra.com/pricing | 128K context. The page does not say what "Turbo" quantization is used |
| ~8B dense: DeepInfra Qwen3.5-9B | 0.10 / 0.15 | USD/MTok | access 2026-11-08 | https://deepinfra.com/pricing | Architecture not stated on the fetched page |
| ~8B dense: Together Qwen3.5 9B | 0.17 / 0.25 | USD/MTok | access 2026-11-08 | https://docs.together.ai/docs/serverless-models | Llama 3.1 8B and Qwen3 8B are not in Together's serverless catalog |
| ~8B dense: Fireworks generic tier 4B–16B | 0.20 (input = output) | USD/MTok | access 2026-11-08 | https://docs.fireworks.ai/serverless/pricing | Applies to models not listed individually. The Llama 3.1 8B model page says serverless is "Not Supported" |
| **~30B MoE** DeepInfra Qwen3-30B-A3B | 0.12 / 0.50 | USD/MTok | access 2026-11-08 | https://deepinfra.com/pricing | 40K context. Qwen3.6-35B-A3B on DeepInfra is $0.10 / $0.95 |
| ~30B MoE: Fireworks NVIDIA Nemotron 3.5 Lightning 30B A3B | 0.05 / 0.20 | USD/MTok | access 2026-11-08 | https://fireworks.ai/pricing | A different model in the same class. The Qwen3-30B-A3B page contradicts itself on serverless |
| ~30B MoE: Fireworks generic tier "MoE up to 56B" | 0.50 | USD/MTok | access 2026-11-08 | https://docs.fireworks.ai/serverless/pricing | Generic tier |
| ~30B MoE: Together | not found | n/a | access 2026-11-08 | https://docs.together.ai/docs/serverless-models | No ~30B A3B MoE is listed serverless (Muse Glimmer 30B is $0.35 / $1.50, architecture not stated) |
| **~70B dense** DeepInfra Llama-3.3-70B-Instruct-Turbo | 0.10 / 0.32 | USD/MTok | access 2026-11-08 | https://deepinfra.com/pricing | 128K context |
| ~70B dense: Together Llama 3.3 70B | 1.04 / 1.04 (listed spec) | USD/MTok | access 2026-11-08 | https://www.together.ai/models/llama-3-3-70b | The page says the model is not on Together's Serverless API, so the listed price may not apply |
| ~70B dense: Fireworks Llama 3.3 70B | not found (serverless "Not Supported") | n/a | access 2026-11-08 | https://fireworks.ai/models/fireworks/llama-v3p3-70b-instruct | On-demand GPUs only. Generic ">16B" tier is $0.90 |
| ~70B dense: Groq Llama 3.3 70B Versatile | "ContactSales" | n/a | access 2026-11-08 | https://console.groq.com/docs/models | 280 tok/s listed speed. No list price is published |
| **Frontier open** DeepSeek V4.1 Flash, Together / Fireworks / DeepInfra | 0.30/1.20; 0.30/1.20; 0.20/0.60 | USD/MTok | access 2026-11-08 | https://www.together.ai/pricing ; https://fireworks.ai/pricing ; https://deepinfra.com/pricing | DeepInfra shows a discounted price (e.g. $0.14 / $0.42); I used the list price |
| Frontier open: DeepSeek V4 Pro, Together / DeepInfra | 1.32/3.96; 1.30/2.60 | USD/MTok | access 2026-11-08 | https://docs.together.ai/docs/serverless-models ; https://deepinfra.com/pricing | Together lists "V4 Pro 0813" |
| Frontier open: Kimi K3, Together / Fireworks / DeepInfra | 3.00/15.00; 3.00/15.00; 2.85/14.25 | USD/MTok | access 2026-11-08 | same three pricing pages | Kimi K3.1 Pro on Together is $2.00 / $10.00 |
| Frontier open: GLM-5.3, Together / Fireworks / DeepInfra | 1.40/4.40; 1.40/4.40; 0.90/4.00 | USD/MTok | access 2026-11-08 | same three pricing pages | DeepInfra also shows 38%-off pricing ($0.5625 / $2.50) |
| gpt-oss-120B, Together / Fireworks / Groq | 0.15/0.60 (all three) | USD/MTok | access 2026-11-08 | https://www.together.ai/pricing ; https://fireworks.ai/pricing ; https://console.groq.com/docs/models | |
| Fireworks batch inference discount | 50 | % off input and output | access 2026-11-08 | https://fireworks.ai/pricing | |

---

## 3. Fine-tuning prices and adapter serving

| Item | Value | Unit | As-of date | Source URL | Notes/uncertainty |
|---|---|---|---|---|---|
| Fireworks LoRA SFT, ≤16B | 0.50 | USD per 1M training tokens | access 2026-11-08 | https://fireworks.ai/pricing | LoRA DPO $1.00; full SFT $1.00; full DPO $2.00. Training tokens are about dataset tokens × epochs |
| Fireworks LoRA SFT, 16.1B–80B | 3.00 | USD/1M training tokens | access 2026-11-08 | same | LoRA DPO $6; full SFT $6 |
| Fireworks LoRA SFT, 80B–300B | 6.00 | USD/1M training tokens | access 2026-11-08 | same | |
| Fireworks LoRA SFT, >300B | 10.00 | USD/1M training tokens | access 2026-11-08 | same | |
| Together SFT ~0.8B–9B (Table 1 / Table 2) | 0.34 / 0.38 | USD/1M training tokens | access 2026-11-08 | https://www.together.ai/pricing | The page has "LoRA" and "Full" tabs but the fetch could not tie the tables to the tabs. Table 1 is probably LoRA: it is cheaper and includes the large MoEs. DPO $0.84 / $0.94 |
| Together SFT ~27–35B (T1 / T2) | 1.05 / 1.16 | USD/1M training tokens | access 2026-11-08 | same | Same caveat |
| Together SFT 70B (Llama 3.3 70B) (T1 / T2) | 2.03 / 2.24 | USD/1M training tokens | access 2026-11-08 | same | |
| Together SFT very large (Kimi K2.6, GLM-5.x) | 15.00–40.00 | USD/1M training tokens | access 2026-11-08 | same | Table 1 only. Minimum charge per job is $4–$60 depending on model |
| Together token-count rule | epochs × train tokens + evals × validation tokens | n/a | access 2026-11-08 | https://docs.together.ai/docs/fine-tuning-pricing | $4 minimum. Failed jobs refunded |
| OpenAI fine-tuning status | being wound down; closed to new users | n/a | access 2026-11-08 | https://developers.openai.com/api/docs/pricing | Fine-tuned models stay available until their base models are deprecated |
| OpenAI gpt-4.1-mini training | 5.00 | USD (unit not stated; section says "per 1M tokens") | access 2026-11-08 | same | gpt-4.1 training is $25.00 |
| OpenAI fine-tuned gpt-4.1-mini inference, input / cached / output | 0.80 / 0.20 / 3.20 | USD/MTok | access 2026-11-08 | same | Batch: $0.40 / $0.10 / $1.60 |
| Tinker (Thinking Machines) Qwen3-8B, 32K ctx: prefill / sample / train | 0.195 / 0.60 / 0.44 | USD/MTok | page "as of October 19, 2026" | https://thinkingmachines.ai/tinker/pricing/ | Cached prefill $0.039. Storage $0.10/GB/month |
| Tinker Qwen3.6-35B-A3B, 64K: prefill / sample / train | 0.45 / 1.335 / 1.177 | USD/MTok | same | same | 256K context: train $2.14 |
| Tinker Kimi-K2.6, 32K: prefill / sample / train | 2.205 / 5.49 / 4.84 | USD/MTok | same | same | Kimi-K3.1-Pro (256K): $6.00 / $25.00 / $32.00 |
| Tinker DeepSeek-V4.1-Flash (256K): prefill / sample / train | 0.60 / 3.60 / 4.98 | USD/MTok | same | same | |
| Fireworks Serverless Training API, Qwen 3.6 35B A3B: prefill / sample / train | 0.45 / 1.335 / 2.14 | USD/MTok | access 2026-11-08 | https://fireworks.ai/pricing | Pay-per-token RL/SFT loop similar to Tinker. Repeated prefixes billed 80% off |
| **Together serverless LoRA inference** | Discontinued | n/a | access 2026-11-08 | https://docs.together.ai/docs/lora-inference | Page says Serverless Multi-LoRA "has been discontinued". Adapters run on dedicated endpoints; several adapters can share one base deployment |
| Together dedicated inference H100 / B200 | 5.49 / 8.99 | USD per GPU-hour | access 2026-11-08 | https://www.together.ai/pricing | H200 is "Contact us" |
| **Fireworks fine-tuned model serving** | Conflicting docs | n/a | access 2026-11-08 | https://docs.fireworks.ai/fine-tuning/fine-tuning-models ; https://docs.fireworks.ai/fine-tuning/deploying-loras | The fine-tuning doc says on-demand deployment is "the only supported method for serving trained models". The LoRA doc says select-base LoRAs can be served serverless and billed per token. Neither says the per-token rate equals the base model's |
| Fireworks multi-LoRA pricing claim (historical) | same price as base model | n/a | 2024-09-18 | https://fireworks.ai/blog/multi-lora | 2024 blog: "Inference prices are the same as a single base model". Not restated in current docs |
| Fireworks: extra cost to serve a trained model? | No; pay GPU-seconds plus the training job | n/a | access 2026-11-08 | https://fireworks.ai/pricing (FAQ) | |

---

## 4. GPU rental prices (on-demand, USD per GPU-hour)

| Item | Value | Unit | As-of date | Source URL | Notes/uncertainty |
|---|---|---|---|---|---|
| **Silicon Data SDH100RT (H100 index)** | 3.00 | USD/GPU-hr | 2026-11-08 | https://www.silicondata.com/products/silicon-index/h100 | Neocloud and marketplace tier. The page shows +6.4% 7-day change, but the table implies about +2.4%. The page FAQ says $2.53, which conflicts with the headline; I used the headline and table. Table range: 2.81 (Oct 15) to 3.04 (Nov 7) |
| SDH100RT history | 2.22–2.26 (Aug 2025); low 2.13 (mid-Sep 2025); about 2.16 (end Sep 2025) | USD/GPU-hr | article 2025-10-20 | https://www.silicondata.com/blog/h100-rental-market-update-september-2025 | |
| SDH100RT history | 2.57 (May 4 2026), 2.73 (end May), 2.75 (Jul 27 2026) | USD/GPU-hr | article 2026-08-04 | https://www.silicondata.com/blog/h200-vs-h100-rental-prices-may-july-2026 | +7.0% from May to July |
| H100 trend, Sep 2025 to Nov 2026 | about +39% (2.16 to 3.00) | % | [derived] | above two sources | |
| Silicon Data H100 Hyperscaler on-demand index | 7.40–7.52 (close 7.48 on Apr 20 2026) | USD/GPU-hr | article 2026-04-27 | https://www.silicondata.com/blog/h100-hyperscaler-index-april-2026 | About a 3x premium over the neocloud tier |
| Silicon Data H200 index | 3.55 | USD/GPU-hr | 2026-11-08 | https://www.silicondata.com/products/silicon-index/h200 | Was $3.43 on 2026-10-10. May–Jul 2026: 2.70 to 3.09 (+14.4%) |
| Silicon Data B200 index (SDB200RT) | 6.81 | USD/GPU-hr | 2026-11-08 | https://www.silicondata.com/products/silicon-index/b200 | Was $5.93 on 2026-10-10 and 5.62 on Sep 1 2026 ("+27.6% YTD"; https://www.silicondata.com/blog/b200-rental-price-august-2026-update). FAQ says $5.45, which conflicts |
| Silicon Data A100 index (SDA100RT) | 1.59 | USD/GPU-hr | 2026-11-08 | https://www.silicondata.com/products/silicon-index/a100 | Was $1.53 on 2026-10-10. FAQ says $1.29, which conflicts |
| Ornn OCPI H100, 1-year contracts | 1.70 (Oct 2025) to 2.35 (Mar 2026), about +40% | USD/GPU-hr | article about late May 2026 | https://cryptobriefing.com/h100-rental-prices-slide-ornn-ai/ | **[secondary]** News report on the Ornn index. On-demand/spot range was $1.38–$11.01 in mid-May 2026. Latest Ornn value not found |
| Lambda HGX B200 | 7.69 | USD/GPU-hr | prices effective 2026-11-16 | https://lambda.com/pricing | The page shows the new prices only; current pricing requires login. Plus tax |
| Lambda HGX H100 (SXM) | 4.59 | USD/GPU-hr | effective 2026-11-16 | same | |
| Lambda HGX A100 80GB / 40GB | 3.19 / 1.99 | USD/GPU-hr | effective 2026-11-16 | same | No H200 or L40S listed |
| RunPod H100 SXM | 3.99 (Secure); from 2.69 (Community) | USD/GPU-hr | page "Updated November 4, 2026" | https://www.runpod.io/pricing | H100 PCIe: $2.89 / from $1.99. H100 NVL: $3.19 |
| RunPod H200 | 5.29 (Secure); from 3.59 (Community) | USD/GPU-hr | 2026-11-04 | same | |
| RunPod B200 | 7.99 | USD/GPU-hr | 2026-11-04 | same | |
| RunPod A100 80GB SXM / PCIe | 1.79 / 1.79 (Secure); from 1.39 / 1.19 (Community) | USD/GPU-hr | 2026-11-04 | same | |
| RunPod L40S | 1.09 | USD/GPU-hr | 2026-11-04 | same | RTX 4090 $0.89; RTX 5090 $1.19. Serverless H100 $5.39/hr |
| CoreWeave HGX H100 (8 GPUs) | 49.24/instance-hr, so about 6.16 per GPU | USD | access 2026-11-08 | https://www.coreweave.com/pricing | Per-GPU figure is [derived] (÷8) |
| CoreWeave HGX H200 / B200 | about 6.31 / 8.60 (50.44 / 68.80 per 8-GPU instance) | USD/GPU-hr | access 2026-11-08 | same | [derived] ÷8. GB200 NVL72: $42.00 per 4 GPUs, about $10.50/GPU |
| CoreWeave A100 80GB / L40S | about 2.70 / 2.25 (21.60 / 18.00 per 8-GPU instance) | USD/GPU-hr | access 2026-11-08 | same | [derived] |
| Nebius HGX H100 / H200 / B200 / B300 | 4.50 / 5.40 / 8.50 / 9.50 | USD/GPU-hr | access 2026-11-08 | https://nebius.com/prices | L40S from $1.55 (Intel) or $1.82 (AMD). No A100 |
| Together GPU clusters H100 / H200 / B200 | 3.99 / 5.99 / 8.19 | USD/GPU-hr | access 2026-11-08 | https://www.together.ai/pricing | Preemptible: 1.99 / 2.99 / 4.09 |
| DeepInfra GPUs A100 80GB / H100 / H200 / B200 | 1.49 / 2.90 / 3.49 / 4.89 | USD/GPU-hr | access 2026-11-08 | https://deepinfra.com/pricing | Listed under "Custom LLMs" |
| Fireworks GPUs H100 / H200 / B200 | 8.00 / 8.00 / 13.00 | USD/GPU-hr | access 2026-11-08 | https://fireworks.ai/pricing | One rate card for serving and training, billed per second. Region-restricted deployments cost 1.5x |
| Vast.ai marketplace | not found | n/a | access 2026-11-08 | https://vast.ai/pricing | The live price grid did not render in the fetch |

---

## 5. Inference throughput benchmarks

| Item | Value | Unit | As-of date | Source URL | Notes/uncertainty |
|---|---|---|---|---|---|
| Llama 3.1 8B FP8, 1x H100 SXM, TensorRT-LLM, ISL/OSL 1000/1000 | 14,991.62 | output tok/s (1 GPU) | docs v1.1.0rc5, updated 2025-09-15 | https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html | Offline max-throughput run: infinite request rate, about 3,000 requests, KV cache 0.95. Batch size not fixed (CUDA graphs up to 8192) |
| same, other ISL/OSL | 128/128: 26,401; 128/2048: 21,413; 500/2000: 17,571; 2048/2048: 9,462 | output tok/s | same | same | H200 TP1, same order plus 1000/1000: 27,028 / 23,102 / 19,759 / 11,822; 1000/1000: 17,162 |
| Llama 3.3 70B FP8, 2x H100 (TP2), TRT-LLM, 1000/1000 | 4,181.06 total, about 2,090 per GPU | output tok/s | 2025-09-15 | same | 128/2048: 5,893 total; 2048/2048: 2,786 total |
| Llama 3.3 70B FP8, H100 TP2, TRT-LLM 1.4.0rc3, 1000/1000 | 2,209 | output tok/s **per GPU** | updated 2026-11-03 | https://nvidia.github.io/TensorRT-LLM/developer-guide/perf-overview.html | 8192/1024: 398/GPU. H200 TP2: 2,587/GPU. B200 FP4 TP1: 6,920/GPU |
| GPT-OSS 20B FP8, 1x H100, TRT-LLM 1.4.0rc3, 1000/1000 | 11,557 | output tok/s/GPU | 2026-11-03 | same | The newest doc has no ~8B dense model. Qwen3 30B A3B appears only at FP4 on Blackwell (B200: 26,971/GPU) |
| Llama 3.1 70B FP8, 4x H100 (TP4), TRT-LLM, 1000/1000 | 4,141.52 total, about 1,035 per GPU | tok/s | undated | https://docs.e2enetworks.com/docs/tir/benchmarks/inference_benchmarks/ | **[secondary]** Vendor docs. No engine version or date. Units only implied |
| Llama 3.1 8B, 1x H100, vLLM serving (default config) | 128:128 at 64 req/s rate gives 8,399 tok/s; 2024:128 gives 19,127 tok/s | tok/s (appears to include input tokens) | undated | same | **[secondary]** Request-rate limited, not saturated. Throughput seems to be input+output (e.g. 8.95 req/s × 2,152 tok ≈ 19.3K). Low confidence |
| Llama 3.1 70B, 4x H100, vLLM serving | 128:128 at 24 req/s gives 3,438 tok/s | tok/s | undated | same | **[secondary]** Same caveats |
| Llama 3.3 70B FP8, H100, 8k/1k, InferenceX | 1,847.6 tok/s/chip at 38 tok/s/user (concurrency about 20); 877.4 at 78 tok/s/user | tok/s/GPU | access 2026-11-08 | https://inferencex.semianalysis.com/compare/llama-3-3-70b-h100-vs-h200 | **[secondary]** The page calls itself an "Unofficial deployment", "Interpolated from real benchmark data". No framework or date. B200: 6,519 at 38 tok/s/user |
| Llama 3.3 70B FP8, **1x** H100, vLLM / TRT-LLM / SGLang, 512 in / 256 out | concurrency 100: 2,400 / 2,780 / 2,460 | output tok/s | 2026-03-23, updated 2026-11-02 | https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks | **[secondary]** Vendor blog. vLLM v0.18.0, TRT-LLM v1.2.0, SGLang v0.5.9. Round numbers; treat with caution |
| vLLM official (v0.6.0) Llama 3 8B 1xH100 and 70B 4xH100 | not found as numbers (charts only) | n/a | 2024-09-05 | https://vllm.ai/blog/2024-09-05-perf-update | ShareGPT: 202 in / 179 out avg, 500 prompts, QPS inf |
| vLLM CI Llama 8B TP1 H100 (ShareGPT, 200 prompts) | 19.50 | requests/s | undated build | https://buildkite.com/vllm/performance-benchmark/builds/9565 | tok/s not reported; precision not stated |
| SGLang official Llama 3.1 benchmarks | not found as numbers (charts only) | n/a | 2024-07-25 | https://lmsys.org/blog/2024-07-25-sglang-llama3/ | Text gives only "up to 5000 tokens per second" for 8B on A100 with short inputs |
| Self-host cost, 8B on H100 at SDH100RT $3.00 | about 0.056 | USD per 1M output tokens | [derived] | from rows above | $3.00 / (14,992 × 3600 / 1e6). Assumes 100% utilisation; input tokens are processed on top (1:1) |
| Self-host cost, 70B on H100 TP2 at $3.00 | about 0.38 | USD per 1M output tokens | [derived] | from rows above | $3.00 / (2,209 × 3600 / 1e6) |

---

## 6. Labor

| Item | Value | Unit | As-of date | Source URL | Notes/uncertainty |
|---|---|---|---|---|---|
| US ML/AI software engineer median total compensation (Levels.fyi) | 251,250 | USD/yr | FAQ labelled 2026; access 2026-11-08 | https://www.levels.fyi/t/software-engineer/focus/ml-ai | Percentiles did not render. Self-reported and skewed toward big tech |
| US computer and information research scientists median wage (BLS) | 140,300 | USD/yr (base wage) | May 2025 data; page modified 2026-08-27 | https://www.bls.gov/ooh/computer-and-information-technology/computer-and-information-research-scientists.htm | Wage only, no equity or benefits. Projected growth 22% (2025–35) |
| US software developers median wage (BLS) | 135,980 | USD/yr | May 2025; modified 2026-08-27 | https://www.bls.gov/ooh/computer-and-information-technology/software-developers.htm | |
| US ML engineer hourly equivalent | about 121 | USD/hr | [derived] | Levels.fyi row | $251,250 / 2,080 h. No overhead or benefits loading |
| South Africa ML engineer average base (PayScale) | 427,293 | ZAR/yr | updated 2026-07-20 | https://www.payscale.com/research/ZA/Job=Machine_Learning_Engineer/Salary | Only 50 profiles. The 10–90th percentile range (R12k–R842k) looks noisy |
| South Africa software engineer average base (PayScale) | 410,649 | ZAR/yr | updated 2026-10-22 | https://www.payscale.com/research/ZA/Job=Software_Engineer/Salary | 655 profiles. Total pay R139k–R819k |
| South Africa software engineer median total comp (Levels.fyi) | 658,057 | ZAR/yr | updated 2026-11-08 | https://www.levels.fyi/t/software-engineer/locations/south-africa | 91 salaries. 25th pct R475K, 75th R945K, 90th R1.33M |
| South Africa ML engineer median total pay (Glassdoor) | 49,000 | ZAR/month | no date | https://www.glassdoor.com/Salaries/south-africa-machine-learning-engineer-salary-SRCH_IL.0,12_IN211_KO13,38.htm | 50 salaries. Range R37K–R54K/month. The page labels its high/low figures "per year", an internal inconsistency |
| Botswana software developer average base (PayScale) | 100,000 | BWP/yr | updated 2026-06-02 | https://www.payscale.com/research/BW/Job=Software_Developer/Salary | Profile count not shown. Ranges look degenerate (P0–P100k). Low confidence |
| Botswana software engineer average base (PayScale) | 25,000 | BWP (labelled annual) | updated 2025-01-05 | https://www.payscale.com/research/BW/Job=Software_Engineer/Salary | Only 10 profiles; implausibly low for an annual figure. Treat as unreliable |
| Botswana software engineer, middle 80% (Paylab) | 4,707–22,003 | BWP/month gross | no date | https://www.paylab.com/bw/salaryinfo/information-technology/software-engineer?lang=en | Fewer than 20 verified respondents. No median given |
| Botswana ML engineer salary | not found | n/a | access 2026-11-08 | n/a | |

---

## 7. Compute for a typical LoRA fine-tune

| Item | Value | Unit | As-of date | Source URL | Notes/uncertainty |
|---|---|---|---|---|---|
| QLoRA Llama 3.1 8B (Unsloth), 100k samples (FineTome-100k), 1 epoch, seq 2048, r=16, effective batch 16, packing on | 4 h 45 min | wall-clock on 1x A100 40GB (Colab) | 2024-07-29 | https://huggingface.co/blog/mlabonne/sft-llama3 | Author estimates about 19h40m on an L4 and about 47 h on a T4 |
| Same, scaled to 10k–50k samples | about 0.5–2.4 | A100-40GB GPU-hours | [derived] | above | Assumes linear scaling with sample count |
| QLoRA paper: 65B model on one 48GB GPU | feasible | n/a | 2023-05-23 | https://arxiv.org/abs/2305.14314 | Abstract claim. Guanaco reached "99.3% of the performance level of ChatGPT" with 24 h of finetuning on a single GPU |
| QLoRA paper: Guanaco training data and schedule | OASST1, 9,209 examples; 1,875 steps; batch 16; r=64 | n/a | 2023 | https://arxiv.org/pdf/2305.14314 | The 33B model trains on a 24GB consumer GPU in under 12 h. No 7B/13B timings reported; GPU model not named |
| NeMo Llama3-8B SFT, 8x H100, FP8, seq 4096 packed | 18,618 | training tokens/s per GPU | NeMo 25.07 docs, updated 2026-01-13 (archived) | https://docs.nvidia.com/nemo-framework/user-guide/25.07/performance/performance-summary.html | Full SFT, not LoRA (no 8B LoRA row on H100). 70B LoRA on 8x H100: 2,754 tok/s/GPU |
| Implied GPU time for 25M training tokens on 8B (NeMo rate) | about 0.37 | H100 GPU-hours | [derived] | above | 25e6 / 18,618 / 3600. Optimised large-scale rate; small single-GPU jobs will be slower |
| Unsloth benchmark claims | 2x faster, >70% less VRAM vs HF + FA2 (Llama 3.1 8B) | n/a | undated | https://unsloth.ai/docs/basics/unsloth-benchmarks | Alpaca, batch 2, grad accum 4, r=32. No wall-clock times given |
| Managed LoRA cost for 25M training tokens on ≤16B | Fireworks about $12.50; Together about $8.50 | USD | [derived] | sections 3 rows | Example only: 25k examples × 1,000 tokens × 1 epoch |

---

## 8. Tokens per request (production usage)

| Item | Value | Unit | As-of date | Source URL | Notes/uncertainty |
|---|---|---|---|---|---|
| OpenRouter average prompt tokens per request | about 1.5K (early 2024) to over 6K (late 2025) | tokens | arXiv v1, 2026-01-15 | https://arxiv.org/html/2601.10088 | Section 4.3, Fig. 14: "increased roughly fourfold" |
| OpenRouter average completion tokens per request | about 150 to 400 | tokens | same | same | Section 4.3, Fig. 15. Growth mostly from reasoning tokens |
| Average sequence length (prompt + completion) | under 2,000 (late 2023) to over 5,400 (late 2025) | tokens | same | same | Section 4.4. The paper's start-date wording is inconsistent with Fig. 14 |
| Programming prompts vs general | 3–4x longer; code tasks "routinely exceed 20K input tokens" | n/a | same | same | Figs. 16 and 18 |
| Share of tokens via reasoning models | over 50% | % of tokens | same | same | Fig. 10. This is tokens routed to reasoning models, not reasoning tokens within outputs |
| Open-weight share of OpenRouter tokens | about one-third (about 30%) by late 2025 | % | same | same | Chinese open models averaged about 13.0% of weekly tokens |
| Cost per average request (6,000 in / 400 out) | Sonnet 5.5 $0.016; Opus 5.5 $0.032; Haiku 5.5 $0.0008; gpt-6-luna $0.0008; DeepInfra Llama 3.3 70B $0.00073; DeepInfra Llama 3.1 8B $0.000136 | USD/request | [derived] | sections 1–2 and this section | List prices, no caching. Caching would cut the input share sharply (Opus/Sonnet 5.5 cache hits are 0.05x) |

---

## Not found / gaps

- No Together or Fireworks serverless price exists for Llama 3.1 8B, Llama 3.3 70B or Qwen3-30B-A3B; both providers have dropped these models from serverless. DeepInfra is the only one of the three still serving them.
- Neither Together nor Fireworks current docs say whether serverless LoRA inference is billed at the base-model price. Together has discontinued serverless LoRA. Fireworks' docs conflict; only its 2024 blog says adapters are billed at the base-model price.
- I could not get vLLM or SGLang official throughput as numbers; they appear only in charts. TensorRT-LLM docs are the primary throughput source used here.
- No latest Ornn OCPI value was found on a primary page. No Vast.ai prices were found (dynamic page).
- No Botswana ML-engineer salary was found. Botswana software-engineer data is sparse and inconsistent.
- No currency conversions (ZAR, BWP to USD) were applied.
- OpenAI fine-tuning training price units are not stated in the table (the section header says "per 1M tokens").
