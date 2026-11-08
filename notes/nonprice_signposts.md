# Non-price reasons to own, and forecasting signposts

Research notes for the model-ownership paper. Compiled 2026-11-08.

How to read these notes:
- **Verified** means the page was opened in this pass and the fact was read from it.
- **Secondary** means a third party reported it and the primary source was not opened (or could not be).
- **Not verified** means the claim could not be confirmed from an opened page. Do not cite these without checking.
- Day counts and percentages marked "computed" are my arithmetic from the cited dates and figures, not the source's.
- Quotes are kept under 15 words. Everything else is paraphrased.

---

## Key takeaways

1. **API models have a shelf life of months, and fine-tunes on closed bases die with their base.** OpenAI guarantees at least 6 months' notice for GA models and 3 months for specialised variants. Anthropic guarantees at least 60 days. Most Anthropic retirements in 2025-26 came 60-76 days after notice. OpenAI says fine-tuned models shut down when their base model does. ([OpenAI deprecations](https://developers.openai.com/api/docs/deprecations), [Anthropic deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations))
2. **OpenAI is winding down self-serve fine-tuning.** New organisations were cut off on 2026-05-07, and existing customers lose job creation on 2027-01-06. Its only RFT base model, o4-mini, was scheduled to shut down on 2026-10-23. This is the clearest example yet of a rented adaptation path being withdrawn. ([OpenAI deprecations](https://developers.openai.com/api/docs/deprecations))
3. **A government directive took a commercial frontier model offline for every customer.** On 2026-06-12 a US export-control directive restricted foreign-national access to Claude Fable 5 and Mythos 5. Anthropic disabled both models for all users. Fable 5 returned globally on 2026-07-01, 19 days later (computed), with a new classifier. Mythos 5 was restored only for a set of US organisations. ([Anthropic, 12 Jun](https://www.anthropic.com/news/fable-mythos-access), [Anthropic, 30 Jun](https://www.anthropic.com/news/redeploying-fable-5))
4. **No major LLM API offers in-country inference in southern Africa.** OpenAI processes data regionally only in the US, the EU and the UAE. Anthropic offers only "global" or "us". Bedrock in Cape Town serves Claude through global cross-Region inference, so computation may happen outside South Africa. For Botswana and South Africa, the only in-country option for capable models is open weights on local or in-region compute. ([OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data), [Claude data residency](https://platform.claude.com/docs/en/build-with-claude/data-residency), [AWS blog](https://aws.amazon.com/blogs/machine-learning/scale-ai-in-south-africa-using-amazon-bedrock-global-cross-region-inference-with-anthropic-claude-4-5-models/))
5. **Open-weight frontier models are plentiful in 2026, but flagship licences are tightening, mostly against hosting competitors.** GLM-5.3, Qwen3.8-2.4T, Kimi K3.1-Pro, Muse Spark Open and Mistral Large 4 use custom "modified MIT"-style licences. These add revenue-threshold conditions on model-as-a-service operators, while internal use stays broadly permitted. Moves the other way: Google put Gemma 4 under Apache 2.0, and Meta returned to open weights (Muse Glimmer, Muse Spark Open) after releasing Muse Spark closed. (Section 4)
6. **The quality gap is small on chat-arena metrics and larger on agentic tasks.** The Stanford AI Index 2026 chart has the top closed model at 1,503 and the top open model at 1,454 on Arena, about 3.3% apart (computed). Mozilla reports a 15.5-point gap on Terminal-Bench 2.1. ([Stanford HAI](https://hai.stanford.edu/ai-index/2026-ai-index-report/technical-performance), [UC Today on Mozilla](https://www.uctoday.com/productivity-automation/mozilla-state-of-open-source-ai-2026-report/))
7. **The memory crunch is raising the cost of owning hardware and, more recently, of renting it.** TrendForce forecast conventional DRAM contract prices up 55-60% QoQ in 1Q26, and projects 2027 HBM blended ASP up 121% YoY. Nvidia reportedly plans server price rises of more than 15% for early-2027 shipments. Silicon Data's H100 rental index reads $3.00/hr on 2026-11-08, up from about $2.37 in May 2025. ([TrendForce Jan](https://www.trendforce.com/presscenter/news/20260105-12860.html), [TrendForce Sep](https://www.trendforce.com/presscenter/news/20260929-13255.html), [Silicon Data](https://www.silicondata.com/products/silicon-index/h100))
8. **Managed adaptation on open bases is growing, and it lets an organisation own its adapters without running infrastructure.** Tinker went GA on 2025-12-12. It charges per token, with checkpoints at $0.10/GB-month that users can download. Its run-rate is reportedly above $100M (secondary). Menlo's 2026 survey finds 10% of AI builders post-train, up from 6%, and 91% of them use outside partners. ([Tinker](https://thinkingmachines.ai/tinker/), [Menlo 2026](https://menlovc.com/perspective/2026-the-state-of-enterprise-ai/))
9. **Usage and spending point in different directions.** In Menlo's estimate, open-source models are 19% of enterprise tokens (up from 11%) but only 4% of spend (down from 6%). a16z finds about 80% of large enterprises are comfortable hosting models directly (up from about 40% in March 2024), yet over a third prefer closed models. ([Menlo 2026](https://menlovc.com/perspective/2026-the-state-of-enterprise-ai/), [a16z](https://www.a16z.news/p/leaders-gainers-and-unexpected-winners))
10. **Derivative activity on Hugging Face is large and rising.** Public model repos grew from 2.43M to 2.96M in the first seven months of 2026. Qwen alone has 151,448 derivatives. In a June 2025 snapshot, 79% of derived models were fine-tunes, including adapters. ([HF State of Open Models](https://huggingface.co/blog/state-of-open-models-summer-2026), [Chen et al., ICSE '26](https://chenbihuan.github.io/paper/icse26-chen-model-supply-chain.pdf))

---

## 1. Model deprecation and behavioural stability

### 1.1 OpenAI

- **Fact:** OpenAI's stated notice periods are at least 6 months for generally available models and at least 3 months for specialised variants (chat, Codex, deep-research). "Preview" models may get much shorter notice, for example about 2 weeks. Safety or compliance concerns can shorten any timeline. `GET /v1/models` returns a `shutdown_date` field.
  - Source: https://developers.openai.com/api/docs/deprecations (Verified)
  - Date: page read 2026-11-08
  - Relevance: This is the formal ceiling on how long a renter can count on a fixed model. Six months is shorter than a typical enterprise validation and procurement cycle in regulated settings.

- **Fact:** Notable shutdowns, with notice periods computed from the listed dates:
  - gpt-4.5-preview: announced 2025-04-14, shut down 2025-07-14 (91 days)
  - o1-preview: announced 2025-04-28, shut down 2025-07-28 (91 days)
  - o1-mini: announced 2025-04-28, shut down 2025-10-27 (182 days)
  - Assistants API: announced 2025-08-26, shut down 2026-08-26 (365 days)
  - chatgpt-4o-latest: announced 2025-11-18, shut down 2026-02-17 (91 days)
  - Large legacy batch (gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4.1-nano, gpt-4o-2024-05-13, o1, o1-pro, o3-mini, o4-mini and others): announced 2026-04-22, shutdown 2026-10-23 (184 days)
  - Codex and chat variants (gpt-5-codex, gpt-5.1-codex family, gpt-5-chat-latest and others): announced 2026-04-22, shutdown 2026-07-23 (92 days)
  - gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-pro, o3, o3-pro: announced 2026-06-11, shutdown 2026-12-11 (183 days)
  - gpt-4o snapshots (2024-08-06, 2024-11-20), gpt-4o-mini, gpt-4.1, gpt-4.1-mini: announced 2026-10-14, shutdown 2027-01-14 (92 days). This looks shorter than the 6-month GA minimum, unless these were already "legacy". Check before citing.
  - gpt-5.4-cyber: announced 2026-09-11, shut down 2026-10-01 (20 days)
  - Source: https://developers.openai.com/api/docs/deprecations (Verified; day counts computed)
  - Date: entries 2025-04 to 2026-11
  - Relevance: gpt-5 was retired about 16 months after its August 2025 launch (launch date from general knowledge, not verified here). Flagship models now turn over in roughly 1-1.5 years, so any product built on one fixed snapshot needs re-validation about once a year.

- **Fact:** Fine-tuned models follow the shutdown schedule of their base models. OpenAI says inference on fine-tunes ends "when the underlying base model is deprecated." This applies to the GPT-5/o3 entry (2026-12-11), the GPT-4o/4.1 entry (2027-01-14) and the 2026-04-22 entry, which lists `ft-o4-mini-2025-04-16`.
  - Source: https://developers.openai.com/api/docs/deprecations (Verified)
  - Date: 2026
  - Relevance: This is the central asymmetry. An adapter or fine-tune on a closed base is not an asset the customer holds. It expires on the vendor's schedule.

- **Fact:** OpenAI's self-serve fine-tuning update has three stages:
  - From 2026-05-07, organisations that have never fine-tuned cannot create jobs.
  - From 2026-07-02, job creation also ends for organisations with no fine-tuned-model inference in the past 60 days.
  - From 2027-01-06, active existing customers can no longer create new jobs.
  - Community threads quote the notice as saying OpenAI is "winding down the fine-tuning API and platform." No official reason was found.
  - Sources: https://developers.openai.com/api/docs/deprecations (Verified); https://community.openai.com/t/openai-s-self-serve-fine-tuning-availability/1380481 (Verified, user post quoting the notice)
  - Date: 2026-05-07 onward
  - Relevance: This is direct evidence for a forecasting claim: renting adaptation from a closed-model vendor exposes customers to the vendor exiting that product line.

### 1.2 Anthropic

- **Fact:** Anthropic commits to at least 60 days' notice before retiring publicly released models. The lifecycle stages are Active, Legacy, Deprecated and Retired. Dates apply to the Claude API, Claude Platform on AWS, Anthropic-managed Claude on Google Cloud, and Microsoft Foundry. Bedrock and Google-managed Vertex set their own schedules.
  - Source: https://platform.claude.com/docs/en/about-claude/model-deprecations (Verified)
  - Date: page read 2026-11-08
  - Relevance: This is the shortest formal notice floor of the three labs. Cloud resellers run separate schedules, so one model can retire on different dates depending on the channel.

- **Fact:** Retirements, with notice periods computed:
  - claude-2.0, claude-2.1, claude-3-sonnet: notice 2025-01-21, retired 2025-07-21 (181 days)
  - claude-3-opus: notice 2025-06-30, retired 2026-01-05 (189 days)
  - claude-3-5-sonnet (both snapshots): notice 2025-08-13, retired 2025-10-28 (76 days)
  - claude-3-7-sonnet: notice 2025-10-28, retired 2026-02-19 (114 days)
  - claude-3-5-haiku: notice 2025-12-19, retired 2026-02-19 (62 days)
  - claude-3-haiku: notice 2026-02-19, retired 2026-04-20 (60 days)
  - claude-opus-4 and claude-sonnet-4: notice 2026-04-14, retired 2026-06-15 (62 days)
  - claude-opus-4-1: notice 2026-06-05, retired 2026-08-05 (61 days)
  - claude-sonnet-4-5: notice 2026-09-30, retirement 2026-11-30 (61 days)
  - claude-mythos-preview: deprecated 2026-06-09, retirement date not yet announced
  - Source: https://platform.claude.com/docs/en/about-claude/model-deprecations (Verified; day counts computed)
  - Date: 2025-2026
  - Relevance: Since late 2025, notice has clustered at the 60-day floor. Claude Sonnet 4.5 (released September 2025) gets about 14 months of availability. Claude Opus 4 and Sonnet 4 (released May 2025) got about 13.

- **Fact:** The API also changed behaviour, not just model availability. `temperature`, `top_p` and `top_k` are removed in API version `2026-10-21`, and requests that include them return a 400 error. On earlier API versions, non-default values were already rejected for Claude 4.7+ and Mythos Preview.
  - Source: https://platform.claude.com/docs/en/about-claude/model-deprecations (Verified)
  - Date: 2026-10-21
  - Relevance: Renters lose control of decoding parameters as well as of model versions. Owners keep full control of sampling.

- **Fact:** On 2025-11-04 Anthropic committed to preserve the weights of all publicly released models for at least "the lifetime of Anthropic as a company." It also committed to produce a post-deployment report and run model "interviews" before retirement. Anthropic says retirements are currently needed because serving costs scale with the number of models offered. It is exploring keeping select retired models publicly available.
  - Source: https://www.anthropic.com/research/deprecation-commitments (Verified)
  - Date: 2025-11-04
  - Relevance: Weights are preserved but not served. Customers get no continued access, so the commitment does not solve the renter's stability problem. Anthropic's own cost rationale supports the argument that vendors cannot afford long version tails.

### 1.3 Google (Gemini API / Vertex)

- **Fact:** The primary Gemini deprecations page could not be read: ai.google.dev redirects to aistudio.google.com/docs/deprecations, which returned only metadata, and the proxy blocked direct access. The Vertex model-versions page also returned only navigation. Google's formal notice policy is therefore **not verified** in this pass.
  - Sources tried: https://aistudio.google.com/docs/deprecations; https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions

- **Fact:** A third-party tracker that says it mirrors Google's page lists these 2026 retirements:
  - Gemini 2.0 Flash, 2.0 Flash-001, 2.0 Flash-Lite and 2.0 Flash-Lite-001: 2026-06-01
  - Gemini 3 Pro Preview: 2026-03-09
  - Gemini 2.5 Flash Preview 09-25: 2026-02-17
  - Gemini 2.5 Computer Use Preview: 2026-07-28
  - Several 2.5/3.1 TTS and Live previews: 2026-11-17
  - Source: https://aiengineerguide.com/tools/model-retirements/gemini/ (Secondary; data last changed 2026-11-02)
  - Relevance: Google's preview and GA models turn over at a similar pace to the other two labs.

- **Fact:** On Google's own developer forum on 2026-03-12, a user reported two conflicting Gemini 2.0 Flash discontinuation dates in Google's documentation: 2026-03-31 and 2026-06-01.
  - Source: https://discuss.ai.google.dev/t/gemini-2-0-flash-discontinuation-date/131389 (Verified; user post)
  - Relevance: Even the published shutdown dates have been unclear.

- **Fact:** On 2026-07-09, `gemini-2.5-flash` returned 404 "no longer available" errors for several hours, well before its published 2026-10-16 deprecation date. Users reported production outages. A Google staff member said the change "has been rolled back" and blamed a "config change issue."
  - Source: https://discuss.ai.google.dev/t/gemini-2-5-flash-deprecated-without-warning-earlier-than-shutdown-date/174217 (Verified)
  - Date: 2026-07-09
  - Relevance: This was a de facto early retirement caused by operational error. It shows renters carry tail risk beyond the published policy.

---

## 2. Availability shocks

### 2.1 Notable outages, 2025-2026

| Date | Provider | What happened | Duration | Source (status) | Relevance |
|---|---|---|---|---|---|
| 2025-06-10 | OpenAI | "Elevated error rates" across 10 API components (Chat Completions, Responses, Fine-tuning, Embeddings, Batch, Realtime and others), plus ChatGPT and Codex. Root-cause report was promised but not described on the page. | About 15h 24m (06:36 to 22:00 on the status page) | https://status.openai.com/incidents/01JXCAW3K3JAE0EP56AEZ7CBG3 (Verified) | The longest API-wide partial outage found. Fine-tuning endpoints were also affected. |
| 2025-06-12 | Google Cloud | An invalid automated quota/policy update crashed Service Control globally (null pointer, no feature flag). "Vertex Gemini API" was among the affected products. | Mini-report says 3h. The status summary spans 10:51 to 18:18 PDT (about 7.5h, computed). us-central1 took up to about 2h40m longer. | https://status.cloud.google.com/incidents/ow5i3PPK96RduMcb1SsW (Verified) | A shared control-plane failure took down hosted model APIs. Hyperscaler resale does not remove the risk. |
| 2025-10-19/20 | AWS us-east-1 | A race condition in DynamoDB's DNS automation left an empty DNS record. EC2, Lambda and NLB were affected. | Some customers had "issues for up to 15 hours." | https://www.infoq.com/news/2025/11/aws-dynamodb-outage-postmortem (Verified) | Bedrock is not named in the InfoQ coverage, so Bedrock impact is **not verified**. Shows region-concentration risk. |
| 2025-11-18 | Cloudflare | A permissions change doubled a Bot Management feature file and crashed the core proxy. | 11:20 to 17:06 UTC (about 5h46m, computed) | https://blog.cloudflare.com/18-november-2025-outage/ (Verified) | Cloudflare's post does not name ChatGPT. Press reports that ChatGPT was disrupted are secondary. Edge/CDN dependencies sit in front of model APIs. |
| 2026-02-03 | OpenAI | API timeouts plus ChatGPT, Codex and image generation. OpenAI blamed a "configuration issue affecting our inference orchestration layer." | About 3-3.5h (15:00 to 18:30 ET) | https://creati.ai/ai-news/2026-02-04/chatgpt-major-outage-february-2026 (Secondary) | 2026 example of an API-affecting outage. |
| 2026-03-02 | Anthropic | claude.ai, apps, Claude Code and login failed. The core API stayed up. | "For hours." Some services were still affected the next day. | https://tbreak.com/claude-ai-outage-march-2026-global/ (Secondary) | Consumer and agent tooling outage, not the API. |
| 2026-07-09 | Google | gemini-2.5-flash returned 404s (see 1.3). | Several hours (rolled back by evening) | https://discuss.ai.google.dev/t/gemini-2-5-flash-deprecated-without-warning-earlier-than-shutdown-date/174217 (Verified) | Availability failure via the lifecycle system. |
| 2026-08-16 | Anthropic | Authentication problems on claude.ai, Claude Code and Cowork. API listed as operational. | About 42 min | https://bleepingcomputer.com/news/artificial-intelligence/anthropic-confirms-claude-is-down-in-major-outage-affecting-multiple-services (Verified) | Short. Included for completeness. |
| 2026-09-03 | OpenAI, Anthropic, Google (coincident) | OpenAI routing error. Claude gateway timeouts. Gemini API problems serving newly created keys. | About 4h overall per the article. Timestamps are inconsistent. | https://tech-insider.org/chatgpt-claude-gemini-down-outage-2026/ (Secondary) | Simultaneous failures across vendors, apparently coincidental. Multi-vendor failover is not a complete hedge. |

Gap: I found no primary status page for a 2026 API-wide outage longer than 4 hours. The aggregators (isdown.app, pingoru.io) list many shorter incidents. They were not used for figures.

### 2.2 Regulatory availability shock: Claude Fable 5 and Mythos 5 (June-July 2026)

- **Fact (suspension):** On 2026-06-12 Anthropic said the US government, "citing national security authorities," had issued an export-control directive. The directive suspended access to Fable 5 and Mythos 5 by foreign nationals inside or outside the US, including Anthropic's own foreign-national employees. Anthropic received it at 5:21pm ET and disabled both models for all customers to comply. Other Claude models were unaffected. Anthropic said the apparent concern was a jailbreak technique, which it said identified only minor, previously known vulnerabilities. It said it disagreed with the action, which it called "a misunderstanding."
  - Source: https://www.anthropic.com/news/fable-mythos-access (Verified)
  - Date: 2026-06-12
  - Note on agency: Anthropic's statements do not name the Department of Commerce or BIS. Third parties attribute the directive to the US Commerce Department: https://www.eisneramper.com/insights/artificial-intelligence-insights/fable-5-suspension-facts-and-timeline-0626/ (Secondary) and https://digitalpolicyalert.org/event/41240-commerce-department-terminated-export-controls-suspending-anthropics-claude-fable-5-and-mythos-5-models (Secondary; records termination on 2026-06-30). Describe the agency as "reported as the Commerce Department" unless a primary government document is found. **No primary government document was found.**

- **Fact (restoration):** On 2026-06-30 Anthropic said the export controls had been lifted.
  - Fable 5 returned globally on 2026-07-01 on the Claude Platform, Claude.ai, Claude Code and Cowork. AWS, Google Cloud and Microsoft Foundry access was to follow "as quickly as possible," with no date.
  - Anthropic had suspended all users because it could not verify nationality in real time.
  - A new classifier, developed with the government, blocks the reported technique in over 99% of cases. Blocked requests go to Opus 4.8. Anthropic acknowledged the classifier flags more benign coding and debugging requests.
  - CAISI, part of Commerce, tested the safeguards.
  - Mythos 5 access was restored for a set of US organisations after government approval on 2026-06-26. Broader Glasswing access was still being coordinated.
  - Source: https://www.anthropic.com/news/redeploying-fable-5 (Verified)
  - Date: 2026-06-30 / 2026-07-01
  - Relevance: This is the clearest non-price case for ownership in the record. A model launched publicly on 2026-06-09 (per [eWeek](https://www.eweek.com/news/anthropic-claude-fable-5-safety-fallbacks/), Verified) was unavailable to all customers for 19 days (computed). It came back with changed behaviour: a new classifier, more false positives, and fallback to another model. Non-US organisations were the stated target of the directive. For them, availability of top hosted models now depends on the nationality and export-control policy of the provider's home country. Present this neutrally: the government's stated rationale was national security, and Anthropic disputed the technical basis.

---

## 3. Data protection and residency

### 3.1 EU AI Act: general-purpose AI (GPAI) models

- **Fact:** GPAI rules, governance and penalty provisions became applicable on 2025-08-02. The GPAI Code of Practice was published on 2025-07-10, after the 2025-05-02 statutory target. The Act becomes generally applicable on 2026-08-02. GPAI models placed on the market before 2025-08-02 must comply by 2027-08-02.
  - Source: https://fpf.org/blog/the-ai-act-implementation-timeline-what-changes-under-the-ai-omnibus (Verified; article 2026-07-28, updated 2026-08-05)
  - Relevance: These are the dates of the compliance regime that applies to anyone who becomes a GPAI "provider."

- **Fact:** The Commission published its GPAI guidelines on 2025-07-18. They set these benchmarks:
  - Indicative GPAI criterion: training compute above 10^23 FLOP.
  - Systemic-risk presumption: above 10^25 FLOP, with notice to the AI Office within two weeks.
  - A downstream modifier becomes a provider if the compute used for modification exceeds one-third of the original model's training compute. Where the original's compute is unknown, the benchmark is one-third of 10^25 FLOP for systemic-risk models and one-third of 10^23 otherwise.
  - Free and open-source models are exempt from some documentation duties, provided they are not monetised and are below the systemic-risk threshold. The copyright policy duty still applies.
  - Commission enforcement with fines starts on 2026-08-02. Fines reach up to €15M or 3% of global turnover.
  - Source: https://www.wilmerhale.com/en/insights/blogs/wilmerhale-privacy-and-cybersecurity-law/20250724-european-commission-issues-guidelines-for-providers-of-general-purpose-ai-models (Verified; law-firm summary of the Commission guidelines, 2025-07-24)
  - Relevance: This is important for the paper. Ordinary LoRA or fine-tuning sits far below one-third of the base model's compute, so an organisation that adapts an open-weight model usually does **not** become a GPAI provider. Adaptation of owned models is therefore lightly regulated at the model layer, though system-level obligations still apply.

- **Fact:** The Commission's one-year grace period for signatories of the GPAI Code of Practice ended on 2026-08-02.
  - Source: https://www.dataprotectionreport.com/2026/07/the-eu-ai-act-when-does-it-become-enforceable-now/ (Verified; Norton Rose Fulbright, 2026-07-24)

- **Fact:** The AI Omnibus amending the AI Act has these milestones:
  - Proposed November 2025, political agreement May 2026.
  - Parliament adoption 2026-06-16, Council adoption 2026-06-29, Official Journal 2026-07-24.
  - Annex III high-risk obligations move to 2027-12-02 (from 2026-08-02). Annex I obligations move to 2028-08-02 (from 2027-08-02).
  - Article 50(2), the synthetic-content transparency duty for GPAI-based systems, moves to 2026-12-02.
  - The AI Office gains exclusive competence over systems built on GPAI models where the model and system come from the same undertaking.
  - Source: https://fpf.org/blog/the-ai-act-implementation-timeline-what-changes-under-the-ai-omnibus (Verified)
  - Relevance: GPAI provider duties were not postponed. High-risk deployer duties were, which reduces near-term compliance pressure on downstream deployers.

### 3.2 Botswana: Data Protection Act, 2024

- **Fact:** The law is the Data Protection Act, Act No. 18 of 2024, which repealed the 2018 Act. The 2018 Act never came into force after repeated extensions. Parliament assented on 2024-08-19. DLA Piper gives commencement as 2024-10-29, and CIPIT describes it as enacted that day. Michalsons says it came into effect on 2025-01-14. **Commencement date conflicts between sources. Check the Gazette.**
  - Sources: https://www.dlapiperdataprotection.com/?c=BW&c2=GB&t=law (Verified); https://cipit.org/?p=3218 (Verified); https://www.michalsons.com/blog/botswanas-data-protection-act-grace-period-extended/60775 (Verified, 2025-01-20)

- **Fact:** The regulator is the Information and Data Protection Commission, with a Data Protection Division and an Access to Information Division. It can approve standard data-protection clauses and binding corporate rules (s 17, per CIPIT).
  - Sources: https://cipit.org/?p=3218 (Verified); https://www.dlapiperdataprotection.com/?c=BW&c2=GB&t=law (Verified)

- **Fact:** Cross-border transfer rules:
  - Transfers out of Botswana are prohibited unless the destination is designated by the Minister through a Government Gazette Order. Michalsons cites s 74.
  - Otherwise the Commissioner must authorise the transfer after assessing the adequacy of protection.
  - Derogations without adequacy include consent, contract necessity, public interest or legal claims, vital interests, and public-register data.
  - DLA Piper lists about 45 designated countries, including South Africa, Kenya, the EU/EEA states, the UK, Switzerland, Japan, the Republic of Korea, Israel, New Zealand, Argentina and Uruguay. **The United States is not on the list.**
  - Under s 54 (per CIPIT), foreign controllers or processors handling Botswana residents' data must appoint a representative. Where they offer goods or services or monitor behaviour, the representative must be based in Botswana.
  - Breach notification is required within 72 hours where feasible. Penalties go up to BWP 500,000 and/or 9 years' imprisonment for security-safeguard failures.
  - Sources: https://www.dlapiperdataprotection.com/?c=BW&c2=GB&t=law (Verified); https://www.michalsons.com/blog/botswanas-data-protection-act-grace-period-extended/60775 (Verified); https://cipit.org/?p=3218 (Verified)
  - Relevance: OpenAI and Anthropic process API data in the US by default, and the US is not designated. Sending personal data from Botswana to a US-processed LLM API needs Commissioner authorisation or a derogation. South Africa is designated, so in-region processing in Johannesburg or Cape Town would be the simpler route. No major LLM API currently offers that (3.5).

### 3.3 South Africa: POPIA cross-border rule (s 72)

- **Fact:** Under POPIA s 72(1), a responsible party may transfer personal information to a third party in a foreign country only if one of five grounds applies:
  1. The recipient is bound by a law, binding corporate rules or an agreement giving substantially similar protection, including on onward transfer.
  2. The data subject consents.
  3. The transfer is necessary to perform a contract with the data subject.
  4. The transfer is necessary for a contract in the data subject's interest.
  5. The transfer benefits the data subject, consent is impracticable, and the data subject would likely consent.
  - Source: https://source.acts.co.za/protection-of-personal-information-act-2013/72__transfers_of_personal_info.php (Verified; statute text)
  - Relevance: This regime is less restrictive than Botswana's. There is no ministerial whitelist, and contractual adequacy suffices.

- **Not verified in this pass:**
  - POPIA's commencement (generally cited as 2020-07-01, with enforcement from 2021-07-01)
  - s 57 prior-authorisation requirements
  - The R10 million maximum fine
  - The regulator's name, the Information Regulator (South Africa), is standard but was not opened here.

### 3.4 Cloud regions in Africa

- **Fact:** AWS Africa (Cape Town), af-south-1, was announced on 2020-04-22 with three Availability Zones.
  - Source: https://aws.amazon.com/about-aws/whats-new/2020/04/announcing-aws-africa-cape-town-region (Verified)
- **Fact:** Microsoft made Azure generally available from data centres in Cape Town and Johannesburg in March 2019. Coverage described Microsoft as the first public cloud provider to deliver cloud services from Africa. The region names "South Africa North" and "South Africa West" are not in this article but are standard Azure names.
  - Source: https://www.infoq.com/news/2019/03/microsoft-azure-africa-regions (Verified; 2019-03-15)
- **Fact:** Google Cloud's Johannesburg region (africa-south1) became operational on 2024-01-31, Google's first in Africa. TechCrunch says Google joins AWS, Azure "and Oracle amongst others" with regions in South Africa.
  - Source: https://techcrunch.com/2024/01/31/googles-first-africa-cloud-region-now-operational/ (Verified)
- **Not verified:** Oracle's Johannesburg region details, Huawei Cloud Johannesburg, any hyperscaler region in Botswana (none known), and any announced Kenya or Nigeria hyperscaler regions.
- Relevance: Hyperscaler capacity in Africa is concentrated in South Africa. For Botswana, South Africa is both the nearest region and a designated transfer destination.

### 3.5 Do the major LLM APIs process data in-region in Africa?

- **OpenAI:** API data residency lists regional storage **and processing** only for the US, Europe (EEA + Switzerland) and the UAE. Australia, Canada, Japan, India, Singapore, South Korea and the UK get storage only. The guide says regional storage "does not imply support for regional processing." No African region appears in the portion read; the page was truncated at 100k characters.
  - Source: https://developers.openai.com/api/docs/guides/your-data (Verified, partial page)
  - Date: page read 2026-11-08. Per-request region switching was dated 2026-08-21 per https://ecorpit.com/openai-per-request-regional-processing-india-storage-only-2026/ (Secondary).
- **Anthropic (first-party):** `inference_geo` accepts only `"global"` (the default) or `"us"`. US-only costs 1.1x on Claude 4.6+. Workspace data-at-rest geography is US only. Bedrock and Google-managed Vertex use their own regions.
  - Source: https://platform.claude.com/docs/en/build-with-claude/data-residency (Verified; undated page)
- **AWS Bedrock (Cape Town):** Claude Sonnet 4.5, Haiku 4.5 and Opus 4.5 can be invoked from af-south-1 through **global** cross-Region inference. AWS says inference "can occur in another Region." Logs, knowledge bases and stored configuration remain in the source region. AWS tells customers to check POPIA fit themselves.
  - Source: https://aws.amazon.com/blogs/machine-learning/scale-ai-in-south-africa-using-amazon-bedrock-global-cross-region-inference-with-anthropic-claude-4-5-models/ (Verified; 2026-01-30)
- **Azure (Foundry):** South Africa North and South Africa West do not appear in any deployment-type table (Global Standard, Data Zone, Standard/Regional, Provisioned, Batch) on Microsoft's model region-availability page, last updated around 2026-09-04. A third-party validation memo (2026-10-11) says the only regional in-country GPT option is gpt-4o 2024-05-13 on Provisioned Throughput, retiring 2026-12-09.
  - Sources: https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure-region-availability (Verified; MEA tab may not have loaded fully); https://know-2nth.pages.dev/sa-llm-residency-validation.html (Secondary)
- **Google Vertex:** The locations page says endpoints "don't guarantee data residency or in-region ML processing" on the global endpoint. africa-south1 did not appear in the portions of the locations page that loaded. The secondary memo says Gemini is not offered in africa-south1, but the region supports custom model inference.
  - Sources: https://docs.cloud.google.com/gemini-enterprise-agent-platform/resources/locations (Verified, partial); https://know-2nth.pages.dev/sa-llm-residency-validation.html (Secondary)
- Relevance: Section 3.2 shows that from Botswana, US-processed APIs need Commissioner authorisation or a derogation, while South Africa is designated. Combined with these findings, the residency argument for owning open weights and serving them in-region (or in-country) is concrete, not hypothetical. This is one of the paper's strongest regional arguments.

---

## 4. The open-weight landscape in 2026

Release dates are Hugging Face `createdAt` timestamps from the HF API unless stated (Verified via https://huggingface.co/api/models?author=...). Licence tags come from HF metadata. Licence terms were read from LICENSE files where noted.

| Family | Notable 2026 releases (HF date) | Licence | Notes |
|---|---|---|---|
| DeepSeek | V4-Pro and V4-Flash (2026-04-22); V4-Pro-0813 (2026-08-13); V4.1-Flash (2026-09-10); V4.1-Flash-1017 (2026-10-17) | MIT | Still fully permissive. A secondary source says V4-Pro has about 1.7T params ([aiweekly](https://aiweekly.co/alerts/deepseek-v4-pro-exits-preview-with-mit-licensed-17t-weights), not opened). |
| Moonshot Kimi | K2.5 (2026-01-01); K2.6 (2026-04-14); K2.7-Code (2026-06-11); K3 (2026-06-13); K3.1-Pro (2026-11-02, 2.8T total / 104B active) | "other" (custom) | K3.1-Pro licence: model-as-a-service (MaaS) operators above $20M revenue in 12 months need a separate agreement. Products above 100M MAU or $20M monthly revenue must display the name. Internal use is exempt. ([LICENSE](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/main/LICENSE), Verified) |
| Z.ai GLM | GLM-5 (2026-02-11); 5.1 (2026-04-03); 5.2 (2026-06-16); 5.3 and 5.3-Flash (2026-08-25; 5.3 is 753B) | 5 to 5.2: MIT. **5.3: custom "glm-5.3".** 5.3-Flash: MIT | GLM-5.3 licence: permissive, but MaaS operators above $10B revenue must pass a Z.ai security review. ([LICENSE](https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE), Verified) This is a shift from MIT at the flagship tier. |
| Alibaba Qwen | Qwen3.8-27B (2026-08-05, Apache-2.0); Qwen3.8-2.4T-A95B (2026-08-08, custom "qwen3.8-max"); Qwen3.8-Flash-Next (2026-08-24, custom) | Mixed | Flagship licence: name display above 100M MAU or $20M monthly revenue. MaaS or "AI Work Assistant" businesses above $50M in 12 months need a separate licence. Internal use is exempt. ([LICENSE](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/LICENSE), Verified) Mid-size models remain Apache-2.0. |
| OpenAI gpt-oss | gpt-oss-120b and gpt-oss-20b (2025-08-05); gpt-oss-safeguard (2025-09) | Apache-2.0 | 120b runs on one 80 GB GPU. 20b needs about 16 GB. ([OpenAI](https://openai.com/index/introducing-gpt-oss/), Verified) No general-purpose successor appears on OpenAI's HF org as of 2026-11-08; the only later item is `privacy-filter`, 2026-04-17. |
| Meta | Muse Spark (2026-04-08), **proprietary** ([Artificial Analysis](https://artificialanalysis.ai/articles/muse-spark-everything-you-need-to-know), Verified). Muse Glimmer 30B (2026-08-09, Apache-2.0). Muse-Spark-Open-816B-A42B (HF 2026-09-15; card gives release as 2026-10-15) | Glimmer: Apache-2.0. Spark Open: custom "Muse Model License" | Spark Open licence: "Built with Muse Spark Open" attribution. MaaS above $20M revenue needs an agreement with Meta. Internal use is excluded from that condition. ([model card](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B), Verified) The last Llama on the `meta-llama` org is Llama 4 (April 2025). |
| Mistral | Mistral Large 4 1T-A52B (2026-09-30); Leanstral-1.5, Shieldstral (Apache-2.0) | Large 4: "Mistral Ecosystem Open-Weight License" (modified MIT) | A separate commercial licence is needed to run a MaaS business. ([model card](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B), Verified) |
| Google Gemma | Gemma 4 (2026-04-02: 31B dense, 26B MoE, E2B/E4B edge) | **Apache-2.0**, replacing the custom Gemma terms | The Register says the old licence prohibited some uses and let Google terminate access. ([The Register](https://www.theregister.com/2026/04/02/googles_gemma_4_open_weights/), Verified) |

Notable shifts (synthesis of the verified items above):
- **Toward closed, then partly back:** Meta released its first frontier model without open weights (Muse Spark, April 2026). It then released Muse Glimmer (Apache-2.0, August) and Muse Spark Open (custom licence, Sept/Oct). Zuckerberg said Muse Spark 1.2 weights would come "soon" ([Tribune India](https://www.tribuneindia.com/news/business/meta-opens-muse-glimmer-weights-zuckerberg-says-muse-spark-1-2-release-coming-soon/), Verified, 2026-08-11).
- **Toward more open:** Google moved Gemma from custom terms to Apache-2.0 (April 2026). The Register reports this was aimed at enterprises worried about access being revoked.
- **Flagship licences tightening against hosting competitors:** Z.ai (GLM-5.3), Alibaba (Qwen3.8 flagship), Moonshot, Meta (Spark Open) and Mistral (Large 4) all add conditions for MaaS operators above revenue thresholds, while smaller or "Flash" variants stay MIT or Apache. For the paper: these clauses target cloud resellers, not organisations running models internally. Qwen and Kimi explicitly exempt internal use. Own-and-adapt for internal use remains legally low-friction.
- **Quality gap:** The Stanford AI Index 2026 Arena chart shows the top closed model at 1,503 and the top open model at 1,454 (49 points, about 3.3%, computed) ([Stanford HAI](https://hai.stanford.edu/ai-index/2026-ai-index-report/technical-performance), Verified). Mozilla's *State of Open Source AI 2026* reports a 3.3-point average gap. The gap was 8.04 points in January 2024 and near zero after DeepSeek-R1 in February 2025 ([Business Standard](https://business-standard.com/technology/artificial-intelligence/open-models-match-closed-ai-deployment-real-hurdle-mozilla-126071600840_1.html), Verified secondary report of Mozilla). On Terminal-Bench 2.1, GPT-5.5 scores 83.4% and the best open model 67.9%, a 15.5-point gap ([UC Today](https://www.uctoday.com/productivity-automation/mozilla-state-of-open-source-ai-2026-report/), Verified secondary).

---

## 5. Hardware and supply signposts

- **Fact:** TrendForce forecast 1Q26 conventional DRAM contract prices up about 55-60% QoQ, and server DRAM up more than 60%. Suppliers are moving advanced nodes and new capacity to server and HBM products. US hyperscalers are taking a larger share of bit supply.
  - Source: https://www.trendforce.com/presscenter/news/20260105-12860.html (Verified; 2026-01-05)
- **Fact:** TrendForce forecast 3Q26 conventional DRAM contract prices up 13-18% QoQ and NAND up 10-15%. Gains are moderating as consumer demand hits affordability limits. Server DRAM stays undersupplied, partly cushioned by long-term agreements. Server shipments are expected to stay robust through 2027.
  - Source: https://www.trendforce.com/presscenter/news/20260703-13134.html (Verified; 2026-07-03)
- **Fact:** TrendForce raised its 2027 HBM outlook to a blended ASP rise of 121% YoY. It expects memory supply to stay tight through 2027, with HBM and conventional DRAM competing for wafer capacity.
  - Source: https://www.trendforce.com/presscenter/news/20260929-13255.html (Verified; 2026-09-29)
- **Fact:** Bloomberg (via TrendForce) reported that Nvidia told major customers server prices will rise by more than 15% for early-2027 shipments (Vera Rubin, Grace Blackwell), with memory a central factor. Morgan Stanley says GPUs' share of AI-server cost has fallen from over 80% to about half in next-generation systems as memory costs climb.
  - Source: https://www.trendforce.com/news/2026/08/24/news-nvidia-reportedly-eyes-more-than-15-price-hikes-for-vera-rubin-grace-blackwell-servers-in-early-2027/ (Verified; 2026-08-24, citing Bloomberg 2026-08-22)
- **Fact:** SemiAnalysis's H100 one-year rental price index:
  - About $1.70/hr/GPU in October 2025 (the low).
  - Above $2/hr in late January 2026.
  - About $2.35/hr in March 2026, nearly +40% from the low.
  - Drivers: memory-driven server repricing and agentic/coding demand. On-demand capacity was described as sold out. B200 spot on AWS p6-b200 reportedly reached $14/hr. Capacity coming online through August-September 2026 was reportedly already booked.
  - Source: https://newsletter.semianalysis.com/p/the-great-gpu-shortage-rental-capacity (Verified; 2026-04-02)
- **Fact:** Silicon Data's H100 rental index (SDH100RT) over time:
  - About $2.37/hr on 2025-05-27 ([IEEE Spectrum](https://spectrum.ieee.org/gpu-prices), Verified)
  - $2.41/hr in late February 2026, with A100 at $1.38 ([OfficeChai](https://officechai.com/ai/surging-rental-costs-of-older-gpus-signals-robust-ai-demand), Verified secondary)
  - $3.00/hr on 2026-11-08 (up from $2.82 on 2026-10-10), with A100 at $1.59 and B200 at $6.81 ([Silicon Data](https://www.silicondata.com/products/silicon-index/h100), Verified). The same page's FAQ gives $2.53, and its stated 7-day change does not match its own table. **Treat the $3.00 headline with caution.**
- Relevance (two-sided): The memory crunch raises the capital cost of owning and running inference hardware, which slows on-prem ownership. It has also pushed rental prices up: about +38% on SemiAnalysis's 1-year H100 index (October 2025 to March 2026) and about +27% on Silicon Data's H100 index (May 2025 to November 2026, caveated). Both are computed, and the indices are not comparable with each other. This erodes the assumption that rented compute keeps getting cheaper. For the paper, ownership via **managed open-weight hosting and adapters** (Section 6) is less exposed to hardware capex than on-prem ownership.

---

## 6. Managed adaptation offerings (own the adapter, not the cluster)

- **Thinking Machines Tinker**
  - Fact: Launched 2025-10-01 in private beta as "a flexible API for fine-tuning language models." It was "free to start," and uses LoRA so runs can share compute.
    - Source: https://thinkingmachines.ai/blog/announcing-tinker/ (Verified)
  - Fact: General availability came on 2025-12-12 ("The waitlist is over!"). The same release added Kimi K2 Thinking, OpenAI-compatible sampling and Qwen3-VL vision input.
    - Source: https://thinkingmachines.ai/blog/tinker-general-availability/ (Verified)
  - Fact: Pricing is per token for prefill, sampling and training, varying by model and context. Checkpoint storage costs $0.10/GB-month. MoE models are priced by active parameters. Examples as of 2026-10-19, per 1M tokens:
    - Qwen3-8B: train $0.44
    - GPT-OSS-120B: train $0.737
    - Kimi-K2.6 (32K context): train $4.84
    - Kimi-K3.1-Pro (256K context): train $32.00
    - DeepSeek-V4.1-Flash: train $4.98
    - Users can download any saved checkpoint via API.
    - Supported models range from 4B to about 2.8T parameters. Some base models are scheduled for retirement (Nemotron-3-Nano-30B-A3B, Qwen3.6-27B, DeepSeek-V3.1 on Oct 23).
    - Sources: https://thinkingmachines.ai/tinker/pricing/ and https://thinkingmachines.ai/tinker/ (Verified)
  - Fact: Tinker reportedly crossed a $100M annual revenue run-rate, per TechCrunch (September 2026) citing an unnamed source.
    - Source: https://www.arr.club/thinking-machine/thinking-machines-100m-run-rate-tinker-oct-2026 (Secondary; TechCrunch and AI Magazine originals not opened)
  - Relevance: This is the template for "own the adapter, rent the GPUs." Even here the platform retires base models. The difference from closed-model fine-tuning is that the weights can be downloaded and run elsewhere.

- **OpenAI reinforcement fine-tuning (RFT)**
  - Fact: RFT became available for o4-mini on 2025-05-08, limited to verified organisations. OpenAI offered a 50% discount for sharing datasets.
    - Source: https://community.openai.com/t/fine-tuning-updates-reinforcement-fine-tuning-now-available-gpt-4-1-nano-fine-tuning/1255539 (Verified; staff post)
  - Fact: o4-mini is in the 2026-04-22 deprecation batch with a 2026-10-23 shutdown, and fine-tunes follow their base models. A secondary source says RFT was available "only for o4-mini," so RFT effectively ended on that date.
    - Sources: https://developers.openai.com/api/docs/deprecations (Verified); https://www.developersdigest.tech/blog/openai-fine-tuning-shutdown-alternatives (Secondary, 2026-10-18)
  - **Not verified:** RFT pricing basis (widely reported as per hour of training; the billing help page failed to load).
  - Relevance: Within about 17 months (computed), a closed-vendor RFT product launched and then lost its only base model. Customers' RL-tuned models do not survive that.

- **AWS (Bedrock / SageMaker)**
  - Fact: Bedrock added reinforcement fine-tuning on 2025-12-03. At launch it supported Amazon Nova 2 Lite, with more models to follow. AWS claims an average 66% accuracy gain over base models. AWS also launched serverless model customisation with RL in SageMaker AI at re:Invent.
    - Sources: https://aws.amazon.com/about-aws/whats-new/2025/12/bedrock-reinforcement-fine-tuning-66-base-models (Verified); https://press.aboutamazon.com/2025/12/aws-simplifies-model-customization-to-help-customers-build-faster-more-efficient-ai-agents (Verified)
  - **Not verified:** whether Bedrock-tuned weights can be exported, and pricing.

- **Google Vertex AI**
  - Fact: The Gemini tuning docs navigation lists supervised, reinforcement-learning and preference tuning. The body did not load, so supported models, pricing and weight export are **not verified**.
    - Source: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-tuning (partial)
  - Secondary: tuned Gemini 3 models are served at 1.5x base price ([DevelopersDigest](https://www.developersdigest.tech/blog/openai-fine-tuning-shutdown-alternatives)). Not verified.
  - Relevance: Tuned Gemini models, like OpenAI fine-tunes, are tied to a closed base. Assume they are non-portable unless shown otherwise.

- **Together AI**
  - Fact: On 2024-12-18 Together launched serverless multi-LoRA. Adapters are billed at the base model's per-token price, and "hundreds" of adapters can share infrastructure. Users can upload their own adapters (for example from HF) and download adapters they train.
    - Source: https://www.together.ai/blog/serverless-multi-lora-fine-tune-and-deploy-hundreds-of-adapters-for-model-customization-at-scale (Verified)

- **Fireworks AI**
  - Fact: On 2024-09-18 Fireworks launched Multi-LoRA. LoRA serving costs the same as the base model (for example $0.20/1M tokens for Llama 3.1 8B at the time). "Hundreds" of LoRAs can run per base model, and a customer (Cresta) reportedly deployed thousands. Fireworks claims 100x cost efficiency against serving 100 separate fine-tuned models, and throughput near 90% of the base model. Users can bring their own adapters.
    - Source: https://fireworks.ai/blog/multi-lora (Verified)

- Relevance (section): Multi-LoRA serving at base-model prices makes the marginal cost of holding many adapters near zero. Organisations can then own adaptation as a portable file without owning a cluster. The contrast with OpenAI's wind-down is the core of the "portable vs. hostage" distinction.

---

## 7. Enterprise surveys newer than December 2025

- **a16z, third annual CIO survey: "Leaders, Gainers, and Unexpected Winners"**
  - Sample: 100 VP-and-above executives at Global 2000 companies (revenue at least $500M; 88% above $1B). Drawn from a vendor database, not randomly. Fieldwork dates not stated.
  - Date: 2026-01-30
  - Figures:
    - 81% use three or more model families, up from 68% less than a year earlier.
    - About 80% are comfortable hosting models directly, up from about 40% in March 2024.
    - "Over a third" prefer closed-source models, and that preference is rising.
    - 65% prefer incumbent vendors' solutions where available.
    - Enterprises have moved away from fine-tuning toward prompting and routing. No percentage is given.
    - Average LLM spend rose from about $4.5M to about $7M over two years, with about $11.6M expected in 2026.
    - Per The Decoder's summary: OpenAI is in production at 78% of firms and Anthropic at 44%.
  - Sources: https://www.a16z.news/p/leaders-gainers-and-unexpected-winners (Verified); https://the-decoder.com/openai-still-leads-enterprise-ai-but-anthropic-is-gaining-fast-according-to-new-study/ (Verified secondary). The Decoder's "$2.5M in 2024" baseline conflicts with a16z's own $4.5M. Use a16z.
  - Relevance: Mixed signals. Self-hosting comfort has doubled, but fine-tuning has reportedly declined and the closed preference is growing.

- **Menlo Ventures, "2026: The State of Enterprise AI"**
  - Sample: "Nearly 400" US enterprise decision-makers controlling more than $4B of AI spend. Fieldwork by Morning Consult, 2026-09-08 to 2026-09-26.
  - Date: 2026-11-03
  - Figures:
    - Enterprise LLM API share: Anthropic 55% (from 40%), OpenAI 31% (from 27%), Google 9% (from 21%).
    - Open-source models: 4% of model spend (from 6%), 19% of tokens/usage (from 11%).
    - 10% of AI builders post-train in 2026 (from 6% in 2025), and 91% of them use outside partners.
    - Deployment spend is $176.8B, up 5.4x from $33B.
    - Build vs buy is not reported.
  - Source: https://menlovc.com/perspective/2026-the-state-of-enterprise-ai/ (Verified)
  - Relevance: These are the best quantitative signposts for the forecast. Open-model usage share rising while spend share falls suggests open models are used for high-volume, low-price workloads. Post-training is rising from a low base and is mostly outsourced, which fits the managed-adaptation thesis.

- **Mozilla, "State of Open Source AI 2026"**
  - Sample: A SlashData developer survey, May 2026. UC Today gives n = 1,494; Business Standard says "more than 1,410." **Check the n in the report.** Also uses OpenRouter token data.
  - Date: 2026-07-14 (report); coverage 2026-07-16 to 2026-07-20
  - Figures:
    - 79% of developers use open models, but only 51% of adopting organisations have them in production, against 63% for closed.
    - At enterprises (over 1,000 employees), production rates are 57% for open and 73% for closed. At small firms, 53% open and 54% closed.
    - With vendor support, 67% of projects reach production. Fully in-house builds reach production 33% of the time.
    - 89% of firms use at least one open-source AI component.
    - Open-weight models were "nearly one-third" of OpenRouter tokens by late 2025.
  - Sources: https://business-standard.com/technology/artificial-intelligence/open-models-match-closed-ai-deployment-real-hurdle-mozilla-126071600840_1.html (Verified secondary); https://www.uctoday.com/productivity-automation/mozilla-state-of-open-source-ai-2026-report/ (Verified secondary). The Mozilla report itself was not opened.
  - Relevance: Deployment capability, not model quality, is the binding constraint. Vendor support doubles the production success rate, which again supports managed adaptation.

- **Amplify Partners, "2026 AI Engineering Report"**
  - Sample: More than 1,000 AI engineers (with Notion and Vercel)
  - Date: listed as "07-02-2026" (ambiguous: 2 July or 7 February)
  - Figures: 45% use open-weight models, out of the box or fine-tuned. More than 90% of those also use closed models.
  - Source: https://www.amplifypartners.com/blog-posts/the-2026-ai-engineering-report (Verified)

- **Stanford AI Index 2026**
  - Date: 2026-04-13 per a secondary source
  - Figures: Arena scores of 1,503 for the top closed model and 1,454 for the top open model. The 88% organisational adoption figure is reported secondhand (tech-insider) and was not verified in the primary.
  - Sources: https://hai.stanford.edu/ai-index/2026-ai-index-report/technical-performance (Verified); https://tech-insider.org/ca/stanford-ai-index-report-2026/ (Secondary)

- **Gaps:**
  - A McKinsey "State of AI 2026" with open-source or fine-tuning figures was not located or verified.
  - No Gartner survey from 2026 with open-vs-closed or fine-tuning shares was found; only 2025 forecasts surfaced.
  - None of the surveys found reports build-vs-buy as a clean percentage for 2026.

---

## 8. Measures of adaptation activity on the Hugging Face Hub

- **Fact:** Chen et al., "A First Look at Model Supply Chain: From the Risk Perspective" (ICSE '26):
  - 1,817,398 HF models crawled as of 2025-06-25, with 540,838 derived models (49,648 base models).
  - Derivation types: fine-tuning, including adapters and PEFT, 427,475 (79.0%); quantisation 100,564 (18.6%); merging 12,799 (2.4%).
  - Across 17 weekly snapshots (2025-03-12 to 2025-07-02), an average of 27,311 models were added per week. Of 7,432 derived models added weekly, 72.8% were fine-tunes.
  - Source: https://chenbihuan.github.io/paper/icse26-chen-model-supply-chain.pdf (Verified)
  - Relevance: This is the cleanest snapshot of the composition of adaptation activity. Fine-tunes dominate derivatives.

- **Fact:** Laufer, Oderinwale and Kleinberg, "Anatomy of a Machine Learning Ecosystem: 2 Million Models on Hugging Face" (arXiv 2508.06811, 2025-08-09):
  - Studies 1.86M models and their family trees.
  - Finds licences drift from restrictive to permissive along lineages, often violating upstream terms.
  - Siblings resemble each other more than parents and children do.
  - Source: https://arxiv.org/abs/2508.06811 (Verified)
  - Relevance: Shows adaptation lineages at scale. Licence drift is a governance caveat when citing derivative counts as legitimate commercial adaptation.

- **Fact:** HuggingGraph (Rahman, Gao, Ji; arXiv 2507.14240, 2025-07-17, v3 2025-09-04) builds a model-dataset graph with 402,654 nodes and 462,524 edges.
  - Source: https://arxiv.org/abs/2507.14240 (Verified, abstract only)

- **Fact:** Hugging Face, "State of Open Models: Summer 2026 Observations":
  - Public model repos grew from 2.43M to 2.96M over seven months of 2026 (+21.5%). Datasets grew from 711k to 1M, and Spaces from 1.00M to 1.44M.
  - Derivatives: Qwen 151,448 (2.6x Meta's total footprint and 4.7x its Llama repos); Google 82,506.
  - Qwen derivatives grew by about 180-210 new repos per day in January-July 2026.
  - There are 28,531 GGUF conversions of Qwen, of which Qwen published 54.
  - Monthly GGUF downloads: Qwen 39.6M, Gemma 20.8M, Llama 7.5M.
  - Hugging Face warns downloads do not capture API usage or private deployments.
  - Source: https://huggingface.co/blog/state-of-open-models-summer-2026 (Verified; 2026-08-14)
  - Relevance: This is the best 2026 time-series data point. Derivative creation is accelerating and concentrating on Chinese bases. Private enterprise fine-tunes are invisible here, so Hub counts are a lower bound on adaptation activity.

- **Gap:** No source gives an adapter-only (LoRA/PEFT) count over time. Chen et al. fold adapters into fine-tunes, and HF does not break them out. The HF API exposes `base_model:adapter:` tags, so a custom count is feasible, but it was not run here because the proxy blocks direct API scripting.

---

## Signposts for the forecasting section

| Signpost | Latest reading | Direction that favours ownership | Source |
|---|---|---|---|
| Notice before API model retirement | Anthropic: about 60 days typical. OpenAI: 3-6 months, with some 92-day GA cases. | Shorter or less reliable notice | Section 1 |
| Closed-vendor fine-tuning availability | OpenAI self-serve fine-tuning ends for all on 2027-01-06 | More vendors exiting or restricting tuning | Section 1.1 |
| Government-ordered model withdrawals | One case (Fable/Mythos, 19 days, June-July 2026) | Repeat cases, or controls extended to other labs | Section 2.2 |
| In-country LLM API processing in Africa | None found for current frontier models | Stays absent (favours ownership) or arrives (weakens the case) | Section 3.5 |
| Open-closed quality gap (Arena) | About 3.3% (2026) | Gap narrowing | Section 4 |
| Flagship open-weight licence restrictiveness | MaaS revenue thresholds spreading; internal use exempt | Internal-use exemptions retained | Section 4 |
| H100 rental index | About $2.41 (Feb 2026), $3.00 (Nov 2026, caveated) | Falling rental prices favour rented-compute ownership; rising prices favour neither | Section 5 |
| HBM/DRAM pricing | 2027 HBM ASP +121% YoY forecast | Easing would cut on-prem ownership costs | Section 5 |
| Share of AI builders who post-train | 10% (2026), up from 6% | Rising | Section 7 (Menlo) |
| Open-source share of enterprise tokens vs spend | 19% tokens, 4% spend | Spend share rising | Section 7 (Menlo) |
| Hub derivative growth | +21.5% repos in 7 months; Qwen +180-210 derivatives/day | Sustained growth | Section 8 |

---

## Verification log: items not verified or with conflicts

- The Department of Commerce as issuer of the Fable/Mythos directive is stated only by secondary sources. Anthropic says "the US government."
- Google's formal Gemini deprecation notice policy: primary page not readable.
- Bedrock impact during the AWS outage of 2025-10-20: not named in the source read.
- Botswana DPA commencement: 2024-10-29 (DLA Piper, CIPIT) vs 2025-01-14 (Michalsons).
- POPIA commencement and enforcement dates, s 57 and maximum fine: not opened.
- Oracle, Huawei and other African cloud regions: not opened.
- Azure South Africa in-country GPT availability: primary tables show none; the PTU exception is secondary.
- Vertex Gemini availability in africa-south1 and Gemini tuning details: partial pages only.
- OpenAI RFT pricing basis: not verified. RFT ending with o4-mini on 2026-10-23 is inferred from the deprecation list plus a secondary source.
- Tinker $100M run-rate: secondary.
- Silicon Data's $3.00 H100 headline conflicts with the $2.53 in its FAQ.
- Mozilla survey n: 1,494 vs "more than 1,410."
- a16z spend baseline: The Decoder ($2.5M) vs a16z ($4.5M). Use a16z.
- The 88% adoption figure in the Stanford AI Index 2026: secondary.
- McKinsey 2026 and Gartner 2026 surveys: not found.
