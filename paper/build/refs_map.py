"""Bibliography entries keyed by the URL the essay links to.

Each value: (bibkey, entry type, fields). Pages without a publication date carry
the year they were accessed (2026). All pages were accessed on 8-9 November 2026.
"""

ACCESSED = "Accessed 8 November 2026"

R = {
    # Markets, prices and trends
    "https://menlovc.com/perspective/2025-the-state-of-generative-ai-in-the-enterprise/":
        ("menlo2025", "misc", dict(author="{Menlo Ventures}", year="2025",
         title="2025: The State of Generative {AI} in the Enterprise")),
    "https://menlovc.com/perspective/2026-the-state-of-enterprise-ai/":
        ("menlo2026", "misc", dict(author="{Menlo Ventures}", year="2026",
         title="2026: The State of Enterprise {AI}")),
    "https://a16z.com/ai-enterprise-2025/":
        ("wang2025cio", "misc", dict(author="Wang, S. and Xu, S. and Kahl, J. and Erten, T.",
         year="2025", title="How 100 Enterprise {CIOs} Are Building and Buying Gen {AI} in 2025",
         howpublished="Andreessen Horowitz (a16z)")),
    "https://arxiv.org/abs/2601.10088":
        ("aubakirova2026openrouter", "misc", dict(
         author="Aubakirova, M. and Atallah, A. and Clark, C. and Summerville, J. and Midha, A.",
         year="2026", title="State of {AI}: An Empirical 100 Trillion Token Study with {OpenRouter}",
         howpublished="arXiv:2601.10088")),
    "https://epoch.ai/publications/the-plunging-price-of-thought":
        ("emberson2026price", "misc", dict(author="Emberson, L. and Roodman, D.", year="2026",
         title="The plunging price of thought", howpublished="Epoch AI")),
    "https://epoch.ai/data-insights/open-closed-eci-gap":
        ("edwards2026gap", "misc", dict(author="Edwards, J. and Emberson, L.", year="2026",
         title="Open models lag state-of-the-art closed models by 4 months", howpublished="Epoch AI")),
    "https://epochai.substack.com/p/whats-the-best-model-you-can-run":
        ("somala2025consumer", "misc", dict(author="Somala and Emberson, L.", year="2025",
         title="What's the best model you can run on a single consumer {GPU}?",
         howpublished="Epoch AI")),
    "https://epoch.ai/data-insights/open-weights-vs-closed-weights-models":
        ("emberson2025lag", "misc", dict(author="Emberson, L.", year="2025",
         title="Open-weight models lag state-of-the-art by around 3 months on average",
         howpublished="Epoch AI")),
    "https://epoch.ai/trends":
        ("epoch2026trends", "misc", dict(author="{Epoch AI}", year="2026", title="Trends")),
    "https://www.silicondata.com/products/silicon-index/h100":
        ("silicondata2026h100", "misc", dict(author="{Silicon Data}", year="2026",
         title="{H100} rental index")),
    "https://www.silicondata.com/products/silicon-index/a100":
        ("silicondata2026a100", "misc", dict(author="{Silicon Data}", year="2026",
         title="{A100} rental index")),
    "https://cryptobriefing.com/nvidia-h100-gpu-rental-costs-surge/":
        ("cryptobriefing2026", "misc", dict(author="{Crypto Briefing}", year="2026",
         title="{NVIDIA} {H100} {GPU} rental costs surge")),
    "https://www.trendforce.com/presscenter/news/20260105-12860.html":
        ("trendforce2026", "misc", dict(author="{TrendForce}", year="2026",
         title="{DRAM} contract price forecast, {January} 2026")),
    "https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html":
        ("nvidia2026trtllm", "misc", dict(author="{NVIDIA}", year="2026",
         title="{TensorRT-LLM} performance overview")),
    "https://www.levels.fyi/t/software-engineer/focus/ml-ai":
        ("levels2026ml", "misc", dict(author="{Levels.fyi}", year="2026",
         title="{ML/AI} software engineer compensation")),
    "https://www.levels.fyi/t/software-engineer/locations/south-africa":
        ("levels2026za", "misc", dict(author="{Levels.fyi}", year="2026",
         title="Software engineer compensation in {South Africa}")),
    "https://www.valutafx.com/history/usd-zar-2026-08-17":
        ("valutafx2026", "misc", dict(author="{ValutaFX}", year="2026",
         title="{USD} to {ZAR} exchange rate history, 2026")),
    "https://platform.claude.com/docs/en/about-claude/pricing":
        ("anthropic2026pricing", "misc", dict(author="{Anthropic}", year="2026", title="Pricing")),
    "https://developers.openai.com/api/docs/pricing":
        ("openai2026pricing", "misc", dict(author="{OpenAI}", year="2026", title="{API} pricing")),
    "https://ai.google.dev/gemini-api/docs/pricing":
        ("google2026pricing", "misc", dict(author="{Google}", year="2026",
         title="{Gemini Developer API} pricing")),
    "https://deepinfra.com/pricing":
        ("deepinfra2026pricing", "misc", dict(author="{DeepInfra}", year="2026", title="Pricing")),
    "https://fireworks.ai/pricing":
        ("fireworks2026pricing", "misc", dict(author="{Fireworks AI}", year="2026", title="Pricing")),
    "https://www.together.ai/pricing":
        ("together2026pricing", "misc", dict(author="{Together AI}", year="2026", title="Pricing")),
    "https://thinkingmachines.ai/tinker/":
        ("tml2026tinker", "misc", dict(author="{Thinking Machines Lab}", year="2026", title="Tinker")),
    # Vendor policies, availability and law
    "https://developers.openai.com/api/docs/deprecations":
        ("openai2026deprecations", "misc", dict(author="{OpenAI}", year="2026", title="Deprecations")),
    "https://developers.openai.com/api/docs/guides/your-data":
        ("openai2026data", "misc", dict(author="{OpenAI}", year="2026", title="Your data")),
    "https://platform.claude.com/docs/en/about-claude/model-deprecations":
        ("anthropic2026deprecations", "misc", dict(author="{Anthropic}", year="2026",
         title="Model deprecations")),
    "https://platform.claude.com/docs/en/build-with-claude/data-residency":
        ("anthropic2026residency", "misc", dict(author="{Anthropic}", year="2026",
         title="Data residency")),
    "https://www.anthropic.com/news/fable-mythos-access":
        ("anthropic2026fable", "misc", dict(author="{Anthropic}", year="2026",
         title="Statement on {Fable 5} and {Mythos 5} access", note="12 June 2026")),
    "https://www.anthropic.com/news/redeploying-fable-5":
        ("anthropic2026redeploy", "misc", dict(author="{Anthropic}", year="2026",
         title="Redeploying {Fable 5}", note="30 June 2026")),
    "https://docs.together.ai/docs/lora-inference":
        ("together2026lora", "misc", dict(author="{Together AI}", year="2026", title="{LoRA} inference")),
    "https://aws.amazon.com/blogs/machine-learning/scale-ai-in-south-africa-using-amazon-bedrock-global-cross-region-inference-with-anthropic-claude-4-5-models/":
        ("aws2026bedrockza", "misc", dict(author="{Amazon Web Services}", year="2026",
         title="Scale {AI} in {South Africa} using {Amazon Bedrock} global cross-{Region} inference")),
    "https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure-region-availability":
        ("microsoft2026regions", "misc", dict(author="{Microsoft}", year="2026",
         title="Foundry models sold directly by {Azure}: region availability")),
    "https://docs.cloud.google.com/gemini-enterprise-agent-platform/resources/locations":
        ("google2026locations", "misc", dict(author="{Google Cloud}", year="2026", title="Locations")),
    "https://www.dlapiperdataprotection.com/?c=BW&c2=GB&t=law":
        ("dlapiper2026botswana", "misc", dict(author="{DLA Piper}", year="2026",
         title="Data protection laws of the world: {Botswana}")),
    # Adaptation and specialization
    "https://arxiv.org/abs/2106.09685":
        ("hu2021lora", "misc", dict(author="Hu, E. and others", year="2021",
         title="{LoRA}: Low-Rank Adaptation of Large Language Models", howpublished="arXiv:2106.09685")),
    "https://arxiv.org/abs/2305.14314":
        ("dettmers2023qlora", "misc", dict(
         author="Dettmers, T. and Pagnoni, A. and Holtzman, A. and Zettlemoyer, L.", year="2023",
         title="{QLoRA}: Efficient Finetuning of Quantized {LLMs}", howpublished="arXiv:2305.14314")),
    "https://thinkingmachines.ai/blog/lora/":
        ("schulman2025lora", "misc", dict(author="Schulman, J. and {Thinking Machines Lab}",
         year="2025", title="{LoRA} Without Regret")),
    "https://huggingface.co/blog/mlabonne/sft-llama3":
        ("labonne2024", "misc", dict(author="Labonne, M.", year="2024",
         title="Fine-tune {Llama 3.1} ultra-efficiently with {Unsloth}", howpublished="Hugging Face blog")),
    "https://arxiv.org/abs/2405.00732":
        ("zhao2024loraland", "misc", dict(author="Zhao, J. and others", year="2024",
         title="{LoRA Land}: 310 Fine-tuned {LLMs} that Rival {GPT-4}", howpublished="arXiv:2405.00732")),
    "https://arxiv.org/abs/2506.02153":
        ("belcak2025slm", "misc", dict(author="Belcak, P. and others", year="2025",
         title="Small Language Models are the Future of Agentic {AI}", howpublished="arXiv:2506.02153")),
    "https://aclanthology.org/2023.emnlp-industry.39":
        ("li2023finance", "inproceedings", dict(author="Li, X. and others", year="2023",
         title="Are {ChatGPT} and {GPT-4} General-Purpose Solvers for Financial Text Analytics?",
         booktitle="Proceedings of EMNLP: Industry Track")),
    "https://aclanthology.org/2024.emnlp-main.15":
        ("ovadia2024", "inproceedings", dict(
         author="Ovadia, O. and Brief, M. and Mishaeli, M. and Elisha, O.", year="2024",
         title="Fine-Tuning or Retrieval? {Comparing} Knowledge Injection in {LLMs}",
         booktitle="Proceedings of EMNLP")),
    "https://www.datacamp.com/blog/composer-2":
        ("datacamp2026composer", "misc", dict(author="{DataCamp}", year="2026", title="{Cursor Composer 2}")),
    "https://huggingface.co/blog/state-of-open-models-summer-2026":
        ("hf2026open", "misc", dict(author="{Hugging Face}", year="2026",
         title="State of open models, summer 2026")),
    "https://chenbihuan.github.io/paper/icse26-chen-model-supply-chain.pdf":
        ("chen2026supply", "inproceedings", dict(author="Chen, B. and others", year="2026",
         title="A First Look at Model Supply Chain: From the Risk Perspective",
         booktitle="Proceedings of ICSE")),
    "https://www.uctoday.com/productivity-automation/mozilla-state-of-open-source-ai-2026-report/":
        ("uctoday2026mozilla", "misc", dict(author="{UC Today}", year="2026",
         title="{Mozilla} State of Open Source {AI} 2026 report")),
    # Data quality
    "https://arxiv.org/abs/2307.08701":
        ("chen2024alpagasus", "inproceedings", dict(author="Chen, L. and others", year="2024",
         title="{AlpaGasus}: Training a Better {Alpaca} with Fewer Data", booktitle="Proceedings of ICLR")),
    "https://arxiv.org/abs/2205.10487":
        ("hernandez2022repeated", "misc", dict(author="Hernandez, D. and others", year="2022",
         title="Scaling Laws and Interpretability of Learning from Repeated Data",
         howpublished="arXiv:2205.10487")),
    "https://arxiv.org/abs/2305.16264":
        ("muennighoff2023", "inproceedings", dict(author="Muennighoff, N. and others", year="2023",
         title="Scaling Data-Constrained Language Models", booktitle="Advances in NeurIPS")),
    "https://arxiv.org/abs/2202.06539":
        ("kandpal2022", "inproceedings", dict(author="Kandpal, N. and Wallace, E. and Raffel, C.",
         year="2022", title="Deduplicating Training Data Mitigates Privacy Risks in Language Models",
         booktitle="Proceedings of ICML")),
    "https://arxiv.org/abs/2302.00539":
        ("lukas2023", "inproceedings", dict(
         author="Lukas, N. and Salem, A. and Sim, R. and Tople, S. and Wutschitz, L. and Zanella-B{\\'e}guelin, S.",
         year="2023", title="Analyzing Leakage of Personally Identifiable Information in Language Models",
         booktitle="IEEE Symposium on Security and Privacy")),
    "https://arxiv.org/abs/2311.04850":
        ("yang2023rephrased", "misc", dict(author="Yang, S. and others", year="2023",
         title="Rethinking Benchmark and Contamination for Language Models with Rephrased Samples",
         howpublished="arXiv:2311.04850")),
    "https://arxiv.org/abs/2405.00332":
        ("zhang2024gsm1k", "inproceedings", dict(author="Zhang, H. and others", year="2024",
         title="A Careful Examination of Large Language Model Performance on Grade School Arithmetic",
         booktitle="NeurIPS Datasets and Benchmarks Track")),
    "https://arxiv.org/abs/2412.14922":
        ("luo2024robustft", "misc", dict(author="Luo, J. and Luo, X. and Ding, K. and Yuan, J. and Xiao, Z. and Zhang, M.",
         year="2024", title="{RobustFT}: Robust Supervised Fine-tuning for Large Language Models under Noisy Response",
         howpublished="arXiv:2412.14922")),
    "https://arxiv.org/abs/2404.09717":
        ("cho2024falsity", "misc", dict(author="Cho, H.", year="2024",
         title="Unveiling Imitation Learning: Exploring the Impact of Data Falsity to Large Language Model",
         howpublished="arXiv:2404.09717")),
    "https://arxiv.org/abs/2204.09371":
        ("zhu2022bert", "inproceedings", dict(
         author="Zhu, D. and Hedderich, M. A. and Zhai, F. and Adelani, D. I. and Klakow, D.", year="2022",
         title="Is {BERT} Robust to Label Noise? {A} Study on Learning with Noisy Labels in Text Classification",
         booktitle="Workshop on Insights from Negative Results in NLP, ACL")),
    "https://arxiv.org/abs/2502.07374":
        ("li2025reason", "misc", dict(author="Li, D. and others", year="2025",
         title="{LLMs} Can Easily Learn to Reason from Demonstrations: Structure, not content, is what matters",
         howpublished="arXiv:2502.07374")),
    "https://arxiv.org/abs/2410.08847":
        ("razin2025", "inproceedings", dict(author="Razin, N. and others", year="2025",
         title="Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization",
         booktitle="Proceedings of ICLR")),
    "https://developers.openai.com/cookbook/examples/chat_finetuning_data_prep":
        ("openai2026cookbook", "misc", dict(author="{OpenAI}", year="2026",
         title="Data preparation and analysis for chat model fine-tuning", howpublished="OpenAI Cookbook")),
    # Data and software
    "https://github.com/gururise/AlpacaDataCleaned":
        ("alpacacleaned", "misc", dict(author="{gururise}", year="2023",
         title="{AlpacaDataCleaned}", howpublished="GitHub repository, main branch")),
    "https://github.com/anthropics/hh-rlhf":
        ("hhrlhf", "misc", dict(author="{Anthropic}", year="2022",
         title="{HH-RLHF}: Human preference data about helpfulness and harmlessness",
         howpublished="GitHub repository, master branch")),
    "https://github.com/Thatayotlhe04/Parallelogram":
        ("parallelogram", "misc", dict(author="Tsenang, T.", year="2026",
         title="Parallelogram: a linter for fine-tuning data", howpublished="GitHub repository, version 1.6.1")),
    "https://github.com/huggingface/trl":
        ("trl", "misc", dict(author="von Werra, L. and others", year="2020",
         title="{TRL}: Transformer Reinforcement Learning", howpublished="GitHub repository, version 1.17.1")),
}

# Link texts that name the bibliography author, so running text can use \citet.
AUTHOR_TEXTS = {"Anthropic", "OpenAI", "Google", "DeepInfra", "Fireworks", "Together", "Tinker",
                "NVIDIA", "Levels.fyi", "Epoch AI", "Silicon Data", "TrendForce", "Crypto Briefing",
                "ValutaFX", "Microsoft", "Google Cloud", "AWS", "DLA Piper", "Hugging Face",
                "UC Today", "Together docs", "DataCamp summary"}


def bibtex():
    out = []
    for url, (key, typ, f) in R.items():
        fields = dict(f)
        fields["url"] = url
        fields.setdefault("note", "")
        fields["note"] = (fields["note"] + ". " if fields["note"] else "") + ACCESSED
        body = ",\n".join(f"  {k:<12} = {{{v}}}" for k, v in fields.items())
        out.append(f"@{typ}{{{key},\n{body}\n}}\n")
    return "\n".join(out)
