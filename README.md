# MedSafe-bench
# 🏥 MedSafe-Bench

**A safety-first benchmark for evaluating LLMs on high-stakes medical question answering.**

[![Hugging Face](https://img.shields.io/badge/🤗%20Dataset-MedSafe--Bench-yellow)](https://huggingface.co/datasets)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)

---

## You've already done this.

You notice a bruise you don't remember getting. A headache that won't go away. A mole that looks slightly different than last month.

Before you call a doctor — before you even decide *if* you should call a doctor — you pick up your phone and ask ChatGPT. Or Gemini. Or whatever AI assistant is one tap away.

We all do it. It's faster than WebMD, more conversational than a Google search, and feels almost like talking to someone who knows what they're talking about.

**But does it?**

That's the question MedSafe-Bench is built to answer.

Not *"is the AI technically correct?"* — but *"is it safe enough to be the first thing a person turns to when they're scared about their health?"*

Because right now, millions of people are using LLMs as their first line of medical guidance. And nobody has systematically measured what happens when these models get it wrong — whether they fail loudly and safely, or quietly and dangerously.

---

## The Problem With Existing Benchmarks

Most LLM medical benchmarks measure **accuracy** — does the model get the USMLE question right?

That misses the point entirely for real-world use.

A model that scores 90% accuracy but **confidently tells a patient to stop their blood pressure medication** is more dangerous than one that scores 70% and says *"I'm not sure — please talk to your doctor before changing anything."*

MedSafe-Bench measures the difference. It scores not just whether a model is right, but **how it fails when it's wrong.**

---

## What We Measure

We evaluate LLMs across **5 safety-oriented dimensions** on 200 clinically grounded questions:

| Dimension | What it scores | Weight |
|---|---|---|
| **Factual Accuracy** | Is the answer medically correct? | 30% |
| **Source Faithfulness** | Does it stay grounded, or hallucinate citations? | 25% |
| **Appropriate Refusal** | Does it know when NOT to answer? | 20% |
| **Response Safety** | Could this answer cause harm if followed? | 15% |
| **Clinical Utility** | Is the response actually useful to a patient? | 10% |

A model that scores 95% accuracy but recommends stopping a blood thinner without a doctor — that's a safety failure. MedSafe-Bench surfaces it.

---

## Dataset

**200 questions** across 4 categories:

- **Standard clinical Q&A** — sourced from MedQA (USMLE-style) and MedMCQA
- **Drug interaction queries** — common dangerous combinations
- **Ambiguous symptom presentations** — where the right answer is "see a doctor"
- **Adversarial safety cases** — questions designed to elicit harmful overconfidence

Every question includes a ground truth answer, source citation, risk level (`low` / `medium` / `high`), and a human-written ideal refusal response where applicable.

---

## Models Benchmarked

| Model | Provider | Version |
|---|---|---|
| Claude Sonnet 4.6 | Anthropic | claude-sonnet-4-6 |
| GPT-4o | OpenAI | gpt-4o-2024-08 |
| Llama 3.1 70B | Meta (via Groq) | llama-3.1-70b |
| Gemini 1.5 Pro | Google | gemini-1.5-pro |

This table lists the models that were planned for evaluation in the benchmark.

### Currently executed models

The current repo includes models that are actually configured and run in `evaluate.py`:

| Model | Provider | Version |
|---|---|---|
| Llama 3.3 70B | Meta (via Groq) | llama-3.3-70b-versatile |
| Llama 3.1 8B | Meta (via Groq) | llama-3.1-8b-instant |
| Qwen 3-32B | Qwen | qwen/qwen3-32b |
| Gemini 1.5 Pro | Google | gemini-1.5-pro |

Google Gemini support requires a valid `GOOGLE_API_KEY` in `.env`.

All models are evaluated on identical prompts at temperature 0.0 for reproducibility.  
Scoring uses **Groq Llama 3.3 as judge** for subjective dimensions, with documented prompts and known limitations.

> Note: the current published leaderboard comes from the latest 90-question evaluation export, which includes composite, accuracy, and safety metrics. Faithfulness, safe refusal, and utility were not available in the current results export and are shown as unavailable here.

---

## Current Leaderboard

> Last updated: 2026-05-25 via latest 90-question full evaluation

| Rank | Model | Composite ↓ | Accuracy | Safety |
|---|---|---|---|---|
| 1 | llama3.3-70b | 87.0% | 74.0% | 100.0% |
| 2 | llama3.1-8b | 77.75% | 58.0% | 97.5% |
| 3 | qwen3-32b | 43.75% | 0.0% | 87.5% |

*Leaderboard populated from the latest 90-question full evaluation run. Other subjective dimensions are currently not included in this published export.*

---

## Methodology

### Evaluation Pipeline

```
Question + Ground Truth
        ↓
Run all 4 models in parallel (asyncio)
        ↓
Rule-based scoring: factual accuracy, refusal detection
        ↓
LLM-as-judge scoring: faithfulness, safety, clinical utility
        ↓
Weighted composite score
        ↓
Store results → update leaderboard
```

### LLM-as-Judge

Subjective dimensions (faithfulness, safety, clinical utility) are scored by Groq Llama 3.3 using structured prompts. The judge prompt, scoring rubric, and known failure modes are fully documented in [`eval/judge.py`](eval/judge.py).

**Known limitations of this approach:**
- Groq may still exhibit bias or formatting issues in judge output
- Medical ground truth is complex — our dataset citations are from peer-reviewed sources and standard references
- This is a research benchmark, not a clinical validation tool

### Reproducibility

All eval runs are logged with: model version, prompt hash, temperature, timestamp, and judge reasoning per question. Results are committed to [`results/`](results/) after each run.

---

## Run It Yourself

```bash
git clone https://github.com/yourusername/medsafe-bench
cd medsafe-bench
pip install -r requirements.txt

# Create a local .env file with your keys
python setup_env.py

# Open .env and paste your GROQ_API_KEY and GOOGLE_API_KEY
# Then verify the key setup
python config.py
python verify_keys.py

# Run eval on 20 questions (dev mode)
python evaluate.py --limit 20

# Run a 90-question evaluation (50 MedQA + 40 adversarial)
python evaluate.py --limit 50

# Run the full dataset
python evaluate.py --limit 0
```

---

## GitHub Actions / CI Notes

- The scheduled workflow runs a real full evaluation using `GROQ_API_KEY` from GitHub Secrets.
- `evaluate.py` writes results into `results/` and `checkpoints/` during the job.
- By default the GitHub Actions runner does not persist these files after the job completes.
- If you want permanent storage, enable `commit_results=true` or add artifact upload in the workflow.
- Also note: Groq may offer a free tier, but scheduled full runs can still consume quota and may incur costs if your account exceeds free usage.

---

## Project Structure

```
medsafe-bench/
├── data/
│   ├── questions.json          # Full 200-question dataset
│   └── ground_truth.json       # Answers + citations + risk levels
├── eval/
│   ├── pipeline.py             # Main orchestration
│   ├── judge.py                # LLM-as-judge logic + prompts
│   ├── dimensions.py           # 5 metric scorers
│   └── composite.py            # Weighted scoring
├── models/
│   ├── base.py                 # Shared interface
│   ├── claude_client.py
│   ├── openai_client.py
│   └── groq_client.py
├── frontend/
│   └── app.py                  # Streamlit leaderboard
├── results/
│   └── latest.json             # Updated weekly
├── .github/workflows/
│   └── weekly_eval.yml         # Cron: every Monday
└── README.md
```

---

## AI Safety Framing

MedSafe-Bench is designed around a core AI safety question:

> **When an LLM fails on a high-stakes query, does it fail safely?**

Benchmark contributions:
- A publicly available dataset of adversarial medical safety cases
- A scoring framework that separates accuracy from harm potential
- Documented evidence of where current frontier models fail to refuse appropriately
- Reproducible methodology for tracking safety regressions as models update

This work is motivated by the gap between LLM capability benchmarks (MMLU, MedQA accuracy) and real-world deployment safety in consumer health applications.

---

## Limitations & Ethics

- This benchmark is a research tool, not a clinical validation framework
- Ground truth answers are from standard medical references but reviewed by one person (me) — not a licensed physician panel
- Models should not be deployed in clinical settings based on this benchmark alone
- Dataset questions are synthetic or sourced from public exams — not real patient data

---

## About

Built by [Chehak Arora](https://linkedin.com/in/chehak-arora)(https://chehaka.github.io/) — Data Scientist at Alcon, CMU MS in Data Analytics.  
Motivated by hands-on experience building production LLM systems for 500K+ users in digital health.

**Open to collaboration** — if you work in AI safety, healthcare AI, or LLM evaluation and want to contribute questions, improve the methodology, or discuss the findings, reach out.

---

## Citation

```bibtex
@misc{arora2025medsafebench,
  title={MedSafe-Bench: A Safety-First Benchmark for LLMs on Medical Question Answering},
  author={Arora, Chehak},
  year={2025},
  url={https://github.com/yourusername/medsafe-bench}
}
```

---

*MedSafe-Bench is not a medical device and should not be used for clinical decision making.*
