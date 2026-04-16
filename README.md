# DeferralX

Experimental pipeline to study **when an LLM should answer autonomously** vs **when it should defer to a human**.

## Target Contribution
- Routing conditioned on `domain x user profile`.
- Utility optimization under severe-risk constraints.
- Comparison with a **Route-To-Reason-inspired** baseline adapted to human deferral.

## Policies Evaluated
- `always_escalate`
- `global_threshold`
- `domain_threshold`
- `profile_threshold`
- `domain_profile_threshold`
- `rtr_style_threshold_router` (Route-To-Reason-inspired)
- `learned_utility_router`

## Experiment CSV Schema
Required columns:
- `example_id`
- `domain`
- `user_profile`
- `correctness` (0/1)
- `severe_if_wrong` (0/1)
- `p_internal` (0..1)
- `p_verbal` (0..1)
- `agreement` (0..1)
- `response_speed` (`fast` or `careful`)

## Publication Workflow (Real Data)

### 1) Prepare real benchmark questions
Expected collection format:
- `example_id,domain,user_profile,prompt,reference_answer,severe_if_wrong`

Example file: [questions_template.csv](/Users/mac/Desktop/A NE JAMAIS EFFACER/DeferralX/data/questions_template.csv)

### 1b) Build questions from online datasets (recommended)
Install dataset loader:

```bash
pip install datasets
```

Inspect dataset columns first:

```bash
PYTHONPATH=src python3 -m deferralx.prepare_questions \
  --dataset cais/mmlu \
  --subset all \
  --split test \
  --output data/mmlu_questions.csv \
  --question-col question \
  --answer-col answer \
  --domain general \
  --print-columns
```

Then export with explicit column mapping.

MMLU example (multiple-choice):
```bash
PYTHONPATH=src python3 -m deferralx.prepare_questions \
  --dataset cais/mmlu \
  --subset all \
  --split test \
  --output data/mmlu_questions.csv \
  --question-col question \
  --choices-col choices \
  --answer-col answer \
  --answer-is-index \
  --shuffle \
  --sample-seed 42 \
  --domain general \
  --domain-mode mmlu_subject \
  --profile-mode cycle \
  --severe-mode by_domain \
  --limit 600
```

FinanceBench example:
```bash
PYTHONPATH=src python3 -m deferralx.prepare_questions \
  --dataset PatronusAI/financebench \
  --split test \
  --output data/finance_questions.csv \
  --question-col question \
  --answer-col answer \
  --domain finance \
  --default-profile cautious_novice \
  --severe-if-wrong 1
```

MedQA example:
```bash
PYTHONPATH=src python3 -m deferralx.prepare_questions \
  --dataset GBaker/MedQA-USMLE-4-options \
  --split test \
  --output data/medqa_questions.csv \
  --question-col question \
  --choices-col options \
  --answer-col answer_idx \
  --answer-is-index \
  --domain medical \
  --default-profile cautious_novice \
  --severe-if-wrong 1
```

Merge files into a single questions CSV with one header.

```bash
PYTHONPATH=src python3 -m deferralx.merge_questions \
  --inputs data/mmlu_questions.csv data/finance_questions.csv data/medqa_questions.csv \
  --output data/article_questions.csv
```

### 2) Collect logs with a real LLM (API mode)

```bash
export OPENAI_API_KEY="..."
PYTHONPATH=src python3 -m deferralx.run collect-real \
  --questions data/questions_template.csv \
  --output data/real_llm_logs.csv \
  --audit-jsonl outputs/audit/real_collection.jsonl \
  --model gpt-4.1-mini
```

Notes:
- `audit-jsonl` stores raw outputs for traceability.
- No synthetic data is generated in this mode.

### 2b) No API key: local Hugging Face model

Install local dependencies:

```bash
pip install torch transformers sentencepiece
```

Run local collection:

```bash
PYTHONPATH=src python3 -m deferralx.run collect-local-hf \
  --questions data/questions_template.csv \
  --output data/real_llm_logs_local.csv \
  --audit-jsonl outputs/audit/real_collection_local_hf.jsonl \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --device auto
```

Fast + interruption-safe variant (recommended):

```bash
PYTHONPATH=src python3 -m deferralx.run collect-local-hf \
  --questions data/mmlu_questions.csv \
  --output data/real_llm_logs_local.csv \
  --audit-jsonl outputs/audit/real_collection_local_hf.jsonl \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --device auto \
  --agreement-samples 1 \
  --skip-confidence-pass \
  --max-tokens 12 \
  --resume \
  --max-examples 100
```

If a run stops mid-way, rerun the exact same command with `--resume`.

CPU-friendly model suggestions:
- `Qwen/Qwen2.5-0.5B-Instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

### 3) Evaluate routing policies

```bash
PYTHONPATH=src python3 -m deferralx.run run \
  --input data/real_llm_logs.csv \
  --outdir outputs/main \
  --utility-config configs/utility_config.json \
  --test-ratio 0.30 \
  --seed 42 \
  --bootstrap 400
```

### 4) Remove split randomness for publication (recommended)
Use a fixed official split:

```bash
PYTHONPATH=src python3 -m deferralx.run run \
  --train-input data/real_train_logs.csv \
  --test-input data/real_test_logs.csv \
  --input data/real_llm_logs.csv \
  --outdir outputs/main_fixedsplit \
  --utility-config configs/utility_config.json \
  --bootstrap 400
```

## Debug Workflow (Not publishable)
Synthetic data only for pipeline testing:

```bash
PYTHONPATH=src python3 -m deferralx.run generate --output data/synthetic_llm_logs.csv --n 6000 --seed 42
PYTHONPATH=src python3 -m deferralx.run run --input data/synthetic_llm_logs.csv --outdir outputs/main_synth
```

## Outputs
In `outputs/main`:
- `metrics_overall.csv`
- `metrics_overall_ci.csv`
- `metrics_by_domain.csv`
- `metrics_by_profile.csv`
- `metrics_by_domain_profile.csv`
- `policy_rankings.csv`
- `selected_thresholds.json`
- `summary.md`
- `decisions_<policy>.csv`

## Google Colab
For heavy runs, use the notebook-style guide:
- [colab_runbook.md](/Users/mac/Desktop/A NE JAMAIS EFFACER/DeferralX/docs/colab_runbook.md)
