# Google Colab Runbook (Recommended for Heavy Jobs)

This guide runs the full DeferralX pipeline on Colab with a small Hugging Face LLM.

## 0) Create Colab Runtime
- Runtime -> Change runtime type -> GPU (T4 is enough for 0.5B models).

## 1) Clone repository
```bash
!git clone <YOUR_REPO_URL>
%cd DeferralX
```

## 2) Install dependencies
```bash
!python -m pip install --upgrade pip
!pip install "numpy<2" torch transformers datasets sentencepiece accelerate
```

## 3) Build real questions from online benchmark (MMLU)
```bash
!PYTHONPATH=src python -m deferralx.prepare_questions \
  --dataset cais/mmlu \
  --subset all \
  --split test \
  --output data/mmlu_questions_500.csv \
  --question-col question \
  --choices-col choices \
  --answer-col answer \
  --answer-is-index \
  --domain general \
  --domain-mode mmlu_subject \
  --profile-mode cycle \
  --severe-mode by_domain \
  --limit 600
```

## 4) Collect local-HF logs
```bash
!PYTHONPATH=src python -m deferralx.run collect-local-hf \
  --questions data/mmlu_questions_600.csv \
  --output data/real_llm_logs_local.csv \
  --audit-jsonl outputs/audit/real_collection_local_hf.jsonl \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --device auto \
  --agreement-samples 1 \
  --skip-confidence-pass \
  --max-tokens 96 \
  --resume \
  --max-examples 100
```

## 5) Evaluate routing policies
```bash
!PYTHONPATH=src python -m deferralx.run run \
  --input data/real_llm_logs_local.csv \
  --outdir outputs/main_colab \
  --utility-config configs/utility_config.json \
  --test-ratio 0.30 \
  --seed 42 \
  --bootstrap 200
```

## 6) View results
```bash
!cat outputs/main_colab/summary.md
!head -n 20 outputs/main_colab/metrics_overall.csv
```

## 7) Optional: export artifacts
```bash
from google.colab import files
files.download('outputs/main_colab/summary.md')
files.download('outputs/main_colab/metrics_overall.csv')
```

## Notes
- Start with `--limit 200` or `--limit 500`, then scale up.
- For finance/medical experiments, repeat with FinanceBench and MedQA datasets and merge question files.
- If a Colab session disconnects, re-run the same collect command with `--resume` to continue from where it stopped.
- If `--max-examples 100` is used, rerun the same collect cell until it prints `No remaining questions to process.`
