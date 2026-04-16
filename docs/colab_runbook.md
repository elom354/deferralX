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
  --output data/mmlu_questions_600.csv \
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

## 4) Collect local-HF logs
```bash
!PYTHONPATH=src python -m deferralx.run collect-local-hf-batched \
  --questions data/mmlu_questions_600.csv \
  --output data/real_llm_logs_local_v2.csv \
  --audit-jsonl outputs/audit/real_collection_local_hf_v2.jsonl \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --device auto \
  --agreement-samples 1 \
  --skip-confidence-pass \
  --max-tokens 12 \
  --batch-size 100
```

## 5) Validate collection before evaluation
```bash
!PYTHONPATH=src python -m deferralx.run inspect-input \
  --input data/real_llm_logs_local_v2.csv \
  --questions data/mmlu_questions_600.csv \
  --min-rows 600 \
  --min-domains 3 \
  --min-profiles 3 \
  --fail-if-not-ready
```

## 6) Evaluate routing policies
```bash
!PYTHONPATH=src python -m deferralx.run run \
  --input data/real_llm_logs_local_v2.csv \
  --outdir outputs/main_colab_v2 \
  --utility-config configs/utility_config.json \
  --test-ratio 0.30 \
  --seed 42 \
  --bootstrap 200
```

## 7) View results
```bash
!cat outputs/main_colab_v2/summary.md
!head -n 20 outputs/main_colab_v2/metrics_overall.csv
```

## 8) Optional: export artifacts
```bash
from google.colab import files
files.download('outputs/main_colab_v2/summary.md')
files.download('outputs/main_colab_v2/metrics_overall.csv')
```

## Notes
- Start with `--limit 200` or `--limit 500`, then scale up.
- For finance/medical experiments, repeat with FinanceBench and MedQA datasets and merge question files.
- If a Colab session disconnects, re-run the same batched collect command; it resumes automatically.
