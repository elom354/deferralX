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

## 4) Collect local-HF logs (Model A)
```bash
!PYTHONPATH=src python -m deferralx.run collect-local-hf-batched \
  --questions data/mmlu_questions_600.csv \
  --output data/real_llm_logs_model_a.csv \
  --audit-jsonl outputs/audit/real_collection_model_a.jsonl \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --device auto \
  --agreement-samples 1 \
  --skip-confidence-pass \
  --max-tokens 12 \
  --batch-size 100
```

## 5) Optional: collect second stronger model (Model B)
Run the exact same protocol with a stronger model id:

```bash
!PYTHONPATH=src python -m deferralx.run collect-local-hf-batched \
  --questions data/mmlu_questions_600.csv \
  --output data/real_llm_logs_model_b.csv \
  --audit-jsonl outputs/audit/real_collection_model_b.jsonl \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --device auto \
  --agreement-samples 1 \
  --skip-confidence-pass \
  --max-tokens 12 \
  --batch-size 100
```

## 6) Validate collection before evaluation
```bash
!PYTHONPATH=src python -m deferralx.run inspect-input \
  --input data/real_llm_logs_model_a.csv \
  --questions data/mmlu_questions_600.csv \
  --min-rows 600 \
  --min-domains 3 \
  --min-profiles 3 \
  --fail-if-not-ready
```

Repeat for `data/real_llm_logs_model_b.csv` if Model B was collected.

## 7) Multi-seed evaluation (primary result)
```bash
!PYTHONPATH=src python -m deferralx.run run-multiseed \
  --input data/real_llm_logs_model_a.csv \
  --outdir outputs/multiseed_model_a \
  --utility-config configs/utility_config.json \
  --test-ratio 0.30 \
  --seeds 7,11,17,23,31 \
  --bootstrap 120
```

Generate report artifacts:

```bash
!PYTHONPATH=src python -m deferralx.run report-multiseed \
  --multiseed-dir outputs/multiseed_model_a \
  --label model_a \
  --outdir outputs/multiseed_model_a/report
```

Repeat the same two commands for Model B with
`data/real_llm_logs_model_b.csv` and `outputs/multiseed_model_b`.

## 8) Compare utility/coverage curves across models
```bash
!PYTHONPATH=src python -m deferralx.run compare-models \
  --runs model_a=outputs/multiseed_model_a model_b=outputs/multiseed_model_b \
  --outdir outputs/model_comparison
```

## 9) View/export key artifacts
```bash
from google.colab import files
files.download('outputs/multiseed_model_a/report/results_section.md')
files.download('outputs/multiseed_model_a/report/utility_coverage_curve.png')
files.download('outputs/model_comparison/model_comparison_results.md')
files.download('outputs/model_comparison/model_utility_coverage_curves.png')
```

## Notes
- Start with `--limit 200` or `--limit 500`, then scale up.
- For finance/medical experiments, repeat with FinanceBench and MedQA datasets and merge question files.
- If a Colab session disconnects, re-run the same batched collect command; it resumes automatically.
