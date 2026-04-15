from __future__ import annotations

import csv
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from deferralx.schema import Record, save_records


QUESTION_REQUIRED_COLUMNS = {
    "example_id",
    "domain",
    "user_profile",
    "prompt",
    "reference_answer",
    "severe_if_wrong",
}


@dataclass
class QuestionRecord:
    example_id: str
    domain: str
    user_profile: str
    prompt: str
    reference_answer: str
    severe_if_wrong: int


class ChatClient(Protocol):
    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        n: int = 1,
        logprobs: bool = False,
    ) -> dict:
        ...


class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", timeout_s: int = 120) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        n: int = 1,
        logprobs: bool = False,
    ) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n,
        }
        if logprobs:
            payload["logprobs"] = True

        return self._post_json("/chat/completions", payload)

    def _post_json(self, route: str, payload: dict) -> dict:
        url = f"{self.base_url}{route}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            method="POST",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} on {url}: {detail}") from e


class LocalHFClient:
    """
    Local Hugging Face backend using transformers.
    Compatible with collect_real_records via the same `chat(...)` interface.
    """

    def __init__(self, model: Any, tokenizer: Any, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        n: int = 1,
        logprobs: bool = False,
    ) -> dict:
        _ = model  # Model is bound at init for local backend.
        choices = []
        for _i in range(max(1, n)):
            answer_text, p_first = self._generate_one(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                want_logprobs=logprobs,
            )
            choice = {"message": {"content": answer_text}}
            if logprobs and p_first is not None:
                choice["logprobs"] = {
                    "content": [{"logprob": math.log(max(p_first, 1e-12))}]
                }
            choices.append(choice)
        return {"choices": choices}

    def _render_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        lines = []
        for m in messages:
            role = m.get("role", "user").strip().upper()
            content = m.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    def _generate_one(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        want_logprobs: bool,
    ) -> tuple[str, float | None]:
        import torch

        prompt = self._render_chat_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        do_sample = temperature > 1e-6
        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask
        if do_sample:
            generate_kwargs["temperature"] = max(temperature, 1e-3)

        with torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)

        new_tokens = output_ids[0, input_ids.shape[1] :]
        answer_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        p_first: float | None = None
        if want_logprobs and new_tokens.numel() > 0:
            with torch.no_grad():
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                token_id = int(new_tokens[0].item())
                p_first = float(probs[0, token_id].item())

        return answer_text, p_first


def load_question_records(path: str | Path) -> list[QuestionRecord]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Question dataset not found: {csv_path}")

    out: list[QuestionRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Question CSV has no header")
        missing = QUESTION_REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Question CSV missing required columns: {sorted(missing)}")

        for row in reader:
            out.append(
                QuestionRecord(
                    example_id=row["example_id"],
                    domain=row["domain"].strip().lower(),
                    user_profile=row["user_profile"].strip().lower(),
                    prompt=row["prompt"],
                    reference_answer=row["reference_answer"],
                    severe_if_wrong=1 if str(row["severe_if_wrong"]).strip() in {"1", "true", "True"} else 0,
                )
            )

    if not out:
        raise ValueError("Question CSV is empty")
    return out


def collect_real_records(
    questions: list[QuestionRecord],
    client: ChatClient,
    model: str,
    max_tokens: int,
    agreement_samples: int,
    agreement_temperature: float,
    fast_latency_s: float,
    system_prompt: str,
    audit_path: str | Path | None,
    output_path: str | Path | None = None,
    append_output: bool = False,
    use_confidence_pass: bool = True,
) -> list[Record]:
    records: list[Record] = []

    audit_file = None
    output_file = None
    output_writer = None
    if audit_path is not None:
        p = Path(audit_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        audit_mode = "a" if append_output and p.exists() else "w"
        audit_file = p.open(audit_mode, encoding="utf-8")
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        exists = p.exists()
        mode = "a" if append_output and exists else "w"
        output_file = p.open(mode, encoding="utf-8", newline="")
        output_writer = csv.writer(output_file)
        if mode == "w":
            output_writer.writerow(
                [
                    "example_id",
                    "domain",
                    "user_profile",
                    "correctness",
                    "severe_if_wrong",
                    "p_internal",
                    "p_verbal",
                    "agreement",
                    "response_speed",
                ]
            )

    try:
        total = len(questions)
        for i, q in enumerate(questions, start=1):
            start = time.time()
            answer_resp = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q.prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                n=1,
                logprobs=True,
            )
            latency_s = time.time() - start

            answer_text = _extract_text(answer_resp, 0)
            p_internal = _extract_first_token_prob(answer_resp, 0)

            confidence_text = "skipped"
            if use_confidence_pass:
                confidence_text = _ask_confidence(
                    client=client,
                    model=model,
                    question=q.prompt,
                    answer=answer_text,
                    max_tokens=80,
                )
                p_verbal = _parse_confidence_number(confidence_text)
            else:
                p_verbal = p_internal if p_internal is not None else 0.5

            agreement = _compute_agreement(
                client=client,
                model=model,
                question=q.prompt,
                reference_answer=answer_text,
                samples=agreement_samples,
                temperature=agreement_temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )

            if p_internal is None:
                # Fallback remains grounded in model outputs, not synthetic labels.
                p_internal = _clamp(0.5 * p_verbal + 0.5 * agreement)

            correctness = 1 if _answers_match(answer_text, q.reference_answer) else 0
            response_speed = "fast" if latency_s <= fast_latency_s else "careful"

            record = Record(
                example_id=q.example_id,
                domain=q.domain,
                user_profile=q.user_profile,
                correctness=correctness,
                severe_if_wrong=q.severe_if_wrong,
                p_internal=p_internal,
                p_verbal=p_verbal,
                agreement=agreement,
                response_speed=response_speed,
            )
            records.append(record)
            if output_writer is not None:
                output_writer.writerow(
                    [
                        record.example_id,
                        record.domain,
                        record.user_profile,
                        record.correctness,
                        record.severe_if_wrong,
                        f"{record.p_internal:.6f}",
                        f"{record.p_verbal:.6f}",
                        f"{record.agreement:.6f}",
                        record.response_speed,
                    ]
                )
                output_file.flush()

            if audit_file is not None:
                audit_payload = {
                    "index": i,
                    "total": total,
                    "example_id": q.example_id,
                    "domain": q.domain,
                    "user_profile": q.user_profile,
                    "prompt": q.prompt,
                    "reference_answer": q.reference_answer,
                    "answer_text": answer_text,
                    "confidence_raw": confidence_text,
                    "p_internal": p_internal,
                    "p_verbal": p_verbal,
                    "agreement": agreement,
                    "correctness": correctness,
                    "response_speed": response_speed,
                    "latency_s": latency_s,
                }
                audit_file.write(json.dumps(audit_payload, ensure_ascii=False) + "\n")

            print(f"[{i}/{total}] {q.example_id} done | correct={correctness} | agreement={agreement:.3f}")
    finally:
        if audit_file is not None:
            audit_file.close()
        if output_file is not None:
            output_file.close()

    return records


def save_real_records(path: str | Path, records: list[Record]) -> None:
    save_records(path, records)


def build_client_from_env(api_key_env: str, base_url: str, timeout_s: int) -> OpenAICompatibleClient:
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env} is empty")
    return OpenAICompatibleClient(api_key=api_key, base_url=base_url, timeout_s=timeout_s)


def build_local_hf_client(
    model_id_or_path: str,
    device: str = "auto",
    use_fp16: bool = False,
) -> LocalHFClient:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Local HF mode requires `torch` and `transformers`. "
            "Install them first, e.g. `pip install torch transformers sentencepiece`."
        ) from e

    resolved_device = _resolve_device(device, torch)

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = torch.float32
    if use_fp16 and resolved_device in {"cuda", "mps"}:
        model_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        dtype=model_dtype,
    )
    model.to(resolved_device)
    model.eval()

    return LocalHFClient(model=model, tokenizer=tokenizer, device=resolved_device)


def _ask_confidence(
    client: ChatClient,
    model: str,
    question: str,
    answer: str,
    max_tokens: int,
) -> str:
    prompt = (
        "Estimate the probability that the proposed answer is correct for the user question. "
        "Return strict JSON with one key only: p_correct (float in [0,1]).\n\n"
        f"Question:\n{question}\n\n"
        f"Proposed answer:\n{answer}\n"
    )
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
        n=1,
        logprobs=False,
    )
    return _extract_text(resp, 0)


def _compute_agreement(
    client: ChatClient,
    model: str,
    question: str,
    reference_answer: str,
    samples: int,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
) -> float:
    if samples <= 0:
        return 0.0

    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        n=samples,
        logprobs=False,
    )

    matches = 0
    ref_norm = _normalize_answer(reference_answer)
    choices = resp.get("choices", [])
    for j in range(min(samples, len(choices))):
        text = _extract_text(resp, j)
        if _normalize_answer(text) == ref_norm:
            matches += 1
    return matches / float(samples)


def _extract_text(resp: dict, index: int) -> str:
    choices = resp.get("choices", [])
    if index >= len(choices):
        return ""
    message = choices[index].get("message", {})
    return str(message.get("content", "")).strip()


def _extract_first_token_prob(resp: dict, index: int) -> float | None:
    choices = resp.get("choices", [])
    if index >= len(choices):
        return None
    choice = choices[index]
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None

    content = logprobs.get("content")
    if not isinstance(content, list) or len(content) == 0:
        return None
    first = content[0]
    lp = first.get("logprob")
    if lp is None:
        return None
    try:
        return _clamp(math.exp(float(lp)))
    except Exception:
        return None


def _parse_confidence_number(text: str) -> float:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "p_correct" in parsed:
            return _clamp(float(parsed["p_correct"]))
    except Exception:
        pass

    match = re.search(r"([01](?:\.\d+)?)", text)
    if match:
        return _clamp(float(match.group(1)))
    return 0.5


def _answers_match(prediction: str, reference: str) -> bool:
    pred_choice = _extract_option(prediction)
    ref_choice = _extract_option(reference)
    if pred_choice is not None and ref_choice is not None:
        return pred_choice == ref_choice

    pred_num = _extract_number(prediction)
    ref_num = _extract_number(reference)
    if pred_num is not None and ref_num is not None:
        tol = max(1e-3, 0.01 * max(1.0, abs(ref_num)))
        return abs(pred_num - ref_num) <= tol

    pred_norm = _normalize_answer(prediction)
    refs = [r.strip() for r in reference.split("||") if r.strip()]
    if not refs:
        refs = [reference]

    for r in refs:
        if pred_norm == _normalize_answer(r):
            return True
    return False


def _extract_option(text: str) -> str | None:
    m = re.search(r"\b([A-E])\b", text.upper())
    if m:
        return m.group(1)
    return None


def _extract_number(text: str) -> float | None:
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _normalize_answer(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9\.\- ]", "", t)
    return t.strip()


def _clamp(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _resolve_device(requested: str, torch_mod: Any) -> str:
    req = requested.strip().lower()
    if req in {"cpu", "cuda", "mps"}:
        return req
    if req != "auto":
        raise ValueError("device must be one of: auto, cpu, cuda, mps")

    if torch_mod.cuda.is_available():
        return "cuda"
    has_mps = hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available()
    if has_mps:
        return "mps"
    return "cpu"
