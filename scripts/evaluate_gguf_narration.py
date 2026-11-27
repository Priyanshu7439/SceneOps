#!/usr/bin/env python3

"""Resumable model-level evaluator for the SceneOps GGUF narration model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

EXPECTED_SHA256 = "20678f5909fd531805ef0f4da152108ca917c67ff4104d87d682cb718441e89a"

DEFAULT_DATASET = r"D:\Desktop\SceneOps\data\processed\sceneops_narration_test.jsonl"

DEFAULT_OUTPUT = r"D:\Desktop\SceneOps\artifacts\evaluation\gguf_blind"

DEFAULT_URL = "http://localhost:8081/v1"

DEFAULT_MODEL = "/mnt/d/Desktop/SceneOps_models/gguf/sceneops-q4_k_m.gguf"

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(lambda: f.read(1024 * 1024), b""):

            h.update(chunk)

    return h.hexdigest()

def load_dataset(path: Path) -> list[dict]:

    rows = []

    with path.open(encoding="utf-8") as f:

        for line_no, line in enumerate(f, 1):

            if not line.strip():

                continue

            try:

                row = json.loads(line)

            except json.JSONDecodeError as e:

                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e

            messages = row.get("messages")

            if not isinstance(messages, list) or [m.get("role") for m in messages] != ["system", "user", "assistant"]:

                raise ValueError(f"Line {line_no}: expected system/user/assistant messages in order")

            if not all(isinstance(m.get("content"), str) and m["content"].strip() for m in messages):

                raise ValueError(f"Line {line_no}: empty or invalid message content")

            rows.append(row)

    if not rows:

        raise ValueError("Dataset is empty")

    return rows

def classify_task(prompt: str) -> str:

    p = prompt.lstrip().casefold()

    if p.startswith("compare") or "compare the recorded" in p[:160]:

        return "temporal"

    if p.startswith("summarize") or "summarize the" in p[:160]:

        return "scene"

    raise ValueError(f"Unrecognized task prompt: {prompt[:120]!r}")

def read_jsonl(path: Path) -> list[dict]:

    if not path.exists():

        return []

    out = []

    with path.open(encoding="utf-8") as f:

        for n, line in enumerate(f, 1):

            if line.strip():

                try:

                    out.append(json.loads(line))

                except json.JSONDecodeError as e:

                    raise ValueError(f"Corrupt {path}, line {n}: {e}") from e

    return out

def append_jsonl(path: Path, row: dict) -> None:

    with path.open("a", encoding="utf-8", newline="\n") as f:

        f.write(json.dumps(row, ensure_ascii=False) + "\n")

        f.flush()

def atomic_json(path: Path, value: dict) -> None:

    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    tmp.replace(path)

def infer(url: str, model: str, messages: list[dict], timeout: float, max_tokens: int) -> tuple[str, str | None]:

    payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens, "stream": False}

    req = Request(url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")

    try:

        with urlopen(req, timeout=timeout) as response:

            data = json.loads(response.read().decode("utf-8"))

    except HTTPError as e:

        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:800]}") from e

    except (URLError, TimeoutError, OSError) as e:

        raise RuntimeError(f"Request failed: {e}") from e

    try:

        choice = data["choices"][0]

        content = choice["message"]["content"]

    except (KeyError, IndexError, TypeError) as e:

        raise RuntimeError(f"Unexpected response: {str(data)[:800]}") from e

    if not isinstance(content, str) or not content.strip():

        raise RuntimeError("Model returned empty content")

    return content.strip(), choice.get("finish_reason")

def score(reference: str, prediction: str) -> dict:

    rt, pt = TOKEN_RE.findall(reference.casefold()), TOKEN_RE.findall(prediction.casefold())

    rc, pc = Counter(rt), Counter(pt)

    overlap = sum((rc & pc).values())

    precision = overlap / len(pt) if pt else 0.0

    recall = overlap / len(rt) if rt else 0.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    norm = lambda s: " ".join(s.casefold().split())

    return {"exact_match": norm(reference) == norm(prediction), "token_precision": precision,

            "token_recall": recall, "token_f1": f1,

            "sequence_similarity": SequenceMatcher(None, norm(reference), norm(prediction)).ratio()}

def aggregate(rows: list[dict]) -> dict:

    if not rows:

        return {"examples": 0}

    keys = ("exact_match", "token_precision", "token_recall", "token_f1", "sequence_similarity")

    result = {"examples": len(rows)}

    for key in keys:

        result[key] = sum(float(r["metrics"][key]) for r in rows) / len(rows)

    return result

def main() -> int:

    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument("--dataset", default=DEFAULT_DATASET)

    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT)

    ap.add_argument("--base-url", default=DEFAULT_URL)

    ap.add_argument("--model", default=DEFAULT_MODEL)

    ap.add_argument("--timeout", type=float, default=180.0)

    ap.add_argument("--max-tokens", type=int, default=256)

    ap.add_argument("--retries", type=int, default=2)

    args = ap.parse_args()

    if args.timeout <= 0 or args.max_tokens < 1 or args.retries < 0:

        ap.error("timeout/max-tokens must be positive and retries non-negative")

    dataset = Path(args.dataset)

    out = Path(args.output_dir)

    out.mkdir(parents=True, exist_ok=True)

    digest = sha256_file(dataset)

    print("Dataset SHA-256:", digest)

    if digest.casefold() != EXPECTED_SHA256:

        raise SystemExit("STOP: test dataset hash mismatch; no inference performed")

    records = load_dataset(dataset)

    if len(records) != 1181:

        raise SystemExit(f"STOP: expected 1181 examples, found {len(records)}")

    items = []

    for i, row in enumerate(records):

        # The dataset stores system, user, and assistant messages. The assistant
        # message is the target/reference and MUST NOT be sent to the model.
        source_messages = row["messages"]
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in source_messages[:2]
        ]
        reference = source_messages[2]["content"]

        if [m["role"] for m in messages] != ["system", "user"]:
            raise SystemExit(f"STOP: invalid inference roles at dataset index {i}")
        if any(m["content"] == reference for m in messages):
            raise SystemExit(
                f"STOP: assistant reference appears verbatim in input messages at index {i}"
            )

        items.append({
            "index": i,
            "task": classify_task(messages[1]["content"]),
            "messages": messages,
            "reference": reference,
        })

    task_counts = Counter(x["task"] for x in items)

    if task_counts != Counter({"scene": 562, "temporal": 619}):

        raise SystemExit(f"STOP: unexpected task distribution: {dict(task_counts)}")

    pred_path, fail_path = out / "predictions.jsonl", out / "failures.jsonl"

    manifest_path, metrics_path = out / "evaluation_manifest.json", out / "metrics.json"

    summary_path = out / "evaluation_summary.md"

    fingerprint = {"dataset_sha256": digest, "examples": len(records), "model": args.model,

                   "base_url": args.base_url.rstrip("/"), "temperature": 0.0,

                   "max_tokens": args.max_tokens, "prompt_policy": "system/user only; assistant target excluded",

                   "metric_version": 1}

    if manifest_path.exists():

        old = json.loads(manifest_path.read_text(encoding="utf-8"))

        if old.get("fingerprint") != fingerprint:

            raise SystemExit("STOP: output directory belongs to another dataset/model/config; choose a new directory")

    else:

        atomic_json(manifest_path, {"created_at_utc": datetime.now(UTC).isoformat(),

                    "fingerprint": fingerprint, "status": "in_progress", "task_distribution": dict(task_counts)})

    existing = read_jsonl(pred_path)

    completed = {}

    for row in existing:

        i = row.get("index")

        if not isinstance(i, int) or not 0 <= i < len(items) or i in completed:

            raise SystemExit("STOP: invalid or duplicate index in predictions.jsonl")

        if row.get("dataset_sha256") != digest or row.get("fingerprint") != fingerprint:

            raise SystemExit("STOP: existing prediction provenance/config mismatch")
        if row.get("task") != items[i]["task"] or row.get("reference") != items[i]["reference"]:
            raise SystemExit(f"STOP: existing prediction target/task mismatch at index {i}")
        if not isinstance(row.get("prediction"), str) or not row["prediction"].strip():
            raise SystemExit(f"STOP: missing/empty prediction at index {i}")

        completed[i] = row

    failures = read_jsonl(fail_path)

    print(f"Examples={len(items)} scene={task_counts['scene']} temporal={task_counts['temporal']} already_done={len(completed)}")

    print("Calling:", args.base_url.rstrip("/") + "/chat/completions")

    for item in items:

        i = item["index"]

        if i in completed:

            continue

        error = None

        for attempt in range(args.retries + 1):

            try:

                prediction, finish_reason = infer(args.base_url, args.model, item["messages"], args.timeout, args.max_tokens)

                row = {"index": i, "task": item["task"], "prediction": prediction,

                       "reference": item["reference"], "finish_reason": finish_reason,

                       "metrics": score(item["reference"], prediction), "dataset_sha256": digest,

                       "fingerprint": fingerprint, "completed_at_utc": datetime.now(UTC).isoformat()}

                append_jsonl(pred_path, row)

                completed[i] = row

                print(f"[{len(completed):4d}/{len(items)}] index={i} {item['task']} F1={row['metrics']['token_f1']:.4f}")

                error = None

                break

            except RuntimeError as exc:

                error = str(exc)

                if attempt < args.retries:

                    time.sleep(2 * (attempt + 1))

        if error is not None:

            failure = {"index": i, "task": item["task"], "error": error,

                       "attempts": args.retries + 1, "dataset_sha256": digest,

                       "fingerprint": fingerprint, "failed_at_utc": datetime.now(UTC).isoformat()}

            append_jsonl(fail_path, failure)

            failures.append(failure)

            print(f"FAILED index={i}: {error}", file=sys.stderr)

    final_rows = read_jsonl(pred_path)

    completed_indices = {r["index"] for r in final_rows}
    if len(completed_indices) != len(final_rows):
        raise SystemExit("STOP: duplicate prediction indices found in final predictions.jsonl")
    if any(not isinstance(i, int) or i < 0 or i >= len(items) for i in completed_indices):
        raise SystemExit("STOP: out-of-range prediction index found in final predictions.jsonl")

    # Keep the failure log as an audit trail, but only count failures that

    # still lack a successful prediction after this (possibly resumed) run.

    failure_history = read_jsonl(fail_path)

    unresolved = {}

    for failure in failure_history:

        if failure.get("index") not in completed_indices:

            unresolved[failure["index"]] = failure

    final_failures = list(unresolved.values())

    scene = [r for r in final_rows if r["task"] == "scene"]

    temporal = [r for r in final_rows if r["task"] == "temporal"]

    report = {"completed_examples": len(final_rows), "failed_examples": len(final_failures),

              "overall": aggregate(final_rows), "by_task": {"scene": aggregate(scene), "temporal": aggregate(temporal)},

              "dataset_sha256": digest, "model": args.model,

              "generation": {"temperature": 0.0, "max_tokens": args.max_tokens,

                             "endpoint": args.base_url.rstrip("/") + "/chat/completions"},

              "metric_note": "Blind inference: only system/user messages were sent; assistant target was excluded. Lexical metrics do not measure factual grounding, clinical correctness, or safety.",

              "finished_at_utc": datetime.now(UTC).isoformat()}

    atomic_json(metrics_path, report)

    done = len(final_rows) == len(items) and not final_failures

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest.update({"status": "completed" if done else "incomplete", "completed_examples": len(final_rows),

                     "failed_examples": len(final_failures), "finished_at_utc": report["finished_at_utc"]})

    atomic_json(manifest_path, manifest)

    lines = ["# SceneOps GGUF Evaluation", "", f"- Status: **{manifest['status']}**",

             f"- Dataset examples: {len(items)}", f"- Completed: {len(final_rows)}",

             f"- Failed: {len(final_failures)}", f"- Dataset SHA-256: `{digest}`",

             f"- Model: `{args.model}`", f"- Generation: temperature 0.0, max_tokens {args.max_tokens}", "",

             "> Lexical metrics do not establish grounding, factual correctness, clinical correctness, or safety.", "",

             "| Split | N | Exact match | Token P | Token R | Token F1 | Sequence similarity |",

             "|---|---:|---:|---:|---:|---:|---:|"]

    for label, values in [("Overall", aggregate(final_rows)), ("Scene", aggregate(scene)), ("Temporal", aggregate(temporal))]:

        if values["examples"]:

            lines.append(f"| {label} | {values['examples']} | {values['exact_match']:.4f} | {values['token_precision']:.4f} | {values['token_recall']:.4f} | {values['token_f1']:.4f} | {values['sequence_similarity']:.4f} |")

        else:

            lines.append(f"| {label} | 0 | â€” | â€” | â€” | â€” | â€” |")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nStatus:", manifest["status"])

    for p in (pred_path, fail_path, metrics_path, manifest_path, summary_path):

        print(p)

    return 0 if done else 2

if __name__ == "__main__":

    raise SystemExit(main())
