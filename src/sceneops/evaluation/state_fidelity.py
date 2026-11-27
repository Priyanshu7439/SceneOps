from __future__ import annotations

import re
import unicodedata
from typing import Any

FIELD_LABELS = {
    "respiratory_rate": "respiratory rate",
    "radial_pulse": "radial pulse",
    "capillary_refill": "capillary refill",
    "mental_status": "mental status",
    "ambulatory_status": "ambulatory status",
    "visible_hemorrhage": "visible hemorrhage",
    "follows_commands": "follows commands",
}
FIELD_ORDER = tuple(FIELD_LABELS)
PATIENT_RE = re.compile(r"\bP\d{3}\b", re.IGNORECASE)


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value.rstrip(" .;:")


def _facts_to_panel(facts: list[dict[str, str]]) -> dict[str, str]:
    panel: dict[str, str] = {}
    for item in facts:
        fact = item["fact"]
        field = fact.rsplit(":", 1)[-1]
        value = re.sub(
            r"\s*\(observed at simulation second \d+\)\s*$",
            "",
            item["value"],
        ).strip()
        if field in panel:
            raise ValueError(f"Duplicate ground-truth field: {field}")
        panel[field] = value
    return panel


def ground_truth_panels(ground_truth: dict[str, Any]) -> tuple[str, dict[int, dict[str, str]]]:
    patient_id = str(ground_truth["patient_id"]).upper()
    if "observations" in ground_truth:
        timestamp = int(ground_truth["observation_time_seconds"])
        return patient_id, {
            timestamp: _facts_to_panel(ground_truth["observations"])
        }

    initial_time = int(ground_truth["initial_time_seconds"])
    follow_time = int(ground_truth["follow_up_time_seconds"])
    if initial_time == follow_time:
        raise ValueError("Temporal ground truth has duplicate timestamps.")
    panels = {
        initial_time: _facts_to_panel(ground_truth["initial_observations"]),
        follow_time: _facts_to_panel(ground_truth["follow_up_observations"]),
    }
    return patient_id, panels


def _parse_panel(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    parts = [part.strip() for part in text.split(";")]
    for part in parts:
        if ":" not in part:
            raise ValueError(f"Malformed observation claim: {part!r}")
        label, value = part.split(":", 1)
        label = normalize(label)
        field = next(
            (key for key, expected_label in FIELD_LABELS.items()
             if normalize(expected_label) == label),
            None,
        )
        if field is None:
            raise ValueError(f"Unsupported observation label: {label!r}")
        if field in result:
            raise ValueError(f"Duplicate observation claim: {field}")
        if not value.strip():
            raise ValueError(f"Empty observation value: {field}")
        result[field] = value.strip()
    return result


def _compare_panel(expected: dict[str, str], actual: dict[str, str], timestamp: int) -> list[dict]:
    issues: list[dict] = []
    for field in FIELD_ORDER:
        if field not in expected:
            continue
        if field not in actual:
            issues.append({
                "type": "missing_observation",
                "timestamp": timestamp,
                "field": field,
                "expected": expected[field],
            })
        elif normalize(actual[field]) != normalize(expected[field]):
            issues.append({
                "type": "incorrect_observation_value",
                "timestamp": timestamp,
                "field": field,
                "expected": expected[field],
                "actual": actual[field],
            })
    for field in actual.keys() - expected.keys():
        issues.append({
            "type": "unsupported_observation",
            "timestamp": timestamp,
            "field": field,
            "actual": actual[field],
        })
    return issues


def _parse_change_claims(text: str, initial: dict[str, str], final: dict[str, str]) -> list[dict]:
    issues: list[dict] = []
    expected_changed = {
        field for field in initial.keys() & final.keys()
        if normalize(initial[field]) != normalize(final[field])
    }
    expected_unchanged = {
        field for field in initial.keys() & final.keys()
        if normalize(initial[field]) == normalize(final[field])
    }

    sections = re.fullmatch(
        r"\s*Observed changes:\s*(.*?)\.\s*Unchanged observations:\s*(.*?)\.\s*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not sections:
        return [{"type": "malformed_change_summary", "actual": text}]

    changed_text, unchanged_text = sections.groups()
    parsed_changed: dict[str, tuple[str, str]] = {}
    if changed_text.strip():
        for claim in changed_text.split(";"):
            match = re.fullmatch(
                r"\s*(respiratory rate|radial pulse|capillary refill|mental status|"
                r"ambulatory status|visible hemorrhage|follows commands)"
                r"\s+changed\s+from\s+(.+?)\s+to\s+(.+?)\s*",
                claim,
                flags=re.IGNORECASE,
            )
            if not match:
                issues.append({"type": "unsupported_or_malformed_change_claim", "claim": claim.strip()})
                continue
            label, old_value, new_value = match.groups()
            field = next(key for key, value in FIELD_LABELS.items() if value == normalize(label))
            if field in parsed_changed:
                issues.append({"type": "duplicate_change_claim", "field": field})
            parsed_changed[field] = (old_value.strip(), new_value.strip())

    parsed_unchanged: dict[str, str] = {}
    if unchanged_text.strip():
        for claim in unchanged_text.split(";"):
            match = re.fullmatch(
                r"\s*(respiratory rate|radial pulse|capillary refill|mental status|"
                r"ambulatory status|visible hemorrhage|follows commands)"
                r"\s+remained\s+(.+?)\s*",
                claim,
                flags=re.IGNORECASE,
            )
            if not match:
                issues.append({"type": "unsupported_or_malformed_unchanged_claim", "claim": claim.strip()})
                continue
            label, value = match.groups()
            field = next(key for key, expected in FIELD_LABELS.items() if expected == normalize(label))
            if field in parsed_unchanged:
                issues.append({"type": "duplicate_unchanged_claim", "field": field})
            parsed_unchanged[field] = value.strip()

    for field in expected_changed:
        expected_pair = (initial[field], final[field])
        actual_pair = parsed_changed.get(field)
        if actual_pair is None:
            issues.append({"type": "missing_change_claim", "field": field})
        elif tuple(map(normalize, actual_pair)) != tuple(map(normalize, expected_pair)):
            issues.append({
                "type": "incorrect_change_claim",
                "field": field,
                "expected_from": expected_pair[0],
                "expected_to": expected_pair[1],
                "actual_from": actual_pair[0],
                "actual_to": actual_pair[1],
            })
    for field in parsed_changed.keys() - expected_changed:
        issues.append({"type": "false_change_claim", "field": field})

    for field in expected_unchanged:
        expected_value = initial[field]
        actual_value = parsed_unchanged.get(field)
        if actual_value is None:
            issues.append({"type": "missing_unchanged_claim", "field": field})
        elif normalize(actual_value) != normalize(expected_value):
            issues.append({
                "type": "incorrect_unchanged_claim",
                "field": field,
                "expected": expected_value,
                "actual": actual_value,
            })
    for field in parsed_unchanged.keys() - expected_unchanged:
        issues.append({"type": "false_unchanged_claim", "field": field})
    return issues


def evaluate_prediction(row: dict[str, Any]) -> dict[str, Any]:
    truth = row["ground_truth"]
    expected_patient, expected_panels = ground_truth_panels(truth)
    prediction = str(row.get("prediction", ""))
    issues: list[dict] = []

    all_patient_mentions = [item.upper() for item in PATIENT_RE.findall(prediction)]
    if not all_patient_mentions:
        issues.append({"type": "missing_patient_id"})
    elif any(item != expected_patient for item in all_patient_mentions):
        issues.append({
            "type": "patient_identity_mismatch",
            "expected": expected_patient,
            "actual_mentions": all_patient_mentions,
        })

    expected_times = sorted(expected_panels)
    timestamp_mentions = [
        int(value)
        for value in re.findall(r"\bsimulation second\s+(\d+)\b", prediction, re.IGNORECASE)
    ]
    # The exact sentence grammar below also ensures each panel timestamp occurs once.
    if sorted(set(timestamp_mentions)) != expected_times:
        issues.append({
            "type": "timestamp_set_mismatch",
            "expected": expected_times,
            "actual": timestamp_mentions,
        })

    panel_claims: dict[int, dict[str, str]] = {}
    change_issues: list[dict] = []

    if len(expected_panels) == 1:
        timestamp = expected_times[0]
        match = re.fullmatch(
            r"\s*At simulation second\s+(\d+),\s+the learner recorded "
            r"(?:the following observations )?for patient\s+(P\d{3}):\s*(.*?)\.\s*",
            prediction,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            # Alternate supported scene phrasing, still closed grammar.
            match = re.fullmatch(
                r"\s*At simulation second\s+(\d+),\s+the learner noted "
                r"(?:the following observations )?for patient\s+(P\d{3}):\s*(.*?)\.\s*",
                prediction,
                flags=re.IGNORECASE | re.DOTALL,
            )
        if not match:
            issues.append({"type": "unsupported_or_extra_narrative_content"})
        else:
            actual_time, patient, panel_text = match.groups()
            if int(actual_time) != timestamp:
                issues.append({"type": "timestamp_mismatch", "expected": timestamp, "actual": int(actual_time)})
            if patient.upper() != expected_patient:
                issues.append({"type": "patient_identity_mismatch", "expected": expected_patient, "actual": patient})
            try:
                panel_claims[int(actual_time)] = _parse_panel(panel_text)
            except ValueError as exc:
                issues.append({"type": "malformed_panel", "detail": str(exc)})
    else:
        times = expected_times
        pattern = re.compile(
            r"\s*For patient\s+(P\d{3}),\s+at simulation second\s+(\d+),\s+"
            r"the learner (?:recorded|noted):\s*(.*?)\.\s*"
            r"At simulation second\s+(\d+),\s+the learner (?:recorded|noted):\s*(.*?)\.\s*"
            r"(?:Observed changes:\s*(.*?)\.\s*)?"
            r"Unchanged observations:\s*(.*?)\.\s*",
            flags=re.IGNORECASE | re.DOTALL,
        )
        match = pattern.fullmatch(prediction)
        if not match:
            issues.append({"type": "unsupported_or_extra_narrative_content"})
        else:
            patient, t1, p1, t2, p2, changed_text, unchanged_text = match.groups()
            changed_text = changed_text or ""
            if patient.upper() != expected_patient:
                issues.append({"type": "patient_identity_mismatch", "expected": expected_patient, "actual": patient})
            if [int(t1), int(t2)] != times:
                issues.append({"type": "timestamp_order_or_value_mismatch", "expected": times, "actual": [int(t1), int(t2)]})
            try:
                panel_claims[int(t1)] = _parse_panel(p1)
                panel_claims[int(t2)] = _parse_panel(p2)
            except ValueError as exc:
                issues.append({"type": "malformed_panel", "detail": str(exc)})
            change_issues = _parse_change_claims(
                f"Observed changes: {changed_text}. Unchanged observations: {unchanged_text}.",
                expected_panels[times[0]],
                expected_panels[times[1]],
            )

    for timestamp, expected_panel in expected_panels.items():
        actual_panel = panel_claims.get(timestamp, {})
        issues.extend(_compare_panel(expected_panel, actual_panel, timestamp))

    issues.extend(change_issues)
    expected_field_count = sum(len(panel) for panel in expected_panels.values())
    actual_present = sum(
        1 for timestamp, panel in panel_claims.items()
        for field in panel if field in expected_panels.get(timestamp, {})
    )
    actual_correct = sum(
        1 for timestamp, panel in panel_claims.items()
        for field, value in panel.items()
        if field in expected_panels.get(timestamp, {})
        and normalize(value) == normalize(expected_panels[timestamp][field])
    )
    identity_ok = not any(issue["type"].startswith("patient_identity") or issue["type"] == "missing_patient_id" for issue in issues)
    timestamp_ok = not any("timestamp" in issue["type"] for issue in issues)
    coverage = actual_present / expected_field_count if expected_field_count else 0.0
    value_accuracy = actual_correct / expected_field_count if expected_field_count else 0.0

    return {
        "index": row.get("index"),
        "task": row.get("task"),
        "scenario_id": truth.get("scenario_id"),
        "patient_id_expected": expected_patient,
        "patient_identity_correct": identity_ok,
        "timestamp_accuracy": timestamp_ok,
        "observation_coverage": coverage,
        "observation_value_accuracy": value_accuracy,
        "state_fidelity_pass": not issues,
        "issues": issues,
    }


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    results = [evaluate_prediction(row) for row in rows]
    total = len(results)
    summary: dict[str, Any] = {
        "examples": total,
        "state_fidelity_pass_rate": (
            sum(result["state_fidelity_pass"] for result in results) / total if total else 0.0
        ),
        "patient_identity_accuracy": (
            sum(result["patient_identity_correct"] for result in results) / total if total else 0.0
        ),
        "timestamp_accuracy": (
            sum(result["timestamp_accuracy"] for result in results) / total if total else 0.0
        ),
        "observation_coverage": (
            sum(result["observation_coverage"] for result in results) / total if total else 0.0
        ),
        "observation_value_accuracy": (
            sum(result["observation_value_accuracy"] for result in results) / total if total else 0.0
        ),
        "tasks": {},
    }
    for task in sorted({result["task"] for result in results}):
        subset = [result for result in results if result["task"] == task]
        summary["tasks"][task] = {
            "examples": len(subset),
            "state_fidelity_pass_rate": sum(r["state_fidelity_pass"] for r in subset) / len(subset),
            "patient_identity_accuracy": sum(r["patient_identity_correct"] for r in subset) / len(subset),
            "timestamp_accuracy": sum(r["timestamp_accuracy"] for r in subset) / len(subset),
            "observation_coverage": sum(r["observation_coverage"] for r in subset) / len(subset),
            "observation_value_accuracy": sum(r["observation_value_accuracy"] for r in subset) / len(subset),
        }
    return {"summary": summary, "details": results}
