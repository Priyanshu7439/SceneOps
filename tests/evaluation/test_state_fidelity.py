from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sceneops.evaluation.state_fidelity import evaluate_prediction, evaluate_rows

MAPPING_PATH = Path("artifacts/final_audit/sceneops_test_ground_truth_mapping.jsonl")


@pytest.fixture(scope="module")
def mapping_rows() -> list[dict]:
    if not MAPPING_PATH.is_file():
        pytest.skip("Full held-out mapping is not present in this checkout.")
    return [
        json.loads(line)
        for line in MAPPING_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture
def temporal_row(mapping_rows: list[dict]) -> dict:
    return copy.deepcopy(next(row for row in mapping_rows if row["task"] == "temporal"))


def test_full_held_out_mapping_passes(mapping_rows: list[dict]) -> None:
    result = evaluate_rows(mapping_rows)
    assert result["summary"]["examples"] == 1181
    assert result["summary"]["state_fidelity_pass_rate"] == 1.0


def test_false_change_summary_is_rejected(temporal_row: dict) -> None:
    prediction = temporal_row["prediction"]
    prediction = prediction.replace(
        "respiratory rate changed from 21 to 30",
        "respiratory rate changed from 21 to 21",
    )
    temporal_row["prediction"] = prediction
    result = evaluate_prediction(temporal_row)
    assert result["state_fidelity_pass"] is False
    assert any(issue["type"] == "incorrect_change_claim" for issue in result["issues"])


def test_false_unchanged_summary_is_rejected(temporal_row: dict) -> None:
    prediction = temporal_row["prediction"]
    prediction = prediction.replace(
        "radial pulse remained present",
        "radial pulse remained absent",
    )
    temporal_row["prediction"] = prediction
    result = evaluate_prediction(temporal_row)
    assert result["state_fidelity_pass"] is False
    assert any(issue["type"] == "incorrect_unchanged_claim" for issue in result["issues"])


def test_unsupported_extra_claim_is_rejected(temporal_row: dict) -> None:
    temporal_row["prediction"] += " The patient is stable and ready for transport."
    result = evaluate_prediction(temporal_row)
    assert result["state_fidelity_pass"] is False
    assert result["issues"], "An unsupported appended claim must produce a rejection reason."


def test_wrong_patient_mention_is_rejected(temporal_row: dict) -> None:
    temporal_row["prediction"] = temporal_row["prediction"].replace("P003", "P999")
    result = evaluate_prediction(temporal_row)
    assert result["state_fidelity_pass"] is False
    assert any("patient_identity" in issue["type"] for issue in result["issues"])


def test_wrong_timestamp_is_rejected(temporal_row: dict) -> None:
    temporal_row["prediction"] = temporal_row["prediction"].replace(
        "simulation second 600", "simulation second 601"
    )
    result = evaluate_prediction(temporal_row)
    assert result["state_fidelity_pass"] is False
    assert any("timestamp" in issue["type"] for issue in result["issues"])


def test_missing_observation_is_rejected(temporal_row: dict) -> None:
    temporal_row["prediction"] = temporal_row["prediction"].replace(
        "; radial pulse: present", ""
    )
    result = evaluate_prediction(temporal_row)
    assert result["state_fidelity_pass"] is False
    assert any(issue["type"] == "missing_observation" for issue in result["issues"])


def test_paraphrased_supported_scaffold_passes(temporal_row: dict) -> None:
    temporal_row["prediction"] = temporal_row["prediction"].replace(
        "the learner recorded:", "the learner noted:"
    )
    result = evaluate_prediction(temporal_row)
    assert result["state_fidelity_pass"] is True


def test_exact_match_to_corrupt_reference_does_not_override_ground_truth(
    temporal_row: dict,
) -> None:
    corrupt = temporal_row["reference"].replace(
        "respiratory rate: 30", "respiratory rate: 21"
    )
    temporal_row["reference"] = corrupt
    temporal_row["prediction"] = corrupt
    result = evaluate_prediction(temporal_row)
    assert result["state_fidelity_pass"] is False
    assert any(issue["type"] == "incorrect_observation_value" for issue in result["issues"])


def test_exact_match_to_corrupt_reference_is_rejected(temporal_row: dict) -> None:
    corrupt = temporal_row["reference"].replace(
        "respiratory rate: 30", "respiratory rate: 21"
    )
    temporal_row["reference"] = corrupt
    temporal_row["prediction"] = corrupt
    result = evaluate_prediction(temporal_row)
    assert result["state_fidelity_pass"] is False
    assert any(issue["type"] == "incorrect_observation_value" for issue in result["issues"])
