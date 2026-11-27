from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneops.evaluation.state_fidelity import evaluate_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("artifacts/final_audit/sceneops_test_ground_truth_mapping.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/final_audit/state_fidelity_v2"),
    )
    args = parser.parse_args()
    if not args.mapping.is_file():
        raise FileNotFoundError(args.mapping)
    rows = [
        json.loads(line)
        for line in args.mapping.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = evaluate_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "evaluator": "sceneops_independent_state_fidelity_v2",
        "ground_truth_source": "regenerated simulator observations in mapping JSONL",
        "reference_text_used_as_ground_truth": False,
        **result["summary"],
        "limitations": [
            "Closed grammar evaluator for the V3 narration template; unseen free-form wording is rejected rather than semantically interpreted.",
            "Checks both timestamped observation panels and the observed-changes/unchanged-observations summaries.",
            "Detects unsupported extra narrative content by requiring a full template match.",
            "Measures consistency with simulator-disclosed observations, not clinical correctness or clinical safety.",
        ],
    }
    (args.output_dir / "state_fidelity_v2_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "state_fidelity_v2_predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for detail in result["details"]:
            handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["state_fidelity_pass_rate"] == 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
