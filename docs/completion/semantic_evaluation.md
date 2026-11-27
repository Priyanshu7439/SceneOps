# Semantic and state-fidelity evaluation

The simulator remains the source of truth. `src/sceneops/evaluation/state_fidelity.py` evaluates predictions against regenerated simulator observations from `artifacts/final_audit/sceneops_test_ground_truth_mapping.jsonl`, not against reference text.

## Held-out result

The independent v2 report covers all 1,181 examples: state-fidelity pass rate 1.0, patient identity accuracy 1.0, timestamp accuracy 1.0, observation coverage 1.0, and observation value accuracy 1.0 for both scene (562) and temporal (619) tasks.

This is a closed-grammar evaluator for the current V3 template. It rejects unsupported free-form wording, checks every timestamped panel and temporal change/unchanged summary, and detects extra narrative content. It measures disclosure consistency, not clinical correctness.

The preserved field audit gives a more permissive view of the GGUF outputs: seven temporal follows-commands attribution errors were identified; visible hemorrhage was explicitly correct in 606/619 cases; capillary-refill omissions occurred in 16 cases where a value was available. These findings should remain visible alongside the strict-template report.

## Adversarial coverage

`tests/evaluation/test_state_fidelity.py` covers false changes, false unchanged claims, unsupported appended claims, wrong patient IDs, wrong timestamps, missing observations, supported paraphrased scaffolds, and corrupted-reference resistance. The rule-based safety reviewer tests also cover hidden/unsupported measurements, diagnosis/treatment/triage language, and temporal attribution.

## Boundary conclusion

The application passes only approved scene facts or timestamped observations into narration requests. Triage, deterioration, resources, transport, and outcomes remain simulator operations. The evaluator does not prove that an arbitrary model endpoint will always obey this boundary; live endpoint testing remains separate.

Additional deterministic boundary tests verify that narration requests contain only disclosed facts, do not expose patient/resource/survival fields, do not mutate simulator snapshots, and leave clock, resources, and patients unchanged when the model raises a timeout.
