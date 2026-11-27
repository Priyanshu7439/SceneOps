# Dataset

Source of truth: data/processed/sceneops_narration_manifest.json.

## Verified manifest

- 5,856 total unique examples
- 320 unique scenarios
- 2,801 scene examples
- 3,055 temporal examples
- Train: 3,554 examples / 192 scenarios
- Validation: 1,121 examples / 64 scenarios
- Test: 1,181 examples / 64 scenarios
- Manifest validation and scenario-level split audit: passed

Scenario-level splitting keeps all examples from a synthetic incident in one split and prevents related observation windows leaking between train, validation, and test.

Primary objectives are scene narration and temporal scene narration. Patient-dialogue datasets and style-selection code remain preserved experimental work.
