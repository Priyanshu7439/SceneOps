# Safety and limitations

SceneOps uses synthetic educational scenarios. It is not clinically validated and must not be used for emergency care or real patient decisions.

Python owns patient state, deterioration, clock, triage, resources, transport, survival, outcomes, and scoring. The model receives authorized disclosures only and generates language. The reviewer rejects unsupported observations, hidden-state leakage, unsafe clinical decisions, and malformed narration. Provider failure returns an error without mutating simulator state.

Known limitations include the fine-tuned lexical-F1 regression, closed-grammar state-fidelity evaluation, unverified live Ray Serve/vLLM/W&B remote paths, and the historical training environment being unavailable locally.
