# Safety and trust boundaries

```mermaid
flowchart LR
  SIM[Simulator-owned state: patients, clock, deterioration, triage, resources, transport, outcomes]
  DISC[Authorized disclosure boundary]
  INPUT[LLM prompt input]
  LLM[Probabilistic language generation]
  OUTPUT[Generated narration]
  REVIEW[Deterministic rule-based safety reviewer]
  UI[Frontend narration output]
  BLOCK[Reject or safe failure]

  SIM --> DISC --> INPUT --> LLM --> OUTPUT --> REVIEW
  REVIEW -->|approved| UI
  REVIEW -->|unsupported, unsafe, malformed| BLOCK
  BLOCK --> UI
  LLM -. prohibited .-> MUTATE[Patient-state mutation]
  LLM -. prohibited .-> TRIAGE[Triage or treatment decision]
  LLM -. prohibited .-> AMB[Ambulance allocation]
  LLM -. prohibited .-> SURV[Survival or scoring decision]
  LLM -. prohibited .-> HIDDEN[Hidden-state access]
```

The application service constructs narration requests from simulator-approved observations. Provider failures and rejected narration do not call simulator mutation methods. This is a software safety boundary, not clinical validation.
