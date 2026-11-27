# Architecture

SceneOps separates deterministic simulation from language generation.

## Request flow

1. React/Vite sends a session or narration request to FastAPI.
2. The application service reads simulator-approved observations.
3. Typed agent contracts carry only task-appropriate disclosures.
4. Scene or temporal agents construct constrained prompts.
5. The configured OpenAI-compatible provider returns text.
6. The rule-based reviewer checks identity, timestamps, values, and unsafe decisions.
7. Only approved narration is returned. Model output is never applied to simulator state.

The simulator owns patient state, deterioration, clock progression, observations, triage, ambulances, transport, survival/outcomes, events, and scoring. Narration failure therefore cannot mutate these values.

## Components

- src/sceneops/simulator: domain state and rules.
- src/sceneops/application: session orchestration.
- src/sceneops/agents: contracts, narration, dialogue, orchestration, safety.
- src/sceneops/api: FastAPI routes and schemas.
- src/sceneops/serving: provider client and optional Ray Serve adapter.
- frontend: learner dashboard and API client.

## Detailed diagrams

- [System overview](architecture/system-overview.md): layered components and data flow.
- [Narration workflow](architecture/narration-workflow.md): request, inference, review, and failure isolation.
- [Dataset pipeline](architecture/dataset-pipeline.md): scenario generation through evaluation artifacts.
- [Model serving](architecture/model-serving.md): provider abstraction, llama.cpp, vLLM client path, and Ray Serve.
- [Safety boundaries](architecture/safety-boundaries.md): simulator authority and prohibited model operations.

## Why the boundaries are deliberate

An LLM-driven simulator would make state transitions, resource allocation, and outcomes probabilistic and difficult to reproduce. SceneOps instead keeps those decisions in typed Python domain code. The model sees only the disclosure assembled for its task and returns text that is checked by a deterministic reviewer. This makes simulator behavior independently testable and lets provider failure be handled as an integration error rather than a state transition.

The service layer also keeps HTTP concerns separate from domain behavior: FastAPI validates request schemas and maps errors to HTTP responses; SceneOpsApplicationService builds simulator-backed agent requests; the engine owns mutation and event recording.
