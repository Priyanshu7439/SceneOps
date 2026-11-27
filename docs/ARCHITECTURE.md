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
