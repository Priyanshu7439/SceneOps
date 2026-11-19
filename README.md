# SceneOps — Mass Casualty Triage Simulation

SceneOps is an AI-powered mass-casualty incident training simulator that places learners in the role of a triage officer managing multiple simulated patients, limited emergency resources, and time-dependent patient deterioration.

## Project Overview

The simulator acts as the ground truth for patient states, observations, resources, and events. AI models generate natural-language scene narration and patient dialogue using only authorized simulator information.

The LLM generates language; it does not make clinical decisions or modify simulator state.

## Architecture

- **Simulation Engine:** Patient state, deterioration, triage, transport, resources, scenarios, and simulation clock.
- **Observation System:** Captures learner-visible patient observations.
- **Scene Narration Agent:** Generates scene descriptions from approved observations.
- **Patient Dialogue Agent:** Generates patient responses from authorized disclosure information.
- **Safety Reviewer:** Reviews generated responses before they are returned through the orchestration layer.
- **Application & API:** Exposes simulation and agent interactions.
- **Training Pipeline:** Dataset generation, validation, splitting, and evaluation.
- **Model Serving:** Backend integration for inference.

## Fine-Tuning Dataset

The validated scene and temporal narration dataset contains **5,856 examples across 320 scenarios**.

| Split | Examples | Scenarios |
|---|---:|---:|
| Training | 3,554 | 192 |
| Validation | 1,121 | 64 |
| Testing | 1,181 | 64 |
| **Total** | **5,856** | **320** |

Scenario-level leakage audit and dataset validation passed.

Dataset files are stored in `data/processed/`:

- `sceneops_narration_master.jsonl`
- `sceneops_narration_train.jsonl`
- `sceneops_narration_validation.jsonl`
- `sceneops_narration_test.jsonl`
- `sceneops_narration_manifest.json`

## Notebooks

The `notebooks/` directory contains:

- `SceneOps_V3_1_Teacher_Data_Generation.ipynb` — narration teacher-data generation.
- `kaggle_sceneops-training-v2.ipynb` — model fine-tuning and training workflow.

## Technology Stack

Python, FastAPI, Hugging Face Transformers, TRL, PEFT, QLoRA, vLLM, AWQ, GGUF, Ray Serve, Weights & Biases, pytest, and Ruff.

## Setup

Tested with Python 3.12.

```bash
git clone https://github.com/Priyanshu7439/SceneOps.git
cd SceneOps

python -m venv .venv
pip install -r requirements.txt
```

## Run Tests

```bash
python -m pytest -q
```

## Code Quality

```bash
ruff check .
ruff format --check .
```

## Disclaimer

SceneOps is an educational simulation project. It is not a medical device and must not be used for real-world diagnosis, treatment, or emergency response.

## Project Status

The project is under active development. Refer to the source code, tests, dataset manifests, and notebooks for implementation details.