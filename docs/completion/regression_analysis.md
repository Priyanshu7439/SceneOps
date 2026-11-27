# Fine-tuned regression analysis

## Confirmed blind results

The valid comparison is `artifacts/evaluation/base_blind` versus `artifacts/evaluation/gguf_blind`. The older `gguf_final` run is excluded because its prompts leaked assistant targets.

| Metric | Base | Fine-tuned | Delta |
|---|---:|---:|---:|
| Overall token F1 | 0.6235269 | 0.5043650 | -0.1191619 |
| Scene token F1 | 0.8181509 | 0.5649590 | -0.2531919 |
| Temporal token F1 | 0.4468247 | 0.4493507 | +0.0025259 |

Paired comparison: fine-tuned wins/losses/ties were 215/966/0 overall, 0/562/0 for scene, and 215/404/0 for temporal.

## Evidence-based findings

- Both blind manifests identify the same 1,181 examples, dataset hash, temperature 0, max tokens 256, and system/user-only prompt policy.
- The notebook's custom collator constructs prompt and full chat-template tokenizations, masks prompt labels with `-100`, masks right padding with `-100`, and raises if the assistant target exceeds 512 tokens. This confirms intended assistant-only loss masking in the recorded pipeline.
- The notebook records `resume_from_checkpoint=checkpoint-223`, checkpoint step 223, expected total step 669, and restored optimizer/scheduler/RNG files in the historical audit. This supports resume-state restoration, but does not independently reproduce the Kaggle run here.
- Fine-tuned outputs are more verbose/prose-like in the preserved evaluation inspection, consistent with lower lexical precision and scene regression. The artifacts do not establish a single causal defect.

## Not confirmed

The local checkout does not contain the original GPU checkpoint, trainer state, or a runnable TRL/PEFT environment. Therefore prompt masking, validation loss, token accuracy, and resume state cannot be independently recomputed here. The unusually strong historical validation figures must not be treated as proof of generalization.

## Recommendation

Do not retrain automatically. If retraining is later authorized, first reproduce the collator audit with a small deterministic batch, compare assistant-only versus full-loss behavior, inspect output-length distributions, and rerun the same blind test plus the state-fidelity suite. Do not change simulator ground truth or dataset splits to improve metrics.
