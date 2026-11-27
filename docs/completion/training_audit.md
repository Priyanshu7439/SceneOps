# Training pipeline audit

Historical training evidence is in `notebooks/kaggle_sceneops-training-v2.ipynb` and `notebook_audit.txt`. The recorded pipeline used Qwen/Qwen2.5-1.5B-Instruct with TRL `SFTTrainer`, PEFT LoRA/QLoRA, and bitsandbytes 4-bit NF4.

## Recorded configuration

- LoRA rank 16, alpha 32, dropout 0.05; target modules q/k/v/o and gate/up/down projections.
- Train/validation/test counts 3,554/1,121/1,181.
- Three epochs; per-device batch 2; gradient accumulation 8; learning rate 2e-4; cosine schedule; warmup 20; weight decay 0.01.
- Maximum sequence length 512; packing disabled; gradient checkpointing enabled; seed 42.
- The notebook verifies 1,777 batches/epoch, 223 optimizer steps/epoch, and 669 total steps, then resumes from checkpoint-223.
- `report_to="none"` in the recorded training configuration.

## Loss and padding audit

The custom collator tokenizes `messages[:-1]` with a generation prompt and all messages without one, verifies the prompt prefix, creates assistant-only target labels, masks prompt and padding labels to `-100`, and keeps the target intact while truncating only the left side of the prompt. This is the intended correct pattern.

The implementation is recorded in the notebook rather than a versioned training module, and the original checkpoint is not present in this checkout. Consequently the historical low validation loss and perfect token-accuracy report are not independently reproducible here.

## Technology status

The notebook records TRL/PEFT, AWQ/GGUF conversion, Ray Serve HTTP smoke cells, and W&B offline initialization. Local package inspection found none of Ray, vLLM, W&B, TRL, PEFT, Transformers, bitsandbytes, or torch installed, so those historical claims are treated as evidence, not live verification.

An optional Axolotl configuration is now structurally validated for the same base model, dataset path, chat-template format, sequence length, and LoRA rank/alpha. It is not evidence of Axolotl execution; launching it would be a new training run and remains intentionally unperformed.
