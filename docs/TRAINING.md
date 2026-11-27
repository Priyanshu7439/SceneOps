# Training

Historical training used Qwen/Qwen2.5-1.5B-Instruct, Transformers, TRL SFTTrainer, PEFT LoRA/QLoRA, bitsandbytes 4-bit NF4, and a Tesla T4 environment.

Recorded configuration:

- LoRA rank 16, alpha 32, dropout 0.05
- q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- 3 epochs, 669 optimization steps, resumed from checkpoint-223
- Sequence length 512; batch 2; gradient accumulation 8
- Learning rate 2e-4; cosine schedule; 20 warmup steps; seed 42

The notebook audit is historical evidence. The original checkpoint and complete environment are not present here, so this is not claimed as a fully reproducible local training run. No retraining is performed.

configs/axolotl_sceneops_lora.yml is an optional structurally validated reference configuration. It was not used for historical training and has not been executed.
