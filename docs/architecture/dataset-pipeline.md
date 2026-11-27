# Dataset and fine-tuning pipeline

This combines current repository code with historical notebook evidence. Training and conversion stages are historical unless a current artifact or test verifies them.

```mermaid
flowchart LR
  SCEN[Deterministic synthetic scenarios]
  OBS[Observation and temporal panel extraction]
  EX[Scene and temporal example generation]
  VALID[Schema, duplicate, narration validators]
  SPLIT[Scenario-level split]
  TRAIN[3,554 train]
  DEV[1,121 validation]
  TEST[1,181 test]
  TOK[Transformers tokenization and label masking]
  QLORA[TRL SFTTrainer + PEFT LoRA/QLoRA + bitsandbytes NF4]
  CKPT[Historical adapter checkpoint-223]
  MERGE[PEFT merge and standalone model]
  AWQ[Historical AWQ export]
  GGUF[Historical GGUF Q4_K_M export]
  EVAL[Blind and state-fidelity evaluation]
  SERVE[Serving artifact]

  SCEN --> OBS --> EX --> VALID --> SPLIT
  SPLIT --> TRAIN
  SPLIT --> DEV
  SPLIT --> TEST
  TRAIN --> TOK --> QLORA --> CKPT --> MERGE
  MERGE --> AWQ
  MERGE --> GGUF
  MERGE --> EVAL
  AWQ --> SERVE
  GGUF --> SERVE
  TEST --> EVAL
```

The manifest records 5,856 examples across 320 scenarios. Historical training records 3 epochs and 669 steps with rank 16, alpha 32, dropout 0.05, and q/k/v/o/gate/up/down modules. The original checkpoint and complete training environment are unavailable locally.
