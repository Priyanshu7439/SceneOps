# SceneOps GGUF Evaluation

- Status: **completed**
- Dataset examples: 1181
- Completed: 1181
- Failed: 0
- Dataset SHA-256: `20678f5909fd531805ef0f4da152108ca917c67ff4104d87d682cb718441e89a`
- Model: `qwen2.5-1.5b-instruct-base`
- Generation: temperature 0.0, max_tokens 256

> Lexical metrics do not establish grounding, factual correctness, clinical correctness, or safety.

| Split | N | Exact match | Token P | Token R | Token F1 | Sequence similarity |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 1181 | 0.0000 | 0.6917 | 0.6058 | 0.6235 | 0.4506 |
| Scene | 562 | 0.0000 | 0.7818 | 0.8583 | 0.8182 | 0.7411 |
| Temporal | 619 | 0.0000 | 0.6099 | 0.3766 | 0.4468 | 0.1868 |
