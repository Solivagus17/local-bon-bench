# local-bon-bench
Local Best-of-N (Generator-Judge) benchmark using Ollama on an 8GB RTX 5050. Evaluates Qwen3:4B (5 drafts) + Qwen3:14B (Judge) vs. Qwen3:14B solo across 50 math and Python tasks. Logs accuracy, latency, Pass@5 oracle ceiling (42%), and verifier collapse (47.6% judge accuracy), showing why 14B solo (88%, 32.4s) beats the pipeline (20%, 62.1s).
