# Where the fixtures come from

Everything under `h200-sweep/` and `h200-ladder.jsonl` comes from Vinayak Baranwal's
public benchmark, github.com/vinayakbaranwal12/speculative-decoding-capacity-tax
(September 2026, MIT License, reproduced in `h200-sweep/LICENSE`), written up as
"Speculative Decoding's Hidden Cost" on DigitalOcean Community (21 September 2026).
It served Qwen/Qwen3-14B on one NVIDIA H200 under vLLM 0.27.1 with
`--max-model-len 2048 --no-enable-prefix-caching --attention-backend FLASHINFER`, plain
and with three speculative configs, each proposing three tokens:

| Arm | `--speculative-config` |
|---|---|
| baseline | none |
| eagle3 | `{"model": "RedHatAI/Qwen3-14B-speculator.eagle3", "num_speculative_tokens": 3, "method": "eagle3"}` |
| draft-0.6B | `{"method": "draft_model", "model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 3}` |
| ngram | `{"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 4, "prompt_lookup_min": 2}` |

## `h200-sweep/`

- `<arm>-2K-sharegpt-c<N>.metrics.txt`: the server's `/metrics` page, verbatim, saved
  by the benchmark after its ShareGPT run at N concurrent requests (`logs/sweep/` in
  the source repo). Each arm ran on its own server boot, with the five concurrency
  levels in order, so the counters are cumulative: the run at 25 is the c25 scrape
  less the c5 scrape, which is what `--since` computes. The c1 scrape is read alone.
  Every request carried about 1,784 prompt tokens and 256 output tokens.
- These scrapes come from the benchmark's first sweep, which ran greedy with
  `ignore_eos`; its author later re-ran the acceptance numbers because that setup
  inflated n-gram acceptance and deflated EAGLE3's. The replay does not depend on
  realistic acceptance: it checks that the tool, reading a run's scrape, predicts
  the throughput that same run measured.
- `measured.csv`: the benchmark's measured output tokens a second and gain over the
  plain server for those runs, copied from `analysis/T3_throughput.csv` (2K regime,
  ShareGPT rows at 1, 5, 25, 50 and 100). Values are unrounded, as published.

## `h200-ladder.jsonl`

The same study's request-level ladder (`results_request_phase/request_phase_summary.jsonl`):
five traces, four arms, eight concurrency levels (1 to 100), with each run's output
tokens a second, mean time per output token and mean acceptance length. Where the
source file holds a pilot row and a later full-ladder row for the same point, the
full-ladder row is kept. Fields other than those six were dropped.

Prompt and output lengths per request for each trace, which `test_spec_window.py`
feeds the model, were derived from the source repo, not from the scrapes:

| Trace | Prompt tokens | Output tokens | How |
|---|---|---|---|
| sharegpt | 192 | 256 | prompts: characters over 4 in `traces/sharegpt_prompts.jsonl`; output: tokens a second x duration over completed requests |
| swebench | 349 | 256 | same |
| cnn_short | 744 | 64 | same |
| cnn_long | 745 | 256 | same |
| edgar_json | 1,720 | 320 | prompts truncated to 1,720 tokens with 320 of output headroom (`RERUN_NOTES.md`, Step 0a) |

## Model folders

`Qwen3-14B/config.json` and `Qwen3-0.6B/config.json` are the Hugging Face configs of
Qwen/Qwen3-14B and Qwen/Qwen3-0.6B (Apache 2.0), as the benchmark recorded them under
`env/`. There is no `model.safetensors.index.json`, so weight bytes are computed from
the config.

## The published table

`test_spec_window.py` also carries the EAGLE-3 paper's vLLM throughput table inline:
Li et al., "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via
Training-Time Test", arXiv 2503.01840, Table 5 (LLaMA-Instruct 3.1 8B, MT-Bench, chain
length 2, no draft tree). Its caption names an A100 and the paragraph above it names
an RTX 3090; the test reads the caption. The Llama 3.1 8B shape numbers in the test are
the published architecture: 32 layers, hidden size 4,096, 8 KV heads of 128, MLP width
14,336 and a 128,256-token vocabulary.
