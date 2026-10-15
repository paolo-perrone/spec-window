# spec-window

For engineers serving an open-weight model on vLLM who have turned on
speculative decoding, or are about to, and want to know the batch size where
it stops paying on their own server.

It reads two things you already have: a scrape of your server's `/metrics`
endpoint and the model's `config.json`. It prints the batch range where a
speculative step still beats a plain one, the share of your engine steps that
ran inside that range, and the batch to stop drafting at, as a vLLM config line.

```
curl -s localhost:8000/metrics > metrics.txt
python3 spec_window.py metrics.txt --model-dir <folder with config.json> --gpu h100-sxm
```

Read-only, standard-library Python, no GPU and no API key: it runs on your
laptop against a scrape copied off the server. A scrape's counters run from
server start, so if your load swings, take two scrapes a few minutes apart in
your busiest hour and read only the traffic between them:

```
curl -s localhost:8000/metrics > before.txt; sleep 300; curl -s localhost:8000/metrics > after.txt
python3 spec_window.py after.txt --since before.txt --model-dir <folder> --gpu h100-sxm
```

## What it prints, on a case with a known answer

`fixtures/h200-sweep/` holds twenty `/metrics` scrapes from a public benchmark:
Qwen3-14B on one H200 under vLLM 0.27.1, served plain and with three drafters,
at 1, 5, 25, 50 and 100 concurrent requests (Vinayak Baranwal,
`speculative-decoding-capacity-tax`, September 2026). With a separate
Qwen3-0.6B draft proposing three tokens, that benchmark measured 1.02x of plain
throughput at 5 concurrent requests and 0.97x at 25, so speculation stopped
paying at about 13. `spec_window` reads only the 25-request run:

```
python3 spec_window.py fixtures/h200-sweep/draft-0.6B-2K-sharegpt-c25.metrics.txt --since fixtures/h200-sweep/draft-0.6B-2K-sharegpt-c5.metrics.txt --model-dir fixtures/Qwen3-14B --gpu h200 --method draft --draft-dir fixtures/Qwen3-0.6B
```

```
spec_window · Qwen/Qwen3-14B · h200 · a draft model proposing 3, 2.94 tokens committed a step
now        batch 22.3 at 51.4 ms a speculative step, 780 steps in this scrape
gain       1.37x at batch 1 · 1.08x at 8 · 0.88x at 22 (now) · 0.81x at 32 · 0.68x at 128
window     speculation pays from batch 1 to 11, break-even at 11.8
steps      9% of this scrape's engine steps ran inside that window
cutoff     draft up to batch 7 (gain 1.10x or better), none above:
           --speculative-config '{"method": "draft_model", "num_speculative_tokens": 3, "num_speculative_tokens_per_batch_size": [[1, 7, 3], [8, 1024, 0]]}'
verdict    turn speculation off: continuous batching already fills this GPU past break-even
```

It put break-even at 11.8 where the benchmark's two measurements put it near
13, and it read 0.88x at the scraped batch where the benchmark measured 0.97x.

## How it works

A decode step reads every weight once plus each sequence's KV cache, and does
two FLOPs per parameter per token. A plain step does that for one token per
sequence. A speculative step verifies the drafted tokens plus one per sequence,
runs the drafter, and commits the acceptance length on average. Speculation
pays while acceptance length x plain step time beats the speculative step time,
and the batch where that stops holding is the break-even.

From a scrape of a server that already drafts, three things are measured, not
modelled:

- tokens committed per sequence per step, from `vllm:generation_tokens_total`
  over the count of `vllm:inter_token_latency_seconds`. vLLM's own acceptance
  counters count per draft, and n-gram lookup skips drafting on steps where it
  finds no match: on the n-gram fixture, per draft reads 3.07 tokens and per
  step reads 1.58;
- the speculative step time and the batch it ran at, the anchor;
- prompt and output lengths and the prefill work riding along each step.

Whatever the model misses at your batch is charged to the drafter, a fixed cost
first and the rest per sequence, so the curve passes through your server's
measured point. Three constants (share of peak bandwidth reached 0.70, share of
peak FLOPs reached 0.86, and how gently a step turns from one limit to the
other) were fitted on the same benchmark's request ladder, described below, and
sit at the top of `spec_window.py`.

From a plain scrape, pass `--accept-len` from a short trial or the drafter's
card. The drafter's fixed cost then comes from that ladder, as a share of a
one-sequence plain step: 0.20 for an EAGLE head, 0.88 for a separate draft
model, 0.88 for n-gram lookup. The report says when it is using that prior.

## Check it before you trust it

`python3 test_spec_window.py` replays three groups of known answers and must
print `3/3 recorded case groups pass` and `9/9 unit checks pass`. Where each
fixture comes from is in `fixtures/README.md`. What the replays show:

- **Fifteen recorded speculative runs, read by the tool from their own scrapes.**
  None of these runs went into fitting the constants, and their prompts ran
  about 1,800 tokens, so this is the out-of-sample check. The gain at the
  scraped batch landed within 0.17 of the measured gain, always on the low
  side. Break-even landed within 2.6 batch slots of the measured 2.6 for the
  EAGLE3 head and within 9 of the measured 13.2 for the 0.6B draft.
  The n-gram runs measured 1.26x, 0.94x, 1.05x, 0.99x and 0.92x, crossing 1.0
  three times, so they have no single break-even to test.
- **Fifty anchored predictions on the same H200's request ladder** (five
  traces, both drafters, anchored at 5, 10, 25, 35 and 50 concurrent requests).
  The constants were fitted on these step times, so this checks the
  break-even arithmetic more than the fit. The largest miss was 11.8 batch
  slots. The 0.6B draft landed early in 22 of 25, by up to 11.4, which errs
  toward drafting less; the other 3 landed within 0.6.
- **The EAGLE-3 paper's vLLM batch table** (Table 5: LLaMA-Instruct 3.1 8B,
  chain length 2, MT-Bench; its caption says A100 and its text says RTX 3090),
  with no scrape to anchor on, given only the gain at batch 2. The model
  tracked the decay within 0.04 through batch 48, then read 1.18x at batch 56
  where the paper measured 1.01x, and 0.89x against 0.71x for the original
  EAGLE. Unanchored, it runs optimistic near the top of the range.

## What it does not do

- Dense models only. A mixture-of-experts step reads a share of its experts
  that grows with the batch, so the curve bends differently: a Red Hat
  benchmark reported EAGLE3 gains holding to 200 concurrent requests on
  gpt-oss-120b. The tool refuses a MoE config rather than guess.
- vLLM only. SGLang's `/metrics` names differ.
- A scrape is one load level. When the steps it covers ran at very different
  batches, the report adds a note and asks for a `--since` window instead.
- A zero draft length does not unload the drafter. vLLM still runs its forward
  pass each step to keep its KV cache in sync, and one operator measured time
  per output token 6.6 percent worse at 2 concurrent requests and 21.6 percent
  worse at 189 with a drafter loaded under a schedule that never drafted (vLLM
  issue 49548). If most of your steps sit past break-even, switching
  speculation off, or serving it from its own replicas, beats the schedule.
- vLLM counts the requests it schedules, prefills included, when it picks the
  draft length, so the cutoff applies to a slightly larger count than the
  decode batch this report prints. The config line's last range ends at 1,024;
  raise it if your `--max-num-seqs` is higher.
- It does not judge acceptance on your data. It reads what your traffic
  produced.

## Support

Tested on 2026-09-29 with Python 3.9.6 and 3.14.6 against vLLM V1 metric names:
the fixtures come from vLLM 0.27.1, and the names and the
`num_speculative_tokens_per_batch_size` key were read from the v0.30.0 source
(`vllm:iteration_tokens_total`, `vllm:inter_token_latency_seconds`,
`vllm:request_prompt_tokens`, `vllm:request_generation_tokens`,
`vllm:request_prefill_kv_computed_tokens`, `vllm:generation_tokens_total`,
`vllm:spec_decode_num_*_total`). No dependencies, so there is no lockfile to
drift. Metric names get re-checked on each vLLM release; open an issue if a
scrape from your version will not parse.
