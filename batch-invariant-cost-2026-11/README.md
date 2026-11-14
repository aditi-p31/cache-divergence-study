# Time per request with vLLM batch-invariant mode, November 2026

How much slower is vLLM 0.33.0 with `VLLM_BATCH_INVARIANT=1` than the default, with prefix caching on and off, when requests run one at a time? This folder answers that from the logs of the [November 2026 recheck](../CURRENCY_CHECK_2026-11.md). No new GPU run was needed.

## Setup

- vLLM 0.33.0, Qwen2.5-7B-Instruct BF16 (revision `a09a354`), one RTX 4090, `--max-num-seqs 1`, one request at a time, so batch size one. Temperature 0 with the model's default repetition penalty (1.05), the same flags as the paper's vLLM runs.
- The 80 BFCL multi-turn episodes in their natural order, with long prompts and short answers.
- Five passes, each on a fresh server: default with cache on, default with cache off (twice), batch-invariant with cache on, batch-invariant with cache off.
- Time is the client's wall time around one non-streaming `/v1/completions` call.
- The batch-invariant pass with cache on did use the cache: 922 of its 923 requests reported cached tokens.
- Only requests that are identical in both passes are compared: same episode and request index, same prompt hash, and the same completion tokens.

[`latency_matched.py`](latency_matched.py) computes the table from the passes' `raw_requests.jsonl` files and printed [`RESULT_2026-11-13.txt`](RESULT_2026-11-13.txt). A second run of the script gave identical output.

## Result

| Comparison | Matched requests | Time ratio (sum of times) | Median ratio per request |
|---|---|---|---|
| Default cache off, run twice (noise) | 956 | 0.91 | 0.91 |
| Default cache on, then batch-invariant cache on | 543 | 2.30 | 2.29 |
| Default cache off, then batch-invariant cache off | 441 | 1.37 | 1.30 |
| Default cache off, then batch-invariant cache on | 441 | 0.80 (0.88 against the repeat) | 0.68 (0.75) |
| Default cache on, then default cache off | 596 | 2.62 | 3.02 |

At batch size one on this workload:

- Batch-invariant mode with prefix caching on took about 2.3 times as long as the default with caching on.
- It took 0.8 to 0.9 of the time of the default with caching off, against about 9 percent drift between two identical passes.
- So for runs that must repeat, keeping prefix caching on in batch-invariant mode cost no more than turning caching off.

## Scope

- The passes ran one after another, about an hour apart, with the batch-invariant passes last, so drift between passes can account for part of a difference.
- The matched requests lean toward the start of each episode, before trajectories split.
- The requests were not streamed, so there is no split into time to first token and decode time.
- One request at a time. Throughput under concurrent load was not measured.

The raw request logs are part of the November run archive (`currency_results.tgz`, sha256 `64525a86...`), which is available on request. Run the script as `python3 latency_matched.py <results dir>`.
