# llm-cachecheck 0.3.0: one-server mode on real servers

This folder holds the GPU check behind one-server mode in llm-cachecheck
0.3.0. It ran on 2026-11-14 (UTC) on one rented RTX 4090.

One-server mode replaces the second, cache-off server with a fresh
`cache_salt` on every recompute request. The question was whether those
salted requests give the same answers as a server with prefix caching off.
In stages V and S, each one-server run was compared, probe by probe, with a
two-servers run against the same cache server in the same session. Stage
SDEF (SGLang with default settings) ran in one-server mode only and has no
two-server comparison.

## Setup

- One RTX 4090 (24 GB), driver 595.91.07, CUDA 13.2.
- vLLM 0.33.0 (torch 2.14.0+cu132) and SGLang 0.5.23 (torch 2.14.1+cu130).
  The full package lists are in [`env_manifest.txt`](env_manifest.txt).
- Qwen2.5-3B-Instruct, BF16, Hugging Face revision `aa8e725`
  ([`SHA256SUMS_model_3B.txt`](SHA256SUMS_model_3B.txt)).
- llm-cachecheck 0.3.0 at commit `f3233f1` (`cachecheck.py` sha256
  `f825c5fd78d44f96...`), the tool's 20 built-in conversations, default
  options. [`compare_reports.py`](compare_reports.py) and
  [`telecheck.py`](telecheck.py) are the scripts that produced
  `compare.txt` and `telemetry_check.txt`.
- One request at a time. Every stage started fresh servers.

Server flags:

| Stage | Server | Flags |
|---|---|---|
| V | vLLM, cache on | `--generation-config vllm --enable-prompt-tokens-details --gpu-memory-utilization 0.45` |
| V | vLLM, cache off | the same, plus `--no-enable-prefix-caching` |
| S | SGLang, cache on | `--dtype bfloat16 --context-length 4096 --max-running-requests 1 --mem-fraction-static 0.60 --max-total-tokens 65536 --disable-cuda-graph` |
| S | SGLang, cache off | the same, plus `--disable-radix-cache` |
| SDEF | SGLang, cache on | `--dtype bfloat16` only, so CUDA graphs on |

Every SGLang server also had `--random-seed 42 --sampling-defaults openai
--enable-cache-report`. Every vLLM server ran with
`VLLM_USE_FLASHINFER_SAMPLER=0`, as in the
[November check](../CURRENCY_CHECK_2026-11.md). That sampler only serves
non-greedy requests, and every request here is greedy.

The decision rules were written down before the run. One-server mode ships
for an engine only if `compare.txt` says `AGREE` for every pair, the
telemetry check is `ok` for every one-server report, and the cache-off
server got no requests during any one-server run.

## Results

| Stage | Runs | Recompute answer identical to the cache-off server | Turn-2 answer differs on the cache path |
|---|---|---|---|
| V: vLLM, one server vs two servers | 2 pairs | 20 of 20 in both pairs | 14 of 20 in all 4 runs |
| S: SGLang, CUDA graphs disabled, one server vs two servers | 2 pairs (both one-server runs against one two-servers run) | 20 of 20 in both pairs | 12 of 20 in all 3 runs |
| SDEF: SGLang, default settings, one server | 2 runs | not compared | 12 of 20 in each run |

- All 20 probes were valid in every run.
- In every pair, turn 1 and the cache-path answer were also identical in
  text between the two modes ([`V_vllm/compare.txt`](V_vllm/compare.txt),
  [`S_sglang/compare.txt`](S_sglang/compare.txt)).
- Each salted recompute request reported 0 cached tokens. On the cache
  path, turn 2 reused 96 to 304 tokens on vLLM and 106 to 315 on SGLang
  (`telemetry_check.txt` in each folder).
- The cache-off server got 0 requests during the one-server runs
  (`requests_per_run.txt`).
- The counts match the November two-server results (14 of 20 on vLLM,
  12 of 20 on SGLang with CUDA graphs disabled).
- With CUDA graphs on (SDEF), the same 12 probes differed as with them
  disabled (S). The SDEF server log shows `'disable_cuda_graph': False`.

To recheck, from this folder:

```bash
python compare_reports.py V_vllm/one_run1.json V_vllm/two_run1.json V_vllm/one_run2.json V_vllm/two_run2.json
python compare_reports.py S_sglang/one_run1.json S_sglang/two_run1.json S_sglang/one_run2.json S_sglang/two_run1.json
python telecheck.py SDEF_sglang
```

## vLLM batch-invariant mode in a second history

The same session repeated November's vLLM batch-invariant pass in a second
history. It used Qwen2.5-7B-Instruct, BF16, revision `a09a354`
([`SHA256SUMS_model_7B.txt`](SHA256SUMS_model_7B.txt)), with the 80 BFCL
multi-turn episodes. It ran the paper's history 1, the episode order in
[`revision/orders/order_1.json`](../revision/orders/order_1.json). November
ran the episodes in their natural order. Each pass had its own fresh
server, with a cache-on pass and a cache-off pass per setting:

| vLLM 0.33.0 | Episodes whose trajectory changed between cache hit and recompute | Shared tokens with a different top-1 log-prob |
|---|---|---|
| `VLLM_BATCH_INVARIANT=1` | 0 of 80 | 0 of 25,900 |
| default | 48 of 80 | 10,338 of 15,372 |

Shared tokens are the answer tokens both passes produced identically,
counted in each episode up to its first difference, so the total is smaller
when episodes diverge. Details are in [`cmp_D2_vllm_bi_path.json`](cmp_D2_vllm_bi_path.json) and
[`cmp_B2_vllm_path.json`](cmp_B2_vllm_path.json). With batch-invariant mode
on, no episode changed in this history either, while default vLLM changed 48
in the same history. The batch-invariant cache-on pass did reuse the cache:
922 of its 923 requests reported cached tokens, 98% of all prompt tokens
(vLLM chose the FLASH_ATTN attention backend). Each cache-off pass was run
once in this history; there was no second cache-off run as a repeat control.
The raw episode logs are kept outside the repository, like November's.

## Other files

- [`run.log`](run.log): the run's own log, with every stage, exit status and
  comparison.
- [`SUMMARY.txt`](SUMMARY.txt): one line per report. Its B2 and D2 lines
  are the episode passes above.
- [`_preflight/`](_preflight/): the short smoke tests that had to pass before
  the measured runs started.
- `srv_*.log` in each stage folder: the server logs, which also record each
  server's full settings.
