# cachecheck

How often does the prompt cache change your server's answers?

Serving stacks reuse the cached key-value state of a shared prompt prefix. A
cache hit and a recomputation run different operations, so with greedy
decoding a small numeric change can flip a token. After that the answer goes
its own way. This is known behaviour: the llama.cpp server README says that
`cache_prompt` "can cause nondeterministic results". What varies is how often
it happens on your model, weights and build. `cachecheck` measures that.

It sends short two-turn conversations with greedy decoding and compares the
second answer on two paths:

- **cache path:** turn 2 reuses the state the server wrote while it generated
  turn 1
- **recompute path:** turn 2 is processed from scratch

Each path runs twice. A probe counts as valid only when all of these hold:

- both paths repeat on their own
- the server's own counts show that the cache hit covered the turn-1 answer
- the server's own counts show that the recompute path reused nothing

## Run it

One file, Python 3.8 or newer, no dependencies. A full run sends 120 requests.

**llama.cpp.** The cache is switched per request. Start the server with one
slot, because the default is several slots that share a cache. Turn the
host-memory prompt cache off to match the study's controlled runs:

```bash
llama-server -m model.gguf -np 1 --cache-ram 0 --port 8080
python cachecheck.py --base-url http://127.0.0.1:8080
```

**vLLM, SGLang or another OpenAI-compatible server.** The cache is a server
setting, so serve the same model twice, once with prefix caching on and once
with it off. Both servers must report cached tokens. For vLLM on one GPU:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --gpu-memory-utilization 0.45 \
    --generation-config vllm --enable-prompt-tokens-details
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 --gpu-memory-utilization 0.45 \
    --generation-config vllm --enable-prompt-tokens-details --no-enable-prefix-caching
python cachecheck.py --mode two-servers --base-url http://127.0.0.1:8000 \
    --nocache-url http://127.0.0.1:8001 --model Qwen/Qwen2.5-7B-Instruct
```

For SGLang, add `--enable-cache-report` to both servers and
`--disable-radix-cache` to the second one.

Two server processes can also differ for other reasons. To check that, start
both with prefix caching off and run with `--skip-hit-check`. Every probe
should come out the same.

Options: `--n` sets the number of probes (default 20), `--max-tokens` the
answer length (default 256), `--out` the JSON report path.

Models that think before answering need thinking turned off for this check.
Their templates often drop past thinking from the history, which shortens the
cache hit, and the tool marks those probes as inconclusive.

## Reading the result

A real run on Apple Silicon (Metal) with Qwen2.5-3B-Instruct at Q4_K_M,
llama.cpp b12138, `-np 1 --cache-ram 0`, default options:

```
probe  1/20: same
...
probe 10/20: DIFFERENT (from character 578)
...
probe 13/20: DIFFERENT (from character 235)
...
probe 15/20: DIFFERENT (from character 376)
...
3 of 20 valid probes gave a different turn-2 answer on the cache path.
```

All 20 probes were valid. On every probe the server reported that the cache
hit covered all of turn 1 and its answer, and that the recompute path reused
nothing. The full report is in
[`examples/qwen2.5-3b-q4km_metal_b12138.json`](examples/qwen2.5-3b-q4km_metal_b12138.json).
It took about 15 minutes on a laptop.

| Exit status | Meaning |
|---|---|
| 0 | No valid probe differed, and at least 80% of probes were valid |
| 1 | At least one valid probe differed |
| 2 | Inconclusive: too few valid probes, or a request failed |

The JSON report keeps every answer, the finish reasons, token counts, cached
token counts and the server build.

The rate is specific to these short probes. It is not comparable to the
episode rates below, where one agent episode chains about ten requests.

## What to do about a difference

For runs that must repeat, turn prefix reuse off and record that setting with
your results:

- llama.cpp: `"cache_prompt": false` per request, or `--no-cache-prompt` for
  the server
- vLLM: `--no-enable-prefix-caching`
- SGLang: `--disable-radix-cache`

A difference is not a bug in your model. It shows that the answer depends on
the server's cache state as well as on the request.

## Background and citation

`cachecheck` adapts the paired cache-on and cache-off design of the study in
this repository. The study used Qwen2.5-7B-Instruct on llama.cpp with greedy
decoding, 80 BFCL multi-turn agent episodes and the host-memory prompt cache
off. The share of episodes whose trajectory changed between a cache hit and a
recomputation was 37.4% with F16 weights and 82.5% with Q4_K_M weights, as a
mean over ten server histories.

If `cachecheck` helps your work, please cite the paper:

```bibtex
@article{patodiya2026samerequest,
  title   = {Same Request, Different Answer: Quantization Amplifies
             Cache-Induced Divergence in LLM Serving},
  author  = {Patodiya, Aditi},
  journal = {IEEE Access},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3749922},
  note    = {Early Access},
  url     = {https://doi.org/10.1109/ACCESS.2026.3749922}
}
```

MIT License, as for the rest of this repository.
