# llm-cachecheck

On PyPI this tool is `llm-cachecheck`. The PyPI package named `cachecheck` is
a different project.

How often does the prompt cache change your server's answers?

Serving stacks reuse the cached key-value state of a shared prompt prefix. A
cache hit and a recomputation run different operations, so with greedy
decoding a small numeric change can flip a token. After that the answer goes
its own way. This is known behavior: the llama.cpp server README says that
`cache_prompt` "can cause nondeterministic results". What varies is how often
it happens on your model, weights and build. `cachecheck` measures that.

It sends short two-turn conversations with greedy decoding and compares the
second answer on two paths:

- **cache path:** turn 2 reuses the state the server wrote while it generated
  turn 1
- **recompute path:** turn 2 is processed from scratch

Each path runs twice. A probe counts as valid only when all of these hold:

- both paths repeat on their own
- the server's own counts show that the cache hit covered the turn-1 answer,
  to within `--hit-tolerance` tokens and at least past the turn-1 prompt
- the server's own counts show that the recompute path reused nothing

## Run it

Python 3.8 or newer, no dependencies. Install it from PyPI:

```bash
pip install llm-cachecheck
llm-cachecheck --help
```

Or download the single file
[`cachecheck.py`](https://github.com/aditi-p31/cache-divergence-study/blob/main/cachecheck/cachecheck.py)
and run `python cachecheck.py` in place of `llm-cachecheck`. Both take the
same options. Each conversation takes six requests, so a run of the 20
built-in conversations sends 120.

**llama.cpp.** The cache is switched per request. Start the server with one
slot, because the default is several slots that share a cache. Turn the
host-memory prompt cache off to match the study's controlled runs:

```bash
llama-server -m model.gguf -np 1 --cache-ram 0 --port 8080
llm-cachecheck --base-url http://127.0.0.1:8080
```

**vLLM or SGLang, one server.** Start the server with prefix caching on (the
default) and with cached-token reporting on. The tool sends every request with
a `cache_salt`. The two turns of the cache path share one salt, so turn 2
reuses the state of turn 1. Each recompute request gets a salt that no other
request has used, so it cannot reuse any cached state.

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000 \
    --generation-config vllm --enable-prompt-tokens-details
llm-cachecheck --mode one-server --base-url http://127.0.0.1:8000 \
    --model Qwen/Qwen2.5-3B-Instruct
```

For SGLang, add `--enable-cache-report` and `--sampling-defaults openai`.
<<FILL: one-server validation sentence>>

The server must honour `cache_salt`. If it does not, the recompute requests
reuse cached state, the tool marks those probes as inconclusive and says so,
and the two-server setup below still works. The server must also report cached
tokens, so `--skip-hit-check` is not available in this mode.

This mode compares a cache hit with a cache miss on one server. From reading
the vLLM 0.33.0 and SGLang 0.5.23 source, a miss on a standard attention model,
served without a KV connector such as LMCache, takes the same code path as on a
server with prefix caching off. For some setups, for example models with Mamba
or sliding-window layers, or speculative decoding, turning prefix caching on
also changes how the server handles requests that miss. To compare with
caching off there, use two servers.

Run it on a server that gets no other traffic, such as a staging copy. On a
busy server, batching with other requests can change answers by itself. The
tool's repeat checks mark many such probes as inconclusive, but they cannot
rule out a difference caused by other traffic.

**vLLM, SGLang or another OpenAI-compatible server, two servers.** The cache
is a server setting, so serve the same model twice, once with prefix caching
on and once with it off. Both servers must report cached tokens. For vLLM on
one 24 GB GPU (a 3B model, so two copies fit):

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000 --gpu-memory-utilization 0.45 \
    --generation-config vllm --enable-prompt-tokens-details
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8100 --gpu-memory-utilization 0.45 \
    --generation-config vllm --enable-prompt-tokens-details --no-enable-prefix-caching
llm-cachecheck --mode two-servers --base-url http://127.0.0.1:8000 \
    --nocache-url http://127.0.0.1:8100 --model Qwen/Qwen2.5-3B-Instruct
```

For SGLang, add `--enable-cache-report` and `--sampling-defaults openai` to
both servers, and `--disable-radix-cache` to the second one.

Make sure each server really started on its own port. If another process
already holds a port, it can answer for it: RunPod's standard images, for
example, run an nginx proxy on port 8001 that forwards to port 8000, so both
URLs would reach the same server.
The tool then reports the recompute path reusing cached tokens and marks every
probe invalid, so check the server logs if that happens.

<<FILL: table lead-in naming both test sessions, prepared in the GPU runbook>>
Tested on real servers on 2026-11-11 and 11-12 (UTC; one RTX 4090; the tool's 20
conversations; details in the [currency note](https://github.com/aditi-p31/cache-divergence-study/blob/main/CURRENCY_CHECK_2026-11.md)):

| Server | Model | Conversations with a different turn-2 answer | Both caches off |
|---|---|---|---|
| vLLM 0.33.0, two servers | Qwen2.5-3B-Instruct, BF16 | 14 of 20, in each of 5 runs | 0 of 20 |
| SGLang 0.5.23, two servers (CUDA graphs disabled) | Qwen2.5-3B-Instruct, BF16 | 12 of 20, in each of 2 runs | 0 of 20 |
| llama.cpp v0.9.0 | Qwen2.5-7B-Instruct, Q4_K_M | 11 of 20, in each of 2 runs | |
<<FILL: one-server table rows>>

If you run it, an [issue](https://github.com/aditi-p31/cache-divergence-study/issues) with what you saw is welcome.

Two server processes can also differ for other reasons. To check that, start
both with prefix caching off and run with `--skip-hit-check`. Every probe
should come out the same.

Options: `--probes` reads your own conversations (next section), `--n` sets
the number of probes (default: all), `--max-tokens` the answer length
(default 256), `--out` the JSON report path. If your server caches in blocks or
pages larger than 32 tokens, set `--hit-tolerance` to that size, or cache hits
can look too short. With large pages, also ask for longer turn-1 answers, so
the hit can reach past the prompt.

Models that think before answering need thinking turned off for this check.
Their templates often drop past thinking from the history, which shortens the
cache hit, and the tool marks those probes as inconclusive.

## Run it on your own conversations

The built-in conversations are short and generic. To check the kind of
prompts your application sends, put your own in a JSON Lines file, one
conversation per line, and pass it with `--probes`:

```bash
llm-cachecheck --base-url http://127.0.0.1:8080 --probes my_conversations.jsonl
```

Each line has one of two forms
([example file](https://github.com/aditi-p31/cache-divergence-study/blob/main/cachecheck/examples/conversations_example.jsonl)):

```json
{"id": "returns-1", "turn1": "A customer wants to return shoes bought 40 days ago. ...", "turn2": "Rewrite the reply so it offers store credit instead."}
{"id": "sql-1", "messages": [{"role": "system", "content": "You are a careful SQL assistant."}, {"role": "user", "content": "Write a query for the ten largest orders."}], "followup": "Change it so it only counts orders created in 2025."}
```

- `turn1` is the first user message and `turn2` the follow-up.
- In the longer form, `messages` is the conversation up to and including the
  turn-1 user message, so it can carry your system prompt and earlier turns.
  `followup` is the turn-2 user message.
- `id` is optional and is copied into the report.
- Text messages only, with the roles system, developer, user and assistant.
  Tool calls and images are not supported.

The tool generates the turn-1 answer itself and builds turn 2 on it, as with
the built-in conversations, and it always decodes greedily. A file that holds
one JSON array of these objects also works.

Let turn 1 ask for an answer of a few sentences. The tool counts a cache hit
only when it reaches past the turn-1 prompt into the answer. A server that
caches in blocks (vLLM uses 16 tokens by default) can stop the hit at the
prompt when the answer is short, and the probe is then inconclusive.

The JSON report saves every answer, so keep it private if your conversations
are. The printed result block holds counts only, with no conversation text.

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
nothing. The full report (made with a pre-release build of 0.2.0) is in
[`examples/qwen2.5-3b-q4km_metal_b12138.json`](https://github.com/aditi-p31/cache-divergence-study/blob/main/cachecheck/examples/qwen2.5-3b-q4km_metal_b12138.json).
It took about 15 minutes on a laptop.

| Exit status | Meaning |
|---|---|
| 0 | No valid probe differed, and at least 80% of probes were valid |
| 1 | At least one valid probe differed |
| 2 | Inconclusive: too few valid probes, or a request failed |

A usage error (a bad or conflicting option) also exits with status 2, before
any request is sent and without writing a report.

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

## Share your result

At the end of a run the tool prints a short result block. It leaves out your
server URL and any key. If you would like to share it, open a
[result issue](https://github.com/aditi-p31/cache-divergence-study/issues/new?template=cachecheck-result.yml)
and paste the block. Results where no probe differed are just as useful.

## Background and citation

`cachecheck` adapts the paired cache-on and cache-off design of the study in
[cache-divergence-study repository](https://github.com/aditi-p31/cache-divergence-study).
The study used Qwen2.5-7B-Instruct on llama.cpp with greedy
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

MIT License. See `LICENSE`.
