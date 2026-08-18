"""Cache-arm handlers: BFCL prompting handlers with controlled caching.

A mixin overrides only the query hook so that:
  - every request carries the paired-arm cache directive (llama.cpp:
    per-request "cache_prompt"; vLLM/SGLang: server-level flag, so the body
    field is omitted there),
  - greedy decoding and a fixed seed are always enforced,
  - the complete raw response (text, token ids via logprobs, usage incl.
    cached_tokens, finish reason) is appended to a JSONL raw log.

The multi-turn orchestration loop, prompt formatting (model chat template),
tool-call decoding, and mock-environment execution are all inherited from
bfcl_eval unchanged, so results remain checker-compatible.
"""

import hashlib
import json
import time
from pathlib import Path

from bfcl_eval.model_handler.local_inference.llama_3_1 import LlamaHandler_3_1
from bfcl_eval.model_handler.local_inference.qwen import QwenHandler
from overrides import override
from transformers import AutoTokenizer

SEED = 42
MAX_NEW_TOKENS = 4096

# Backends where the cache arm is set per request vs per server process.
PER_REQUEST_CACHE_BACKENDS = {"llamacpp"}
SERVER_FLAG_CACHE_BACKENDS = {"vllm", "sglang"}


class CacheArmMixin:
    def _init_cache_arm(
        self,
        base_url: str,
        served_model_name: str,
        backend: str,
        cache_arm: bool,
        raw_log_path,
        max_context_length: int,
        tokenizer_hf_id: str,
    ) -> None:
        assert backend in PER_REQUEST_CACHE_BACKENDS | SERVER_FLAG_CACHE_BACKENDS
        self.backend = backend
        self.cache_arm = cache_arm
        self.raw_log_path = Path(raw_log_path)
        self.raw_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Attributes normally set inside bfcl's own batch_inference pipeline.
        self.model_path_or_id = served_model_name
        self.max_context_length = max_context_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_hf_id)

        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="EMPTY", max_retries=0)

        # Per-episode bookkeeping filled in by the runner before each episode.
        self.current_episode_id = None
        self.request_counter = 0

    def _cache_arm_query(self, inference_data: dict):
        function = inference_data["function"]
        message = inference_data["message"]
        formatted_prompt: str = self._format_prompt(message, function)
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        input_token_count = len(self.tokenizer.tokenize(formatted_prompt))
        if self.max_context_length < input_token_count + 2:
            leftover_tokens_count = 1000
        else:
            leftover_tokens_count = min(
                MAX_NEW_TOKENS, self.max_context_length - input_token_count - 2
            )

        extra_body = {"seed": SEED}
        if self.backend in PER_REQUEST_CACHE_BACKENDS:
            extra_body["cache_prompt"] = self.cache_arm

        start = time.time()
        api_response = self.client.completions.create(
            model=self.model_path_or_id,
            temperature=self.temperature,
            prompt=formatted_prompt,
            max_tokens=leftover_tokens_count,
            logprobs=1,
            extra_body=extra_body,
            timeout=600,
        )
        latency = time.time() - start

        record = {
            "episode_id": self.current_episode_id,
            "request_idx": self.request_counter,
            "arm": "C+" if self.cache_arm else "C-",
            "backend": self.backend,
            "prompt_sha256": hashlib.sha256(formatted_prompt.encode()).hexdigest(),
            "prompt_chars": len(formatted_prompt),
            "prompt_tokens_local_count": input_token_count,
            "latency_s": latency,
            "response": api_response.model_dump(),
        }
        with open(self.raw_log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        self.request_counter += 1

        return api_response, latency


def _make_handler(base_cls):
    class Handler(CacheArmMixin, base_cls):
        def __init__(
            self,
            model_hf_id: str,
            base_url: str,
            served_model_name: str,
            backend: str,
            cache_arm: bool,
            raw_log_path,
            max_context_length: int = 16384,
            temperature: float = 0.0,
        ) -> None:
            base_cls.__init__(
                self,
                model_name=model_hf_id,
                temperature=temperature,
                registry_name=model_hf_id,
                is_fc_model=False,
            )
            self._init_cache_arm(
                base_url=base_url,
                served_model_name=served_model_name,
                backend=backend,
                cache_arm=cache_arm,
                raw_log_path=raw_log_path,
                max_context_length=max_context_length,
                tokenizer_hf_id=model_hf_id,
            )

    # Assigned after class creation: bfcl's EnforceOverrides metaclass only
    # inspects methods defined in the class body, and the @override decorator
    # cannot resolve bases inside a dynamically created class.
    Handler._query_prompting = CacheArmMixin._cache_arm_query
    return Handler


CacheArmQwenHandler = _make_handler(QwenHandler)
CacheArmLlamaHandler = _make_handler(LlamaHandler_3_1)

FAMILY_HANDLERS = {
    "qwen": CacheArmQwenHandler,
    "llama": CacheArmLlamaHandler,
}
