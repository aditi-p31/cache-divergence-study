#!/bin/bash
# Validation gate (POD_RUNBOOK.md step 1). Run detached; writes
# /workspace/gate.log and exits 0 only if all three checks pass.
set -euo pipefail
cd /workspace/study
export PATH=/usr/local/cuda-12.6/bin:$PATH

MODEL=/workspace/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
SRV=/workspace/llama.cpp/build/bin/llama-server

echo "=== GATE 1: llama.cpp CUDA smoke (2 episodes, both arms) ==="
"$SRV" -m "$MODEL" --alias qwen7b-q4km --host 127.0.0.1 --port 8080 \
  --parallel 1 -c 16384 -ngl 99 --seed 42 --slots > /workspace/srv_gate.log 2>&1 &
SRV_PID=$!
for i in $(seq 1 60); do
  curl -s http://127.0.0.1:8080/health | grep -q '"ok"' && break
  sleep 2
done
curl -s http://127.0.0.1:8080/health | grep -q '"ok"' || { echo "GATE FAIL: server never healthy"; exit 1; }

T0=$(date +%s)
PYTHONPATH=harness uv run python harness/run_episodes.py \
  --model-hf-id Qwen/Qwen2.5-7B-Instruct --served-model qwen7b-q4km \
  --base-url http://127.0.0.1:8080/v1 --backend llamacpp --family qwen \
  --arm on --n-episodes 2 --out-dir /workspace/results/gate 2>&1 | grep -E '^\[|done'
T1=$(date +%s)
PYTHONPATH=harness uv run python harness/run_episodes.py \
  --model-hf-id Qwen/Qwen2.5-7B-Instruct --served-model qwen7b-q4km \
  --base-url http://127.0.0.1:8080/v1 --backend llamacpp --family qwen \
  --arm off --n-episodes 2 --out-dir /workspace/results/gate 2>&1 | grep -E '^\[|done'
T2=$(date +%s)
echo "TIMING: arm_on 2ep $((T1-T0))s, arm_off 2ep $((T2-T1))s"
uv run python harness/compare_arms.py /workspace/results/gate

echo "=== GATE 2: GSM8K bridge n=10 on CUDA ==="
PYTHONPATH=harness uv run python harness/run_gsm8k.py \
  --model-hf-id Qwen/Qwen2.5-7B-Instruct --served-model qwen7b-q4km \
  --base-url http://127.0.0.1:8080/v1 --backend llamacpp --cache on \
  --n 10 --out-dir /workspace/results/gate_gsm8k 2>&1 | tail -3

kill $SRV_PID 2>/dev/null || true
echo "=== GATE 3: throughput extract ==="
grep -oE 'prompt processing, n_tokens = +[0-9]+.*tokens per second' /workspace/srv_gate.log | tail -3

echo "=== VALIDATION GATE COMPLETE ==="
