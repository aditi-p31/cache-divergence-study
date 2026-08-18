#!/bin/bash
# One-time pod setup for the cache-divergence study. Idempotent-ish.
# Run: nohup bash /workspace/study/harness/pod_setup.sh > /workspace/setup.log 2>&1 &
set -euo pipefail
cd /workspace

echo "=== [1/6] apt tooling ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq cmake build-essential rsync tmux > /dev/null

echo "=== [2/6] CUDA toolkit 12.6 ==="
if ! command -v nvcc >/dev/null; then
  curl -sLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb
  apt-get update -qq
  apt-get install -y -qq cuda-toolkit-12-6 > /dev/null
fi
export PATH=/usr/local/cuda-12.6/bin:$PATH
nvcc --version | tail -1

echo "=== [3/6] llama.cpp b10434 CUDA build ==="
if [ ! -x /workspace/llama.cpp/build/bin/llama-server ]; then
  git clone --depth 1 --branch b10434 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
  cd /workspace/llama.cpp
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF > /dev/null
  cmake --build build --target llama-server -j "$(nproc)" 2>&1 | tail -2
  cd /workspace
fi
/workspace/llama.cpp/build/bin/llama-server --version 2>&1 | head -2

echo "=== [4/6] harness venv ==="
cd /workspace/study
uv sync --quiet
uv run python -c "import bfcl_eval, importlib.metadata as m; print('bfcl-eval', m.version('bfcl-eval'))"

echo "=== [5/6] vLLM venv ==="
if [ ! -d /workspace/venv-vllm ]; then
  uv venv /workspace/venv-vllm --python 3.12 --quiet
  VIRTUAL_ENV=/workspace/venv-vllm uv pip install --quiet vllm
fi
VIRTUAL_ENV=/workspace/venv-vllm uv pip show vllm | head -2

echo "=== [6/6] SGLang venv ==="
if [ ! -d /workspace/venv-sglang ]; then
  uv venv /workspace/venv-sglang --python 3.12 --quiet
  VIRTUAL_ENV=/workspace/venv-sglang uv pip install --quiet "sglang[all]"
fi
VIRTUAL_ENV=/workspace/venv-sglang uv pip show sglang | head -2

echo "=== SETUP COMPLETE ==="
