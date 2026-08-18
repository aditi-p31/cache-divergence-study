#!/bin/bash
# Setup for the repair pod. Pins the SAME stack as the original collection,
# deliberately: the repair run exists to compare execution orderings, so the
# engine versions must not differ from the runs it is being compared against.
# This pod's driver (580) would support newer vLLM; we do not use it.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
cd /workspace

echo "=== [1/5] apt + cuda toolkit ==="
apt-get update -qq
apt-get install -y -qq cmake build-essential rsync git wget > /dev/null
rm -f /etc/apt/sources.list.d/cuda.list   # template ships a duplicate unsigned entry
if ! /usr/local/cuda-12.6/bin/nvcc --version >/dev/null 2>&1; then
  curl -sLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null 2>&1
  apt-get update -qq
  apt-get install -y -qq cuda-nvcc-12-6 cuda-cudart-dev-12-6 libcublas-dev-12-6 >/dev/null 2>&1 \
    || apt-get install -y -qq cuda-nvcc-12-6 cuda-cudart-dev-12-6 >/dev/null 2>&1
fi
export PATH=/usr/local/cuda-12.6/bin:$PATH
nvcc --version | tail -1

echo "=== [2/5] llama.cpp b10434 (same commit as the original runs) ==="
if [ ! -x /workspace/llama.cpp/build/bin/llama-server ]; then
  git clone --depth 1 --branch b10434 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp 2>&1 | tail -1
  cd /workspace/llama.cpp
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF > /dev/null
  cmake --build build --target llama-server -j "$(nproc)" 2>&1 | tail -2
  cd /workspace
fi
/workspace/llama.cpp/build/bin/llama-server --version 2>&1 | head -2

echo "=== [3/5] harness venv ==="
cd /workspace/study && uv sync --quiet
uv run python -c "import bfcl_eval, importlib.metadata as m; print('bfcl-eval', m.version('bfcl-eval'))"

echo "=== [4/5] vLLM venv, pinned to the original stack ==="
if [ ! -d /workspace/venv-vllm ]; then uv venv /workspace/venv-vllm --python 3.12 --quiet; fi
VIRTUAL_ENV=/workspace/venv-vllm uv pip install --quiet "vllm==0.11.0" --torch-backend=cu128
VIRTUAL_ENV=/workspace/venv-vllm uv pip install --quiet "transformers<5" ninja
ln -sf /workspace/venv-vllm/bin/ninja /usr/local/bin/ninja
/workspace/venv-vllm/bin/python -c "import vllm,torch,transformers;print('vllm',vllm.__version__,'torch',torch.__version__,'tf',transformers.__version__,'cuda',torch.cuda.is_available())"

echo "=== [5/5] models ==="
mkdir -p /workspace/models
GGUF=/workspace/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
[ -f "$GGUF" ] || wget -q "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf" -O "$GGUF"
ls -la "$GGUF"
cd /workspace/study && uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct')
print('hf weights ready')"

echo "=== SETUP COMPLETE ==="
