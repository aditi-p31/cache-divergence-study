#!/bin/bash
# Phase B setup. Models and build live on the container disk (/root/work);
# /workspace is a 20 GB network volume used only as a results mirror.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
W=/root/work; MIR=/mnt/persistent/phaseB
mkdir -p "$W/models" "$MIR"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$W/setup.log"; cp "$W/setup.log" "$MIR/setup.log" 2>/dev/null; }
fail(){ log "SETUP_FAIL: $*"; touch "$W/SETUP_FAIL"; cp "$W/SETUP_FAIL" "$MIR/" 2>/dev/null; exit 1; }
rm -f "$W/SETUP_DONE" "$W/SETUP_FAIL"

log "[1/5] apt: cmake rsync"
apt-get install -y -qq cmake rsync build-essential >/dev/null 2>&1 || fail "apt install"

log "[2/5] models (parallel download, sha256 verified against HF LFS oids)"
declare -A SHA=(
 [Qwen2.5-7B-Instruct-f16.gguf]=863c978275bca3fdacdde06cbfc1a65f2bd65210bb7ecfd9ded3b24219d81b54
 [Qwen2.5-7B-Instruct-Q8_0.gguf]=9c6a6e61664446321d9c0dd7ee28a0d03914277609e21bc0e1fce4abe780ce1b
 [Qwen2.5-7B-Instruct-Q4_K_M.gguf]=65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423
 [Qwen2.5-7B-Instruct-Q3_K_M.gguf]=6738a2d4f9b280c55b2a19a7ab27334a75d2cafc6ef82a11a075f4b5613eb736 )
for f in "${!SHA[@]}"; do
  ( [ -f "$W/models/$f" ] || wget -q -c "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/$f" -O "$W/models/$f" ) &
done

log "[3/5] llama.cpp b10434 build (CUDA 12.6, sm_89) while models download"
export PATH=/usr/local/cuda-12.6/bin:$PATH
if [ ! -x "$W/llama.cpp/build/bin/llama-server" ]; then
  git clone -q --depth 1 --branch b10434 https://github.com/ggml-org/llama.cpp "$W/llama.cpp" || fail "clone"
  C=$(git -C "$W/llama.cpp" rev-parse --short=7 HEAD); [ "$C" = "7e4c0a9" ] || fail "commit $C != 7e4c0a9"
  cmake -S "$W/llama.cpp" -B "$W/llama.cpp/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF >>"$W/build.log" 2>&1 || fail "cmake configure"
  cmake --build "$W/llama.cpp/build" --target llama-server -j 21 >>"$W/build.log" 2>&1 || fail "cmake build"
fi
"$W/llama.cpp/build/bin/llama-server" --version 2>&1 | head -2 | tee -a "$W/setup.log"

log "[4/5] harness venv + tokenizer"
cd "$W/study" && uv sync --quiet >>"$W/setup.log" 2>&1 || fail "uv sync"
uv run python -c "
from transformers import AutoTokenizer; import bfcl_eval, importlib.metadata as m
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct'); print('tokenizer ok; bfcl-eval', m.version('bfcl-eval'))" >>"$W/setup.log" 2>&1 || fail "tokenizer/bfcl"
tail -1 "$W/setup.log"

wait
log "[5/5] sha256 verification"
: > "$W/models/SHA256SUMS"
for f in "${!SHA[@]}"; do
  got=$(sha256sum "$W/models/$f" | cut -d" " -f1)
  echo "$got  $f" >> "$W/models/SHA256SUMS"
  [ "$got" = "${SHA[$f]}" ] || fail "sha mismatch $f: $got"
  log "  ok $f"
done
cp "$W/models/SHA256SUMS" "$MIR/"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a "$W/setup.log"
log "SETUP_DONE"; touch "$W/SETUP_DONE"; cp "$W/SETUP_DONE" "$MIR/"
