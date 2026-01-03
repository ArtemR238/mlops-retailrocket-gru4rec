#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:-artifacts/model.onnx}"
TRT_PATH="${2:-artifacts/model.trt}"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec is not available. Install TensorRT and ensure trtexec is in PATH."
  exit 1
fi

mkdir -p "$(dirname "$TRT_PATH")"

# Basic FP16 build (adjust as needed)
trtexec --onnx="$ONNX_PATH" --saveEngine="$TRT_PATH" --fp16

echo "Saved TensorRT engine to: $TRT_PATH"
