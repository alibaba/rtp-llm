#!/usr/bin/env bash
# After 'bazel build //rtp_llm/bailian/proto:predict_v2_py', create symlinks in this
# directory to the generated *_pb2.py (same as model_rpc/proto symlinks to bazel-out).
# Run from repo root.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
BAZEL_BIN="$(bazel info bazel-bin 2>/dev/null || echo "bazel-out/k8-opt/bin")"
TARGET_DIR="$BAZEL_BIN/rtp_llm/bailian/proto"
for f in model_config_pb2.py model_config_pb2_grpc.py predict_v2_pb2.py predict_v2_pb2_grpc.py __init__.py; do
  if [ -f "$TARGET_DIR/$f" ]; then
    ln -sf "$(readlink -f "$TARGET_DIR/$f")" "$SCRIPT_DIR/$f"
    echo "linked $SCRIPT_DIR/$f -> $TARGET_DIR/$f"
  else
    echo "skip $f (run: bazel build //rtp_llm/bailian/proto:predict_v2_py)" >&2
  fi
done
