#!/usr/bin/env bash
set -Eeuo pipefail
RUNTIME=/home/admin/rtp-hol/runtime/rtp-master-d4d9bf18b
[[ -d "$RUNTIME" ]] && { echo "exists"; exit 0; }
stage="${RUNTIME}.staging.$$"
trap 'rm -rf "$stage"' EXIT
wheel=/home/admin/.cache/bazel/_bazel_admin/d1da4251353000c992adb5dcea5c1d71/execroot/rtp_llm/bazel-out/k8-opt/bin/rtp_llm/rtp_llm-0.2.0-cp310-cp310-manylinux1_x86_64.whl
test -s "$wheel"
mkdir -p "$stage/site-packages"
/opt/conda310/bin/python -m pip install --no-deps -q --target "$stage/site-packages" "$wheel"
/opt/conda310/bin/python -m pip install -q --target "$stage/site-packages" "pydantic==2.13.4"
test -s "$stage/site-packages/rtp_llm/libs/libth_transformer.so"
echo d4d9bf18b+grpc-build-patch > "$stage/source_commit.txt"
mv "$stage" "$RUNTIME"
trap - EXIT
echo "RUNTIME_OK $RUNTIME"
