#!/bin/bash
# Generate the gRPC / protobuf Python stubs a source checkout needs (they are build outputs, not tracked):
# rtp_llm/cpp/model_rpc/proto/*_pb2*.py and rtp_llm/dash_sc/proto/*_pb2*.py. Needs grpcio-tools (the engine
# environment has it). Run from anywhere; writes next to the .proto files.
# usage: [PYTHON=python3] bash gen_protos.sh [repo_root]
set -e
WS=${1:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PYTHON:-python3}
cd "$WS/rtp_llm/cpp/model_rpc/proto"
for p in model_rpc_service.proto flexlb_schedule_service.proto; do
  $PY -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. "$p"
done
cd "$WS/rtp_llm/dash_sc/proto"
$PY create_grpc_proto.py .
ls "$WS"/rtp_llm/cpp/model_rpc/proto/*_pb2*.py "$WS"/rtp_llm/dash_sc/proto/*_pb2*.py
