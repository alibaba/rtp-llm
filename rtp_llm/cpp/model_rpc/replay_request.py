"""Replay a P-worker request log or a prefill model-input snapshot on a dev server.

On the P worker, ENABLE_PREFILL_REQUEST_LOG=1 writes complete native requests to
LOG_PATH/prefill_requests/prefill_request_r<RANK>_p<PID>.log.
ENABLE_MODEL_INPUTS_LOG=1 also writes
forward snapshots, with full prefill token IDs, to LOG_PATH/model_inputs/*.pt.

python -m rtp_llm.cpp.model_rpc.replay_request prefill_request_r0_p123.log \
    --request-id 123 --address localhost:8089
python -m rtp_llm.cpp.model_rpc.replay_request model_inputs_r0_s0_p123_000.pt \
    --record-sequence 7 --prefill-index 0 --address localhost:8089

Without --address, print the reconstructed native RPC request as JSON. Snapshot
replay uses greedy generation of one token; request-log replay retains generation
settings. Both rebuild the input on the dev server, without production routing.
"""

import argparse
import json
import time
from pathlib import Path

from google.protobuf.json_format import MessageToJson, ParseDict
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import GenerateInputPB


def load_request_log(path: Path, request_id: int | None = None) -> GenerateInputPB:
    selected = None
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") != "prefill_request":
                continue
            request = ParseDict(record["request"], GenerateInputPB())
            if request_id is not None and request.request_id != request_id:
                continue
            if selected is not None:
                raise ValueError(
                    "Multiple requests match; select --request-id or extract one log line"
                )
            selected = request
    if selected is None:
        raise ValueError("No matching prefill request in log")
    return selected


def load_model_inputs(
    path: Path, record_sequence: int | None = None, prefill_index: int = 0
) -> GenerateInputPB:
    import torch

    chunk = torch.load(path, map_location="cpu", weights_only=False)
    records = (
        chunk["records"]
        if chunk.get("record_type") == "model_inputs_chunk"
        else [chunk]
    )
    snapshots = [
        record
        for record in records
        if record.get("record_type") == "gpt_model_inputs_snapshot"
        and record.get("all_input_token_ids") is not None
        and (record_sequence is None or record["record_sequence"] == record_sequence)
    ]
    if len(snapshots) != 1:
        raise ValueError(
            "Select one prefill snapshot with --record-sequence; "
            "the dump must contain all_input_token_ids (older dumps cannot recover cached prefixes)"
        )
    snapshot = snapshots[0]
    full_tokens = snapshot["all_input_token_ids"]
    if not 0 <= prefill_index < len(full_tokens):
        raise ValueError("--prefill-index is outside the context batch")
    modalities = snapshot.get("request_is_multimodal")
    if modalities is None or len(modalities) != len(full_tokens):
        raise ValueError(
            "The dump must contain request_is_multimodal for every prefill request; "
            "older dumps cannot safely distinguish cached multimodal inputs"
        )
    if modalities[prefill_index]:
        raise ValueError(
            "Token-only snapshot replay cannot reconstruct a multimodal request, "
            "including images already in the prefix cache"
        )
    if snapshot.get("input_embeddings") is not None:
        raise ValueError(
            "Token-only snapshot replay does not reconstruct input embeddings"
        )
    request = GenerateInputPB(
        request_id=int(snapshot["request_id"][prefill_index]),
        token_ids=full_tokens[prefill_index].tolist(),
    )
    config = request.generate_config
    config.max_new_tokens = 1
    config.num_beams = 1
    config.num_return_sequences = 1
    config.top_k = 1
    config.top_p = 1.0
    config.temperature = 1.0
    config.repetition_penalty = 1.0
    trace_ids = snapshot.get("trace_ids", [])
    if len(trace_ids) >= len(full_tokens):
        config.trace_id.value = trace_ids[
            len(trace_ids) - len(full_tokens) + prefill_index
        ]
    return request


def prepare_local_request(request: GenerateInputPB) -> GenerateInputPB:
    replay = GenerateInputPB()
    replay.CopyFrom(request)
    replay.start_time = time.time_ns() // 1000
    replay.ClearField("client_id")
    replay.ClearField("group_id")
    replay.ClearField("group_size")
    replay.generate_config.ClearField("role_addrs")
    replay.generate_config.can_use_pd_separation = False
    return replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--request-id", type=int)
    parser.add_argument("--record-sequence", type=int)
    parser.add_argument("--prefill-index", type=int, default=0)
    parser.add_argument(
        "--address", help="Dev backend native RPC address, not the Dash gRPC port"
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--timeout-ms", type=int, default=60000)
    args = parser.parse_args()
    if args.timeout_ms <= 0 or (
        args.max_new_tokens is not None and args.max_new_tokens <= 0
    ):
        parser.error("--timeout-ms and --max-new-tokens must be positive")
    if args.input_path.suffix == ".pt":
        if args.request_id is not None:
            parser.error(
                "Use --record-sequence and --prefill-index for a model-input dump"
            )
        request = load_model_inputs(
            args.input_path, args.record_sequence, args.prefill_index
        )
    else:
        if args.record_sequence is not None or args.prefill_index != 0:
            parser.error("Use --request-id for a prefill request log")
        request = load_request_log(args.input_path, args.request_id)
    request = prepare_local_request(request)
    request.generate_config.timeout_ms = args.timeout_ms
    if args.max_new_tokens is not None:
        request.generate_config.max_new_tokens = args.max_new_tokens
    if not args.address:
        print(MessageToJson(request, preserving_proto_field_name=True))
        return

    import grpc
    from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub

    with grpc.insecure_channel(
        args.address,
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    ) as channel:
        for response in RpcServiceStub(channel).GenerateStreamCall(
            request, timeout=args.timeout_ms / 1000
        ):
            print(MessageToJson(response, preserving_proto_field_name=True))


if __name__ == "__main__":
    main()
