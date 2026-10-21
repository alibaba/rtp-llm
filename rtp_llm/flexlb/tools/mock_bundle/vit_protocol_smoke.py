"""Exercise a running main Master + Mock ViT/P/D using the native protobuf contracts.

This protocol client does not replace real frontend HTTP/native-runtime acceptance.
Generate model/schedule Python protobufs before running it.
"""
import argparse
import base64
import json
import struct
import time
import urllib.request
import zlib
from pathlib import Path

import grpc

from rtp_llm.cpp.model_rpc.proto import flexlb_schedule_service_pb2 as schedule
from rtp_llm.cpp.model_rpc.proto import flexlb_schedule_service_pb2_grpc as schedule_rpc
from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2_grpc as rpc


def multimodal_key(image):
    # Linux libstdc++ string identity used by MultimodalInput::cache_key.
    data = image.multimodal_url.encode()
    mask, multiplier = (1 << 64) - 1, 0xC6A4A7935BD1E995
    result = 0xC70F6907 ^ (len(data) * multiplier & mask)
    full = len(data) // 8 * 8
    for offset in range(0, full, 8):
        word = int.from_bytes(data[offset:offset + 8], 'little') * multiplier & mask
        word ^= word >> 47
        result = (result ^ (word * multiplier & mask)) * multiplier & mask
    if full < len(data):
        result = (result ^ int.from_bytes(data[full:], 'little')) * multiplier & mask
    result ^= result >> 47
    result = result * multiplier & mask
    result ^= result >> 47
    return str(result) + '_1_-1_-1_-1_-1_-1_-1_-1__-1'


def png_url():
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 56, 56, 8, 2, 0, 0, 0))
    data += chunk(b'IDAT', zlib.compress((b'\0' + b'\x11\x22\x33' * 56) * 56)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(data).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master', required=True, help='host:grpc-port')
    parser.add_argument('--control', required=True, help='Mock HTTP base URL')
    parser.add_argument('--processor-identity', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    results = []
    with grpc.insecure_channel(args.master) as channel:
        master = schedule_rpc.FlexlbServiceStub(channel)
        for index, kind in enumerate(('url', 'prepared_v41', 'prepared_text')):
            rid = int(time.time_ns() // 1000) + index
            image = pb.MultimodalInputPB(multimodal_url=png_url(), multimodal_type=1,
                mm_preprocess_config=pb.MMPreprocessConfigPB(width=-1, height=-1, min_pixels=-1, max_pixels=-1,
                    fps=-1, min_frames=-1, max_frames=-1, max_long_side_pixel=-1))
            keys = [multimodal_key(image)] if kind == 'url' else []
            selected = None
            if kind != 'prepared_text':
                chosen = master.Schedule(schedule.FlexlbScheduleRequestPB(request_id=rid, seq_len=1,
                    generate_timeout=30000, request_time_ms=int(time.time() * 1000), vit_route_only=True,
                    cache_affinity_keys=keys, cache_key_block_size=2), timeout=5)
                assert chosen.success and len(chosen.server_status) == 1, chosen
                assert not chosen.enqueued_by_master and not chosen.HasField('lifecycle'), chosen
                selected = chosen.server_status[0]
            request = pb.GenerateInputPB(request_id=rid, generate_config=pb.GenerateConfigPB(
                max_new_tokens=4, num_beams=1, num_return_sequences=1, timeout_ms=30000))
            if kind == 'url':
                with grpc.insecure_channel(f'{selected.server_ip}:{selected.grpc_port}') as vit_channel:
                    vit = rpc.MultimodalRpcServiceStub(vit_channel)
                    metadata = vit.GetMultimodalHashes(pb.MultimodalHashRequestPB(keys=keys,
                        inputs=pb.MultimodalInputsPB(request_id=rid, multimodal_inputs=[image]), timeout_ms=5000), timeout=6)
                    entry = metadata.entries[0]
                    assert entry.hash_hit and list(entry.split_size) == [4], metadata
                    ids = list(struct.unpack('<4i', entry.feature_hashes))
                    request.token_ids.extend([10] + ids + [11])
                    request.multimodal_inputs.append(image)
                    request.multimodal_token_layout.spans.add(offset=1, length=4)
            elif kind == 'prepared_v41':
                typed_image = pb.V41ImageInputPB(start=1, n_vit_h=3, n_vit_w=6, types=[0, 1, 1, 2, 3],
                    content_sha256='a' * 64, processor_identity=args.processor_identity,
                    patches=pb.TensorPB(data_type=pb.TensorPB.BF16, shape=[18, 3, 14, 14], bf16_data=bytes(18 * 3 * 14 * 14 * 2)))
                request.token_ids.extend([42] + [129264] * 5 + [43])
                request.v41_inputs.CopyFrom(pb.V41RequestInputsPB(schema_version=1, images=[typed_image],
                    token_types=[-1, 0, 1, 1, 2, 3, -1], image_mask=[False, True, True, True, True, True, False]))
            else:
                request.token_ids.extend([42, 43])
                request.v41_inputs.CopyFrom(pb.V41RequestInputsPB(schema_version=1, token_types=[-1, -1], image_mask=[False, False]))
            route_request = schedule.FlexlbScheduleRequestPB(request_id=rid, generate_input=request.SerializeToString(),
                seq_len=len(request.token_ids), generate_timeout=30000, request_time_ms=int(time.time() * 1000),
                max_new_tokens=4, num_beams=1, cache_key_block_size=2)
            if selected is not None:
                route_request.selected_vit.CopyFrom(selected)
            routed = master.Schedule(route_request, timeout=30)
            assert routed.success, routed
            roles = {status.role: status for status in routed.server_status}
            assert {'PREFILL', 'DECODE'} <= roles.keys(), routed
            if selected is not None:
                assert roles['VIT'].server_ip == selected.server_ip and roles['VIT'].grpc_port == selected.grpc_port, routed
            for role, status in roles.items():
                request.generate_config.role_addrs.add(role_str=role, ip=status.server_ip,
                    grpc_port=status.grpc_port, http_port=status.http_port)
            prefill = roles['PREFILL']
            with grpc.insecure_channel(f'{prefill.server_ip}:{prefill.grpc_port}') as p_channel:
                engine = rpc.RpcServiceStub(p_channel)
                stream = (engine.FetchResponse(pb.FetchRequestPB(request_id=rid), timeout=30)
                          if routed.enqueued_by_master else engine.GenerateStreamCall(request, timeout=30))
                frames = list(stream)
            assert frames and all(not frame.error_info.ByteSize() for frame in frames), frames
            assert any(frame.flatten_output.finished and all(frame.flatten_output.finished) for frame in frames), frames
            tokens = sum(len(frame.flatten_output.output_ids.int32_data) // 4 for frame in frames)
            assert tokens == 4, (kind, tokens, frames)
            assert frames[-1].flatten_output.aux_info[0].input_len == len(request.token_ids), frames
            results.append(dict(case=kind, prompt_tokens=len(request.token_ids), completion_tokens=tokens,
                                dispatcher='BATCH' if routed.enqueued_by_master else 'NON_BATCH'))
    deadline = time.monotonic() + 10
    while True:
        with urllib.request.urlopen(args.control.rstrip('/') + '/snapshot', timeout=5) as response:
            snapshot = json.load(response)
        engines = snapshot['engines']
        totals = {key: sum(int(engine.get(key, 0)) for engine in engines)
                  for key in ('running', 'waiting', 'held_blocks', 'referenced_blocks')}
        if not any(totals.values()):
            break
        assert time.monotonic() < deadline, totals
        time.sleep(.1)
    output = dict(status='PASS', scope='protobuf protocol smoke; no real frontend HTTP', cases=results, drain=totals)
    Path(args.output).write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
