"""Exercise frozen release frontend metadata RPC -> Master -> Mock P/D.

This is a protocol probe. It does not validate HTTP rendering, GPU/RDMA, or an incident cause.
"""
import argparse
import asyncio
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import grpc
import torch
from PIL import Image
from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.cpp.model_rpc.proto import flexlb_schedule_service_pb2 as schedule
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
from rtp_llm.cpp.model_rpc.proto.flexlb_schedule_service_pb2_grpc import FlexlbServiceStub
from rtp_llm.models.multimodal.deepseek_v41_processor import V41ImageProcessorConfig, image_token_types
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.utils.grpc_host_channel_pool import GrpcHostChannelPool
from rtp_llm.utils.grpc_util import trans_from_tensor


def cache_keys(ids, block_size, typed=None):
    result, value, mask = [], 0, (1 << 64) - 1
    for index, token in enumerate(ids):
        value = (value ^ ((token + 0x9E3779B97F4A7C15 + (value << 12) + (value >> 32)) & mask)) & mask
        if value >= 1 << 63:
            value -= 1 << 64
        if (index + 1) % block_size == 0:
            if typed is not None:
                begin, end = index + 1 - block_size, index + 1
                for image in typed.images:
                    start, image_end = image.start, image.start + len(image.types)
                    if image_end <= begin or start >= end:
                        continue
                    words = [-41, start, image_end, image.n_vit_h, image.n_vit_w]
                    for digest in (image.content_sha256, image.processor_identity):
                        words.extend(int(digest[i:i + 8], 16) for i in range(0, len(digest) - 7, 8))
                    for word in words:
                        word = word if word < 1 << 31 else word - (1 << 32)
                        value = (value ^ ((word + 0x9E3779B97F4A7C15 + (value << 12) + (value >> 32)) & mask)) & mask
                        if value >= 1 << 63:
                            value -= 1 << 64
            result.append(value)
    return result


async def run(args):
    visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
    visitor._vit_channel_pool = GrpcHostChannelPool()
    results = []
    with grpc.insecure_channel(args.master_grpc) as channel:
        master = FlexlbServiceStub(channel)
        for packed in (False, True):
            for repeat in range(2):
                rid = time.time_ns() % ((1 << 62) - 1)
                raw = pb.GenerateInputPB(request_id=rid, token_ids=[11, 6, 12])
                raw.generate_config.max_new_tokens = 4
                raw.generate_config.reuse_cache = True
                if packed:
                    types = image_token_types(9, 13).tolist()
                    raw.ClearField('token_ids')
                    raw.token_ids.extend([20 + i for i in range(len(types))] + [12])
                    typed = raw.v41_inputs
                    typed.schema_version = 1
                    typed.token_types.extend(types + [-1])
                    typed.image_mask.extend([True] * len(types) + [False])
                    config = V41ImageProcessorConfig(max_seq_len=16384)
                    patches = torch.zeros((27 * 39, 3, 14, 14), dtype=torch.bfloat16)
                    image = typed.images.add(start=0, n_vit_h=27, n_vit_w=39, types=types,
                        content_sha256=hashlib.sha256(patches.view(torch.uint8).numpy().tobytes()).hexdigest(),
                        processor_identity=config.identity)
                    image.patches.CopyFrom(trans_from_tensor(patches))
                else:
                    data = io.BytesIO()
                    Image.new('RGB', (224, 224), (90, 120, 50)).save(data, format='PNG')
                    raw.multimodal_inputs.add(multimodal_url='data:image/png;base64,' + base64.b64encode(data.getvalue()).decode(),
                        multimodal_type=1, mm_preprocess_config=pb.MMPreprocessConfigPB(width=-1, height=-1,
                        min_pixels=-1, max_pixels=-1, fps=-1, min_frames=-1, max_frames=-1))
                request = schedule.FlexlbScheduleRequestPB(request_id=rid, seq_len=len(raw.token_ids),
                    max_new_tokens=4, generate_timeout=30000, cache_key_block_size=64, vit_only=True)
                vision = master.Schedule(request, timeout=5)
                assert vision.success and len(vision.server_status) == 1, vision
                vit = vision.server_status[0]
                raw.generate_config.role_addrs.add(role_str='VIT', role=pb.RoleAddrPB.VIT,
                    ip=vit.server_ip, grpc_port=vit.grpc_port, http_port=vit.http_port)
                if packed:
                    routing_ids = list(raw.token_ids)
                else:
                    address = RoleAddr(role=RoleType.VIT, ip=vit.server_ip, grpc_port=vit.grpc_port, http_port=vit.http_port)
                    hashes = await visitor._get_vit_token_ids(address, raw, 5)
                    assert len(hashes[0]) == 64
                    routing_ids = [11] + hashes[0] + [12]
                request.vit_only = False
                request.seq_len = len(routing_ids)
                request.block_cache_keys.extend(cache_keys(routing_ids, 64, raw.v41_inputs if packed else None))
                request.generate_input = raw.SerializeToString()
                route = master.Schedule(request, timeout=10)
                assert route.success, route
                prefill = next(item for item in route.server_status if item.role == 'PREFILL')
                for item in route.server_status:
                    if item.role != 'VIT':
                        raw.generate_config.role_addrs.add(role_str=item.role, ip=item.server_ip,
                            grpc_port=item.grpc_port, http_port=item.http_port)
                with grpc.insecure_channel(f'{prefill.server_ip}:{prefill.grpc_port}') as worker_channel:
                    worker = RpcServiceStub(worker_channel)
                    stream = (worker.FetchResponse(pb.FetchRequestPB(request_id=rid), timeout=10)
                        if route.enqueued_by_master else worker.GenerateStreamCall(raw, timeout=10))
                    frames = list(stream)
                    assert len(frames) == 2 and frames[-1].flatten_output.finished[0], frames
                    if repeat:
                        assert frames[-1].flatten_output.aux_info[0].prefix_len == (128 if packed else 64), frames
                    for frame in frames:
                        assert frame.request_id == rid, frame
                        assert frame.flatten_output.aux_info[0].input_len == len(routing_ids), frame
                    results.append({'packed_v41': packed, 'repeat': repeat, 'input_len': len(routing_ids),
                        'enqueued_by_master': route.enqueued_by_master, 'frames': len(frames),
                        'prefix_len': frames[-1].flatten_output.aux_info[0].prefix_len})
    await visitor._vit_channel_pool.close()
    output = {'passed': len(results), 'requests': results}
    Path(args.output).write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master-grpc', required=True)
    parser.add_argument('--output', required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
