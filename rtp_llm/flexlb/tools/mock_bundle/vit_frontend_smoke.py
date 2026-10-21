"""Check a running real frontend + Master + Mock ViT/P/D deployment.

The frontend must use a Qwen2-VL tokenizer/config with remote ViT enabled.
Run inside the frontend environment, with generated protobufs on PYTHONPATH.
"""
import argparse
import base64
import concurrent.futures
import io
import json
from pathlib import Path

import grpc
import requests
from PIL import Image
from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import MultimodalRpcServiceStub


def image(width, height, color):
    data = io.BytesIO()
    Image.new('RGB', (width, height), color).save(data, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontend', required=True)
    parser.add_argument('--vit-grpc', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--require-reuse', action='store_true')
    args = parser.parse_args()
    small, large = image(56, 56, (110, 30, 170)), image(448, 448, (40, 180, 60))
    cases = [('text', [], 0), ('small', [small], 4), ('large', [large], 256),
             ('duplicate', [small, small], 8), ('mixed', [large, small], 260)]

    def send(case):
        name, images, expected = case
        content = [{'type': 'text', 'text': 'tok9 tok10'}]
        content += [{'type': 'image_url', 'image_url': {'url': url}} for url in images]
        response = requests.post(args.frontend.rstrip('/') + '/v1/chat/completions',
            json={'model': 'mock-vit', 'messages': [{'role': 'user', 'content': content}],
                  'max_tokens': 4, 'stream': False, 'extra_configs': {'reuse_cache': True}}, timeout=30)
        response.raise_for_status()
        result = response.json()
        usage, aux = result['usage'], result['aux_info']
        assert usage.get('prompt_tokens_details', {}).get('image_tokens', 0) == expected, result
        assert sum(aux['multimodal_lengths'].values()) == expected, result
        assert usage['prompt_tokens'] == aux['input_len'], result
        assert usage['total_tokens'] == aux['input_len'] + aux['output_len'], result
        assert usage['completion_tokens'] == 4, result
        assert {'PREFILL', 'DECODE'} <= {a['role'] for a in aux['role_addrs']}, result
        return {'case': name, 'image_tokens': expected, 'input_len': aux['input_len'],
                'prefix_len': aux['prefix_len'], 'output_len': aux['output_len']}

    rows = [send(case) for case in cases]
    rows.extend(send(case) for case in cases)  # warm cache
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows.extend(pool.map(send, [cases[4]] * 8))
    with grpc.insecure_channel(args.vit_grpc) as channel:
        vit = MultimodalRpcServiceStub(channel)
        status = vit.GetWorkerStatus(pb.StatusVersionPB(), timeout=5)
        assert status.alive and status.role == 'VIT'
        assert not vit.GetCacheStatus(pb.CacheVersionPB(), timeout=5).HasField('multimodal_cache')
        directory = vit.GetCacheStatus(pb.CacheVersionPB(need_cache_keys=True), timeout=5).multimodal_cache
        assert directory.worker_instance and len(directory.keys) >= 2
    if args.require_reuse:
        assert any(row['case'] == 'large' and row['prefix_len'] > 0 for row in rows[5:]), rows
    output = {'passed': len(rows), 'requests': rows, 'vit_worker_instance': directory.worker_instance,
              'hash_keys': len(directory.keys), 'cpu_embedding_keys': len(directory.cpu_embedding_keys)}
    Path(args.output).write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
