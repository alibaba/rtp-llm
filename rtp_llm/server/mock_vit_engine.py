"""CPU forward backend for the frozen dsv41 release ViT server.

Keep the release RPC server, admission, status and token-cache implementation;
replace only model execution and inject a CPU feature-hash kernel per worker.
"""
import argparse
import base64
import hashlib
import io
import math
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace

import torch
from PIL import Image, UnidentifiedImageError

from rtp_llm.models.multimodal.deepseek_v41_processor import V41ImageProcessorConfig
from rtp_llm.utils.mm_process_engine import MMEmbeddingRes, MMProcessEngine

_MASK = (1 << 64) - 1


def _mix(value):
    value &= _MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK
    return value ^ (value >> 31)


def cpu_feature_hash(embedding):
    """FeatureHash.h CPU algorithm, including BF16 bytes and signed int32 IDs."""
    data = embedding.contiguous().view(torch.uint8).numpy().tobytes()
    stride = len(data) // embedding.shape[0]
    values = []
    for start in range(0, len(data), stride):
        row = data[start:start + stride]
        result = _mix(0x243F6A8885A308D3 ^ stride)
        for index in range(0, stride, 8):
            result ^= _mix(int.from_bytes(row[index:index + 8], 'little') ^ _mix(0x9E3779B97F4A7C15 + index // 8))
        result = _mix(result ^ 0x9E3779B97F4A7C15)
        token = (result ^ (result >> 32)) & 0xFFFFFFFF
        values.append(token if token < 1 << 31 else token - (1 << 32))
    return torch.tensor(values, dtype=torch.int32)


class MockLegacyMMEngine:
    def __init__(self, hidden_size=128, max_seq_len=16384, delay_ms=1,
                 cache_bytes=64 << 20, prepared_v41=False, concurrency=4,
                 model_path=None, token_cache_items=10000):
        if hidden_size <= 0 or max_seq_len <= 0 or delay_ms < 0 or cache_bytes < 0 or concurrency <= 0:
            raise ValueError('invalid Mock ViT limits')
        processor = V41ImageProcessorConfig(max_seq_len=max_seq_len)
        if model_path:
            from rtp_llm.config.dsv41_config import V41Config
            config = V41Config.from_path(model_path)
            processor = V41ImageProcessorConfig.from_model_config(config)
            hidden_size = int(config.text['hidden_size'])
            max_seq_len = processor.max_seq_len
            prepared_v41 = True
        self.model = SimpleNamespace(
            model_config=SimpleNamespace(hidden_size=hidden_size, compute_dtype=torch.bfloat16,
                max_seq_len=max_seq_len, model_type='deepseek_v41' if prepared_v41 else 'qwen2_vl'),
            mm_part=SimpleNamespace(processor_config=processor))
        self.vit_config = SimpleNamespace(mm_transport_mode='grpc', vit_max_concurrent_requests=concurrency,
            vit_token_cache_item_num=token_cache_items, vit_token_cache_time_window_ms=1800000)
        self._scheduler = self  # Retain release's configured concurrent admission path.
        self.delay_ms, self.cache_limit = delay_ms, cache_bytes
        self._cache, self._bytes = OrderedDict(), 0
        self._lock = threading.Lock()
        self.forward_calls = 0

    _check_request = staticmethod(MMProcessEngine._check_request)

    def _forward(self, payload, rows, deadline=None, cancelled=None):
        self._check_request(deadline, cancelled)
        if rows <= 0 or rows >= self.model.model_config.max_seq_len:
            raise ValueError('Mock image token count exceeds sequence length')
        key = hashlib.sha256(payload + str(rows).encode()).digest()
        with self._lock:
            found = self._cache.pop(key, None)
            if found is not None:
                self._cache[key] = found
                return found
        end = time.monotonic() + self.delay_ms / 1000
        while time.monotonic() < end:
            self._check_request(deadline, cancelled)
            time.sleep(min(.01, max(0, end - time.monotonic())))
        generator = torch.Generator().manual_seed(int.from_bytes(key[:8], 'little') & ((1 << 63) - 1))
        tensor = torch.rand((rows, self.model.model_config.hidden_size), generator=generator).to(torch.bfloat16)
        charge = tensor.numel() * tensor.element_size()
        self._check_request(deadline, cancelled)
        with self._lock:
            self.forward_calls += 1
            if charge <= self.cache_limit:
                previous = self._cache.pop(key, None)
                if previous is not None:
                    self._bytes -= previous.numel() * previous.element_size()
                self._cache[key], self._bytes = tensor, self._bytes + charge
                while self._bytes > self.cache_limit:
                    _, victim = self._cache.popitem(last=False)
                    self._bytes -= victim.numel() * victim.element_size()
        return tensor

    def submit(self, urls, types=None, tensors=None, preprocess_configs=None, deadline=None, cancelled=None):
        results = []
        for index, url in enumerate(urls):
            self._check_request(deadline, cancelled)
            if types is not None and int(types[index].value) != 1:
                raise ValueError('Mock URL backend supports IMAGE only')
            if not url.startswith('data:'):
                raise ValueError('legacy CPU fixture requires a base64 data image')
            raw = base64.b64decode(url.split(',', 1)[1], validate=True)
            if len(raw) > 16 << 20:
                raise ValueError('Mock image exceeds byte limit')
            try:
                decoded = Image.open(io.BytesIO(raw))
            except UnidentifiedImageError as error:
                raise ValueError('invalid Mock image') from error
            with decoded as image:
                if image.width * image.height > 32 << 20:
                    raise ValueError('Mock image exceeds decoded-pixel limit')
                config = preprocess_configs[index] if preprocess_configs else [-1] * 8
                width, height = config[:2]
                width = width if width > 0 else image.width
                height = height if height > 0 else image.height
                min_pixels = config[2] if config[2] > 0 else 4 * 28**2
                max_pixels = config[3] if config[3] > 0 else 16384 * 28**2
                if min_pixels > max_pixels or min(width, height) <= 0 or max(width, height) / min(width, height) > 200:
                    raise ValueError('invalid Mock Qwen image geometry')
                h, w = max(28, round(height / 28) * 28), max(28, round(width / 28) * 28)
                if h * w > max_pixels:
                    beta = math.sqrt(height * width / max_pixels)
                    h, w = math.floor(height / beta / 28) * 28, math.floor(width / beta / 28) * 28
                elif h * w < min_pixels:
                    beta = math.sqrt(min_pixels / (height * width))
                    h, w = math.ceil(height * beta / 28) * 28, math.ceil(width * beta / 28) * 28
                seed = image.convert('RGB').tobytes() + str((h, w, config)).encode()
                results.append(self._forward(seed, h // 28 * (w // 28), deadline, cancelled))
        return MMEmbeddingRes(results, max_batch_size=1, gpu_forwards=len(results))

    def submit_v41(self, images):
        results = [self._forward(image['patches'].contiguous().view(torch.uint8).numpy().tobytes()
                                + image['processor_identity'].encode(), image['types'].numel()) for image in images]
        return MMEmbeddingRes(results, max_batch_size=1, gpu_forwards=len(results))

    def stop(self):
        with self._lock:
            self._cache.clear()
            self._bytes = 0


def make_service(**kwargs):
    from rtp_llm.server import vit_rpc_server
    # This process runs a CPU Mock backend; retain the release server's control flow.
    return vit_rpc_server.MultimodalRpcServer(MockLegacyMMEngine(**kwargs), feature_hash_fn=cpu_feature_hash)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--prepared-v41', action='store_true')
    parser.add_argument('--delay-ms', type=int, default=1)
    parser.add_argument('--model-path')
    parser.add_argument('--health-port', type=int)
    parser.add_argument('--cache-bytes', type=int, default=64 << 20)
    parser.add_argument('--token-cache-items', type=int, default=10000)
    parser.add_argument('--shutdown-timeout', type=float, default=30)
    args = parser.parse_args()
    from rtp_llm.server.vit_rpc_server import _create_rpc_server, _serve_rpc_server
    service = make_service(hidden_size=args.hidden_size, prepared_v41=args.prepared_v41,
                           concurrency=args.concurrency, delay_ms=args.delay_ms,
                           model_path=args.model_path, cache_bytes=args.cache_bytes,
                           token_cache_items=args.token_cache_items)
    server, executor = _create_rpc_server(service, args.concurrency)
    bound = server.add_insecure_port(f'[::]:{args.port}')
    if not bound:
        raise RuntimeError('failed to bind Mock ViT port')
    print(f'Legacy Mock ViT ready port={bound} protocol=dsv41_ced_release', flush=True)
    _serve_rpc_server(server, service.engine, executor, args.shutdown_timeout, args.health_port)


if __name__ == '__main__':
    main()
