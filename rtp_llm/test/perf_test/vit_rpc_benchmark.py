"""Exercise an independent ViT service from multiple client processes."""

import argparse
import base64
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import grpc
import numpy as np

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputsPB,
    TensorPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceStub,
)


def tensor_array(tensor):
    if tensor.data_type == TensorPB.BF16:
        values = np.frombuffer(tensor.bf16_data, dtype=np.uint16).astype(np.uint32)
        values = (values << 16).view(np.float32)
    elif tensor.data_type == TensorPB.FP16:
        values = np.frombuffer(tensor.fp16_data, dtype=np.float16).astype(np.float32)
    elif tensor.data_type == TensorPB.FP32:
        values = np.frombuffer(tensor.fp32_data, dtype=np.float32)
    else:
        raise ValueError(f"unexpected ViT dtype: {tensor.data_type}")
    if len(tensor.shape) != 2 or not np.isfinite(values).all():
        raise ValueError("ViT output must be a finite two-dimensional tensor")
    return values.reshape(tuple(tensor.shape))


def make_request(urls, phase):
    request = MultimodalInputsPB()
    for index, url in enumerate(urls):
        item = request.multimodal_inputs.add(multimodal_url=url, multimodal_type=1)
        item.mm_preprocess_config.image_block_start_mod4 = (phase + index) % 4
    return request


def _client_process(address, urls, references, count, barrier, timeout, atol, rtol):
    channel = grpc.insecure_channel(
        address,
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
        ],
    )
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
        stub = MultimodalRpcServiceStub(channel)
        barrier.wait(timeout=timeout)
        latencies, batch_sizes, max_error = [], [], 0.0
        for index in range(count):
            phase = index % 4
            started = time.perf_counter()
            response, call = stub.RemoteMultimodalEmbedding.with_call(
                make_request(urls, phase), timeout=timeout
            )
            latencies.append(time.perf_counter() - started)
            if len(response.multimodal_outputs) != len(urls):
                raise AssertionError("ViT response image count changed")
            for item, expected in zip(response.multimodal_outputs, references[phase]):
                actual = tensor_array(item.multimodal_embedding)
                np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
                max_error = max(max_error, float(np.max(np.abs(actual - expected))))
            metadata = dict(call.trailing_metadata())
            batch_sizes.append(int(metadata.get("vit-max-batch-images", "0")))
        return latencies, batch_sizes, max_error
    finally:
        channel.close()


def benchmark(
    address,
    urls,
    processes=8,
    requests_per_process=8,
    timeout=120,
    atol=0.05,
    rtol=0.02,
    require_batch=True,
):
    if processes < 1 or requests_per_process < 1 or not urls:
        raise ValueError(
            "positive client/request counts and at least one image required"
        )
    with grpc.insecure_channel(
        address,
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
        ],
    ) as channel:
        grpc.channel_ready_future(channel).result(timeout=timeout)
        stub = MultimodalRpcServiceStub(channel)
        references = []
        for phase in range(4):
            result = stub.RemoteMultimodalEmbedding(
                make_request(urls, phase), timeout=timeout
            )
            if len(result.multimodal_outputs) != len(urls):
                raise AssertionError("ViT reference image count changed")
            references.append(
                [
                    tensor_array(item.multimodal_embedding)
                    for item in result.multimodal_outputs
                ]
            )
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        barrier = manager.Barrier(processes + 1)
        with ProcessPoolExecutor(max_workers=processes, mp_context=context) as pool:
            futures = [
                pool.submit(
                    _client_process,
                    address,
                    urls,
                    references,
                    requests_per_process,
                    barrier,
                    timeout,
                    atol,
                    rtol,
                )
                for _ in range(processes)
            ]
            started = time.perf_counter()
            barrier.wait(timeout=timeout)
            ready = time.perf_counter()
            results = [future.result() for future in futures]
            elapsed = time.perf_counter() - ready
    latencies = np.array([value for result in results for value in result[0]])
    batch_sizes = [value for result in results for value in result[1]]
    if require_batch and max(batch_sizes, default=0) <= 1:
        raise AssertionError("No GPU forward contained multiple images")
    return {
        "processes": processes,
        "requests": processes * requests_per_process,
        "images_per_request": len(urls),
        "elapsed_s": elapsed,
        "client_startup_s": ready - started,
        "requests_per_s": len(latencies) / elapsed,
        "images_per_s": len(latencies) * len(urls) / elapsed,
        "rpc_p50_ms": float(np.percentile(latencies, 50) * 1000),
        "rpc_p95_ms": float(np.percentile(latencies, 95) * 1000),
        "max_gpu_batch_images": max(batch_sizes, default=0),
        "observed_request_batch_sizes": sorted(set(batch_sizes)),
        "max_abs_error_vs_serial": max(result[2] for result in results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--requests-per-process", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--allow-single-image-forward", action="store_true")
    args = parser.parse_args()
    urls = [
        "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
        for path in args.image
    ]
    print(
        json.dumps(
            benchmark(
                args.address,
                urls,
                args.processes,
                args.requests_per_process,
                args.timeout,
                args.atol,
                args.rtol,
                not args.allow_single_image_forward,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
