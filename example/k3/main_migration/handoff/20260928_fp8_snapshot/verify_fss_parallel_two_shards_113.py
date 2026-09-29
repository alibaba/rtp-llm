#!/usr/bin/env python3
"""Verify that a single FastSafetensors reader switches same-size K3 shards."""

import gc
import hashlib
import json
import os
import time

import fast_safetensors
import torch


root = "/mnt/hf3fs/3fs/models/kimi/kimi-k3-4layers"
paths = [f"{root}/model-{index:05d}-of-000007.safetensors" for index in (2, 3)]
reader = fast_safetensors.LoadWithShm(2 * 1024**3, "cuda:0", True)
torch.cuda.set_device(0)
for path in paths:
    start = time.monotonic()
    tensors = reader.load_safetensors_to_device(path)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    keys = sorted(tensors)
    samples = []
    for key in (keys[0], keys[len(keys) // 2], keys[-1]):
        value = tensors[key].contiguous().view(torch.uint8).reshape(-1)
        first = value[:1024].cpu().numpy().tobytes()
        last = value[-1024:].cpu().numpy().tobytes()
        samples.append({"key": key, "shape": list(tensors[key].shape),
                        "dtype": str(tensors[key].dtype),
                        "sample_sha256": hashlib.sha256(first + last).hexdigest()})
    print(json.dumps({"path": path, "size": os.stat(path).st_size,
                      "seconds": elapsed, "tensor_count": len(keys),
                      "samples": samples}), flush=True)
    del tensors
    gc.collect()
    torch.cuda.empty_cache()
