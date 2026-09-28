#!/usr/bin/env python3
"""Compare the shipped FastSafetensors device APIs on a K3 3FS shard."""

import json
import hashlib
import os
import sys
import time

import fast_safetensors
import torch


path = "/mnt/hf3fs/3fs/models/kimi/kimi-k3-4layers/model-00002-of-000007.safetensors"
use_shm = bool(int(sys.argv[1]))
torch.cuda.set_device(0)
start = time.monotonic()
tensors = fast_safetensors.load_safetensors_to_device(
    path, max_buf_size=2 * 1024**3, device="cuda:0", direct_io=True, use_shm=use_shm
)
torch.cuda.synchronize()
elapsed = time.monotonic() - start
name = sorted(tensors)[0]
t = tensors[name]
all_digest = None
if os.environ.get("HASH_ALL_TENSORS") == "1":
    digest = hashlib.sha256()
    for key in sorted(tensors):
        item = tensors[key].contiguous()
        digest.update(key.encode() + b"\0")
        digest.update(str(item.dtype).encode() + b"\0")
        digest.update(str(tuple(item.shape)).encode() + b"\0")
        digest.update(item.view(torch.uint8).cpu().numpy().tobytes())
    all_digest = digest.hexdigest()
print(json.dumps({"path": path, "use_shm": use_shm,
                  "file_bytes": os.stat(path).st_size, "seconds": elapsed,
                  "count": len(tensors), "first_tensor": name,
                  "first_shape": list(t.shape), "first_dtype": str(t.dtype),
                  "first_1024_sum": float(t.reshape(-1)[:1024].sum()),
                  "first_device": str(t.device), "all_tensor_sha256": all_digest}), flush=True)
