#!/usr/bin/env python3
"""Read one verified K3 shard with disjoint O_DIRECT requests; never mutate it."""

import concurrent.futures
import json
import mmap
import os
import time


PATH = "/mnt/hf3fs/3fs/models/kimi/kimi-k3-4layers/model-00002-of-000007.safetensors"
BLOCK = 4 * 1024 * 1024
COUNTS = (1, 8, 32, 64, 32, 8, 1)


def read_partition(fd, worker, workers, blocks):
    buf = mmap.mmap(-1, BLOCK)
    view = memoryview(buf)
    nbytes = 0
    check = 0
    try:
        for block in range(worker, blocks, workers):
            n = os.preadv(fd, [view], block * BLOCK)
            if n != BLOCK:
                raise OSError(f"short read at block {block}: {n}")
            nbytes += n
            check = (check + view[0] + view[BLOCK - 1]) & 0xFFFFFFFF
    finally:
        view.release()
        buf.close()
    return nbytes, check


def main():
    size = os.stat(PATH).st_size
    blocks = size // BLOCK
    fd = os.open(PATH, os.O_RDONLY | os.O_DIRECT)
    print(json.dumps({"path": PATH, "file_bytes": size, "block_bytes": BLOCK,
                      "scanned_bytes": blocks * BLOCK, "counts": COUNTS}), flush=True)
    try:
        for workers in COUNTS:
            start = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                parts = list(pool.map(lambda i: read_partition(fd, i, workers, blocks), range(workers)))
            elapsed = time.monotonic() - start
            nbytes = sum(x[0] for x in parts)
            print(json.dumps({"workers": workers, "seconds": elapsed, "bytes": nbytes,
                              "MiB_per_s": nbytes / (1024 ** 2) / elapsed,
                              "checksum": sum(x[1] for x in parts) & 0xFFFFFFFF}), flush=True)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
