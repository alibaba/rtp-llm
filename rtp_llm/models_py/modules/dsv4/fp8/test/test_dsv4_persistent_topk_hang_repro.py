"""Reproduce the DSV4 persistent-topk cooperative-barrier hang.

This test intentionally runs the large-row radix path with a launch shape that
matches the production hang signature: TopK=512, vec_size=4, and about
31 CTAs per row group.  The buggy kernel can hang inside the inter-CTA
arrival_counter barrier.  To keep Bazel responsive, the actual CUDA work runs
in a child process; if the child stops making progress, the parent kills it and
reports the hang.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


TOP_K = 512
TARGET_CTAS_PER_GROUP = 31
NUM_ROWS = 4
WORKSPACE_BYTES = 1024 * 1024
DEFAULT_CHILD_TIMEOUT_SEC = 45.0
DEFAULT_ITERS = 256


def _fixed_smem_large() -> int:
    radix = 256
    return ((radix + radix + 5) * 4 + 15) & ~15


def _derive_stride_for_ctas_per_group(max_smem_per_block: int) -> tuple[int, int]:
    """Mirror the host launch math enough to force ctas_per_group ~= 31."""
    vec_size = 4
    max_chunk_elements = (max_smem_per_block - _fixed_smem_large()) // 4
    max_chunk_elements = (max_chunk_elements // vec_size) * vec_size
    min_chunk = vec_size * 1024
    if max_chunk_elements < min_chunk:
        max_chunk_elements = min_chunk

    stride = max_chunk_elements * (TARGET_CTAS_PER_GROUP - 1) + min_chunk
    stride = ((stride + vec_size - 1) // vec_size) * vec_size
    ctas_per_group = (stride + max_chunk_elements - 1) // max_chunk_elements
    return stride, ctas_per_group


def _child_main(iters: int) -> int:
    import torch

    from rtp_llm.ops.compute_ops import rtp_llm_ops

    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available", flush=True)
        return 0
    if not hasattr(rtp_llm_ops, "dsv4_persistent_topk"):
        print("SKIP: rtp_llm_ops.dsv4_persistent_topk is not built", flush=True)
        return 0

    props = torch.cuda.get_device_properties(0)
    max_smem = int(props.shared_memory_per_block_optin)
    stride, ctas_per_group = _derive_stride_for_ctas_per_group(max_smem)
    num_sms = int(props.multi_processor_count)
    expected_num_groups = max(1, min(NUM_ROWS, (num_sms - 1) // ctas_per_group))
    expected_grid_x = expected_num_groups * ctas_per_group

    print(
        "hang-repro launch: "
        f"device={props.name!r}, sms={num_sms}, max_smem={max_smem}, "
        f"N={NUM_ROWS}, T={stride}, K={TOP_K}, "
        f"ctas_per_group={ctas_per_group}, expected_grid_x={expected_grid_x}, "
        f"iters={iters}",
        flush=True,
    )

    if ctas_per_group != TARGET_CTAS_PER_GROUP:
        print(
            f"SKIP: expected ctas_per_group={TARGET_CTAS_PER_GROUP}, "
            f"got {ctas_per_group}",
            flush=True,
        )
        return 0

    generator = torch.Generator(device="cuda").manual_seed(20260519)
    logits = torch.randn(NUM_ROWS, stride, device="cuda", generator=generator)
    lengths = torch.full((NUM_ROWS,), stride, dtype=torch.int32, device="cuda")
    output = torch.full((NUM_ROWS, TOP_K), -1, dtype=torch.int32, device="cuda")
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()

    start = time.monotonic()
    for i in range(iters):
        if i % 16 == 0:
            print(f"child iteration {i}", flush=True)
        rtp_llm_ops.dsv4_persistent_topk(
            logits,
            lengths,
            output,
            workspace,
            TOP_K,
            stride,
        )
        torch.cuda.synchronize()

    elapsed = time.monotonic() - start
    print(f"child completed {iters} iterations in {elapsed:.3f}s", flush=True)
    return 0


def _parent_main(iters: int, timeout_sec: float) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, __file__, "--child", "--iters", str(iters)]
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: list[str] = []
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout_sec
    while True:
        line = proc.stdout.readline()
        if line:
            print(line, end="", flush=True)
            lines.append(line.rstrip())

        rc = proc.poll()
        if rc is not None:
            tail = "\n".join(lines[-20:])
            if rc != 0:
                raise AssertionError(f"child exited with {rc}; output tail:\n{tail}")
            return 0

        if time.monotonic() > deadline:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            tail = "\n".join(lines[-40:])
            raise AssertionError(
                "dsv4_persistent_topk hang reproduced: child did not finish "
                f"within {timeout_sec:.1f}s. Output tail:\n{tail}"
            )

        if not line:
            time.sleep(0.1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument(
        "--iters",
        type=int,
        default=int(os.environ.get("DSV4_TOPK_HANG_REPRO_ITERS", DEFAULT_ITERS)),
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=float(
            os.environ.get(
                "DSV4_TOPK_HANG_REPRO_TIMEOUT_SEC", DEFAULT_CHILD_TIMEOUT_SEC
            )
        ),
    )
    args = parser.parse_args()

    if args.child:
        return _child_main(args.iters)
    return _parent_main(args.iters, args.timeout_sec)


if __name__ == "__main__":
    raise SystemExit(main())
