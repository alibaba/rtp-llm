"""Kernel-level benchmark: FlashInfer trtllm-gen MLA decode vs baseline.

Compares the attention kernel latency of two decode paths on the K3 MLA
geometry (96 heads, kv_lora_rank=512, rope_head_dim=64, page_size=64):

  baseline : flashinfer.BatchMLAPagedAttentionWrapper (backend="auto"),
             the kernel used by the current MlaFlashInferDecodeOp path
  trtllm   : flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla,
             the kernel used by TrtllmGenMlaDecodeOp

Only the attention kernel is timed; the query-absorption and output
projection BMMs are identical in both integration paths and excluded.

Usage:
    python trtllm_gen_mla_decode_benchmark.py [--batches 1,8,32,64,128]
        [--seq-lens 1024,4096,8192,16384,32768] [--iters 50]
"""

import argparse
import math

import torch
from flashinfer import BatchMLAPagedAttentionWrapper
from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

NUM_HEADS = 96
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
PAGE_SIZE = 64
SCALE = (QK_ROPE_HEAD_DIM + 128) ** -0.5  # K3 qk_head_dim = 192
BLOCK_ALIGN_TOKENS = 128

DEFAULT_BATCHES = [1, 8, 32, 64, 128]
DEFAULT_SEQ_LENS = [1024, 4096, 8192, 16384, 32768]


class BaselineRunner:
    """BatchMLAPagedAttentionWrapper with backend="auto"."""

    def __init__(self, max_bs: int, max_pages: int, device: torch.device):
        self.workspace = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device=device)
        self.qo_indptr_h = torch.arange(max_bs + 1, dtype=torch.int32, device="cpu")
        self.kv_indptr_h = torch.zeros(max_bs + 1, dtype=torch.int32, device="cpu")
        self.kv_len_arr_h = torch.zeros(max_bs, dtype=torch.int32, device="cpu")
        self.kv_indices_d = torch.zeros(max_pages, dtype=torch.int32, device=device)
        self.wrapper = BatchMLAPagedAttentionWrapper(
            self.workspace,
            backend="auto",
            use_cuda_graph=False,
            qo_indptr=self.qo_indptr_h,
            kv_indptr=self.kv_indptr_h,
            kv_indices=self.kv_indices_d,
            kv_len_arr=self.kv_len_arr_h,
        )

    def setup(
        self,
        batch_size: int,
        seq_len: int,
        kv_cache: torch.Tensor,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
    ):
        pages_per_req = kv_cache.size(0) // batch_size
        self.kv_indptr_h[: batch_size + 1] = (
            torch.arange(batch_size + 1, dtype=torch.int32) * pages_per_req
        )
        self.kv_len_arr_h[:batch_size] = seq_len
        self.kv_indices_d[: batch_size * pages_per_req] = torch.arange(
            batch_size * pages_per_req, dtype=torch.int32, device=kv_cache.device
        )
        self.wrapper.plan(
            self.qo_indptr_h[: batch_size + 1],
            self.kv_indptr_h[: batch_size + 1],
            self.kv_indices_d[: batch_size * pages_per_req],
            self.kv_len_arr_h[:batch_size],
            NUM_HEADS,
            KV_LORA_RANK,
            QK_ROPE_HEAD_DIM,
            PAGE_SIZE,
            True,  # causal
            SCALE,
            torch.bfloat16,
            torch.bfloat16,
        )
        self.compressed_kv, self.k_pe = torch.split(
            kv_cache, [KV_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1
        )
        self.q_nope = q_nope
        self.q_pe = q_pe
        self.attn_output = torch.empty_like(q_nope)

    def run(self):
        self.wrapper.run(
            self.q_nope, self.q_pe, self.compressed_kv, self.k_pe, self.attn_output
        )


class TrtllmRunner:
    """flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla."""

    def __init__(self, max_bs: int, max_blocks_per_req: int, device: torch.device):
        self.workspace = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.block_tables = torch.zeros(
            (max_bs, max_blocks_per_req), dtype=torch.int32, device=device
        )
        self.seq_lens = torch.zeros(max_bs, dtype=torch.int32, device=device)

    def setup(
        self, batch_size: int, seq_len: int, kv_cache: torch.Tensor, query: torch.Tensor
    ):
        pages_per_req = kv_cache.size(0) // batch_size
        blocks_per_align = max(1, BLOCK_ALIGN_TOKENS // PAGE_SIZE)
        padded = (
            (pages_per_req + blocks_per_align - 1) // blocks_per_align
        ) * blocks_per_align
        table = torch.zeros((batch_size, padded), dtype=torch.int32)
        for b in range(batch_size):
            table[b, :pages_per_req] = torch.arange(
                b * pages_per_req, (b + 1) * pages_per_req, dtype=torch.int32
            )
        self.block_tables[:batch_size, :padded].copy_(table)
        self.seq_lens[:batch_size] = seq_len
        self.batch_size = batch_size
        self.padded = padded
        self.max_seq_len = seq_len
        self.kv_cache = kv_cache
        self.query = query

    def run(self):
        trtllm_batch_decode_with_kv_cache_mla(
            query=self.query,
            kv_cache=self.kv_cache,
            workspace_buffer=self.workspace,
            qk_nope_head_dim=KV_LORA_RANK,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            block_tables=self.block_tables[: self.batch_size, : self.padded],
            seq_lens=self.seq_lens[: self.batch_size],
            max_seq_len=self.max_seq_len,
            bmm1_scale=SCALE,
            bmm2_scale=1.0,
        )


def time_kernel(run, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # us


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batches",
        default=",".join(map(str, DEFAULT_BATCHES)),
        help="comma-separated batch sizes",
    )
    parser.add_argument(
        "--seq-lens",
        default=",".join(map(str, DEFAULT_SEQ_LENS)),
        help="comma-separated kv sequence lengths",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    batches = [int(x) for x in args.batches.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    torch.manual_seed(args.seed)
    device = torch.device("cuda", torch.cuda.current_device())
    cap = torch.cuda.get_device_capability(device)
    print(
        f"GPU: {torch.cuda.get_device_name(device)} (sm{cap[0]}{cap[1]}), "
        f"heads={NUM_HEADS}, kv_lora_rank={KV_LORA_RANK}, page={PAGE_SIZE}, "
        f"bf16, warmup={args.warmup}, iters={args.iters}"
    )

    max_bs = max(batches)
    max_pages_per_req = max((s + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lens)
    blocks_per_align = max(1, BLOCK_ALIGN_TOKENS // PAGE_SIZE)
    max_padded = (
        (max_pages_per_req + blocks_per_align - 1) // blocks_per_align
    ) * blocks_per_align

    baseline = BaselineRunner(max_bs, max_bs * max_pages_per_req, device)
    trtllm = TrtllmRunner(max_bs, max_padded, device)

    print()
    header = f"{'bs':>4} {'seq_len':>8} | {'baseline_us':>11} {'trtllm_us':>10} {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for seq_len in seq_lens:
        pages_per_req = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        for bs in batches:
            kv_cache = torch.randn(
                (bs * pages_per_req, PAGE_SIZE, HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
            )
            q_nope = torch.randn(
                (bs, NUM_HEADS, KV_LORA_RANK), dtype=torch.bfloat16, device=device
            )
            q_pe = torch.randn(
                (bs, NUM_HEADS, QK_ROPE_HEAD_DIM), dtype=torch.bfloat16, device=device
            )
            query = torch.randn(
                (bs, 1, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
            )

            try:
                baseline.setup(bs, seq_len, kv_cache, q_nope, q_pe)
                us_base = time_kernel(baseline.run, args.warmup, args.iters)
                base_str = f"{us_base:11.2f}"
            except RuntimeError as e:
                base_str = f"{'FAIL':>11}"
                print(f"  baseline failed bs={bs} seq={seq_len}: {e}")

            try:
                trtllm.setup(bs, seq_len, kv_cache, query)
                us_trtllm = time_kernel(trtllm.run, args.warmup, args.iters)
                trtllm_str = f"{us_trtllm:10.2f}"
                speedup_str = (
                    f"{us_base / us_trtllm:7.2f}x"
                    if "FAIL" not in base_str
                    else "    n/a"
                )
            except RuntimeError as e:
                trtllm_str = f"{'UNSUPPORTED':>10}"
                speedup_str = "    n/a"
                print(f"  trtllm failed bs={bs} seq={seq_len}: {e}")

            print(f"{bs:>4} {seq_len:>8} | {base_str} {trtllm_str} {speedup_str}")
            del kv_cache, q_nope, q_pe, query


if __name__ == "__main__":
    main()
