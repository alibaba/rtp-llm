"""Instance-owned Prefill TopK with an immutable selection policy."""

import torch


class PpuPrefillTopK:
    def __init__(self, options):
        self.backend = options.get("DSV4_INDEXER_TOPK_BACKEND", "auto").strip().lower()
        if self.backend not in ("auto", "native", "torch", "sglang"):
            raise ValueError(f"Unsupported PPU Prefill TopK backend {self.backend!r}")
        self.canonicalize = options.get(
            "DSV4_INDEXER_TOPK_CANONICALIZE", ""
        ).strip().lower() in ("1", "true", "yes", "on")
        self.fast = options.get("DSV4_PREFILL_FAST_TOPK", "1") != "0"
        self.force_radix = options.get("DSV4_PREFILL_TOPK_FORCE_RADIX", "1") != "0"
        if self.backend == "sglang":
            from rtp_llm.platforms.ppu.kernels.cuda.ppu_sglang_topk import topk_prefill

            self._select = topk_prefill
        elif self.backend == "torch":
            from rtp_llm.models_py.modules.dsv4.fp8.indexer import (
                _run_prefill_topk_torch,
            )

            self._select = _run_prefill_topk_torch
        else:
            from rtp_llm.ops import rtp_llm_ops

            self._ops = rtp_llm_ops
            self.fast = self.fast and hasattr(rtp_llm_ops, "fast_topk_v2_variable")

    def __call__(self, logits, starts, ends, out, topk, compress_ratio):
        if self.backend in ("sglang", "torch"):
            self._select(logits, starts, ends, out, topk)
        elif (
            self.fast
            and int(topk) in (512, 1024, 2048)
            and logits.size(1) * compress_ratio <= 12 * 1024
        ):
            self._ops.fast_topk_v2_variable(
                logits,
                out,
                (ends - starts).contiguous(),
                starts.contiguous(),
                int(topk),
            )
        else:
            self._ops.dsv4_top_k_per_row_prefill(
                logits,
                starts,
                ends,
                out,
                logits.size(0),
                logits.stride(0),
                logits.stride(1),
                int(topk),
                self.force_radix,
            )
        if self.canonicalize:
            sentinel = torch.iinfo(out.dtype).max
            sortable = torch.where(out >= 0, out, sentinel)
            sorted_idx = torch.sort(sortable, dim=-1).values
            out.copy_(torch.where(sorted_idx == sentinel, -1, sorted_idx))
