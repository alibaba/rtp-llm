"""Strict native group-128 FP8 MQA adapter for the K3.1 Indexer policy.

Only QK scoring is adapted here. PageRR, KV budgets, TopK and DCP remain in
the shared execution path. There is no compatibility/BF16 scoring fallback.
"""

import importlib

import torch

from rtp_llm.models_py.kernels.cuda.indexer_topk_keys import (
    sanitize_indexer_scores, write_indexer_score_wire,
)
from rtp_llm.models_py.utils.module_profiler import profile_scope


class NativeFp8IndexerScore:
    def __init__(self, backend=None):
        required = ("fp8_mqa_logits", "fp8_paged_mqa_logits", "get_paged_mqa_logits_metadata")
        if backend is None:
            current = importlib.import_module("deep_gemm")
            # Preserve the shared branch's newer GEMM/RS dependency. Its SM10
            # MX MQA does not implement the group-128 FP32-weight contract.
            if all(callable(getattr(current, name, None)) for name in required):
                backend = current
            else:
                try:
                    backend = importlib.import_module("deep_gemm_mqa")
                except ModuleNotFoundError as error:
                    if error.name != "deep_gemm_mqa":
                        raise
                    raise RuntimeError(
                        "K3.1 requires the native group-128 FP8 MQA dependency; "
                        "use the CUDA13 x86 build with deep_gemm_mqa"
                    ) from error
        if getattr(backend, "__name__", "") == "deep_gemm_mqa" and not getattr(backend, "__native_globals_isolated__", False):
            raise RuntimeError("K3.1 MQA dependency must isolate native JIT globals from main DeepGEMM")
        self.backend = backend
        missing = [name for name in required if not callable(getattr(self.backend, name, None))]
        if missing:
            raise RuntimeError("K3.1 requires native FP8 MQA entry points: " + ", ".join(missing))
        self._dense = self.backend.fp8_mqa_logits
        self._paged = self.backend.fp8_paged_mqa_logits
        self._metadata = self.backend.get_paged_mqa_logits_metadata

    @staticmethod
    def _tensor(value, dtype, device, name):
        if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                or value.device != device or not value.is_contiguous()):
            raise ValueError(f"{name} must be contiguous {dtype} on {device}")

    def supports_direct_pages(self, page_size, device):
        return (page_size in (64, 128)
                and getattr(self.backend, "__name__", "") == "deep_gemm_mqa"
                and torch.cuda.get_device_capability(device)[0] == 10)

    def prepare_paged_metadata(self, context_lens, *, num_sms, page_size=64):
        """Prepare the pinned native API's per-request, per-query lengths."""
        self._tensor(context_lens, torch.int32, context_lens.device, "Paged MQA context lengths")
        if context_lens.ndim != 2 or not context_lens.is_cuda or type(num_sms) is not int or num_sms <= 0:
            raise ValueError("Prepare CUDA per-query lengths and a positive SM count")
        if page_size != 64 and not self.supports_direct_pages(page_size, context_lens.device):
            raise ValueError("Native paged MQA does not support this physical page size")
        return self._metadata(context_lens, page_size, num_sms)

    def _query(self, query, weights):
        if query.ndim not in (3, 4) or query.shape[-2:] != (32, 128) or not query.is_cuda:
            raise ValueError("K3.1 FP8 Q must retain all32 heads of128 dimensions")
        self._tensor(query, torch.float8_e4m3fn, query.device, "Indexer Q")
        self._tensor(weights, torch.float32, query.device, "FP32 folded head weights")
        rows = query.numel() // (32 * 128)
        if weights.shape != (rows, 32):
            raise ValueError("Indexer head weights must include one Q scale per token/head")
        return rows

    def _finish(self, score, positions, starts, ends, shape, out, sanitize=True, *, wire=None):
        if score.dtype != torch.float32 or score.shape != shape or score.stride(-1) != 1:
            raise RuntimeError("Native FP8 MQA returned an incompatible score layout")
        if (positions.dtype != torch.int32 or positions.shape != (shape[1],)
                or positions.device != score.device or not positions.is_contiguous()):
            raise ValueError("Indexer columns require INT32 global positions")
        if wire is not None:
            self._tensor(wire, torch.float32, score.device, "Indexer DCP wire")
            self._tensor(ends, torch.int32, score.device, "Paged MQA score ends")
            if wire.ndim != 2 or wire.shape[0] < shape[0] or wire.shape[1] != shape[1]:
                raise ValueError("Indexer DCP wire shape mismatch")
            write_indexer_score_wire(score, positions, ends, wire)
            return wire
        if out is not None:
            self._tensor(out, torch.float32, score.device, "Indexer score workspace")
            if out.shape != shape:
                raise ValueError("Indexer score workspace shape mismatch")
            out.copy_(score)
            score = out
        if sanitize:
            sanitize_indexer_scores(score, positions, starts=starts, ends=ends)
        return score

    def dense(self, query, key, key_scales, weights, starts, ends, positions, *, out=None,
              sanitize=True):
        rows = self._query(query, weights)
        if query.ndim != 3 or key.ndim != 2 or key.shape[1] != 128:
            raise ValueError("Native non-paged MQA requires Q[M,32,128] and K[N,128]")
        self._tensor(key, torch.float8_e4m3fn, query.device, "Indexer K")
        self._tensor(key_scales, torch.float32, query.device, "Indexer K scales")
        if key_scales.shape != (key.shape[0],):
            raise ValueError("Indexer K requires one FP32 scale per128 dimensions")
        for name, value in (("ks", starts), ("ke", ends)):
            self._tensor(value, torch.int32, query.device, name)
            if value.shape != (rows,):
                raise ValueError("Indexer ks/ke require one bound per query")
        # A positive max_seqlen_k selects compressed rows addressed relative
        # to ks. The shared KV/TopK loop consumes absolute workspace columns.
        with profile_scope("RTP::attention.dsa.indexer.fp8.qk"):
            score = self._dense(query, (key, key_scales), weights, starts, ends,
                                clean_logits=False, max_seqlen_k=0)
        return self._finish(score, positions, starts, ends, (rows, key.shape[0]), out,
                            sanitize=sanitize)

    def paged(self, query, cache, weights, context_lens, block_table, metadata,
              max_context_len, positions, *, out=None, wire=None):
        rows = self._query(query, weights)
        if out is not None and wire is not None:
            raise ValueError("Paged MQA accepts one score destination")
        if query.ndim != 4:
            raise ValueError("Paged MQA requires Q[B,next_n,32,128]")
        self._tensor(cache, torch.uint8, query.device, "FP8 paged cache")
        if (cache.ndim != 4 or cache.shape[2:] != (1, 132)
                or (cache.size(1) != 64 and not self.supports_direct_pages(cache.size(1), cache.device))):
            raise ValueError("Native paged MQA needs a supported block-planar page size")
        self._tensor(context_lens, torch.int32, query.device, "Paged MQA context lengths")
        self._tensor(block_table, torch.int32, query.device, "Paged MQA block table")
        if (context_lens.shape != query.shape[:2] or block_table.ndim != 2
                or block_table.shape[0] != query.shape[0]):
            raise ValueError("Prepare per-query lengths and one owner block table per request")
        with profile_scope("RTP::attention.dsa.indexer.fp8.paged_qk"):
            score = self._paged(query, cache, weights, context_lens, block_table,
                                metadata, max_context_len, clean_logits=False)
        return self._finish(score, positions, None, context_lens.view(rows),
                            (rows, max_context_len), out, wire=wire)
