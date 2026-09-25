"""K3 BF16 absorbed MLA using vLLM's fused epilogue and FlashInfer backends.

Metadata and workspace belong to the RTP attention instance. Keeping cache
insertion separate lets that instance publish CacheStore writes before reading
attention, without an extra query-concatenation or cache-scatter kernel.
"""

import inspect
import os
from pathlib import Path

import torch


def _load_fused_epilogue():
    name = "fused_kimi_k3_mla_decode_q_concat_kv_cache_insert"
    if not hasattr(torch.ops._C, name):
        path = os.environ.get("KIMI_K3_VLLM_STABLE_LIBRARY")
        if not path:
            raise RuntimeError(
                "K3 native MLA requires the vLLM stable-libtorch epilogue; "
                "set KIMI_K3_VLLM_STABLE_LIBRARY to its ABI-compatible library"
            )
        torch.ops.load_library(str(Path(path).resolve(strict=True)))
    if not hasattr(torch.ops._C, name):
        raise RuntimeError("The selected vLLM library does not provide K3 MLA")
    return getattr(torch.ops._C, name)


class NativeMlaDecode:
    """NoPE BF16 MLA for ordinary decode and compact multi-query verification."""

    def __init__(
        self, *, num_heads, kv_lora_rank, nope_dim, pe_dim, page_size,
        softmax_extra_scale, workspace, max_batch=1,
    ):
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        required = {"cum_seq_lens_q", "max_q_len", "backend", "out"}
        if not required.issubset(
            inspect.signature(trtllm_batch_decode_with_kv_cache_mla).parameters
        ):
            raise RuntimeError("K3 requires FlashInfer with compact CuTe DSL MLA")
        if (kv_lora_rank, nope_dim, pe_dim) != (512, 128, 64) or page_size not in (64, 128):
            raise ValueError("K3 native BF16 MLA requires latent512/NoPE128/PE64 and page64 or page128")
        if workspace.dtype != torch.uint8 or not workspace.is_cuda:
            raise ValueError("K3 MLA workspace must be a CUDA uint8 tensor")
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.nope_dim = nope_dim
        self.pe_dim = pe_dim
        self.page_size = page_size
        self.scale = (nope_dim + pe_dim) ** -0.5 * softmax_extra_scale
        self.workspace = workspace
        self.kernel = trtllm_batch_decode_with_kv_cache_mla
        self.backend = "trtllm-gen" if page_size == 64 else "cute-dsl"
        self.max_batch = max_batch
        self.counter = None
        if self.backend == "trtllm-gen":
            from flashinfer.utils import (
                get_device_sm_count, get_trtllm_gen_multi_ctas_kv_counter_bytes,
            )
            counter_bytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(
                max_batch, num_heads, get_device_sm_count(workspace.device),
            )
            # The native kernel resets its semaphores after every launch.
            # Allocate once before capture; no per-step zero kernel is needed.
            self.counter = torch.zeros(counter_bytes, dtype=torch.uint8, device=workspace.device)
        self.fused_epilogue = _load_fused_epilogue()

    def write_cache(self, q, kv, k_pe, cache, slot_mapping, k_weight):
        """Return absorbed Q while inserting this step's K/V in-place."""
        if any(t.dtype != torch.bfloat16 for t in (q, kv, k_pe, cache, k_weight)):
            raise ValueError("BF16 K3 MLA cannot reuse resources of another precision")
        if q.shape[1:] != (self.num_heads, self.nope_dim + self.pe_dim):
            raise ValueError("Unexpected K3 MLA query layout")
        if slot_mapping.numel() != q.shape[0] or slot_mapping.dtype != torch.int64:
            raise ValueError("K3 MLA requires one int64 cache slot per physical query")
        latent_q = torch.bmm(q[..., :self.nope_dim].transpose(0, 1), k_weight)
        absorbed_q = torch.empty(
            (q.shape[0], self.num_heads, self.kv_lora_rank + self.pe_dim),
            dtype=q.dtype, device=q.device,
        )
        self.fused_epilogue(
            latent_q.transpose(0, 1), q[..., self.nope_dim:], kv, k_pe,
            absorbed_q, cache.view(-1, self.page_size, self.kv_lora_rank + self.pe_dim),
            slot_mapping, self.page_size, None, None,
        )
        return absorbed_q

    def attend(
        self, query, cache, v_weight, *, block_tables, seq_lens,
        cu_query_lens, max_query_len, max_seq_len,
    ):
        """Consume graph-stable RTP metadata; never derive bounds on the GPU."""
        if v_weight.dtype != torch.bfloat16:
            raise ValueError("BF16 K3 MLA requires BF16 absorption matrices")
        if self.counter is not None and seq_lens.numel() > self.max_batch:
            raise ValueError("K3 MLA batch exceeds the allocated semaphore capacity")
        backend_args = {} if self.counter is None else {"multi_ctas_kv_counter_buffer": self.counter}
        output = self.kernel(
            query=query,
            kv_cache=cache.view(-1, 1, self.page_size, self.kv_lora_rank + self.pe_dim),
            workspace_buffer=self.workspace,
            qk_nope_head_dim=self.nope_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.pe_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            backend=self.backend,
            **backend_args,
            return_lse=False,
            cum_seq_lens_q=cu_query_lens,
            max_q_len=max_query_len,
        )
        return torch.bmm(output.transpose(0, 1), v_weight).transpose(0, 1)
