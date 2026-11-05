"""Qwen4 SM120 prefill admission and engine cache adaptation."""

import torch
import triton

from rtp_llm.models_py.kernels.cuda.gdn_sm120.prefill import (
    sm120_dsl_available,
    sm120_prefill_prepared,
)
from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNetPrefill
from rtp_llm.models_py.triton_kernels.fla.block import (
    load_initial_state_from_block_map,
    store_page_checkpoints_kernel,
)


class Qwen4Sm120GdnPrefill(Qwen3NextGatedDeltaNetPrefill):
    def _fla(
        self,
        mixed_qkv,
        b,
        a,
        kv_cache_tensor,
        seq_size_per_block,
        attn_inputs,
        attn_meta,
    ):
        host_lengths = attn_inputs.input_lengths
        host_prefixes = attn_inputs.prefix_lengths
        supported = (
            mixed_qkv.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(mixed_qkv.device) == (12, 0)
            and mixed_qkv.dtype == torch.bfloat16
            and 2048 <= mixed_qkv.shape[0] <= 32768
            and self.head_k_dim == self.head_v_dim == 128
            and seq_size_per_block == 128
            and not attn_inputs.is_cuda_graph
            and host_lengths.device.type == "cpu"
            and host_prefixes.device.type == "cpu"
            and sm120_dsl_available()
        )
        if not supported:
            return super()._fla(
                mixed_qkv,
                b,
                a,
                kv_cache_tensor,
                seq_size_per_block,
                attn_inputs,
                attn_meta,
            )
        lengths = tuple(int(n) for n in host_lengths.tolist())
        prefixes = tuple(int(n) for n in host_prefixes.tolist())
        if any(n < 0 or n % 128 for n in prefixes) or any(n <= 0 for n in lengths):
            return super()._fla(
                mixed_qkv,
                b,
                a,
                kv_cache_tensor,
                seq_size_per_block,
                attn_inputs,
                attn_meta,
            )
        from rtp_llm.models_py.triton_kernels.fla.gdn_prefill_prepare import (
            is_supported as preparation_supported,
        )

        if not preparation_supported(
            mixed_qkv,
            a,
            b,
            self.alog,
            self.dt_bias,
            self.local_num_k_heads,
            self.local_num_v_heads,
        ):
            return super()._fla(
                mixed_qkv,
                b,
                a,
                kv_cache_tensor,
                seq_size_per_block,
                attn_inputs,
                attn_meta,
            )
        states = (
            self._get_ssm_states(kv_cache_tensor)
            if kv_cache_tensor is not None
            else None
        )
        initial = None
        if states is not None:
            initial = torch.empty(
                (len(lengths), self.local_num_v_heads, 128, 128),
                device=mixed_qkv.device,
                dtype=self.ssm_state_dtype,
            )
            load_initial_state_from_block_map(
                attn_inputs.prefix_lengths_device,
                attn_inputs.kv_cache_kernel_block_id_device,
                states,
                initial,
                128,
            )
        from rtp_llm.models_py.triton_kernels.fla.gdn_prefill_prepare import (
            prepare_sm120_prefill,
        )

        q, k, v, alpha, beta = prepare_sm120_prefill(
            mixed_qkv,
            a,
            b,
            self.alog,
            self.dt_bias,
            self.local_num_k_heads,
            self.local_num_v_heads,
        )
        output, checkpoints, final, starts = sm120_prefill_prepared(
            q, k, v, alpha, beta, initial, attn_inputs.cu_seqlens_device, lengths
        )
        if states is not None:
            store_page_checkpoints_kernel[
                (
                    len(lengths),
                    triton.cdiv(max(lengths), 128),
                    triton.cdiv(self.local_num_v_heads * 128 * 128, 1024),
                )
            ](
                checkpoints,
                final,
                starts,
                attn_inputs.cu_seqlens_device,
                attn_inputs.prefix_lengths_device,
                attn_inputs.kv_cache_kernel_block_id_device,
                states,
                TABLE_STRIDE=attn_inputs.kv_cache_kernel_block_id_device.stride(0),
                CACHE_STRIDE=states.stride(0),
                HEADS=self.local_num_v_heads,
                PAGE=128,
                BLOCK=1024,
            )
        return output.squeeze(0)
