"""Qwen4 GDN backend selection and paged cache adaptation."""

import os

import torch
import triton

from rtp_llm.models_py.kernels.cuda.gdn_sm120.prefill import (
    sm120_dsl_available,
    sm120_prefill_prepared,
)
from rtp_llm.models_py.model_desc.qwen3_next import (
    Qwen3NextGatedDeltaNetDecode,
    Qwen3NextGatedDeltaNetPrefill,
)
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


class Qwen4FlashInferGdnPrefill(Qwen4Sm120GdnPrefill):
    def _conv_and_fla(
        self, mixed_qkv, b, a, kv_cache_tensor, seq_size_per_block, inputs, meta
    ):
        lengths = inputs.input_lengths
        prefixes = inputs.prefix_lengths
        cu = inputs.cu_seqlens_device
        if not (
            mixed_qkv.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(mixed_qkv.device)[0] == 10
            and mixed_qkv.dtype == torch.bfloat16
            and self.head_k_dim == self.head_v_dim == 128
            and not inputs.is_cuda_graph
            and not meta.is_target_verify
            and not meta.is_cp_linear_attn
            and lengths.device.type == prefixes.device.type == "cpu"
            and lengths.numel() > 0
            and cu is not None
            and cu.numel() == lengths.numel() + 1
            and (
                kv_cache_tensor is None
                or (seq_size_per_block == 128 and self.ssm_state_dtype == torch.float32)
            )
        ):
            return super()._conv_and_fla(
                mixed_qkv, b, a, kv_cache_tensor, seq_size_per_block, inputs, meta
            )

        from flashinfer.gdn_prefill import chunk_gated_delta_rule

        from rtp_llm.models_py.triton_kernels.common.scatter_qkv import scatter_qkv
        from rtp_llm.models_py.triton_kernels.fla.exact_qk_norm import (
            fused_l2norm_qk_exact,
            supports_exact_qk_norm,
        )
        from rtp_llm.models_py.triton_kernels.fla.gdn_gating_prefill import (
            gdn_gating_prefill,
            supports_gdn_gating_prefill,
        )
        from rtp_llm.models_py.triton_kernels.fla.gdn_prefill_prepare import (
            checkpoint_starts,
        )

        if not supports_gdn_gating_prefill(self.alog, a, b, self.dt_bias):
            return super()._conv_and_fla(
                mixed_qkv, b, a, kv_cache_tensor, seq_size_per_block, inputs, meta
            )

        # Qwen35 Omega also owns FlashInfer offsets per forward. The host arrays
        # are scheduler metadata, so no device copy is needed to use them.
        plan = getattr(meta, "_qwen4_flashinfer_prefill_plan", None)
        if plan is None or plan[0] is not cu:
            lengths_host = lengths.tolist()
            prefixes_host = prefixes.tolist()
            num_checkpoints = sum(length // 128 for length in lengths_host)
            valid = (
                all(length > 0 for length in lengths_host)
                and sum(lengths_host) == mixed_qkv.shape[0]
                and (
                    kv_cache_tensor is None
                    or all(
                        prefix >= 0 and prefix % 128 == 0 for prefix in prefixes_host
                    )
                )
            )
            starts = (
                checkpoint_starts(cu, len(lengths_host))
                if valid and kv_cache_tensor is not None and num_checkpoints
                else None
            )
            plan = cu, valid, max(lengths_host), num_checkpoints, starts
            meta._qwen4_flashinfer_prefill_plan = plan
        _, valid, max_length, num_checkpoints, starts = plan
        if not valid:
            return super()._conv_and_fla(
                mixed_qkv, b, a, kv_cache_tensor, seq_size_per_block, inputs, meta
            )

        # Omega's convolution writes separate Q/K/V, preserving the BF16
        # rounding before Q/K L2 normalization.
        fused_heads = (
            (self.local_num_k_heads, self.local_num_v_heads)
            if mixed_qkv.shape[0] >= 2048
            and self.conv_weights.dtype == torch.bfloat16
            and self.local_num_k_heads % 2 == self.local_num_v_heads % 2 == 0
            else None
        )
        conv_output = self._conv1d(
            mixed_qkv,
            kv_cache_tensor,
            seq_size_per_block,
            inputs,
            metadata=meta.get_prefill_conv1d_meta(),
            fused_qkv_heads=fused_heads,
        )
        normalized = fused_heads is not None
        if normalized:
            q, k, v = conv_output
        else:
            if not conv_output.is_contiguous():
                return super()._fla(
                    conv_output, b, a, kv_cache_tensor, seq_size_per_block, inputs, meta
                )
            q, k, v = scatter_qkv(
                conv_output, self.local_num_k_heads, self.local_num_v_heads, 128, 128
            )
            normalized = supports_exact_qk_norm(q, k)
            if normalized:
                q, k = fused_l2norm_qk_exact(q, k)
        gate, beta = gdn_gating_prefill(
            self.alog,
            a,
            b,
            self.dt_bias,
            flashinfer=True,
        )

        states = initial = checkpoints = None
        batch = lengths.numel()
        state_shape = (self.local_num_v_heads, 128, 128)
        if kv_cache_tensor is not None:
            states = self._get_ssm_states(kv_cache_tensor)
            initial = torch.empty(
                (batch, *state_shape), device=q.device, dtype=torch.float32
            )
            load_initial_state_from_block_map(
                inputs.prefix_lengths_device,
                inputs.kv_cache_kernel_block_id_device,
                states,
                initial,
                seq_size_per_block,
            )
            if starts is not None:
                checkpoints = initial.new_empty((num_checkpoints, *state_shape))

        result = chunk_gated_delta_rule(
            q[0],
            k[0],
            v[0],
            g=gate[0],
            beta=beta[0],
            initial_state=initial,
            output_final_state=states is not None,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=not normalized,
            state_checkpoints=checkpoints,
            checkpoint_cu_starts=starts,
            checkpoint_every_n_tokens=(
                seq_size_per_block if checkpoints is not None else 0
            ),
            backend="flashinfer",
            use_cp=False,
            max_seqlen=max_length,
        )
        if states is None:
            return result
        output, final = result
        # The cache writer only reads checkpoints for non-final pages.
        if checkpoints is None:
            checkpoints = final
            starts = cu
        table = inputs.kv_cache_kernel_block_id_device
        store_page_checkpoints_kernel[
            (batch, triton.cdiv(max_length, 128), triton.cdiv(final[0].numel(), 1024))
        ](
            checkpoints,
            final,
            starts,
            cu,
            inputs.prefix_lengths_device,
            table,
            states,
            TABLE_STRIDE=table.stride(0),
            CACHE_STRIDE=states.stride(0),
            HEADS=self.local_num_v_heads,
            PAGE=128,
            BLOCK=1024,
        )
        return output


class Qwen4FlashInferGdnDecode(Qwen3NextGatedDeltaNetDecode):
    def _fla(self, mixed_qkv, b, a, kv_cache_tensor, seq_size_per_block, inputs, meta):
        if not (
            mixed_qkv.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(mixed_qkv.device)[0] == 10
            and mixed_qkv.dtype == torch.bfloat16
            and self.head_k_dim == self.head_v_dim == 128
            and self.ssm_state_dtype == torch.float32
            and not self._is_target_verify(inputs, meta)
            and inputs.kv_cache_kernel_block_id_device is not None
        ):
            return super()._fla(
                mixed_qkv, b, a, kv_cache_tensor, seq_size_per_block, inputs, meta
            )
        from rtp_llm.models_py.triton_kernels.fla.flashinfer_decode import (
            flashinfer_gdn_decode,
        )

        heads = (self.local_num_k_heads, self.local_num_k_heads, self.local_num_v_heads)
        q, k, v = mixed_qkv.reshape(-1, 1, sum(heads), 128).split(heads, dim=2)
        return flashinfer_gdn_decode(
            q,
            k,
            v,
            a,
            b,
            self.alog,
            self.dt_bias,
            self._get_ssm_states(kv_cache_tensor),
            inputs.kv_cache_kernel_block_id_device,
            inputs.sequence_lengths_plus_1_device,
            seq_size_per_block,
        ).squeeze(1)


def install_qwen4_gdn_backends(attention):
    backend = os.environ.get("QWEN4_GDN_BACKEND", "native")
    if backend not in ("native", "flashinfer"):
        raise ValueError("QWEN4_GDN_BACKEND must be native or flashinfer")
    prefill_cls = Qwen4Sm120GdnPrefill
    if backend == "flashinfer":
        prefill_cls = Qwen4FlashInferGdnPrefill
        decode = attention.decode_gdn
        attention.decode_gdn = Qwen4FlashInferGdnDecode(
            decode.linear_attn_config, decode.parallelism_config, decode.weights
        )
    prefill = attention.prefill_gdn
    attention.prefill_gdn = prefill_cls(
        prefill.linear_attn_config, prefill.parallelism_config, prefill.weights
    )
