"""K3's native KDA core behind RTP's convolution and CacheStore lifecycle."""

import torch

from rtp_llm.models_py.model_desc.kimi_linear import KimiLinearKDAPrefill
from rtp_llm.models_py.modules.kimi_k3.native_kda import (
    flash_kda_paged_prefill,
    vllm_kda_paged_prefill,
    cula_kda_paged_prefill,
)


class KimiK3NativeKDAPrefill(KimiLinearKDAPrefill):
    core = staticmethod(flash_kda_paged_prefill)

    def __init__(
        self, config, parallelism, weights, backend="flashkda", use_paged_conv=False
    ):
        super().__init__(config, parallelism, weights)
        self.use_paged_conv = use_paged_conv
        if backend == "vllm_triton":
            self.core = vllm_kda_paged_prefill
        elif backend == "cula":
            self.core = cula_kda_paged_prefill
        elif backend != "flashkda":
            raise ValueError(f"Unsupported native KDA prefill backend: {backend}")
        # Native KDA consumes the gate parameters in FP32, independently of the
        # ordinary projection precision. Conversion happens once, at creation.
        self.alog = self.alog.float().contiguous()
        self.dt_bias = self.dt_bias.float().contiguous()

    @staticmethod
    def _validate_prefix_state(attn_inputs, seq_size_per_block):
        # Validate cached-prefix ownership before a kernel can read the state.
        prefixes = attn_inputs.prefix_lengths
        table = attn_inputs.kv_cache_kernel_block_id
        if prefixes.device.type != "cpu" or table.device.type != "cpu":
            raise ValueError("K3 convolution requires host cache metadata mirrors")
        for row, prefix in enumerate(prefixes.tolist()):
            if prefix > 0:
                position = (prefix - 1) // seq_size_per_block
                if position >= table.shape[1] or int(table[row, position]) <= 0:
                    raise ValueError("K3 convolution is missing a cached prefix state")

    def _conv1d(
        self, mixed_qkv, kv_cache_tensor, seq_size_per_block, attn_inputs, metadata=None
    ):
        from rtp_llm.models_py.modules.kimi_k3.native_conv import causal_conv1d_fn

        states = (
            self._get_conv_states(kv_cache_tensor).transpose(1, 2)
            if kv_cache_tensor is not None else None
        )
        if states is not None:
            self._validate_prefix_state(attn_inputs, seq_size_per_block)
        return causal_conv1d_fn(
            mixed_qkv.transpose(0, 1), self.conv_weights, None, states,
            attn_inputs.cu_seqlens_device,
            attn_inputs.kv_cache_kernel_block_id_device,
            attn_inputs.prefix_lengths_device, seq_size_per_block, metadata,
        ).transpose(0, 1)

    def _fla(
        self,
        mixed_qkv,
        forget_gate,
        beta,
        kv_cache_tensor,
        seq_size_per_block,
        attn_inputs,
    ):
        cu = attn_inputs.cu_seqlens
        prefixes = attn_inputs.prefix_lengths
        if cu.device.type != "cpu" or prefixes.device.type != "cpu":
            raise ValueError("K3 native KDA prefill requires RTP host metadata mirrors")
        cu, prefixes = cu.tolist(), prefixes.tolist()
        if len(prefixes) != len(cu) - 1:
            raise ValueError("K3 native KDA prefix metadata does not match the batch")
        shape = (-1, self.local_num_v_heads, self.head_k_dim)
        if isinstance(mixed_qkv, tuple):
            q, k, v = (x.reshape(shape) for x in mixed_qkv)
            tokens, device = mixed_qkv[0].shape[0], mixed_qkv[0].device
        else:
            q, k, v = (x.reshape(shape) for x in mixed_qkv.chunk(3, dim=-1))
            tokens, device = mixed_qkv.shape[0], mixed_qkv.device
        if kv_cache_tensor is None:
            if any(prefixes):
                raise ValueError("A cached KDA prefix requires recurrent cache storage")
            count = len(prefixes)
            logical = attn_inputs.logical_request_count or count
            block_table = [[i + 1] if i < logical else [0] for i in range(count)]
            seq_size_per_block = max([1] + [b - a for a, b in zip(cu, cu[1:])])
            states = torch.zeros(
                (count + 1, self.local_num_v_heads, self.head_k_dim, self.head_v_dim),
                dtype=torch.float32,
                device=device,
            )
        else:
            table = attn_inputs.kv_cache_kernel_block_id
            if table.device.type != "cpu":
                raise ValueError("K3 native KDA block planning requires host block IDs")
            block_table = table.tolist()
            states = self._get_ssm_states(kv_cache_tensor)
        return self.core(
            q,
            k,
            v,
            forget_gate.reshape(shape),
            beta,
            self.alog,
            self.dt_bias,
            self.gate_lower_bound,
            states,
            cu,
            prefixes,
            block_table,
            seq_size_per_block,
        ).reshape(tokens, -1)

    def forward(
        self, mixed_qkv, forget_gate, beta, attn_inputs, kv_cache, attn_meta
    ):
        if not self.use_paged_conv or kv_cache is None:
            return super().forward(
                mixed_qkv, forget_gate, beta, attn_inputs, kv_cache, attn_meta
            )
        from rtp_llm.models_py.triton_kernels.causal_conv1d.paged_short_conv_prefill import (
            paged_short_conv_prefill,
        )

        metadata = attn_meta.prefill_paged_conv_meta
        if metadata is None:
            raise ValueError("paged convolution metadata is required for prefill")
        kv_cache_tensor = kv_cache.kv_cache_base.reshape(
            kv_cache.kv_cache_base.shape[0], -1
        )
        page_size = kv_cache.seq_size_per_block
        # A checkpoint inside a 64-token program cannot be published safely.
        # Keep the existing conv for unaligned cached prefixes.
        if any(int(prefix) % 64 for prefix in attn_inputs.prefix_lengths.tolist()):
            return super().forward(
                mixed_qkv, forget_gate, beta, attn_inputs, kv_cache, attn_meta
            )
        self._validate_prefix_state(attn_inputs, page_size)
        q, k, v, _, packed_beta = paged_short_conv_prefill(
            mixed_qkv,
            self.conv_weights,
            self._get_conv_states(kv_cache_tensor),
            attn_inputs.kv_cache_kernel_block_id_device,
            attn_inputs.prefix_lengths_device,
            attn_inputs.cu_seqlens_device,
            page_size,
            metadata,
            aux=beta,
        )
        output = self._fla(
            (q, k, v), forget_gate, packed_beta, kv_cache_tensor, page_size,
            attn_inputs,
        )
        if (
            attn_inputs.cache_store_inputs is not None
            and attn_inputs.cache_store_writer is not None
        ):
            attn_inputs.cache_store_writer.write(
                attn_inputs.cache_store_inputs, kv_cache
            )
        return output


KimiK3FlashKDAPrefill = KimiK3NativeKDAPrefill
