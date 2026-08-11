"""Kimi K3 Prefill execution using short convolution and cuLA."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
from torch import nn

from rtp_llm.models_py.modules.kimi_k3.kda.cache import KimiK3KDACache
from rtp_llm.models_py.modules.kimi_k3.kda.state import KimiKDAState
from rtp_llm.models_py.modules.kimi_k3.sequence import sequence_offsets
from rtp_llm.models_py.triton_kernels.kimi_kda import kimi_kda_short_conv_prefill
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W


_CULA_LOGGED_DEVICES: set[int] = set()


@torch.compiler.disable
def _packed_causal_depthwise_conv1d_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    *,
    use_initial_state: Optional[bool] = None,
    sequence_ranges: Optional[list[tuple[int, int]]] = None,
    output_target: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run KDA short convolution over packed Prefill sequences."""

    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[0]:
        raise ValueError(
            "packed causal conv expects x=[tokens,channels] and "
            "weight=[channels,kernel]"
        )
    if not x.is_cuda:
        raise RuntimeError("Kimi K3 short convolution requires CUDA")
    ranges = (
        sequence_ranges
        if sequence_ranges is not None
        else sequence_offsets(cu_seqlens, x.shape[0])
    )
    channels, kernel_size = weight.shape
    history_size = kernel_size - 1
    expected_state = (len(ranges), channels, history_size)
    had_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = x.new_zeros(expected_state)
    elif tuple(initial_state.shape) != expected_state:
        raise ValueError(
            f"conv state must have shape {expected_state}, got "
            f"{tuple(initial_state.shape)}"
        )

    if output_target is not None and (
        tuple(output_target.shape) != tuple(x.shape)
        or output_target.dtype != x.dtype
        or output_target.device != x.device
        or not output_target.is_contiguous()
    ):
        raise ValueError(
            "KDA short conv output target must be contiguous and match "
            f"the input: input={tuple(x.shape)}/{x.dtype}/{x.device}, "
            f"output={tuple(output_target.shape)}/{output_target.dtype}/"
            f"{output_target.device}"
        )

    outputs: list[torch.Tensor] = []
    final_states: list[torch.Tensor] = []
    for sequence_idx, (start, end) in enumerate(ranges):
        sequence = x[start:end]
        history = initial_state[sequence_idx].to(dtype=x.dtype)
        if end == start:
            output = x.new_empty((0, channels))
            combined = torch.cat((history, sequence.transpose(0, 1)), dim=-1)
            final_state = (
                combined[:, -history_size:] if history_size else combined[:, :0]
            )
        else:
            output, final_state = kimi_kda_short_conv_prefill(
                sequence,
                weight,
                history,
                use_history=(
                    had_initial_state
                    if use_initial_state is None
                    else use_initial_state
                ),
                output=(None if output_target is None else output_target[start:end]),
                final_state=None,
            )
        outputs.append(output)
        final_states.append(final_state)
    if len(outputs) == 1:
        return outputs[0], final_states[0].unsqueeze(0)
    return torch.cat(outputs, dim=0), torch.stack(final_states, dim=0)


class KimiK3KDAPrefill(nn.Module):
    """Prefill-only KDA executor."""

    def __init__(
        self,
        *,
        weights: Dict[str, torch.Tensor],
        cache: KimiK3KDACache,
        local_heads: int,
        head_dim: int,
        projection_size: int,
        history_size: int,
        gate_lower_bound: Optional[float],
        q_conv: torch.Tensor,
        k_conv: torch.Tensor,
        v_conv: torch.Tensor,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.cache = cache
        self.local_heads = local_heads
        self.head_dim = head_dim
        self.projection_size = projection_size
        self.history_size = history_size
        self.gate_lower_bound = gate_lower_bound
        self.q_conv = q_conv
        self.k_conv = k_conv
        self.v_conv = v_conv
        self._segment_cu_seqlens: dict[tuple[int, int], torch.Tensor] = {}
        self._segment_cu_seqlens_cpu: dict[int, torch.Tensor] = {}

    def _cu_seqlens_for_segment(
        self, segment_length: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device_index = device.index if device.index is not None else 0
        device_key = (device_index, segment_length)
        device_cu = self._segment_cu_seqlens.get(device_key)
        if device_cu is None:
            device_cu = torch.tensor(
                [0, segment_length], dtype=torch.int32, device=device
            )
            self._segment_cu_seqlens[device_key] = device_cu
        return device_cu, self._host_cu_seqlens_for_segment(segment_length)

    def _host_cu_seqlens_for_segment(
        self, segment_length: int
    ) -> torch.Tensor:
        host_cu = self._segment_cu_seqlens_cpu.get(segment_length)
        if host_cu is None:
            host_cu = torch.tensor([0, segment_length], dtype=torch.int32)
            self._segment_cu_seqlens_cpu[segment_length] = host_cu
        return host_cu

    def _cula(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        recurrent_state: Optional[torch.Tensor],
        *,
        cu_seqlens: torch.Tensor,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
        output_target: Optional[torch.Tensor] = None,
        checkpoint_interval: Optional[int] = None,
        checkpoint_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run cuLA and return its canonical FP32 final state."""

        if not q.is_cuda:
            raise RuntimeError("Kimi K3 cuLA Prefill requires CUDA")
        if checkpoint_interval is None and checkpoint_states is not None:
            raise ValueError("checkpoint_states requires checkpoint_interval")
        if recurrent_state is not None and (
            recurrent_state.dtype != torch.float32
            or not recurrent_state.is_contiguous()
        ):
            raise ValueError("K3 cuLA state must be contiguous FP32")
        try:
            import cula
            from cula.kda import chunk_kda as cula_chunk_kda
        except Exception as error:
            raise RuntimeError(
                "Prefill requires cuLA but the cuda-linear-attention package "
                f"could not be imported: {type(error).__name__}: {error}"
            ) from error
        if self.gate_lower_bound is None:
            raise RuntimeError("cuLA requires K3's finite gate lower bound")

        device_index = q.device.index if q.device.index is not None else 0
        if device_index not in _CULA_LOGGED_DEVICES:
            logging.info(
                "[KimiK3 cuLA] enabled device=%s package=%s version=%s",
                q.device,
                getattr(cula, "__file__", "<unknown>"),
                getattr(cula, "__version__", "<unknown>"),
            )
            _CULA_LOGGED_DEVICES.add(device_index)
        single_sequence = int(cu_seqlens.numel()) == 2
        cula_cu_seqlens = None if single_sequence else cu_seqlens.contiguous()
        with torch.inference_mode():
            cula_result = cula_chunk_kda(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                raw_gate.to(dtype=q.dtype).contiguous(),
                raw_beta.to(dtype=q.dtype),
                scale=self.head_dim**-0.5,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                cu_seqlens=cula_cu_seqlens,
                cu_seqlens_cpu=(
                    None if cula_cu_seqlens is None else cu_seqlens_cpu
                ),
                safe_gate=True,
                lower_bound=float(self.gate_lower_bound),
                disable_recompute=False,
                use_intracard_cp=(
                    False if checkpoint_interval is not None else "auto"
                ),
                A_log=self.weights[W.linear_attn_alog].float().contiguous(),
                dt_bias=self.weights[W.linear_attn_dt_b_kda].float().contiguous(),
                checkpoint_interval=checkpoint_interval,
                checkpoint_states=checkpoint_states,
            )
            if checkpoint_interval is None:
                output, final_state = cula_result
            else:
                output, final_state, published_checkpoints = cula_result
                if (
                    checkpoint_states is None
                    or published_checkpoints.data_ptr() != checkpoint_states.data_ptr()
                ):
                    raise RuntimeError(
                        "cuLA did not publish into the requested FP32 checkpoint buffer"
                    )
            if (
                final_state is None
                or final_state.dtype != torch.float32
                or not final_state.is_contiguous()
            ):
                raise RuntimeError("cuLA must return a contiguous FP32 final state")
            if output_target is not None:
                output_target.copy_(output)
                output = output_target
        return output.to(dtype=q.dtype), final_state

    def _aligned_checkpoint_prefill(
        self,
        q_projected: torch.Tensor,
        k_projected: torch.Tensor,
        v_projected: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        initial_state: KimiKDAState,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
        *,
        past_length: int,
        page_size: int,
        block_map: list[list[int]],
    ) -> tuple[torch.Tensor, KimiKDAState]:
        """Run one long cuLA invocation and publish exact page checkpoints."""

        token_count = int(q_projected.shape[0])
        if token_count <= 0:
            raise ValueError("cuLA checkpoint prefill requires at least one token")
        if page_size % 64:
            raise ValueError(
                "cuLA checkpoint page size must be a multiple of 64 tokens, "
                f"got {page_size}"
            )
        if past_length % page_size:
            raise ValueError(
                "cuLA checkpoint prefill requires a page-aligned prefix, "
                f"got past_length={past_length}, page_size={page_size}"
            )
        checkpoint_count = (token_count + page_size - 1) // page_size
        q_conv = torch.empty_like(q_projected)
        k_conv = torch.empty_like(k_projected)
        v_conv = torch.empty_like(v_projected)
        sequence_range = [(0, token_count)]
        q_result, q_final = _packed_causal_depthwise_conv1d_prefill(
            q_projected,
            self.q_conv,
            cu_seqlens,
            initial_state.q_conv_state,
            use_initial_state=past_length > 0,
            sequence_ranges=sequence_range,
            output_target=q_conv,
        )
        k_result, k_final = _packed_causal_depthwise_conv1d_prefill(
            k_projected,
            self.k_conv,
            cu_seqlens,
            initial_state.k_conv_state,
            use_initial_state=past_length > 0,
            sequence_ranges=sequence_range,
            output_target=k_conv,
        )
        v_result, v_final = _packed_causal_depthwise_conv1d_prefill(
            v_projected,
            self.v_conv,
            cu_seqlens,
            initial_state.v_conv_state,
            use_initial_state=past_length > 0,
            sequence_ranges=sequence_range,
            output_target=v_conv,
        )
        if (
            q_result.data_ptr() != q_conv.data_ptr()
            or k_result.data_ptr() != k_conv.data_ptr()
            or v_result.data_ptr() != v_conv.data_ptr()
        ):
            raise RuntimeError("K3 short convolution did not use its output target")

        recurrent_checkpoints = torch.empty(
            (
                1,
                checkpoint_count,
                self.local_heads,
                self.head_dim,
                self.head_dim,
            ),
            dtype=torch.float32,
            device=q_projected.device,
        )
        cu_seqlens_cpu = self._host_cu_seqlens_for_segment(token_count)
        head_shape = (1, token_count, self.local_heads, self.head_dim)
        output, recurrent_final = self._cula(
            q_conv.reshape(head_shape),
            k_conv.reshape(head_shape),
            v_conv.reshape(head_shape),
            raw_gate.reshape(head_shape),
            raw_beta.reshape(1, token_count, self.local_heads),
            None if past_length == 0 else initial_state.recurrent_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            checkpoint_interval=page_size,
            checkpoint_states=recurrent_checkpoints,
        )

        for checkpoint_index in range(checkpoint_count):
            end = min((checkpoint_index + 1) * page_size, token_count)
            absolute_end = past_length + end
            if end >= self.history_size:
                q_checkpoint = (
                    q_projected.narrow(
                        0, end - self.history_size, self.history_size
                    )
                    .transpose(0, 1)
                    .unsqueeze(0)
                )
                k_checkpoint = (
                    k_projected.narrow(
                        0, end - self.history_size, self.history_size
                    )
                    .transpose(0, 1)
                    .unsqueeze(0)
                )
                v_checkpoint = (
                    v_projected.narrow(
                        0, end - self.history_size, self.history_size
                    )
                    .transpose(0, 1)
                    .unsqueeze(0)
                )
            else:
                q_checkpoint = q_final
                k_checkpoint = k_final
                v_checkpoint = v_final
            self.cache.store_position(
                KimiKDAState(
                    q_conv_state=q_checkpoint,
                    k_conv_state=k_checkpoint,
                    v_conv_state=v_checkpoint,
                    recurrent_state=recurrent_checkpoints[:, checkpoint_index],
                ),
                0,
                kv_cache,
                attention_inputs,
                0,
                absolute_end - 1,
                block_map=block_map,
            )

        return output, KimiKDAState(
            q_conv_state=q_final,
            k_conv_state=k_final,
            v_conv_state=v_final,
            recurrent_state=recurrent_final,
        )

    def _paged_prefill(
        self,
        q_projected: torch.Tensor,
        k_projected: torch.Tensor,
        v_projected: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        initial_state: KimiKDAState,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
    ) -> tuple[torch.Tensor, KimiKDAState]:
        """Run cuLA page-by-page and persist every reusable boundary."""

        ranges = sequence_offsets(
            cu_seqlens,
            q_projected.shape[0],
            cu_seqlens_host=getattr(attention_inputs, "cu_seqlens_host", None),
        )
        past_lengths = self.cache.prefix_lengths(attention_inputs, cu_seqlens)
        block_map = self.cache.block_map(attention_inputs)
        page_size = int(kv_cache.seq_size_per_block)
        if page_size <= 0:
            raise ValueError("linear cache seq_size_per_block must be positive")
        if len(ranges) == 1 and past_lengths[0] % page_size == 0:
            return self._aligned_checkpoint_prefill(
                q_projected,
                k_projected,
                v_projected,
                raw_gate,
                raw_beta,
                cu_seqlens,
                initial_state,
                kv_cache,
                attention_inputs,
                past_length=past_lengths[0],
                page_size=page_size,
                block_map=block_map,
            )

        fused_output = q_projected.new_empty(
            1, q_projected.shape[0], self.local_heads, self.head_dim
        )
        q_finals: list[torch.Tensor] = []
        k_finals: list[torch.Tensor] = []
        v_finals: list[torch.Tensor] = []
        recurrent_finals: list[torch.Tensor] = []
        for sequence_idx, ((start, end), past_length) in enumerate(
            zip(ranges, past_lengths)
        ):
            q_state = initial_state.q_conv_state[sequence_idx : sequence_idx + 1]
            k_state = initial_state.k_conv_state[sequence_idx : sequence_idx + 1]
            v_state = initial_state.v_conv_state[sequence_idx : sequence_idx + 1]
            recurrent_state = initial_state.recurrent_state[
                sequence_idx : sequence_idx + 1
            ]
            cursor = start
            absolute_position = past_length
            while cursor < end:
                tokens_to_page_end = page_size - (absolute_position % page_size)
                segment_end = min(end, cursor + tokens_to_page_end)
                segment_length = segment_end - cursor
                segment_cu, segment_cu_cpu = self._cu_seqlens_for_segment(
                    segment_length, cu_seqlens.device
                )
                segment_ranges = [(0, segment_length)]
                q, q_state = _packed_causal_depthwise_conv1d_prefill(
                    q_projected[cursor:segment_end],
                    self.q_conv,
                    segment_cu,
                    q_state,
                    use_initial_state=absolute_position > 0,
                    sequence_ranges=segment_ranges,
                )
                k, k_state = _packed_causal_depthwise_conv1d_prefill(
                    k_projected[cursor:segment_end],
                    self.k_conv,
                    segment_cu,
                    k_state,
                    use_initial_state=absolute_position > 0,
                    sequence_ranges=segment_ranges,
                )
                v, v_state = _packed_causal_depthwise_conv1d_prefill(
                    v_projected[cursor:segment_end],
                    self.v_conv,
                    segment_cu,
                    v_state,
                    use_initial_state=absolute_position > 0,
                    sequence_ranges=segment_ranges,
                )
                head_shape = (
                    1,
                    segment_length,
                    self.local_heads,
                    self.head_dim,
                )
                segment_output, recurrent_state = self._cula(
                    q.reshape(head_shape),
                    k.reshape(head_shape),
                    v.reshape(head_shape),
                    raw_gate[cursor:segment_end].reshape(head_shape),
                    raw_beta[cursor:segment_end].reshape(
                        1, segment_length, self.local_heads
                    ),
                    recurrent_state,
                    cu_seqlens=segment_cu,
                    cu_seqlens_cpu=segment_cu_cpu,
                    output_target=fused_output[:, cursor:segment_end],
                )
                segment_state = KimiKDAState(
                    q_conv_state=q_state,
                    k_conv_state=k_state,
                    v_conv_state=v_state,
                    recurrent_state=recurrent_state,
                )
                self.cache.store_position(
                    segment_state,
                    0,
                    kv_cache,
                    attention_inputs,
                    sequence_idx,
                    absolute_position + segment_length - 1,
                    block_map=block_map,
                )
                cursor = segment_end
                absolute_position += segment_length

            q_finals.append(q_state[0])
            k_finals.append(k_state[0])
            v_finals.append(v_state[0])
            recurrent_finals.append(recurrent_state[0])

        if len(ranges) == 1:
            return fused_output, KimiKDAState(
                q_conv_state=q_state,
                k_conv_state=k_state,
                v_conv_state=v_state,
                recurrent_state=recurrent_state,
            )
        return fused_output, KimiKDAState(
            q_conv_state=torch.stack(q_finals),
            k_conv_state=torch.stack(k_finals),
            v_conv_state=torch.stack(v_finals),
            recurrent_state=torch.stack(recurrent_finals),
        )

    def forward(
        self,
        q_projected: torch.Tensor,
        k_projected: torch.Tensor,
        v_projected: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        state: Optional[KimiKDAState],
        kv_cache: Optional[LayerKVCache],
        attention_inputs: Optional[PyAttentionInputs],
    ) -> tuple[torch.Tensor, KimiKDAState]:
        if attention_inputs is not None and getattr(
            attention_inputs, "is_target_verify", False
        ):
            raise RuntimeError(
                "Kimi K3 target verify requires the direct paged Decode path"
            )
        if kv_cache is not None:
            if attention_inputs is None:
                raise ValueError("attention_inputs are required with a KDA cache")
            state = self.cache.load_state(kv_cache, attention_inputs, cu_seqlens)
            return self._paged_prefill(
                q_projected,
                k_projected,
                v_projected,
                raw_gate,
                raw_beta,
                cu_seqlens,
                state,
                kv_cache,
                attention_inputs,
            )

        sequence_ranges = sequence_offsets(
            cu_seqlens,
            q_projected.shape[0],
            cu_seqlens_host=(
                getattr(attention_inputs, "cu_seqlens_host", None)
                if attention_inputs is not None
                else None
            ),
        )
        q, q_final = _packed_causal_depthwise_conv1d_prefill(
            q_projected,
            self.q_conv,
            cu_seqlens,
            None if state is None else state.q_conv_state,
            sequence_ranges=sequence_ranges,
        )
        k, k_final = _packed_causal_depthwise_conv1d_prefill(
            k_projected,
            self.k_conv,
            cu_seqlens,
            None if state is None else state.k_conv_state,
            sequence_ranges=sequence_ranges,
        )
        v, v_final = _packed_causal_depthwise_conv1d_prefill(
            v_projected,
            self.v_conv,
            cu_seqlens,
            None if state is None else state.v_conv_state,
            sequence_ranges=sequence_ranges,
        )
        token_count = q_projected.shape[0]
        head_shape = (1, token_count, self.local_heads, self.head_dim)
        output, recurrent_final = self._cula(
            q.reshape(head_shape),
            k.reshape(head_shape),
            v.reshape(head_shape),
            raw_gate.reshape(head_shape),
            raw_beta.reshape(1, token_count, self.local_heads),
            None if state is None else state.recurrent_state,
            cu_seqlens=cu_seqlens,
        )
        return output, KimiKDAState(
            q_conv_state=q_final,
            k_conv_state=k_final,
            v_conv_state=v_final,
            recurrent_state=recurrent_final,
        )


__all__ = ["KimiK3KDAPrefill"]
