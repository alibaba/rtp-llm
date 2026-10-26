from statistics import median
from time import perf_counter
from typing import Callable, Tuple

import torch

from rtp_llm.models_py.model_desc.kimi_linear import KimiLinearKDAPrefill
from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNetPrefill
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    prepare_causal_conv1d_metadata,
)
from rtp_llm.ops import DataType, LinearAttentionConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W

AuxTensors = Tuple[torch.Tensor, torch.Tensor]
ForwardFn = Callable[
    [object, torch.Tensor, AuxTensors, object, LayerKVCache, object],
    torch.Tensor,
]


class _AttnInputsWrapper:
    """Override readonly device mirrors for a focused model test."""

    def __init__(self, wrapped: PyAttentionInputs, overrides: dict):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    def __setattr__(self, name, value):
        try:
            setattr(object.__getattribute__(self, "_wrapped"), name, value)
        except AttributeError:
            object.__getattribute__(self, "_overrides")[name] = value


def _linear_config(heads: int, head_dim: int, conv_width: int) -> LinearAttentionConfig:
    config = LinearAttentionConfig()
    config.linear_num_key_heads = heads
    config.linear_num_value_heads = heads
    config.linear_key_head_dim = head_dim
    config.linear_value_head_dim = head_dim
    config.linear_conv_kernel_dim = conv_width
    config.ssm_state_dtype = DataType.TYPE_BF16
    config.conv_state_dtype = DataType.TYPE_BF16
    return config


def _parallel_config() -> ParallelismConfig:
    config = ParallelismConfig()
    config.tp_size = 1
    config.tp_rank = 0
    return config


def _make_gdn_module(device: torch.device) -> Qwen3NextGatedDeltaNetPrefill:
    heads, head_dim, conv_width = 2, 32, 4
    qkv_size = heads * head_dim * 2 + heads * head_dim
    weights = {
        W.linear_attn_conv1d_w: torch.randn(
            qkv_size, 1, conv_width, device=device, dtype=torch.bfloat16
        ),
        W.linear_attn_dt_b: torch.randn(heads, device=device, dtype=torch.bfloat16),
        W.linear_attn_alog: torch.randn(heads, device=device, dtype=torch.bfloat16),
    }
    return Qwen3NextGatedDeltaNetPrefill(
        _linear_config(heads, head_dim, conv_width), _parallel_config(), weights
    )


def _make_kda_module(device: torch.device) -> KimiLinearKDAPrefill:
    heads, head_dim, conv_width = 2, 32, 4
    qkv_size = heads * head_dim * 2 + heads * head_dim
    weights = {
        W.linear_attn_conv1d_w: torch.randn(
            qkv_size, 1, conv_width, device=device, dtype=torch.bfloat16
        ),
        W.linear_attn_dt_b_kda: torch.randn(heads, device=device, dtype=torch.bfloat16),
        W.linear_attn_alog: torch.randn(heads, device=device, dtype=torch.bfloat16),
    }
    return KimiLinearKDAPrefill(
        _linear_config(heads, head_dim, conv_width), _parallel_config(), weights
    )


def _make_cache(module, blocks: int, device: torch.device) -> LayerKVCache:
    converter = module.linear_cache_converter
    block_size = converter.block_size_bytes
    assert block_size % 2 == 0
    base = torch.zeros(blocks, block_size // 2, device=device, dtype=torch.bfloat16)
    return LayerKVCache(base, 256)


def _copy_linear_block(module, cache: LayerKVCache, source: int, destination: int):
    base = cache.kv_cache_base.reshape(cache.kv_cache_base.shape[0], -1)
    converter = module.linear_cache_converter
    ssm = converter.get_ssm_state_tensor(base)
    conv = converter.get_conv_state_tensor(base)
    ssm[destination].copy_(ssm[source])
    conv[destination].copy_(conv[source])


def _attention_inputs(
    prefix: int,
    length: int,
    block_map: torch.Tensor,
    device: torch.device,
    positions: torch.Tensor | None = None,
    block_ids: torch.Tensor | None = None,
):
    inputs = PyAttentionInputs()
    inputs.is_prefill = True
    inputs.input_lengths = torch.tensor([length], dtype=torch.int32, device="cpu")
    inputs.prefix_lengths = torch.tensor([prefix], dtype=torch.int32, device="cpu")
    inputs.cu_seqlens_device = torch.tensor(
        [0, length], dtype=torch.int32, device=device
    )
    inputs.kv_cache_kernel_block_id_device = block_map.to(device)
    if positions is not None:
        inputs.cache_save_positions = positions
    if block_ids is not None:
        inputs.cache_save_block_ids = block_ids
    return _AttnInputsWrapper(
        inputs, {"prefix_lengths_device": inputs.prefix_lengths.to(device)}
    )


def _gdn_forward(module, tokens, aux, attention_inputs, cache, metadata):
    b, a = aux
    return module(tokens, b, a, attention_inputs, cache, metadata)


def _kda_forward(module, tokens, aux, attention_inputs, cache, metadata):
    forget_gate, beta = aux
    return module(tokens, forget_gate, beta, attention_inputs, cache, metadata)


def _exercise_sparse_save_restore(
    module,
    forward: ForwardFn,
    metadata_factory,
    full_tokens: torch.Tensor,
    aux_full: AuxTensors,
    device: torch.device,
):
    cache = _make_cache(module, 16, device)

    def run(
        prefix: int,
        length: int,
        block_map: torch.Tensor,
        positions: torch.Tensor | None = None,
        block_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = _attention_inputs(
            prefix, length, block_map, device, positions=positions, block_ids=block_ids
        )
        metadata = metadata_factory(
            prepare_causal_conv1d_metadata(inputs.cu_seqlens_device, device=device)
        )
        return forward(
            module,
            full_tokens[prefix : prefix + length],
            tuple(tensor[prefix : prefix + length] for tensor in aux_full),
            inputs,
            cache,
            metadata,
        )

    block_map_source = torch.tensor([[0, 1, 2, 3, 4, 5, -1, -1]], dtype=torch.int32)
    run(0, 1300, block_map_source)

    _copy_linear_block(module, cache, 5, 8)
    source_snapshot = cache.kv_cache_base[5].clone()
    unused_snapshot = cache.kv_cache_base[[13, 14, 15]].clone()
    block_map_continue = torch.tensor(
        [[-1, -1, -1, -1, -1, 8, 9, 7]], dtype=torch.int32
    )
    save_positions = torch.tensor(
        [[1536, 1792, 2000]], device=device, dtype=torch.int64
    )
    save_block_ids = torch.tensor([[10, 11, 12]], device=device, dtype=torch.int32)

    # Prime the selected-state path once before the two comparison runs. The
    # ordinary Triton kernels are not bit-stable across different CUDA
    # allocation histories, so this keeps the selected-path comparison stable.
    run(
        1300,
        700,
        block_map_continue,
        positions=save_positions,
        block_ids=save_block_ids,
    )
    _copy_linear_block(module, cache, 5, 8)

    # Run the selected-state path twice with only the restored source block
    # reset between calls. This checks model-level selected-path determinism
    # without mixing different CUDA allocator histories.
    first_output = run(
        1300,
        700,
        block_map_continue,
        positions=save_positions,
        block_ids=save_block_ids,
    )
    _copy_linear_block(module, cache, 5, 8)
    save_output = run(
        1300,
        700,
        block_map_continue,
        positions=save_positions,
        block_ids=save_block_ids,
    )

    torch.testing.assert_close(first_output, save_output, rtol=0, atol=0)
    torch.testing.assert_close(cache.kv_cache_base[5], source_snapshot, rtol=0, atol=0)
    torch.testing.assert_close(
        cache.kv_cache_base[[13, 14, 15]], unused_snapshot, rtol=0, atol=0
    )

    final_snapshot = cache.kv_cache_base[7].clone()
    torch.testing.assert_close(cache.kv_cache_base[12], final_snapshot, rtol=0, atol=0)
    selected_1536_snapshot = cache.kv_cache_base[10].clone()
    selected_1792_snapshot = cache.kv_cache_base[11].clone()

    # Generate independent no-save endpoint states. The ordinary path only
    # writes the final state of each run, so 1536 and 1792 need separate
    # continuations. Blocks 13/14 receive the ordinary-path final states at
    # those endpoints; comparing them to the selected blocks verifies
    # model-level wiring without relying on cross-path output bitwise equality,
    # which is not guaranteed across CUDA allocator histories.
    no_save_1536_map = torch.tensor(
        [[-1, -1, -1, -1, -1, 13, -1, -1]], dtype=torch.int32
    )
    _copy_linear_block(module, cache, 5, 13)
    run(1300, 236, no_save_1536_map)
    torch.testing.assert_close(
        cache.kv_cache_base[10], cache.kv_cache_base[13], rtol=0, atol=0
    )
    _copy_linear_block(module, cache, 13, 15)
    no_save_1792_map = torch.tensor(
        [[-1, -1, -1, -1, -1, 13, 14, -1]], dtype=torch.int32
    )
    _copy_linear_block(module, cache, 5, 13)
    run(1300, 492, no_save_1792_map)
    torch.testing.assert_close(
        cache.kv_cache_base[11], cache.kv_cache_base[14], rtol=0, atol=0
    )
    _copy_linear_block(module, cache, 14, 6)

    # Restore from 1536. Block 15 holds the independently computed no-save
    # endpoint, while block 13 receives the selected snapshot. Comparing them
    # avoids the known 1-ULP selected/no-save allocator-history difference
    # while still validating real model continuation from the saved state.
    _copy_linear_block(module, cache, 10, 13)
    reference_1536_map = torch.tensor(
        [[-1, -1, -1, -1, -1, 15, 14, 7]], dtype=torch.int32
    )
    reference_1536_output = run(1536, 464, reference_1536_map)
    reference_1536_final_snapshot = cache.kv_cache_base[7].clone()
    restore_1536_map = torch.tensor(
        [[-1, -1, -1, -1, -1, 13, 14, 7]], dtype=torch.int32
    )
    restore_1536_output = run(1536, 464, restore_1536_map)
    torch.testing.assert_close(
        restore_1536_output, reference_1536_output, rtol=0, atol=0
    )
    torch.testing.assert_close(
        cache.kv_cache_base[7], reference_1536_final_snapshot, rtol=0, atol=0
    )
    torch.testing.assert_close(
        cache.kv_cache_base[10], selected_1536_snapshot, rtol=0, atol=0
    )
    torch.testing.assert_close(
        cache.kv_cache_base[11], selected_1792_snapshot, rtol=0, atol=0
    )

    # Block 6 holds the independent no-save 1792 endpoint, because the 1536
    # continuation above reuses block 14 as a runtime destination.
    reference_1792_map = torch.tensor(
        [[-1, -1, -1, -1, -1, -1, 6, 7]], dtype=torch.int32
    )
    reference_1792_output = run(1792, 208, reference_1792_map)
    reference_1792_final_snapshot = cache.kv_cache_base[7].clone()
    restore_1792_map = torch.tensor(
        [[-1, -1, -1, -1, -1, -1, 11, 7]], dtype=torch.int32
    )
    restore_1792_output = run(1792, 208, restore_1792_map)
    torch.testing.assert_close(
        restore_1792_output, reference_1792_output, rtol=0, atol=0
    )
    torch.testing.assert_close(
        cache.kv_cache_base[7], reference_1792_final_snapshot, rtol=0, atol=0
    )
    torch.testing.assert_close(
        cache.kv_cache_base[10], selected_1536_snapshot, rtol=0, atol=0
    )
    torch.testing.assert_close(
        cache.kv_cache_base[11], selected_1792_snapshot, rtol=0, atol=0
    )

    # Report whole-layer latency separately from the selected-state kernel
    # stage measured by T08. Keep the source immutable and reset the runtime
    # continuation block before every sample; timing is not a correctness gate.
    def timed_forward(with_save: bool) -> tuple[float, float]:
        _copy_linear_block(module, cache, 5, 8)
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = perf_counter()
        start.record()
        run(
            1300,
            700,
            block_map_continue,
            positions=save_positions if with_save else None,
            block_ids=save_block_ids if with_save else None,
        )
        end.record()
        end.synchronize()
        return (perf_counter() - wall_start) * 1000, start.elapsed_time(end)

    for _ in range(2):
        timed_forward(False)
        timed_forward(True)
    baseline_samples = [timed_forward(False) for _ in range(5)]
    selected_samples = [timed_forward(True) for _ in range(5)]
    print(
        f"MODEL_COST {type(module).__name__} G1300 N2000 S3 "
        f"baseline_wall_ms={median(x[0] for x in baseline_samples):.3f} "
        f"selected_wall_ms={median(x[0] for x in selected_samples):.3f} "
        f"baseline_cuda_ms={median(x[1] for x in baseline_samples):.3f} "
        f"selected_cuda_ms={median(x[1] for x in selected_samples):.3f}",
        flush=True,
    )


def _exercise_internal_f_last_same_slot(
    module,
    forward: ForwardFn,
    metadata_factory,
    full_tokens: torch.Tensor,
    aux_full: AuxTensors,
    device: torch.device,
):
    """Keep an interior F1320 state independent of N1340 in one physical slot."""
    cache = _make_cache(module, 16, device)

    def run(start: int, end: int, block_id: int, positions=None, destinations=None):
        block_map = torch.tensor([[0, 1, 2, 3, 4, block_id, -1, -1]], dtype=torch.int32)
        inputs = _attention_inputs(
            start,
            end - start,
            block_map,
            device,
            positions=positions,
            block_ids=destinations,
        )
        metadata = metadata_factory(
            prepare_causal_conv1d_metadata(inputs.cu_seqlens_device, device=device)
        )
        return forward(
            module,
            full_tokens[start:end],
            tuple(tensor[start:end] for tensor in aux_full),
            inputs,
            cache,
            metadata,
        )

    run(0, 1300, 5)
    source = cache.kv_cache_base[5].clone()
    _copy_linear_block(module, cache, 5, 8)
    # The registration keeps N first. F is appended last, even though its
    # absolute endpoint is earlier; selected columns keep their own IDs.
    positions = torch.tensor([[1340, 1320]], dtype=torch.int64, device=device)
    destinations = torch.tensor([[9, 10]], dtype=torch.int32, device=device)
    run(1300, 1340, 8, positions, destinations)
    saved_f = cache.kv_cache_base[10].clone()
    saved_n = cache.kv_cache_base[9].clone()
    torch.testing.assert_close(cache.kv_cache_base[5], source, rtol=0, atol=0)

    _copy_linear_block(module, cache, 5, 11)
    run(1300, 1320, 11)  # Independent no-save state at the actual F.
    torch.testing.assert_close(
        cache.kv_cache_base[10], cache.kv_cache_base[11], rtol=0, atol=0
    )
    _copy_linear_block(module, cache, 5, 12)
    run(1300, 1340, 12)
    torch.testing.assert_close(
        cache.kv_cache_base[9], cache.kv_cache_base[12], rtol=0, atol=0
    )
    torch.testing.assert_close(cache.kv_cache_base[10], saved_f, rtol=0, atol=0)
    torch.testing.assert_close(cache.kv_cache_base[9], saved_n, rtol=0, atol=0)

    _copy_linear_block(module, cache, 10, 13)
    reused = run(1320, 1340, 13)
    _copy_linear_block(module, cache, 11, 14)
    reference = run(1320, 1340, 14)
    torch.testing.assert_close(reused, reference, rtol=0, atol=0)
    torch.testing.assert_close(
        cache.kv_cache_base[13], cache.kv_cache_base[14], rtol=0, atol=0
    )
    torch.testing.assert_close(cache.kv_cache_base[5], source, rtol=0, atol=0)


def _exercise_tail_boundaries(
    module,
    forward: ForwardFn,
    metadata_factory,
    full_tokens: torch.Tensor,
    aux_full: AuxTensors,
    device: torch.device,
):
    """Check tail-only, aligned N, empty save slots, and aligned-chunk G."""
    for prefix, endpoint in ((1300, 1320), (1300, 1536), (1344, 1360)):
        cache = _make_cache(module, 16, device)

        def run(block_map, start, length, positions=None, block_ids=None):
            inputs = _attention_inputs(
                start, length, block_map, device, positions, block_ids
            )
            metadata = metadata_factory(
                prepare_causal_conv1d_metadata(inputs.cu_seqlens_device, device=device)
            )
            return forward(
                module,
                full_tokens[start : start + length],
                tuple(tensor[start : start + length] for tensor in aux_full),
                inputs,
                cache,
                metadata,
            )

        source_map = torch.tensor([[0, 1, 2, 3, 4, 5, -1, -1]], dtype=torch.int32)
        run(source_map, 0, prefix)
        source = cache.kv_cache_base[5].clone()
        _copy_linear_block(module, cache, 5, 8)
        runtime_map = torch.tensor([[-1, -1, -1, -1, -1, 8, 9, 7]], dtype=torch.int32)
        save_positions = torch.tensor(
            [[0, endpoint, 0]], dtype=torch.int64, device=device
        )
        save_blocks = torch.tensor([[-1, 12, -1]], dtype=torch.int32, device=device)
        run(runtime_map, prefix, endpoint - prefix, save_positions, save_blocks)
        torch.testing.assert_close(cache.kv_cache_base[5], source, rtol=0, atol=0)
        torch.testing.assert_close(
            cache.kv_cache_base[12], cache.kv_cache_base[8], rtol=0, atol=0
        )
        saved = cache.kv_cache_base[12].clone()

        # An S=0 row suppresses optional destinations and still runs the
        # ordinary producer. Reset C from A so this is a valid continuation.
        _copy_linear_block(module, cache, 5, 8)
        run(
            runtime_map,
            prefix,
            endpoint - prefix,
            torch.empty((1, 0), dtype=torch.int64, device=device),
            torch.empty((1, 0), dtype=torch.int32, device=device),
        )
        torch.testing.assert_close(cache.kv_cache_base[12], saved, rtol=0, atol=0)

        _copy_linear_block(module, cache, 5, 13)
        no_save_map = torch.tensor(
            [[-1, -1, -1, -1, -1, 13, -1, -1]], dtype=torch.int32
        )
        run(no_save_map, prefix, endpoint - prefix)
        torch.testing.assert_close(
            cache.kv_cache_base[12], cache.kv_cache_base[13], rtol=0, atol=0
        )
        torch.testing.assert_close(cache.kv_cache_base[5], source, rtol=0, atol=0)


def _exercise_mixed_prefix_batch(
    module,
    forward: ForwardFn,
    metadata_factory,
    full_tokens: torch.Tensor,
    aux_full: AuxTensors,
    device: torch.device,
):
    """Two requests with different G share one selected-state launch."""
    cache = _make_cache(module, 40, device)

    def run(prefixes, lengths, block_map, positions=None, block_ids=None):
        inputs = PyAttentionInputs()
        inputs.is_prefill = True
        inputs.input_lengths = torch.tensor(lengths, dtype=torch.int32)
        inputs.prefix_lengths = torch.tensor(prefixes, dtype=torch.int32)
        inputs.cu_seqlens_device = torch.tensor(
            [0, lengths[0], sum(lengths)], dtype=torch.int32, device=device
        )
        inputs.kv_cache_kernel_block_id_device = block_map.to(device)
        if positions is not None:
            inputs.cache_save_positions = positions
        if block_ids is not None:
            inputs.cache_save_block_ids = block_ids
        wrapped = _AttnInputsWrapper(
            inputs, {"prefix_lengths_device": inputs.prefix_lengths.to(device)}
        )
        metadata = metadata_factory(
            prepare_causal_conv1d_metadata(inputs.cu_seqlens_device, device=device)
        )
        tokens = torch.cat(
            [full_tokens[p : p + n] for p, n in zip(prefixes, lengths)], dim=0
        )
        aux = tuple(
            torch.cat([tensor[p : p + n] for p, n in zip(prefixes, lengths)], dim=0)
            for tensor in aux_full
        )
        return forward(module, tokens, aux, wrapped, cache, metadata)

    source_map = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, -1, -1],
            [16, 17, 18, 19, 20, 21, -1, -1],
        ],
        dtype=torch.int32,
    )
    run([0, 0], [1300, 1344], source_map)
    _copy_linear_block(module, cache, 5, 8)
    _copy_linear_block(module, cache, 21, 24)
    old_sources = cache.kv_cache_base[[5, 21]].clone()
    runtime_map = torch.tensor(
        [
            [-1, -1, -1, -1, -1, 8, 9, 7],
            [-1, -1, -1, -1, -1, 24, 25, 23],
        ],
        dtype=torch.int32,
    )
    save_positions = torch.tensor(
        [[0, 1536, 0, 1792, 2000], [0, 1536, 0, 1792, 2000]],
        dtype=torch.int64,
        device=device,
    )
    save_blocks = torch.tensor(
        [[-1, 10, -1, 11, 12], [-1, 26, -1, 27, 28]],
        dtype=torch.int32,
        device=device,
    )
    run([1300, 1344], [700, 656], runtime_map, save_positions, save_blocks)
    torch.testing.assert_close(
        cache.kv_cache_base[[5, 21]], old_sources, rtol=0, atol=0
    )
    for save_id, runtime_id in ((12, 7), (28, 23)):
        torch.testing.assert_close(
            cache.kv_cache_base[save_id],
            cache.kv_cache_base[runtime_id],
            rtol=0,
            atol=0,
        )

    for endpoint, first_destinations, second_destinations in (
        (1536, (10, 26), (13, 29)),
        (1792, (11, 27), (14, 30)),
    ):
        _copy_linear_block(module, cache, 5, 13)
        _copy_linear_block(module, cache, 21, 29)
        no_save_map = torch.tensor(
            [
                [-1, -1, -1, -1, -1, 13, 14, -1],
                [-1, -1, -1, -1, -1, 29, 30, -1],
            ],
            dtype=torch.int32,
        )
        run([1300, 1344], [endpoint - 1300, endpoint - 1344], no_save_map)
        for selected_id, ordinary_id in zip(first_destinations, second_destinations):
            torch.testing.assert_close(
                cache.kv_cache_base[selected_id],
                cache.kv_cache_base[ordinary_id],
                rtol=0,
                atol=0,
            )
