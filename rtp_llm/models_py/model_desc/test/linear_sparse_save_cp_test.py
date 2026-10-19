"""Real NCCL CP prefill save and continuation for the Next LINEAR layer."""

import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.model_desc.qwen3_next import (
    Qwen3NextGatedDeltaNet,
    Qwen3NextMetadata,
)
from rtp_llm.models_py.model_desc.test._linear_sparse_save_common import (
    _AttnInputsWrapper,
    _copy_linear_block,
    _linear_config,
    _make_cache,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    prepare_causal_conv1d_metadata,
)
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyContextParallelParams
from rtp_llm.utils.model_weight import W

_WORLD_SIZE = 2


def _rank_positions(length: int, rank: int) -> list[int]:
    assert length % (2 * _WORLD_SIZE) == 0
    half = length // (2 * _WORLD_SIZE)
    return list(range(rank * half, (rank + 1) * half)) + list(
        range(length - (rank + 1) * half, length - rank * half)
    )


def _make_module(rank: int, device: torch.device) -> Qwen3NextGatedDeltaNet:
    heads, dim, hidden = 4, 32, 128
    local_heads = heads // _WORLD_SIZE
    qkv_size = 3 * local_heads * dim
    weights = {
        W.linear_attn_conv1d_w: torch.randn(
            qkv_size, 1, 4, device=device, dtype=torch.bfloat16
        ),
        W.linear_attn_dt_b: torch.randn(
            local_heads, device=device, dtype=torch.bfloat16
        ),
        W.linear_attn_alog: torch.randn(
            local_heads, device=device, dtype=torch.bfloat16
        ),
        W.linear_attn_norm_w: torch.randn(dim, device=device, dtype=torch.bfloat16),
        W.linear_attn_qkvz_w: torch.randn(
            hidden, qkv_size + local_heads * dim, device=device, dtype=torch.bfloat16
        ),
        W.linear_attn_ba_w: torch.randn(
            hidden, local_heads * 2, device=device, dtype=torch.bfloat16
        ),
        W.linear_attn_out_w: torch.randn(
            local_heads * dim, hidden, device=device, dtype=torch.bfloat16
        ),
    }
    config = ParallelismConfig()
    config.tp_size = _WORLD_SIZE
    config.tp_rank = rank
    return Qwen3NextGatedDeltaNet(
        _linear_config(heads, dim, 4), config, weights, layernorm_eps=1e-6
    ).to(device)


def _cp_inputs_and_metadata(
    prefix: int,
    length: int,
    rank: int,
    block_map: torch.Tensor,
    device: torch.device,
    positions: torch.Tensor | None,
    block_ids: torch.Tensor | None,
) -> tuple[_AttnInputsWrapper, Qwen3NextMetadata, torch.Tensor]:
    padded_length = ((length + 2 * _WORLD_SIZE - 1) // (2 * _WORLD_SIZE)) * (
        2 * _WORLD_SIZE
    )
    rank_positions = [_rank_positions(padded_length, r) for r in range(_WORLD_SIZE)]
    gather_order = rank_positions[0] + rank_positions[1]
    restore = torch.empty(padded_length, dtype=torch.int32, device=device)
    restore[torch.tensor(gather_order, device=device)] = torch.arange(
        padded_length, dtype=torch.int32, device=device
    )
    local_positions = torch.tensor(rank_positions[rank], device=device)
    local_length = padded_length // _WORLD_SIZE
    full_cu = torch.tensor([0, length], dtype=torch.int32, device=device)

    inputs = PyAttentionInputs()
    inputs.is_prefill = True
    inputs.input_lengths = torch.tensor([local_length], dtype=torch.int32)
    inputs.prefix_lengths = torch.tensor([prefix], dtype=torch.int32)
    inputs.cu_seqlens_device = torch.tensor(
        [0, local_length], dtype=torch.int32, device=device
    )
    inputs.kv_cache_kernel_block_id_device = block_map.to(device)
    cp_info = PyContextParallelParams()
    cp_info.prefill_qkv_padding_mask = torch.zeros(
        padded_length, dtype=torch.int32, device=device
    )
    cp_info.prefill_qkv_padding_mask[:length] = 1
    cp_info.prefill_qkv_restore_indice = restore
    cp_info.prefill_actual_input_lengths_cpu = torch.tensor([length], dtype=torch.int32)
    inputs.context_parallel_info = cp_info
    if positions is not None:
        inputs.cache_save_positions = positions
    if block_ids is not None:
        inputs.cache_save_block_ids = block_ids
    wrapped = _AttnInputsWrapper(
        inputs, {"prefix_lengths_device": inputs.prefix_lengths.to(device)}
    )
    unpad_restore = restore[:length]
    inv_restore = torch.full((padded_length,), -1, dtype=torch.long, device=device)
    inv_restore[unpad_restore.long()] = torch.arange(length, device=device)
    local_start = rank * local_length
    local_inv = inv_restore[local_start : local_start + local_length]
    local_valid_mask = local_inv >= 0
    metadata = Qwen3NextMetadata(
        full_prefill_conv1d_meta=prepare_causal_conv1d_metadata(full_cu, device=device),
        full_prefill_cu_seqlens=full_cu,
        cp_restore_indices=restore,
        cp_local_extract_indices=local_inv[local_valid_mask],
        cp_local_valid_mask=local_valid_mask,
    )
    return wrapped, metadata, local_positions


def _worker(rank: int, store_path: str) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(minutes=10),
    )
    collective_torch._group_map[Group.DP_AND_TP] = dist.group.WORLD
    collective_torch._parallelism_config = SimpleNamespace(
        tp_size=_WORLD_SIZE, dp_size=1, world_size=_WORLD_SIZE
    )
    collective_torch._initialized = True
    try:
        torch.manual_seed(1729)
        torch.cuda.manual_seed_all(1729)
        module = _make_module(rank, device)
        cache = _make_cache(module.prefill_gdn, 16, device)
        full_hidden = torch.randn(2004, 128, device=device, dtype=torch.bfloat16)

        def run(
            prefix: int,
            length: int,
            block_map: torch.Tensor,
            positions: torch.Tensor | None = None,
            block_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            inputs, metadata, local_positions = _cp_inputs_and_metadata(
                prefix, length, rank, block_map, device, positions, block_ids
            )
            local_hidden = full_hidden[prefix + local_positions].contiguous()
            local_hidden[local_positions >= length] = 0
            with torch.no_grad():
                return module(local_hidden, None, cache, inputs, metadata)

        source_map = torch.tensor([[0, 1, 2, 3, 4, 5, -1, -1]], dtype=torch.int32)
        run(0, 1300, source_map)
        _copy_linear_block(module.prefill_gdn, cache, 5, 8)
        source = cache.kv_cache_base[5].clone()
        continuation_map = torch.tensor(
            [[-1, -1, -1, -1, -1, 8, 9, 7]], dtype=torch.int32
        )
        positions = torch.tensor([[1536, 1792, 2000]], dtype=torch.int64, device=device)
        destinations = torch.tensor([[10, 11, 12]], dtype=torch.int32, device=device)
        run(1300, 700, continuation_map, positions, destinations)
        torch.testing.assert_close(cache.kv_cache_base[5], source, rtol=0, atol=0)
        torch.testing.assert_close(
            cache.kv_cache_base[12], cache.kv_cache_base[7], rtol=0, atol=0
        )
        selected_1536 = cache.kv_cache_base[10].clone()
        selected_1792 = cache.kv_cache_base[11].clone()

        _copy_linear_block(module.prefill_gdn, cache, 5, 13)
        run(
            1300,
            236,
            torch.tensor([[-1, -1, -1, -1, -1, 13, -1, -1]], dtype=torch.int32),
        )
        torch.testing.assert_close(
            cache.kv_cache_base[10], cache.kv_cache_base[13], rtol=0, atol=0
        )
        _copy_linear_block(module.prefill_gdn, cache, 13, 15)
        _copy_linear_block(module.prefill_gdn, cache, 5, 13)
        run(
            1300,
            492,
            torch.tensor([[-1, -1, -1, -1, -1, 13, 14, -1]], dtype=torch.int32),
        )
        torch.testing.assert_close(
            cache.kv_cache_base[11], cache.kv_cache_base[14], rtol=0, atol=0
        )
        _copy_linear_block(module.prefill_gdn, cache, 14, 6)

        # Compare ordinary CP continuation from the independently computed
        # no-save endpoint against the selected snapshot.
        _copy_linear_block(module.prefill_gdn, cache, 10, 13)
        reference_map = torch.tensor(
            [[-1, -1, -1, -1, -1, 15, 14, 7]], dtype=torch.int32
        )
        reference = run(1536, 464, reference_map)
        reference_final = cache.kv_cache_base[7].clone()
        restore_map = torch.tensor([[-1, -1, -1, -1, -1, 13, 14, 7]], dtype=torch.int32)
        restored = run(1536, 464, restore_map)
        torch.testing.assert_close(restored, reference, rtol=0, atol=0)
        torch.testing.assert_close(
            cache.kv_cache_base[7], reference_final, rtol=0, atol=0
        )
        torch.testing.assert_close(
            cache.kv_cache_base[10], selected_1536, rtol=0, atol=0
        )
        torch.testing.assert_close(
            cache.kv_cache_base[11], selected_1792, rtol=0, atol=0
        )

        reference_1792_map = torch.tensor(
            [[-1, -1, -1, -1, -1, -1, 6, 7]], dtype=torch.int32
        )
        reference_1792 = run(1792, 208, reference_1792_map)
        reference_1792_final = cache.kv_cache_base[7].clone()
        restore_1792_map = torch.tensor(
            [[-1, -1, -1, -1, -1, -1, 11, 7]], dtype=torch.int32
        )
        restored_1792 = run(1792, 208, restore_1792_map)
        torch.testing.assert_close(restored_1792, reference_1792, rtol=0, atol=0)
        torch.testing.assert_close(
            cache.kv_cache_base[7], reference_1792_final, rtol=0, atol=0
        )
        torch.testing.assert_close(
            cache.kv_cache_base[11], selected_1792, rtol=0, atol=0
        )

        # The real CP processor pads a 701-token suffix to 704. Selected
        # positions must remain in the unpadded, full-suffix coordinate system.
        cache = _make_cache(module.prefill_gdn, 32, device)
        run(0, 1300, source_map)
        _copy_linear_block(module.prefill_gdn, cache, 5, 8)
        padded_source = cache.kv_cache_base[5].clone()
        padded_positions = torch.tensor(
            [[1536, 1792, 2001]], dtype=torch.int64, device=device
        )
        padded_destinations = torch.tensor(
            [[20, 21, 22]], dtype=torch.int32, device=device
        )
        run(
            1300,
            701,
            torch.tensor([[-1, -1, -1, -1, -1, 8, 9, 7]], dtype=torch.int32),
            padded_positions,
            padded_destinations,
        )
        torch.testing.assert_close(
            cache.kv_cache_base[5], padded_source, rtol=0, atol=0
        )
        torch.testing.assert_close(
            cache.kv_cache_base[22], cache.kv_cache_base[7], rtol=0, atol=0
        )
        _copy_linear_block(module.prefill_gdn, cache, 5, 13)
        run(
            1300,
            236,
            torch.tensor([[-1, -1, -1, -1, -1, 13, -1, -1]], dtype=torch.int32),
        )
        torch.testing.assert_close(
            cache.kv_cache_base[20], cache.kv_cache_base[13], rtol=0, atol=0
        )
        _copy_linear_block(module.prefill_gdn, cache, 13, 15)
        _copy_linear_block(module.prefill_gdn, cache, 5, 13)
        run(
            1300,
            492,
            torch.tensor([[-1, -1, -1, -1, -1, 13, 14, -1]], dtype=torch.int32),
        )
        torch.testing.assert_close(
            cache.kv_cache_base[21], cache.kv_cache_base[14], rtol=0, atol=0
        )
        _copy_linear_block(module.prefill_gdn, cache, 20, 13)
        padded_reference = run(
            1536,
            465,
            torch.tensor([[-1, -1, -1, -1, -1, 15, 17, 18]], dtype=torch.int32),
        )
        padded_final = cache.kv_cache_base[18].clone()
        padded_restored = run(
            1536,
            465,
            torch.tensor([[-1, -1, -1, -1, -1, 13, 17, 19]], dtype=torch.int32),
        )
        torch.testing.assert_close(padded_restored, padded_reference, rtol=0, atol=0)
        torch.testing.assert_close(
            cache.kv_cache_base[19], padded_final, rtol=0, atol=0
        )
        torch.testing.assert_close(
            cache.kv_cache_base[5], padded_source, rtol=0, atol=0
        )

        # G1100 starts in logical slot 4. CP must use the unscaled LINEAR
        # global slot geometry for its owner copy and all three full points.
        cache = _make_cache(module.prefill_gdn, 32, device)
        run(0, 1100, torch.tensor([[0, 1, 2, 3, 4, -1, -1, -1]], dtype=torch.int32))
        _copy_linear_block(module.prefill_gdn, cache, 4, 8)
        g1100_source = cache.kv_cache_base[4].clone()
        run(
            1100,
            900,
            torch.tensor([[-1, -1, -1, -1, 8, 9, 10, 7]], dtype=torch.int32),
            torch.tensor([[1280, 1536, 1792, 2000]], dtype=torch.int64, device=device),
            torch.tensor([[20, 21, 22, 23]], dtype=torch.int32, device=device),
        )
        torch.testing.assert_close(cache.kv_cache_base[4], g1100_source, rtol=0, atol=0)
        torch.testing.assert_close(
            cache.kv_cache_base[23], cache.kv_cache_base[7], rtol=0, atol=0
        )
        for endpoint, destination, runtime_map in (
            (1280, 20, [13, -1, -1, -1]),
            (1536, 21, [13, 14, -1, -1]),
            (1792, 22, [13, 14, 15, -1]),
        ):
            _copy_linear_block(module.prefill_gdn, cache, 4, 13)
            run(
                1100,
                endpoint - 1100,
                torch.tensor([[-1, -1, -1, -1] + runtime_map], dtype=torch.int32),
            )
            last_slot = (endpoint - 1) // 256
            torch.testing.assert_close(
                cache.kv_cache_base[destination],
                cache.kv_cache_base[runtime_map[last_slot - 4]],
                rtol=0,
                atol=0,
            )
            if endpoint == 1536:
                _copy_linear_block(module.prefill_gdn, cache, 14, 16)
        _copy_linear_block(module.prefill_gdn, cache, 21, 17)
        g1100_reference = run(
            1536,
            464,
            torch.tensor([[-1, -1, -1, -1, 13, 16, 18, 19]], dtype=torch.int32),
        )
        g1100_final = cache.kv_cache_base[19].clone()
        g1100_restored = run(
            1536,
            464,
            torch.tensor([[-1, -1, -1, -1, 13, 17, 18, 24]], dtype=torch.int32),
        )
        torch.testing.assert_close(g1100_restored, g1100_reference, rtol=0, atol=0)
        torch.testing.assert_close(cache.kv_cache_base[24], g1100_final, rtol=0, atol=0)
        torch.testing.assert_close(cache.kv_cache_base[4], g1100_source, rtol=0, atol=0)

        # Two requests with different G values share one actual NCCL all-gather.
        # Their save slots and restore paths must remain row-local.
        cache = _make_cache(module.prefill_gdn, 64, device)
        mixed_hidden = torch.randn(2, 2000, 128, device=device, dtype=torch.bfloat16)

        def run_mixed(
            prefixes: tuple[int, int],
            lengths: tuple[int, int],
            maps: torch.Tensor,
            positions: torch.Tensor | None = None,
            block_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            assert all(length % (2 * _WORLD_SIZE) == 0 for length in lengths)
            local_lengths = [length // _WORLD_SIZE for length in lengths]
            full_cu = torch.tensor(
                [0, lengths[0], sum(lengths)], dtype=torch.int32, device=device
            )
            offsets = [0, lengths[0]]
            gather_order = [
                offsets[b] + position
                for cp_rank in range(_WORLD_SIZE)
                for b in range(2)
                for position in _rank_positions(lengths[b], cp_rank)
            ]
            restore = torch.empty(sum(lengths), dtype=torch.int32, device=device)
            restore[torch.tensor(gather_order, device=device)] = torch.arange(
                sum(lengths), dtype=torch.int32, device=device
            )
            local_hidden = torch.cat(
                [
                    mixed_hidden[
                        b, [prefixes[b] + p for p in _rank_positions(lengths[b], rank)]
                    ]
                    for b in range(2)
                ],
                dim=0,
            ).contiguous()
            inputs = PyAttentionInputs()
            inputs.is_prefill = True
            inputs.input_lengths = torch.tensor(local_lengths, dtype=torch.int32)
            inputs.prefix_lengths = torch.tensor(prefixes, dtype=torch.int32)
            inputs.cu_seqlens_device = torch.tensor(
                [0, local_lengths[0], sum(local_lengths)],
                dtype=torch.int32,
                device=device,
            )
            inputs.kv_cache_kernel_block_id_device = maps.to(device)
            cp_info = PyContextParallelParams()
            cp_info.prefill_qkv_padding_mask = torch.ones(
                sum(lengths), dtype=torch.int32, device=device
            )
            cp_info.prefill_qkv_restore_indice = restore
            cp_info.prefill_actual_input_lengths_cpu = torch.tensor(
                lengths, dtype=torch.int32
            )
            inputs.context_parallel_info = cp_info
            if positions is not None:
                inputs.cache_save_positions = positions
            if block_ids is not None:
                inputs.cache_save_block_ids = block_ids
            wrapped = _AttnInputsWrapper(
                inputs, {"prefix_lengths_device": inputs.prefix_lengths.to(device)}
            )
            local_start = rank * sum(local_lengths)
            inv_restore = torch.empty(sum(lengths), dtype=torch.long, device=device)
            inv_restore[restore.long()] = torch.arange(sum(lengths), device=device)
            metadata = Qwen3NextMetadata(
                full_prefill_conv1d_meta=prepare_causal_conv1d_metadata(
                    full_cu, device=device
                ),
                full_prefill_cu_seqlens=full_cu,
                cp_restore_indices=restore,
                cp_local_extract_indices=inv_restore[
                    local_start : local_start + sum(local_lengths)
                ],
                cp_local_valid_mask=torch.ones(
                    sum(local_lengths), dtype=torch.bool, device=device
                ),
            )
            with torch.no_grad():
                return module(local_hidden, None, cache, wrapped, metadata)

        run_mixed(
            (0, 0),
            (1300, 1344),
            torch.tensor(
                [
                    [0, 1, 2, 3, 4, 5, -1, -1],
                    [16, 17, 18, 19, 20, 21, -1, -1],
                ],
                dtype=torch.int32,
            ),
        )
        _copy_linear_block(module.prefill_gdn, cache, 5, 8)
        _copy_linear_block(module.prefill_gdn, cache, 21, 24)
        mixed_sources = cache.kv_cache_base[[5, 21]].clone()
        run_mixed(
            (1300, 1344),
            (700, 656),
            torch.tensor(
                [
                    [-1, -1, -1, -1, -1, 8, 9, 7],
                    [-1, -1, -1, -1, -1, 24, 25, 23],
                ],
                dtype=torch.int32,
            ),
            torch.tensor(
                [[1536, 1792, 2000], [1536, 1792, 2000]],
                dtype=torch.int64,
                device=device,
            ),
            torch.tensor(
                [[40, 41, 42], [43, 44, 45]], dtype=torch.int32, device=device
            ),
        )
        torch.testing.assert_close(
            cache.kv_cache_base[[5, 21]], mixed_sources, rtol=0, atol=0
        )
        torch.testing.assert_close(
            cache.kv_cache_base[[42, 45]], cache.kv_cache_base[[7, 23]], rtol=0, atol=0
        )
        _copy_linear_block(module.prefill_gdn, cache, 5, 30)
        _copy_linear_block(module.prefill_gdn, cache, 21, 31)
        run_mixed(
            (1300, 1344),
            (236, 192),
            torch.tensor(
                [
                    [-1, -1, -1, -1, -1, 30, -1, -1],
                    [-1, -1, -1, -1, -1, 31, -1, -1],
                ],
                dtype=torch.int32,
            ),
        )
        torch.testing.assert_close(
            cache.kv_cache_base[[40, 43]], cache.kv_cache_base[[30, 31]], rtol=0, atol=0
        )
        _copy_linear_block(module.prefill_gdn, cache, 30, 34)
        _copy_linear_block(module.prefill_gdn, cache, 31, 35)
        _copy_linear_block(module.prefill_gdn, cache, 5, 30)
        _copy_linear_block(module.prefill_gdn, cache, 21, 31)
        run_mixed(
            (1300, 1344),
            (492, 448),
            torch.tensor(
                [
                    [-1, -1, -1, -1, -1, 30, 32, -1],
                    [-1, -1, -1, -1, -1, 31, 33, -1],
                ],
                dtype=torch.int32,
            ),
        )
        torch.testing.assert_close(
            cache.kv_cache_base[[41, 44]], cache.kv_cache_base[[32, 33]], rtol=0, atol=0
        )
        _copy_linear_block(module.prefill_gdn, cache, 40, 36)
        _copy_linear_block(module.prefill_gdn, cache, 43, 37)
        mixed_reference = run_mixed(
            (1536, 1536),
            (464, 464),
            torch.tensor(
                [
                    [-1, -1, -1, -1, -1, 34, 38, 39],
                    [-1, -1, -1, -1, -1, 35, 46, 47],
                ],
                dtype=torch.int32,
            ),
        )
        mixed_final = cache.kv_cache_base[[39, 47]].clone()
        mixed_restored = run_mixed(
            (1536, 1536),
            (464, 464),
            torch.tensor(
                [
                    [-1, -1, -1, -1, -1, 36, 38, 48],
                    [-1, -1, -1, -1, -1, 37, 46, 49],
                ],
                dtype=torch.int32,
            ),
        )
        torch.testing.assert_close(mixed_restored, mixed_reference, rtol=0, atol=0)
        torch.testing.assert_close(
            cache.kv_cache_base[[48, 49]], mixed_final, rtol=0, atol=0
        )
        torch.testing.assert_close(
            cache.kv_cache_base[[5, 21]], mixed_sources, rtol=0, atol=0
        )

    finally:
        dist.destroy_process_group()
        collective_torch._group_map.clear()
        collective_torch._parallelism_config = None
        collective_torch._initialized = False


class LinearSparseSaveCPTest(unittest.TestCase):
    def test_next_cp2_sparse_save_and_restore(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE:
            self.skipTest("two CUDA devices are required")
        if not dist.is_available() or not dist.is_nccl_available():
            self.skipTest("NCCL torch.distributed backend is required")
        with tempfile.TemporaryDirectory(prefix="linear_sparse_cp_") as temp_dir:
            mp.spawn(
                _worker,
                args=(os.path.join(temp_dir, "nccl_file_store"),),
                nprocs=_WORLD_SIZE,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
