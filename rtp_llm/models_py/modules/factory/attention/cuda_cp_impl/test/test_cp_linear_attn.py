"""
Unit tests for CP linear attention (GatedDeltaNet) per-layer all-gather path.

Tests:
  1. Metadata and forward: production CP metadata maps local zigzag tokens correctly
  2. Relay: real TP subgroups match non-CP output and cache states
"""

import contextlib
import logging
import math
import unittest
from typing import List
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.test.cp_test_utils import (
    build_cp_attn_inputs,
    build_shuffle_indices,
    compute_rank_positions,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    causal_conv1d_fn,
    prepare_causal_conv1d_metadata,
)
from rtp_llm.test.utils.port_util import PortManager

logging.basicConfig(level=logging.INFO, format="%(message)s")


class _AttnInputsWrapper:
    """Thin wrapper to override readonly pybind11 attributes for testing."""

    def __init__(self, wrapped, overrides: dict):
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


def _add_device_tensors(inputs, device: torch.device):
    """Wrap PyAttentionInputs with device tensors that C++ normally creates."""
    return _AttnInputsWrapper(
        inputs,
        {
            "prefix_lengths_device": inputs.prefix_lengths.to(device),
            "input_lengths_device": inputs.input_lengths.to(device),
            "kv_cache_kernel_block_id_device": (
                inputs.kv_cache_kernel_block_id.to(device)
                if inputs.kv_cache_kernel_block_id is not None
                else None
            ),
        },
    )


def _make_linear_module(device, parallelism_config):
    from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet
    from rtp_llm.ops import DataType, LinearAttentionConfig
    from rtp_llm.utils.model_weight import W

    num_heads, head_dim, hidden_size, conv_width = 2, 64, 128, 4
    config = LinearAttentionConfig()
    config.linear_num_key_heads = num_heads
    config.linear_num_value_heads = num_heads
    config.linear_key_head_dim = head_dim
    config.linear_value_head_dim = head_dim
    config.linear_conv_kernel_dim = conv_width
    config.ssm_state_dtype = DataType.TYPE_BF16
    config.conv_state_dtype = DataType.TYPE_BF16

    qkv_dim = head_dim * num_heads * 3
    torch.manual_seed(123)
    weights = {
        W.linear_attn_conv1d_w: torch.randn(
            qkv_dim, 1, conv_width, device=device, dtype=torch.bfloat16
        ),
        W.linear_attn_dt_b: torch.randn(num_heads, device=device, dtype=torch.bfloat16),
        W.linear_attn_alog: torch.randn(num_heads, device=device, dtype=torch.bfloat16),
        W.linear_attn_norm_w: torch.randn(
            head_dim, device=device, dtype=torch.bfloat16
        ),
        W.linear_attn_qkvz_w: torch.randn(
            hidden_size,
            qkv_dim + head_dim * num_heads,
            device=device,
            dtype=torch.bfloat16,
        ),
        W.linear_attn_qkvz_s: None,
        W.linear_attn_ba_w: torch.randn(
            hidden_size, num_heads * 2, device=device, dtype=torch.bfloat16
        ),
        W.linear_attn_out_w: torch.randn(
            head_dim * num_heads,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        ),
        W.linear_attn_out_s: None,
    }
    return Qwen3NextGatedDeltaNet(
        config, parallelism_config, weights, layernorm_eps=1e-6
    ).to(device)


def _make_nocp_inputs(new_length, prefix_length, device):
    from rtp_llm.ops.compute_ops import PyAttentionInputs

    total_length = prefix_length + new_length
    block_count = math.ceil(total_length / 64)
    block_ids = torch.arange(1, block_count + 1, dtype=torch.int32).view(1, -1)
    inputs = PyAttentionInputs()
    inputs.is_prefill = True
    inputs.cu_seqlens_device = torch.tensor(
        [0, new_length], dtype=torch.int32, device=device
    )
    inputs.input_lengths = torch.tensor([new_length], dtype=torch.int32)
    inputs.sequence_lengths = torch.tensor([total_length], dtype=torch.int32)
    inputs.prefix_lengths = torch.tensor([prefix_length], dtype=torch.int32)
    inputs.kv_cache_kernel_block_id = block_ids
    inputs.context_parallel_info = None
    return _add_device_tensors(inputs, device)


def _make_cp_metadata(inputs, cp_size, cp_rank, device):
    from types import SimpleNamespace

    from rtp_llm.models_py.model_desc.qwen3_next import (
        Qwen3NextMetadata,
        Qwen3NextModel,
    )

    metadata_builder = SimpleNamespace(
        parallelism_config=SimpleNamespace(tp_size=cp_size, tp_rank=cp_rank)
    )
    metadata = Qwen3NextModel._build_cp_linear_attn_metadata(
        metadata_builder, inputs, device
    )
    return Qwen3NextMetadata(
        full_prefill_conv1d_meta=metadata[0],
        full_prefill_cu_seqlens=metadata[1],
        cp_restore_indices=metadata[2],
        cp_local_extract_indices=metadata[3],
        cp_local_valid_mask=metadata[4],
    )


def _new_linear_cache(module, total_length, device):
    from rtp_llm.ops.compute_ops import LayerKVCache

    converter = module.prefill_gdn.linear_cache_converter
    row_elements = math.ceil(converter.block_size_bytes / 2)
    block_count = math.ceil(total_length / 64)
    base = torch.zeros(
        block_count + 1,
        row_elements,
        dtype=torch.bfloat16,
        device=device,
    )
    return LayerKVCache(base, 64)


def _run_cp_relay_case(module, device, cp_rank, new_length, prefix_length, with_cache):
    from rtp_llm.models_py.distributed.collective_torch import Group
    from rtp_llm.models_py.distributed.collective_torch import (
        all_reduce as distributed_all_reduce,
    )
    from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextMetadata

    total_length = prefix_length + new_length
    reference_cache = cp_cache = None
    if with_cache:
        initial_cache = _new_linear_cache(module, total_length, device)
        if prefix_length:
            prefix_hidden = torch.randn(
                prefix_length, 128, device=device, dtype=torch.bfloat16
            )
            prefix_inputs = _make_nocp_inputs(prefix_length, 0, device)
            prefix_cu = prefix_inputs.cu_seqlens_device
            prefix_meta = Qwen3NextMetadata(
                prefill_conv1d_meta=prepare_causal_conv1d_metadata(prefix_cu, device)
            )
            module(prefix_hidden, None, initial_cache, prefix_inputs, prefix_meta)
        reference_cache = _new_linear_cache(module, total_length, device)
        reference_cache.kv_cache_base.copy_(initial_cache.kv_cache_base)
        cp_cache = _new_linear_cache(module, total_length, device)
        cp_cache.kv_cache_base.copy_(initial_cache.kv_cache_base)

    full_hidden = torch.randn(new_length, 128, device=device, dtype=torch.bfloat16)
    reference_inputs = _make_nocp_inputs(new_length, prefix_length, device)
    reference_cu = reference_inputs.cu_seqlens_device
    reference_meta = Qwen3NextMetadata(
        prefill_conv1d_meta=prepare_causal_conv1d_metadata(reference_cu, device)
    )
    with torch.no_grad():
        reference_output = module(
            full_hidden, None, reference_cache, reference_inputs, reference_meta
        )

    cp_size = 2
    segment_alignment = 64
    padded_length = math.ceil(new_length / (2 * cp_size * segment_alignment)) * (
        2 * cp_size * segment_alignment
    )
    cp_chunk_length = padded_length // cp_size
    positions = build_shuffle_indices(
        [new_length], [cp_chunk_length], cp_size, cp_rank
    ).to(device)
    valid = positions < new_length
    local_hidden = full_hidden.new_zeros(cp_chunk_length, full_hidden.shape[1])
    local_hidden[valid] = full_hidden.index_select(0, positions[valid].long())
    cp_inputs = build_cp_attn_inputs(
        [total_length],
        [cp_chunk_length],
        cp_size,
        64,
        prefix_lengths=[prefix_length],
        block_id_start=1,
        device=device,
    )
    cp_inputs.context_parallel_info.prefill_shuffle_indices = positions.cpu()
    cp_inputs = _add_device_tensors(cp_inputs, device)
    assert (cp_inputs.context_parallel_info.prefill_qkv_padding_mask == 0).any()
    cp_meta = _make_cp_metadata(cp_inputs, cp_size, cp_rank, device)

    def out_of_place_all_reduce(tensor, group):
        assert group == Group.TP
        result = tensor.clone()
        return distributed_all_reduce(result, group=group, inplace=True)

    patch_target = "rtp_llm.models_py.model_desc.qwen3_next.all_reduce"
    reduce_context = (
        patch(patch_target, side_effect=out_of_place_all_reduce)
        if with_cache
        else contextlib.nullcontext()
    )
    with reduce_context, torch.no_grad():
        cp_output = module(local_hidden, None, cp_cache, cp_inputs, cp_meta)

    torch.testing.assert_close(
        cp_output[valid].float(),
        reference_output.index_select(0, positions[valid].long()).float(),
        rtol=1e-2,
        atol=1e-2,
    )
    if cp_cache is not None:
        reference_ssm = module.prefill_gdn._get_ssm_states(
            reference_cache.kv_cache_base
        )
        cp_ssm = module.prefill_gdn._get_ssm_states(cp_cache.kv_cache_base)
        torch.testing.assert_close(
            cp_ssm.float(), reference_ssm.float(), rtol=2e-2, atol=2e-2
        )
        reference_conv = module.prefill_gdn._get_conv_states(
            reference_cache.kv_cache_base
        )
        cp_conv = module.prefill_gdn._get_conv_states(cp_cache.kv_cache_base)
        torch.testing.assert_close(cp_conv, reference_conv, rtol=0, atol=0)


def _cp_relay_worker(world_rank, master_port):
    from rtp_llm.models_py.distributed.collective_torch import (
        destroy_distributed_environment,
        init_distributed_environment,
    )
    from rtp_llm.ops import CPRotateMethod, NcclCommConfig, ParallelismConfig

    device = torch.device(f"cuda:{world_rank}")
    torch.cuda.set_device(device)
    cp_rank = world_rank % 2
    parallelism_config = ParallelismConfig()
    parallelism_config.tp_size = 2
    parallelism_config.tp_rank = cp_rank
    parallelism_config.dp_size = 2
    parallelism_config.dp_rank = world_rank // 2
    parallelism_config.world_size = 4
    parallelism_config.world_rank = world_rank
    parallelism_config.local_world_size = 4
    parallelism_config.local_rank = world_rank
    parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
    init_distributed_environment(
        parallelism_config,
        NcclCommConfig(nccl_ip="127.0.0.1"),
        master_port,
        timeout=60,
    )
    try:
        module = _make_linear_module(device, parallelism_config)
        for new_length, prefix_length, with_cache in (
            (1, 0, False),
            (2, 0, False),
            (65, 0, True),
            (65, 64, True),
        ):
            _run_cp_relay_case(
                module,
                device,
                cp_rank,
                new_length,
                prefix_length,
                with_cache,
            )
    finally:
        destroy_distributed_environment()


class TestCPLinearAttnMetadata(unittest.TestCase):
    def test_cache_ends_support_short_and_aligned_sequences(self):
        from rtp_llm.models_py.model_desc.qwen3_next import _cp_cache_ends

        self.assertEqual(_cp_cache_ends(128, 2048, torch.device("cpu")).tolist(), [128])
        self.assertEqual(
            _cp_cache_ends(4096, 2048, torch.device("cpu")).tolist(), [2048, 4096]
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCPLinearAttnForward(unittest.TestCase):
    """Verify that CP GatedDeltaNet forward matches non-CP reference on a single GPU."""

    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def test_segmented_conv_matches_full_sequence(self):
        """Carrying the last width-1 inputs preserves causal-conv results."""
        width = 4
        dim = 256
        lengths = [64, 1, 2, 65]
        total = sum(lengths)
        x = torch.randn(total, dim, device=self.device, dtype=torch.bfloat16)
        weight = torch.randn(dim, width, device=self.device, dtype=torch.bfloat16)
        prefix_lengths = torch.zeros(1, dtype=torch.int32, device=self.device)

        full_cu = torch.tensor([0, total], dtype=torch.int32, device=self.device)
        full = causal_conv1d_fn(
            x=x.transpose(0, 1),
            weight=weight,
            bias=None,
            conv_states=None,
            query_start_loc=full_cu,
            block_map=None,
            prefix_lengths=prefix_lengths,
            seq_size_per_block=1,
            metadata=prepare_causal_conv1d_metadata(full_cu, self.device),
        ).transpose(0, 1)

        carry = x.new_zeros(width - 1, dim)
        outputs = []
        start = 0
        for length in lengths:
            raw_segment = x[start : start + length]
            segment_input = torch.cat([carry, raw_segment])
            segment_cu = torch.tensor(
                [0, segment_input.shape[0]], dtype=torch.int32, device=self.device
            )
            segment_output = causal_conv1d_fn(
                x=segment_input.transpose(0, 1),
                weight=weight,
                bias=None,
                conv_states=None,
                query_start_loc=segment_cu,
                block_map=None,
                prefix_lengths=prefix_lengths,
                seq_size_per_block=1,
                metadata=prepare_causal_conv1d_metadata(segment_cu, self.device),
            ).transpose(0, 1)
            outputs.append(segment_output[width - 1 :])
            carry = segment_input[-(width - 1) :].contiguous()
            start += length

        torch.testing.assert_close(torch.cat(outputs), full, rtol=0, atol=0)

    def _run_cp_vs_nocp(
        self,
        sequence_lengths: List[int],
        cp_size: int = 2,
        cp_rank: int = 0,
        num_k_heads: int = 4,
        num_v_heads: int = 4,
        head_k_dim: int = 64,
        head_v_dim: int = 64,
        hidden_size: int = 256,
        conv_kernel_dim: int = 4,
    ):
        """Test that CP linear attn forward matches non-CP on the same data."""
        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextGatedDeltaNet,
            Qwen3NextMetadata,
        )
        from rtp_llm.models_py.triton_kernels.causal_conv1d import (
            prepare_causal_conv1d_metadata,
        )
        from rtp_llm.ops import DataType, LinearAttentionConfig, ParallelismConfig
        from rtp_llm.ops.compute_ops import PyAttentionInputs

        assert all(sl % (cp_size * 2) == 0 for sl in sequence_lengths)
        cp_chunk_lengths = [sl // cp_size for sl in sequence_lengths]
        total_tokens = sum(sequence_lengths)
        batch_size = len(sequence_lengths)

        linear_cfg = LinearAttentionConfig()
        linear_cfg.linear_num_key_heads = num_k_heads
        linear_cfg.linear_num_value_heads = num_v_heads
        linear_cfg.linear_key_head_dim = head_k_dim
        linear_cfg.linear_value_head_dim = head_v_dim
        linear_cfg.linear_conv_kernel_dim = conv_kernel_dim
        linear_cfg.ssm_state_dtype = DataType.TYPE_BF16
        linear_cfg.conv_state_dtype = DataType.TYPE_BF16

        par_cfg = ParallelismConfig()
        par_cfg.tp_size = 1
        par_cfg.tp_rank = 0

        qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
        z_dim = head_v_dim * num_v_heads
        qkvz_dim = qkv_dim + z_dim
        ba_dim = num_v_heads * 2

        torch.manual_seed(123)
        conv_w = torch.randn(
            qkv_dim, 1, conv_kernel_dim, device=self.device, dtype=torch.bfloat16
        )
        dt_b = torch.randn(num_v_heads, device=self.device, dtype=torch.bfloat16)
        alog = torch.randn(num_v_heads, device=self.device, dtype=torch.bfloat16)
        norm_w = torch.randn(head_v_dim, device=self.device, dtype=torch.bfloat16)

        from rtp_llm.utils.model_weight import W

        qkvz_w = torch.randn(
            hidden_size, qkvz_dim, device=self.device, dtype=torch.bfloat16
        )
        ba_w = torch.randn(
            hidden_size, ba_dim, device=self.device, dtype=torch.bfloat16
        )
        out_w = torch.randn(
            num_v_heads * head_v_dim,
            hidden_size,
            device=self.device,
            dtype=torch.bfloat16,
        )

        weights = {
            W.linear_attn_conv1d_w: conv_w,
            W.linear_attn_dt_b: dt_b,
            W.linear_attn_alog: alog,
            W.linear_attn_norm_w: norm_w,
            W.linear_attn_qkvz_w: qkvz_w,
            W.linear_attn_qkvz_s: None,
            W.linear_attn_ba_w: ba_w,
            W.linear_attn_out_w: out_w,
            W.linear_attn_out_s: None,
        }

        module = Qwen3NextGatedDeltaNet(
            linear_cfg, par_cfg, weights, layernorm_eps=1e-6
        ).to(self.device)

        full_hidden = torch.randn(
            total_tokens, hidden_size, device=self.device, dtype=torch.bfloat16
        )

        # --- Non-CP reference ---
        full_cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        for i, sl in enumerate(sequence_lengths):
            full_cu[i + 1] = full_cu[i] + sl

        nocp_inputs = PyAttentionInputs()
        nocp_inputs.is_prefill = True
        nocp_inputs.cu_seqlens_device = full_cu
        nocp_inputs.input_lengths = torch.tensor(
            sequence_lengths, dtype=torch.int32, device="cpu"
        )
        nocp_inputs.prefix_lengths = torch.zeros(
            batch_size, dtype=torch.int32, device="cpu"
        )
        nocp_inputs.context_parallel_info = None
        nocp_inputs = _add_device_tensors(nocp_inputs, self.device)

        nocp_conv_meta = prepare_causal_conv1d_metadata(
            query_start_loc=full_cu, device=self.device
        )
        nocp_meta = Qwen3NextMetadata(prefill_conv1d_meta=nocp_conv_meta)

        with torch.no_grad():
            ref_output = module(full_hidden, None, None, nocp_inputs, nocp_meta)

        # --- CP path (mocked all_gather) ---
        all_rank_pos = compute_rank_positions(sequence_lengths, cp_size)
        rank_positions = all_rank_pos[cp_rank]
        rank_idx = torch.tensor(rank_positions, device=self.device)
        local_hidden = full_hidden[rank_idx].contiguous()

        cp_attn_inputs = build_cp_attn_inputs(
            sequence_lengths,
            cp_chunk_lengths,
            cp_size,
            tokens_per_block=16,
            device=self.device,
        )
        cp_attn_inputs = _add_device_tensors(cp_attn_inputs, self.device)

        all_rank_packed: List[torch.Tensor] = []
        with torch.no_grad():
            for r in range(cp_size):
                r_pos = torch.tensor(all_rank_pos[r], device=self.device)
                r_hidden = full_hidden[r_pos]
                # Use the projection helper so the test runs under both the
                # fused (single-GEMM) and 2-GEMM dispatch paths.
                r_qkvz, r_ba = module._input_project(r_hidden)
                r_mixed_qkv, r_z, r_b, r_a = module.fix_query_key_value_ordering(
                    r_qkvz, r_ba
                )
                all_rank_packed.append(torch.cat([r_mixed_qkv, r_b, r_a], dim=-1))

        cp_meta = _make_cp_metadata(
            cp_attn_inputs, cp_size, cp_rank, self.device
        )

        def mock_ag(tensor, group=None):
            return torch.cat(all_rank_packed, dim=0)

        AG_MODULE = "rtp_llm.models_py.model_desc.qwen3_next"
        with patch(f"{AG_MODULE}.all_gather", side_effect=mock_ag):
            with torch.no_grad():
                cp_output = module(local_hidden, None, None, cp_attn_inputs, cp_meta)

        ref_local = ref_output[rank_idx]
        diff = (cp_output.float() - ref_local.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        logging.info(f"max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
        self.assertTrue(
            torch.allclose(cp_output.float(), ref_local.float(), rtol=1e-2, atol=1e-2),
            f"CP vs non-CP mismatch: max_diff={max_diff}, mean_diff={mean_diff}",
        )

    def test_single_seq_cp2(self):
        self._run_cp_vs_nocp(sequence_lengths=[32], cp_size=2, cp_rank=0)

    def test_single_seq_cp2_rank1(self):
        self._run_cp_vs_nocp(sequence_lengths=[32], cp_size=2, cp_rank=1)

    def test_multi_batch_cp2(self):
        self._run_cp_vs_nocp(sequence_lengths=[16, 32], cp_size=2, cp_rank=0)

    @unittest.skipUnless(torch.cuda.device_count() >= 4, "four CUDA devices required")
    def test_cp2_relay_short_tail_prefix_and_cache(self):
        ports, locks = PortManager().get_consecutive_ports(1)
        try:
            mp.spawn(_cp_relay_worker, args=(ports[0],), nprocs=4, join=True)
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
