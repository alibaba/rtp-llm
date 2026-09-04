"""Two-GPU smoke test for NCCL and user-buffer linear CP relay."""

import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed import collective_torch, user_buffers
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.model_desc.qwen3_next import (
    Qwen3NextGatedDeltaNet,
    Qwen3NextMetadata,
    fused_gdn_gating,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.linear_attn_utils import (
    ZigzagCPPlan,
    get_segment_valid_lengths,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.test.cp_test_utils import (
    build_cp_attn_inputs,
    compute_rank_positions,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    causal_conv1d_fn,
    prepare_causal_conv1d_metadata,
)

_WORLD_SIZE = 2


def _install_world_as_tp_group() -> None:
    collective_torch._group_map.clear()
    collective_torch._group_map[Group.DP_AND_TP] = dist.group.WORLD
    collective_torch._parallelism_config = SimpleNamespace(
        tp_size=_WORLD_SIZE,
        dp_size=1,
        world_size=_WORLD_SIZE,
    )
    collective_torch._initialized = True


def _clear_collective_state() -> None:
    collective_torch._group_map.clear()
    collective_torch._parallelism_config = None
    collective_torch._initialized = False


class _RealRelayHarness(Qwen3NextGatedDeltaNet):
    def __init__(self, rank, conv_weights, conv_cache, ssm_cache, use_user_buffers):
        torch.nn.Module.__init__(self)
        self.parallelism_config = SimpleNamespace(
            tp_size=_WORLD_SIZE,
            tp_rank=rank,
            use_ub_comm=use_user_buffers,
            prefill_cp_config=SimpleNamespace(kv_cache_sharded=False),
        )
        self.prefill_gdn = SimpleNamespace(
            alog=torch.zeros(1, dtype=torch.bfloat16, device=conv_weights.device),
            dt_bias=torch.zeros(1, dtype=torch.bfloat16, device=conv_weights.device),
            local_num_k_heads=1,
            local_num_v_heads=1,
            head_k_dim=64,
            head_v_dim=64,
            linear_conv_kernel_dim=4,
            conv_weights=conv_weights,
            _get_conv_states=lambda _: conv_cache,
            _get_ssm_states=lambda _: ssm_cache,
        )
        self.head_v_dim = 64
        self.local_num_v_heads = 1
        self.norm = lambda attn_out, _: attn_out
        self.out_proj = torch.nn.Identity()


def _run_real_prefix_reuse(rank: int, use_user_buffers: bool) -> None:
    device = torch.device(f"cuda:{rank}")
    prefix_tokens = 64
    new_tokens = 256
    block_size = 64
    state_len = 3
    qkv_dim = 3 * 64

    torch.manual_seed(20260902)
    conv_weights = torch.randn(qkv_dim, 4, dtype=torch.bfloat16, device=device)
    full_mixed_qkv = torch.randn(
        prefix_tokens + new_tokens,
        qkv_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    full_b = torch.randn(new_tokens, 1, dtype=torch.bfloat16, device=device)
    full_a = torch.randn_like(full_b)
    prefix_ssm_state = torch.randn(1, 64, 64, dtype=torch.bfloat16, device=device)

    conv_cache = torch.zeros(6, state_len, qkv_dim, dtype=torch.bfloat16, device=device)
    ssm_cache = torch.zeros(6, 1, 64, 64, dtype=torch.bfloat16, device=device)
    conv_cache[1].copy_(full_mixed_qkv[prefix_tokens - state_len : prefix_tokens])
    ssm_cache[1].copy_(prefix_ssm_state)
    cached_prefix_conv = conv_cache[1].clone()
    cached_prefix_ssm = ssm_cache[1].clone()
    harness = _RealRelayHarness(
        rank, conv_weights, conv_cache, ssm_cache, use_user_buffers
    ).to(device)

    full_cu = torch.tensor(
        [0, prefix_tokens + new_tokens], dtype=torch.int32, device=device
    )
    full_conv_meta = prepare_causal_conv1d_metadata(full_cu, device)
    with torch.no_grad():
        expected_conv = causal_conv1d_fn(
            x=full_mixed_qkv.transpose(0, 1),
            weight=conv_weights,
            bias=None,
            conv_states=None,
            query_start_loc=full_cu,
            block_map=None,
            prefix_lengths=torch.zeros(1, dtype=torch.int32, device=device),
            seq_size_per_block=1,
            metadata=full_conv_meta,
        ).transpose(0, 1)[prefix_tokens:]
        full_g, full_beta = fused_gdn_gating(
            harness.prefill_gdn.alog,
            full_a.contiguous(),
            full_b.contiguous(),
            harness.prefill_gdn.dt_bias,
        )
        expected_output, expected_chunks, expected_final = (
            harness._run_linear_cp_gdn_segment(
                expected_conv,
                full_g,
                full_beta,
                prefix_ssm_state.float().unsqueeze(0),
            )
        )

    rank_positions = compute_rank_positions([new_tokens], _WORLD_SIZE)[rank]
    rank_indices = torch.tensor(rank_positions, dtype=torch.long, device=device)
    new_mixed_qkv = full_mixed_qkv[prefix_tokens:]
    local_conv_cu = torch.tensor([0, 67, 134], dtype=torch.int32, device=device)
    cp_inputs = build_cp_attn_inputs(
        sequence_lengths=[prefix_tokens + new_tokens],
        cp_chunk_lengths=[new_tokens // _WORLD_SIZE],
        cp_size=_WORLD_SIZE,
        tokens_per_block=block_size,
        prefix_lengths=[prefix_tokens],
        device=device,
    )
    block_ids = torch.arange(1, 6, dtype=torch.int32, device=device).unsqueeze(0)
    attention_inputs = SimpleNamespace(
        input_lengths=cp_inputs.input_lengths,
        prefix_lengths=cp_inputs.prefix_lengths,
        context_parallel_info=cp_inputs.context_parallel_info,
        kv_cache_kernel_block_id=block_ids.cpu(),
        kv_cache_kernel_block_id_device=block_ids,
        cache_store_inputs=None,
        cache_store_writer=None,
        is_cuda_graph=False,
    )
    cp_plan = ZigzagCPPlan(cp_size=_WORLD_SIZE, cp_rank=rank)
    attn_meta = Qwen3NextMetadata(
        cp_plan=cp_plan,
        cp_segment_valid_lengths=get_segment_valid_lengths(
            new_tokens, new_tokens // (2 * _WORLD_SIZE), _WORLD_SIZE
        ),
        cp_local_conv1d_meta=prepare_causal_conv1d_metadata(local_conv_cu, device),
        cp_local_conv_cu_seqlens=local_conv_cu,
        cp_local_conv_prefix_lengths=torch.zeros(2, dtype=torch.int32, device=device),
        cp_local_valid_mask=torch.ones(
            new_tokens // _WORLD_SIZE, dtype=torch.bool, device=device
        ),
    )
    kv_cache = SimpleNamespace(
        kv_cache_base=torch.empty(1, 1, dtype=torch.bfloat16, device=device),
        seq_size_per_block=block_size,
    )

    with torch.no_grad():
        actual_output = harness._forward_cp_prefill(
            new_mixed_qkv[rank_indices].contiguous(),
            torch.zeros(new_tokens // _WORLD_SIZE, 64, device=device),
            full_b[rank_indices].contiguous(),
            full_a[rank_indices].contiguous(),
            attention_inputs,
            kv_cache,
            attn_meta,
        )

    torch.testing.assert_close(
        actual_output,
        expected_output.reshape(new_tokens, 64)[rank_indices],
        rtol=2e-2,
        atol=2e-2,
    )
    block_ends = torch.arange(block_size, new_tokens + 1, block_size, device=device)
    tail_offsets = torch.arange(-state_len, 0, device=device)
    expected_conv_states = new_mixed_qkv[block_ends[:, None] + tail_offsets[None, :]]
    expected_ssm_states = torch.cat([expected_chunks[0, 1:], expected_final]).to(
        torch.bfloat16
    )
    torch.testing.assert_close(conv_cache[2:6], expected_conv_states)
    torch.testing.assert_close(
        ssm_cache[2:6], expected_ssm_states, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(conv_cache[1], cached_prefix_conv)
    torch.testing.assert_close(ssm_cache[1], cached_prefix_ssm)


def _relay_worker(rank: int, world_size: int, init_file: str) -> None:
    if world_size != _WORLD_SIZE:
        raise AssertionError(f"expected world_size={_WORLD_SIZE}, got {world_size}")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=120),
        )
        _install_world_as_tp_group()
        _run_real_prefix_reuse(rank, use_user_buffers=False)
        dist.barrier()

        user_buffers.init_user_buffers_communicator(
            dist.group.WORLD, rank, world_size, buffer_size=1 << 20
        )
        _run_real_prefix_reuse(rank, use_user_buffers=True)
        torch.cuda.synchronize(device)
        dist.barrier()
    finally:
        try:
            user_buffers.destroy_user_buffers_communicator()
        except RuntimeError:
            pass
        if dist.is_initialized():
            dist.destroy_process_group()
        _clear_collective_state()


class CPRelayDistributedTest(unittest.TestCase):
    def test_two_rank_nccl_and_user_buffer_relay(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE:
            self.skipTest("two CUDA devices are required")
        if not dist.is_available() or not dist.is_nccl_available():
            self.skipTest("NCCL torch.distributed backend is required")

        with tempfile.TemporaryDirectory(prefix="qwen35_cp_relay_") as temp_dir:
            mp.spawn(
                _relay_worker,
                args=(_WORLD_SIZE, os.path.join(temp_dir, "nccl_store")),
                nprocs=_WORLD_SIZE,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
