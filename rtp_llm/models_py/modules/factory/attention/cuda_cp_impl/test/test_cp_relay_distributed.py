"""Two-rank integration test for the Qwen3.5 linear-attention CP relay."""

import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed import collective_torch, user_buffers
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.model_desc.qwen3_next import (
    Qwen3NextGatedDeltaNet,
    Qwen3NextMetadata,
    Qwen3NextModel,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.linear_attn_utils import (
    ZigzagCPPlan,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.test.cp_test_utils import (
    build_cp_attn_inputs,
    compute_rank_positions,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    prepare_causal_conv1d_metadata,
)
from rtp_llm.ops import (
    CPRotateMethod,
    DataType,
    LinearAttentionConfig,
    ParallelismConfig,
)
from rtp_llm.utils.model_weight import W

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


def _run_user_buffer_exchange(rank: int) -> None:
    communicator = user_buffers.get_user_buffers_communicator()
    peer = 1 - rank
    source = torch.empty(1 << 16, dtype=torch.float32, device=f"cuda:{rank}")
    destination = torch.empty_like(source)
    communication_stream = torch.cuda.Stream(device=rank)
    with torch.cuda.stream(communication_stream):
        source.fill_(rank + 1)
        destination.fill_(-1)
        if not communicator.send_recv(
            source,
            dst=peer,
            recv_tensor=destination,
            src=peer,
        ):
            raise AssertionError("UserBuffer exchange unexpectedly exceeded capacity")
    communication_stream.synchronize()
    torch.testing.assert_close(destination, torch.full_like(destination, peer + 1))


def _run_relay(rank: int, use_user_buffers: bool) -> None:
    device = torch.device(f"cuda:{rank}")
    local_qkv = torch.arange(4, dtype=torch.bfloat16, device=device).unsqueeze(1)
    z = torch.zeros_like(local_qkv)
    g = torch.zeros(1, 4, 1, dtype=torch.float32, device=device)
    beta = torch.zeros(1, 4, 1, dtype=torch.bfloat16, device=device)
    initial_states = []

    class _RelayHarness:
        _linear_cp_relay_logged = True

    harness = _RelayHarness()
    harness.parallelism_config = SimpleNamespace(use_ub_comm=use_user_buffers)
    harness.prefill_gdn = SimpleNamespace(
        alog=torch.zeros(1, dtype=torch.bfloat16, device=device),
        dt_bias=torch.zeros(1, dtype=torch.bfloat16, device=device),
        local_num_v_heads=1,
        head_k_dim=1,
        head_v_dim=1,
    )
    harness.head_v_dim = 1
    harness.local_num_v_heads = 1
    harness.norm = lambda attn_out, _: attn_out
    harness.out_proj = lambda attn_out: attn_out

    def run_segment(mixed_qkv, segment_g, segment_beta, initial_state):
        del segment_g, segment_beta
        initial_states.append(float(initial_state.item()))
        final_state = initial_state + 1
        segment_out = torch.full(
            (mixed_qkv.shape[0], 1, 1),
            float(final_state.item()),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )
        chunk_states = torch.empty(
            1, 1, 1, 1, 1, dtype=torch.float32, device=mixed_qkv.device
        )
        return segment_out, chunk_states, final_state

    harness._run_linear_cp_gdn_segment = run_segment
    with patch(
        "rtp_llm.models_py.model_desc.qwen3_next.fused_gdn_gating",
        return_value=(g, beta),
    ):
        output = Qwen3NextGatedDeltaNet._forward_linear_cp_relay(
            harness,
            local_qkv,
            z,
            torch.empty(4, 1, device=device),
            torch.empty(4, 1, device=device),
            prefix_ssm_state=None,
            kv_cache_tensor=None,
            cache_block_ends=None,
            cache_block_ids=None,
            cp_plan=ZigzagCPPlan(_WORLD_SIZE, rank),
            segment_valid_lengths=(2,) * (2 * _WORLD_SIZE),
        )

    expected_states = [0.0, 2.0] if rank == 0 else [1.0]
    if initial_states != expected_states:
        raise AssertionError(
            f"rank {rank} initial states {initial_states} != {expected_states}"
        )
    expected_output = (
        torch.tensor([1, 1, 3, 3], dtype=torch.bfloat16, device=device)
        if rank == 0
        else torch.full((4,), 2, dtype=torch.bfloat16, device=device)
    ).unsqueeze(1)
    torch.testing.assert_close(output, expected_output)


def _make_linear_attention_module(
    rank: int, device: torch.device
) -> tuple[Qwen3NextGatedDeltaNet, LinearAttentionConfig, ParallelismConfig]:
    head_dim = 64
    hidden_size = 64
    qkv_dim = 3 * head_dim

    linear_config = LinearAttentionConfig()
    linear_config.linear_num_key_heads = 1
    linear_config.linear_num_value_heads = 1
    linear_config.linear_key_head_dim = head_dim
    linear_config.linear_value_head_dim = head_dim
    linear_config.linear_conv_kernel_dim = 4
    linear_config.ssm_state_dtype = DataType.TYPE_BF16
    linear_config.conv_state_dtype = DataType.TYPE_BF16

    parallelism_config = ParallelismConfig()
    parallelism_config.tp_size = _WORLD_SIZE
    parallelism_config.tp_rank = rank
    parallelism_config.dp_size = 1
    parallelism_config.world_size = _WORLD_SIZE
    parallelism_config.world_rank = rank
    parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER

    torch.manual_seed(20260901)
    def randn(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device=device)

    weights = {
        W.linear_attn_conv1d_w: randn(qkv_dim, 1, 4),
        W.linear_attn_dt_b: randn(1),
        W.linear_attn_alog: randn(1),
        W.linear_attn_norm_w: randn(head_dim),
        W.linear_attn_qkvz_w: randn(hidden_size, 4 * head_dim),
        W.linear_attn_qkvz_s: None,
        W.linear_attn_ba_w: randn(hidden_size, 2),
        W.linear_attn_out_w: randn(head_dim, hidden_size),
        W.linear_attn_out_s: None,
    }
    module = Qwen3NextGatedDeltaNet(
        linear_config, parallelism_config, weights, layernorm_eps=1e-6
    ).to(device)
    return module, linear_config, parallelism_config


def _make_prefill_inputs(
    new_tokens: int,
    prefix_tokens: int,
    block_ids: torch.Tensor,
    device: torch.device,
    context_parallel_info=None,
):
    return SimpleNamespace(
        is_prefill=True,
        is_target_verify=False,
        is_cuda_graph=False,
        cu_seqlens=torch.tensor([0, new_tokens], dtype=torch.int32, device=device),
        input_lengths=torch.tensor([new_tokens], dtype=torch.int32),
        prefix_lengths=torch.tensor([prefix_tokens], dtype=torch.int32),
        prefix_lengths_d=torch.tensor([prefix_tokens], dtype=torch.int32, device=device),
        kv_cache_block_id_host=block_ids.cpu(),
        kv_cache_kernel_block_id_host=block_ids.cpu(),
        kv_cache_kernel_block_id_device=block_ids,
        kv_cache_kernel_block_id_device_by_group=None,
        cache_store_inputs=None,
        context_parallel_info=context_parallel_info,
    )


def _run_prefix_reuse_model(rank: int) -> None:
    device = torch.device(f"cuda:{rank}")
    module, linear_config, parallelism_config = _make_linear_attention_module(
        rank, device
    )
    prefix_tokens = 64
    new_tokens = 256
    tokens_per_block = 64
    block_ids = torch.arange(1, 6, dtype=torch.int32, device=device).unsqueeze(0)

    converter = module.prefill_gdn.linear_cache_converter
    cache_elements = converter.block_size_bytes // converter.dtype_size_bytes(
        torch.bfloat16
    )

    def make_cache(base):
        return SimpleNamespace(
            kv_cache_base=base,
            seq_size_per_block=tokens_per_block,
        )

    seed_cache = make_cache(
        torch.zeros(6, cache_elements, dtype=torch.bfloat16, device=device)
    )
    seed_ssm_states = module.prefill_gdn._get_ssm_states(seed_cache.kv_cache_base)
    seed_conv_states = module.prefill_gdn._get_conv_states(seed_cache.kv_cache_base)
    torch.manual_seed(20260902)
    prefix_ssm_state = torch.randn_like(seed_ssm_states[1])
    prefix_conv_state = torch.randn_like(seed_conv_states[1])
    seed_ssm_states[1].copy_(prefix_ssm_state)
    seed_conv_states[1].copy_(prefix_conv_state)
    new_hidden = torch.randn(new_tokens, 64, dtype=torch.bfloat16, device=device)
    cp_cache = make_cache(seed_cache.kv_cache_base.clone())
    reference_cache = make_cache(seed_cache.kv_cache_base.clone())

    reference_inputs = _make_prefill_inputs(
        new_tokens, prefix_tokens, block_ids, device
    )
    reference_meta = Qwen3NextMetadata(
        prefill_conv1d_meta=prepare_causal_conv1d_metadata(
            reference_inputs.cu_seqlens, device
        )
    )
    with torch.no_grad(), patch(
        "rtp_llm.models_py.model_desc.qwen3_next.compute_ops.write_cache_store"
    ):
        reference_output = module(
            new_hidden, None, reference_cache, reference_inputs, reference_meta
        )

    rank_positions = compute_rank_positions([new_tokens], _WORLD_SIZE)[rank]
    local_indices = torch.tensor(rank_positions, dtype=torch.long, device=device)
    local_hidden = new_hidden[local_indices].contiguous()
    built_cp_inputs = build_cp_attn_inputs(
        sequence_lengths=[prefix_tokens + new_tokens],
        cp_chunk_lengths=[new_tokens // _WORLD_SIZE],
        cp_size=_WORLD_SIZE,
        tokens_per_block=tokens_per_block,
        prefix_lengths=[prefix_tokens],
        device=device,
    )
    cp_inputs = _make_prefill_inputs(
        new_tokens // _WORLD_SIZE,
        prefix_tokens,
        block_ids,
        device,
        context_parallel_info=built_cp_inputs.context_parallel_info,
    )

    captured_metadata = []

    def decoder_layer(
        hidden_states,
        residual,
        fmha_impl,
        kv_cache,
        attention_inputs,
        attn_meta,
    ):
        captured_metadata.append(attn_meta)
        return (
            module(hidden_states, fmha_impl, kv_cache, attention_inputs, attn_meta),
            residual,
        )

    def build_cp_metadata(attention_inputs, metadata_device):
        return Qwen3NextModel._build_cp_linear_attn_metadata(
            model, attention_inputs, metadata_device
        )

    model = SimpleNamespace(
        embed_tokens=lambda _: local_hidden,
        config=SimpleNamespace(linear_attention_config=linear_config),
        parallelism_config=parallelism_config,
        layers=[decoder_layer],
        kv_cache=SimpleNamespace(get_layer_cache=lambda _: cp_cache),
        norm=lambda hidden_states, residual: (hidden_states, residual),
    )
    model._build_cp_linear_attn_metadata = build_cp_metadata
    model_inputs = SimpleNamespace(
        input_ids=torch.arange(local_hidden.shape[0], device=device),
        attention_inputs=cp_inputs,
    )
    with torch.no_grad(), patch.object(
        module,
        "_forward_linear_cp_conv",
        wraps=module._forward_linear_cp_conv,
    ) as run_conv, patch.object(
        module,
        "_run_linear_cp_gdn_segment",
        wraps=module._run_linear_cp_gdn_segment,
    ) as run_segment:
        cp_output = Qwen3NextModel.forward(
            model, model_inputs, SimpleNamespace(fmha_params=None)
        ).hidden_states

    if len(captured_metadata) != 1:
        raise AssertionError(f"rank {rank} did not receive exactly one metadata object")
    cp_metadata = captured_metadata[0]
    torch.testing.assert_close(
        cp_metadata.cp_local_extract_indices,
        local_indices,
    )
    if not bool(cp_metadata.cp_local_valid_mask.all().item()):
        raise AssertionError(f"rank {rank} unexpectedly received padded CP tokens")
    torch.testing.assert_close(
        cp_output,
        reference_output[local_indices],
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(run_conv.call_args.args[1], prefix_conv_state)

    if rank == 0:
        torch.testing.assert_close(
            run_segment.call_args_list[0].args[3],
            prefix_ssm_state.float().unsqueeze(0),
        )

    cp_ssm_states = module.prefill_gdn._get_ssm_states(cp_cache.kv_cache_base)
    cp_conv_states = module.prefill_gdn._get_conv_states(cp_cache.kv_cache_base)
    reference_ssm_states = module.prefill_gdn._get_ssm_states(
        reference_cache.kv_cache_base
    )
    reference_conv_states = module.prefill_gdn._get_conv_states(
        reference_cache.kv_cache_base
    )
    torch.testing.assert_close(cp_ssm_states[1], prefix_ssm_state)
    torch.testing.assert_close(cp_conv_states[1], prefix_conv_state)
    # The general prefill writer does not materialize the first post-prefix
    # intermediate SSM boundary; later boundaries remain a valid oracle.
    torch.testing.assert_close(
        cp_ssm_states[3:6], reference_ssm_states[3:6], rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        cp_conv_states[2:6], reference_conv_states[2:6], rtol=1e-2, atol=1e-2
    )
    if torch.count_nonzero(cp_ssm_states[2:6]).item() == 0:
        raise AssertionError(f"rank {rank} did not write new SSM cache states")
    if torch.count_nonzero(cp_conv_states[2:6]).item() == 0:
        raise AssertionError(f"rank {rank} did not write new convolution cache states")


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

        _run_relay(rank, use_user_buffers=False)
        dist.barrier()
        _run_prefix_reuse_model(rank)
        dist.barrier()

        user_buffers.init_user_buffers_communicator(
            dist.group.WORLD,
            rank,
            world_size,
            buffer_size=1 << 20,
        )
        _run_user_buffer_exchange(rank)
        dist.barrier()
        _run_relay(rank, use_user_buffers=True)
        torch.cuda.synchronize(device)
        dist.barrier()
    finally:
        try:
            user_buffers.destroy_user_buffers_communicator()
        except (NameError, RuntimeError):
            pass
        if dist.is_initialized():
            dist.destroy_process_group()
        _clear_collective_state()


class CPRelayDistributedTest(unittest.TestCase):
    def test_user_buffer_exchange_launches_before_stream_waits(self) -> None:
        communicator = object.__new__(user_buffers.UserBufferCommunicator)
        communicator.local_rank = 0
        communicator.per_rank_buffer_size = 1024
        communicator._ub_handle = 1
        communicator._communicator_ptr = 2
        communicator._rank_offsets = {0: 0, 1: 512}
        send_stream = mock.Mock(cuda_stream=11)
        recv_stream = mock.Mock(cuda_stream=12)
        current_stream = mock.Mock()
        communicator._send_streams = {1: send_stream}
        communicator._recv_stream = recv_stream
        communicator.cleanup = mock.Mock()
        timeline = []
        send_stream.wait_stream.side_effect = lambda _: timeline.append("send-ready")
        recv_stream.wait_stream.side_effect = lambda _: timeline.append("recv-ready")
        current_stream.wait_stream.side_effect = lambda stream: timeline.append(
            "send-done" if stream is send_stream else "recv-done"
        )

        with patch.object(
            torch.cuda, "current_stream", return_value=current_stream
        ), patch.object(
            user_buffers,
            "userbuffers_send",
            side_effect=lambda *args: timeline.append("send"),
        ), patch.object(
            user_buffers,
            "userbuffers_recv",
            side_effect=lambda *args: timeline.append("recv"),
        ):
            tensor = torch.empty(4, device="cuda")
            self.assertTrue(
                communicator.send_recv(tensor, 1, torch.empty_like(tensor), 1)
            )

        self.assertEqual(
            timeline,
            ["send-ready", "send", "recv-ready", "recv", "send-done", "recv-done"],
        )

    def test_two_rank_nccl_and_user_buffer_relay(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE:
            self.skipTest("two CUDA devices are required")
        if not dist.is_available() or not dist.is_nccl_available():
            self.skipTest("NCCL torch.distributed backend is required")

        with tempfile.TemporaryDirectory(prefix="qwen35_cp_relay_") as temp_dir:
            init_file = os.path.join(temp_dir, "nccl_file_store")
            mp.spawn(
                _relay_worker,
                args=(_WORLD_SIZE, init_file),
                nprocs=_WORLD_SIZE,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
