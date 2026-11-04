"""Projection/collective hooks and row layouts, with state-free attention cores.

The small synthetic heads isolate model wiring; these tests do not validate
attention or recurrent mathematics. Projections, normalization and collectives
execute their production CUDA implementations.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.multiprocessing as mp
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    destroy_distributed_environment,
    get_process_group,
    init_distributed_environment,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.bf16_comm_fusion import (
    Bf16CollectiveProjection,
    prepare_bf16_projection,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.modules.hybrid.kda_attention import KdaAttention
from rtp_llm.models_py.modules.hybrid.query_layout import QueryLayout
from rtp_llm.models_py.modules.kimi_k3.dsa import KimiK31DSA
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import FusedRMSNormGated
from rtp_llm.ops import NcclCommConfig, ParallelismConfig
from rtp_llm.test.utils.port_util import PortsContext
from torch import nn

_HIDDEN = 7168
_WIDTH = 128


class _Inputs:
    def __init__(self, *, is_prefill, valid_token_mask=None):
        self.is_prefill = is_prefill
        self.is_mtp_draft_update = False
        self.valid_token_mask = valid_token_mask


class _Owner:
    def __init__(self):
        self.releases = 0

    def release_eager_buffers(self):
        self.releases += 1


def _linear(k, n, device):
    return CudaF16Linear(
        torch.randn(k, n, dtype=torch.bfloat16, device=device) * 0.015625
    )


class _DsaHook(KimiK31DSA):
    def __init__(self, device):
        nn.Module.__init__(self)
        self.prefill_query_cp = False
        self.tp_size, self.heads, self.v_dim = 2, 1, _WIDTH
        self.q_rank, self.kv_rank, self.suffix_dim = 128, 64, 64
        self.input = _linear(_HIDDEN, 384, device)
        self.output = _linear(_WIDTH, _HIDDEN, device)
        self._bf16_collective = None
        self._mtp_bf16_collectives = None

    def _attend(self, qkv, fmha, cache, hidden):
        # Consume gathered hidden, as the real Indexer's K/head projections do.
        return (
            qkv[:, :_WIDTH] * 0.25 + hidden[:, :_WIDTH] + hidden[:, _WIDTH:256] * 0.125
        )


class _KdaCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.parallelism_config = SimpleNamespace(tp_size=2)
        self.local_num_k_heads = self.local_num_v_heads = 1
        self.head_k_dim = self.head_v_dim = _WIDTH

    def forward(self, qkv, forget, beta, attention_inputs, cache, metadata):
        q, k, v = qkv.split(_WIDTH, dim=-1)
        return q + k * 0.5 + v * 0.25 + forget * torch.sigmoid(beta)


class _LegacyMappedNorm(nn.Module):
    """Independent pre-change norm -> zero/index_copy layout reference."""

    def __init__(self, norm, layout):
        super().__init__()
        self.norm, self.layout = norm, layout

    def forward(self, x, gate, *, row_indices, out):
        natural = self.norm(x, gate).reshape(-1, _WIDTH)
        restored, _ = self.layout.scatter_rankmajor(natural)
        return restored.reshape(out.shape)


def _kda(device):
    input_projection = _linear(_HIDDEN, 577, device)
    prepare_bf16_projection(input_projection, alignment=128)
    return KdaAttention(
        input_projection=input_projection,
        forget_projection=_linear(64, _WIDTH, device),
        output_projection=_linear(_WIDTH, _HIDDEN, device),
        fa_width=64,
        norm=FusedRMSNormGated(
            torch.rand(_WIDTH, device=device, dtype=torch.bfloat16) + 0.5, 1e-6
        ),
        prefill=_KdaCore(),
        decode=_KdaCore(),
    )


def _layout(rank, device, physical, lengths):
    true_rows = sum(lengths)
    padding = torch.arange(physical, device=device) < true_rows
    restore = torch.arange(physical, device=device).roll(3).flip(0)
    valid = torch.ones(true_rows, dtype=torch.bool, device=device)
    valid[2] = False
    info = SimpleNamespace(
        prefill_actual_input_lengths_cpu=torch.tensor(lengths, dtype=torch.int32),
        prefill_prefix_lengths_cpu=torch.tensor([0, 5], dtype=torch.int32),
        prefill_qkv_restore_indice=restore,
        prefill_qkv_padding_mask=padding,
    )
    layout = QueryLayout(info, 2, rank, device, valid_mask=valid)
    layout.prepare_kda_buffers(640, _WIDTH)
    return layout


def _compare(actual, expected):
    if not torch.isfinite(actual).all():
        raise AssertionError("nonfinite hook output")
    # Push RS accumulates BF16 partials in FP32, unlike NCCL's BF16 tree.
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.02)


def _graph_replay(candidate, reference, hidden):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            candidate()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = candidate()
    torch.cuda.current_stream().wait_stream(stream)
    for shift in (0.375, -0.625):
        hidden.copy_(torch.randn_like(hidden) + shift)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        snapshot = actual.clone()
        _compare(snapshot, reference())


def _decode_cases(model, device, call):
    collective = Bf16CollectiveProjection(
        get_process_group(Group.TP),
        device,
        max_m=32,
        hidden_size=_HIDDEN,
        output_projections=[model.output],
        decode_staging=True,
    )
    if collective.custom_ag is None or collective.push_rs is None:
        raise AssertionError("Decode hook test requires custom AG and Push RS")
    for verify in (False, True):
        inputs = _Inputs(is_prefill=verify)
        metadata = SimpleNamespace(is_target_verify=verify)
        for local_rows in (4, 16, 24):
            hidden = torch.randn(
                local_rows, _HIDDEN, device=device, dtype=torch.bfloat16
            )

            def forward(backend):
                model._bf16_collective = backend
                return call(hidden, inputs, metadata)

            with patch.object(
                collective, "all_gather_gemm", wraps=collective.all_gather_gemm
            ) as ag:
                with patch.object(
                    collective,
                    "gemm_reduce_scatter",
                    wraps=collective.gemm_reduce_scatter,
                ) as rs:
                    actual = forward(collective).clone()
                    expected_calls = int(local_rows * 2 <= collective.max_m)
                    if (
                        ag.call_count != expected_calls
                        or rs.call_count != expected_calls
                    ):
                        raise AssertionError(
                            "Decode collective eligibility/fallback was not honored"
                        )
            _compare(actual, forward(None))
            if local_rows == 4:
                _graph_replay(
                    lambda: forward(collective), lambda: forward(None), hidden
                )


def _check_dsa(device):
    model = _DsaHook(device)
    owner = _Owner()
    fmha = {"full": SimpleNamespace(workspace_owner=owner)}

    def call(hidden, inputs, metadata):
        before = owner.releases
        output = model(hidden, fmha, None, inputs, metadata)
        if owner.releases != before + 1:
            raise AssertionError(
                "DSA must release eager leases exactly once per forward"
            )
        return output

    _decode_cases(model, device, call)


def _check_kda_decode(device):
    model = _kda(device)
    _decode_cases(
        model,
        device,
        lambda hidden, inputs, metadata: model(hidden, None, None, inputs, metadata),
    )


def _check_kda(rank, device):
    model = _kda(device)
    collective = Bf16CollectiveProjection(
        get_process_group(Group.TP),
        device,
        max_m=512,
        hidden_size=_HIDDEN,
        output_projections=[model.output],
    )
    if collective.custom_ag is None or collective.push_rs is None:
        raise AssertionError("KDA hook test requires custom AG and Push RS")
    norm = model.norm
    for physical, lengths in ((16, [7, 3]), (32, [11, 9])):
        candidate_layout = _layout(rank, device, physical, lengths)
        reference_layout = _layout(rank, device, physical, lengths)
        inputs = _Inputs(is_prefill=True)
        hidden = torch.randn(
            physical // 2, _HIDDEN, device=device, dtype=torch.bfloat16
        )
        candidate_metadata = SimpleNamespace(
            is_target_verify=False, query_layout=candidate_layout
        )
        reference_metadata = SimpleNamespace(
            is_target_verify=False, query_layout=reference_layout
        )
        reference_norm = _LegacyMappedNorm(norm, reference_layout)

        def candidate():
            model._bf16_collective, model.norm = collective, norm
            return model(hidden, None, None, inputs, candidate_metadata)

        def reference():
            model._bf16_collective, model.norm = None, reference_norm
            return model(hidden, None, None, inputs, reference_metadata)

        with patch.object(
            collective, "all_gather_gemm", wraps=collective.all_gather_gemm
        ) as ag:
            with patch.object(
                collective, "gemm_reduce_scatter", wraps=collective.gemm_reduce_scatter
            ) as rs:
                actual = candidate().clone()
                if ag.call_count != 1 or rs.call_count != 1:
                    raise AssertionError(
                        "KDA did not execute both collective projection hooks"
                    )
        _compare(actual, reference())
        restored = candidate_layout.reserve("output", physical, _WIDTH, torch.bfloat16)
        padding = candidate_layout.rankmajor_to_natural < 0
        if torch.count_nonzero(restored[padding]).item():
            raise AssertionError("mapped norm must zero padded physical rows")
        hole = candidate_layout.natural_to_rankmajor[2]
        if torch.count_nonzero(restored[hole]).item():
            raise AssertionError("mapped norm must retain invalid natural rows as zero")
        _graph_replay(candidate, reference, hidden)


def _worker(rank, port):
    torch.cuda.set_device(rank)
    torch.manual_seed(431 + rank)
    parallelism = ParallelismConfig()
    parallelism.world_rank, parallelism.world_size = rank, 2
    parallelism.local_rank, parallelism.tp_size, parallelism.dp_size = rank, 2, 1
    try:
        init_distributed_environment(
            parallelism,
            NcclCommConfig(
                nccl_ip="127.0.0.1",
                tp_nccl_port=port + 9,
                dp_tp_nccl_port=port + 1,
                ffn_tp_nccl_port=port + 6,
            ),
            port,
            disable_custom_all_reduce=True,
        )
        device = torch.device("cuda", rank)
        _check_dsa(device)
        _check_kda_decode(device)
        _check_kda(rank, device)
        torch.cuda.synchronize()
    finally:
        destroy_distributed_environment()


class Bf16AttentionHooksTest(unittest.TestCase):
    def test_real_projection_and_collective_hooks(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2, "requires two CUDA GPUs")
        with PortsContext(num_ports=12) as ports:
            mp.spawn(_worker, args=(ports[0],), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
