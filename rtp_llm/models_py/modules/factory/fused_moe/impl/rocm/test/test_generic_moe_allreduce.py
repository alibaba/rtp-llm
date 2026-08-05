"""ROCm tests for GenericMoe unified TP all-reduce wiring."""

import multiprocessing as mp
import socket
from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.model_desc import generic_moe as generic_moe_module
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers import (
    pure_tp_router as pure_tp_router_module,
)
from rtp_llm.models_py.modules.hybrid import dense_mlp as dense_mlp_module
from rtp_llm.ops import ActivationType, MoeConfig, NcclCommConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W


class _FixedSelectTopk(nn.Module):
    def forward(self, logits, topk_ids, topk_weights):
        topk_ids.zero_()
        topk_weights.fill_(1.0)


class _ZeroRouterGate(nn.Module):
    def __init__(self, expert_num):
        super().__init__()
        self.expert_num = expert_num

    def forward(self, hidden_states):
        return torch.zeros(
            hidden_states.size(0),
            self.expert_num,
            dtype=torch.float32,
            device=hidden_states.device,
        )


class _SharedUpProjection(nn.Module):
    def forward(self, hidden_states):
        return torch.cat((hidden_states, hidden_states), dim=-1)


class _SharedDownProjection(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, activated):
        return activated[..., : self.hidden_size]


class _SharedGateProjection(nn.Module):
    def forward(self, hidden_states):
        return hidden_states[..., :1]


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _make_real_parallelism(rank, world_size):
    parallelism_config = ParallelismConfig()
    parallelism_config.tp_size = world_size
    parallelism_config.tp_rank = rank
    parallelism_config.ffn_tp_size = world_size
    parallelism_config.ffn_tp_rank = rank
    parallelism_config.ep_size = 1
    parallelism_config.ep_rank = 0
    parallelism_config.dp_size = 1
    parallelism_config.dp_rank = 0
    parallelism_config.world_size = world_size
    parallelism_config.world_rank = rank
    parallelism_config.local_rank = rank
    parallelism_config.local_world_size = world_size
    return parallelism_config


def _build_real_layer(rank, parallelism_config, with_gate):
    device = torch.device(f"cuda:{rank}")
    hidden_size = 512
    inter_size = 1024
    expert_num = 32

    config = ModelConfig()
    config.hidden_size = hidden_size
    config.inter_size = inter_size
    config.expert_num = expert_num
    config.moe_k = 4
    config.activation_type = ActivationType.Swiglu
    config.moe_style = 2
    config.quant_config = None

    # Give each rank a different routed-expert contribution so the test checks
    # an actual cross-rank sum instead of two identical local tensors.
    torch.manual_seed(1000 + rank)
    weights = {
        W.moe_gate: torch.empty(1, device=device),
        W.moe_w1: torch.randn(
            expert_num, 2 * inter_size, hidden_size, device=device, dtype=torch.bfloat16
        )
        * 0.02,
        W.moe_w2: torch.randn(
            expert_num, hidden_size, inter_size, device=device, dtype=torch.bfloat16
        )
        * 0.02,
        W.ffn_w13: torch.empty(1, device=device),
        W.ffn_w2: torch.empty(1, device=device),
    }
    if with_gate:
        weights[W.shared_expert_gate] = torch.empty(1, device=device)

    moe_config = MoeConfig()
    moe_config.use_all_gather = True
    moe_config.fake_balance_expert = False

    def make_linear(_weights, weight_key, *args, **kwargs):
        if weight_key == W.moe_gate:
            return _ZeroRouterGate(expert_num)
        if weight_key == W.ffn_w13:
            return _SharedUpProjection()
        if weight_key == W.ffn_w2:
            return _SharedDownProjection(hidden_size)
        if weight_key == W.shared_expert_gate:
            return _SharedGateProjection()
        raise AssertionError(f"unexpected test linear weight key: {weight_key}")

    with patch(
        "rtp_llm.models_py.model_desc.generic_moe.LinearFactory.create_linear_from_weights",
        side_effect=make_linear,
    ):
        layer = GenericMoeLayer(
            config,
            parallelism_config,
            weights,
            moe_config,
        )

    if type(layer.fused_moe.router).__name__ != "PureTpRouterNoQuant":
        raise AssertionError(
            "real TP integration test selected "
            f"{type(layer.fused_moe.router).__name__}, expected PureTpRouterNoQuant"
        )
    layer.select_topk = _FixedSelectTopk()
    return layer


def _run_real_two_gpu_case(rank, world_size, port, with_gate):
    parallelism_config = _make_real_parallelism(rank, world_size)
    initialized = False
    try:
        torch.cuda.set_device(rank)
        init_distributed_environment(
            parallelism_config,
            NcclCommConfig(
                nccl_ip="127.0.0.1",
                tp_nccl_port=port + 1,
                dp_tp_nccl_port=port + 2,
                ffn_tp_nccl_port=port + 3,
            ),
            nccl_init_port=port,
            backend="nccl",
            timeout=300,
        )
        initialized = True
        layer = _build_real_layer(rank, parallelism_config, with_gate)
        # TP ranks consume the same replicated hidden states.  Keep the routed
        # expert weights rank-specific so the collective still sums distinct
        # partial outputs, while the shared-expert gate remains rank-consistent.
        torch.manual_seed(2024)
        hidden_states = torch.randn(
            32, 512, device=torch.device(f"cuda:{rank}"), dtype=torch.bfloat16
        )
        dist.broadcast(hidden_states, src=0)
        counts = {"all_reduce": 0}

        def counted_all_reduce(tensor, group):
            if group is not Group.TP:
                raise AssertionError(f"unexpected collective group: {group}")
            counts["all_reduce"] += 1
            return collective_torch.all_reduce(tensor, group)

        with (
            patch.object(
                generic_moe_module, "all_reduce", side_effect=counted_all_reduce
            ),
            patch.object(
                dense_mlp_module, "all_reduce", side_effect=counted_all_reduce
            ),
            patch.object(
                pure_tp_router_module, "all_reduce", side_effect=counted_all_reduce
            ),
        ):
            layer.use_unified_tp_allreduce = True
            unified_output = layer(hidden_states.clone())
            unified_calls = counts["all_reduce"]
            dist.barrier()

            layer.use_unified_tp_allreduce = False
            legacy_output = layer(hidden_states.clone())
            legacy_calls = counts["all_reduce"] - unified_calls
            dist.barrier()

        if unified_calls != 1 or legacy_calls != 2:
            raise AssertionError(
                f"unexpected TP collective counts: unified={unified_calls}, "
                f"legacy={legacy_calls}"
            )
        torch.testing.assert_close(
            unified_output,
            legacy_output,
            rtol=2e-2,
            atol=2e-3,
        )
        if rank == 0:
            print(
                f"[real_tp_unified] gate={with_gate} unified_calls={unified_calls} "
                f"legacy_calls={legacy_calls} ✓"
            )
    finally:
        if initialized:
            destroy_distributed_environment()


def _launch_real_two_gpu_case(with_gate):
    world_size = 2
    port = _get_free_port()
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_run_real_two_gpu_case,
            args=(rank, world_size, port, with_gate),
            name=f"generic-moe-rank-{rank}",
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=300)
    failed = [process for process in processes if process.exitcode != 0]
    if failed:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise RuntimeError(
            "real GenericMoe two-GPU worker failed: "
            + ", ".join(
                f"{process.name} exitcode={process.exitcode}" for process in failed
            )
        )


def _make_layer(
    *,
    supports_skip_tp_allreduce=True,
    ffn_tp_size=2,
    attn_tp_size=2,
    ep_size=1,
    moe_style=2,
    with_shared_expert_gate=False,
):
    config = SimpleNamespace(
        hidden_size=8,
        inter_size=16,
        expert_num=4,
        moe_k=2,
        quant_config=None,
        activation_type="SiGLU",
        moe_style=moe_style,
        eplb_config=SimpleNamespace(phy_exp_num=lambda count: count),
    )
    parallelism_config = SimpleNamespace(
        ep_size=ep_size,
        dp_rank=0,
        dp_size=1,
        get_ffn_tp_size=lambda: ffn_tp_size,
        get_attn_tp_size=lambda: attn_tp_size,
    )
    moe_config = SimpleNamespace(fake_balance_expert=False)
    weights = {
        W.moe_w1: torch.empty(4, 2, 8),
        W.moe_w2: torch.empty(4, 8, 2),
    }
    if with_shared_expert_gate:
        weights[W.shared_expert_gate] = torch.empty(8, 1)
    fused_moe = SimpleNamespace(
        topk_ids_dtype=torch.int32,
        router=SimpleNamespace(supports_skip_tp_allreduce=supports_skip_tp_allreduce),
    )

    with (
        patch(
            "rtp_llm.models_py.model_desc.generic_moe.LinearFactory.create_linear_from_weights",
            return_value=Mock(),
        ),
        patch(
            "rtp_llm.models_py.model_desc.generic_moe.SelectTopk",
            return_value=Mock(),
        ),
        patch(
            "rtp_llm.models_py.model_desc.generic_moe.DenseMLP",
            return_value=Mock(),
        ),
        patch("rtp_llm.models_py.model_desc.generic_moe.MoEConfigAdapter"),
        patch(
            "rtp_llm.models_py.model_desc.generic_moe.FusedMoeFactory"
        ) as fused_moe_factory,
    ):
        fused_moe_factory.return_value.create_fused_moe.return_value = fused_moe
        return GenericMoeLayer(config, parallelism_config, weights, moe_config)


def _configure_forward(layer, *, gate_enabled=False):
    hidden_states = torch.randn(4, 8)
    routed_output = torch.randn_like(hidden_states)
    shared_output = torch.randn_like(hidden_states)
    gate_output = torch.full((4, 1), 2.0)

    layer.gate = Mock(return_value=torch.zeros(4, 4))
    layer.select_topk = Mock(
        side_effect=lambda logits, topk_ids, topk_weights: (
            topk_ids.zero_(),
            topk_weights.fill_(0.5),
        )
    )
    fused_moe = Mock(return_value=routed_output)
    fused_moe.topk_ids_dtype = torch.int32
    layer.fused_moe = fused_moe
    layer.shared_expert = Mock(return_value=shared_output)
    if gate_enabled:
        layer.shared_expert_gate = Mock(return_value=gate_output)
        layer.sigmoid_gate_scale_add = Mock(
            side_effect=lambda gate, shared, output: output.add_(
                torch.sigmoid(gate) * shared
            )
        )
    else:
        layer.shared_expert_gate = None
        layer.sigmoid_gate_scale_add = None
    layer.correction_bias = None
    return hidden_states, routed_output, shared_output, gate_output, fused_moe


class GenericMoeInitializationTest(TestCase):
    def test_unified_decision_covers_all_predicate_terms(self):
        cases = (
            ("pure_tp_shared_supported", True, 2, 1, 2, True),
            ("ffn_tp_one", True, 1, 1, 2, False),
            ("ep_mode", True, 2, 2, 2, False),
            ("no_shared_expert", True, 2, 1, 1, False),
            ("unsupported_router", False, 2, 1, 2, False),
        )
        for name, supports, ffn_tp, ep_size, moe_style, expected in cases:
            with self.subTest(name=name):
                layer = _make_layer(
                    supports_skip_tp_allreduce=supports,
                    ffn_tp_size=ffn_tp,
                    ep_size=ep_size,
                    moe_style=moe_style,
                )
                self.assertEqual(layer.use_unified_tp_allreduce, expected)

    def test_attn_tp_size_is_part_of_the_constructor_contract(self):
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            _make_layer(ffn_tp_size=4, attn_tp_size=2)

        self.assertTrue(
            _make_layer(ffn_tp_size=2, attn_tp_size=4).use_unified_tp_allreduce
        )

    def test_shared_expert_gate_is_assembled_by_init(self):
        layer = _make_layer(with_shared_expert_gate=True)
        self.assertIsNotNone(layer.shared_expert_gate)
        self.assertIsNotNone(layer.sigmoid_gate_scale_add)
        self.assertTrue(layer.use_unified_tp_allreduce)


class GenericMoeUnifiedAllreduceTest(TestCase):
    @patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce")
    def test_pure_tp_combines_partial_outputs_before_reduce(self, mock_all_reduce):
        layer = _make_layer()
        hidden_states, routed_output, shared_output, _, fused_moe = _configure_forward(
            layer
        )
        expected_input = routed_output + shared_output
        mock_all_reduce.side_effect = lambda tensor, group: tensor * 2

        result = layer(hidden_states)

        mock_all_reduce.assert_called_once()
        reduce_input = mock_all_reduce.call_args.args[0]
        self.assertIs(mock_all_reduce.call_args.kwargs["group"], Group.TP)
        torch.testing.assert_close(reduce_input, expected_input)
        torch.testing.assert_close(result, expected_input * 2)
        self.assertTrue(fused_moe.call_args.kwargs["skip_tp_allreduce"])
        self.assertTrue(layer.shared_expert.call_args.kwargs["skip_allreduce"])

    @patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce")
    def test_pure_tp_gate_is_merged_in_place_before_reduce(self, mock_all_reduce):
        layer = _make_layer()
        hidden_states, routed_output, shared_output, gate_output, fused_moe = (
            _configure_forward(layer, gate_enabled=True)
        )
        expected_input = routed_output.clone()
        expected_input.add_(torch.sigmoid(gate_output) * shared_output)
        mock_all_reduce.side_effect = lambda tensor, group: tensor * 2

        result = layer(hidden_states)

        reduce_input = mock_all_reduce.call_args.args[0]
        self.assertIs(reduce_input, routed_output)
        self.assertIs(layer.shared_expert_gate.call_args.args[0], hidden_states)
        torch.testing.assert_close(reduce_input, expected_input)
        torch.testing.assert_close(result, expected_input * 2)
        self.assertTrue(fused_moe.call_args.kwargs["skip_tp_allreduce"])

    @patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce")
    def test_ep_reduces_shared_output_only(self, mock_all_reduce):
        layer = _make_layer(ep_size=2)
        hidden_states, routed_output, shared_output, _, fused_moe = _configure_forward(
            layer
        )
        mock_all_reduce.side_effect = lambda tensor, group: tensor * 2

        result = layer(hidden_states)

        reduce_input = mock_all_reduce.call_args.args[0]
        torch.testing.assert_close(reduce_input, shared_output)
        torch.testing.assert_close(result, routed_output + shared_output * 2)
        self.assertFalse(fused_moe.call_args.kwargs["skip_tp_allreduce"])
        self.assertTrue(layer.shared_expert.call_args.kwargs["skip_allreduce"])

    @patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce")
    def test_ffn_tp_one_does_not_add_collective(self, mock_all_reduce):
        layer = _make_layer(ffn_tp_size=1)
        hidden_states, routed_output, shared_output, _, fused_moe = _configure_forward(
            layer
        )

        result = layer(hidden_states)

        mock_all_reduce.assert_not_called()
        torch.testing.assert_close(result, routed_output + shared_output)
        self.assertFalse(fused_moe.call_args.kwargs["skip_tp_allreduce"])
        self.assertFalse(layer.shared_expert.call_args.kwargs["skip_allreduce"])


@skipUnless(torch.cuda.is_available(), "ROCm is not available")
class GenericMoeRealAllreduceTest(TestCase):
    def test_two_gpu_real_layer_matches_legacy_for_gated_and_ungated(self):
        for with_gate in (False, True):
            with self.subTest(with_gate=with_gate):
                _launch_real_two_gpu_case(with_gate)


if __name__ == "__main__":
    main()
