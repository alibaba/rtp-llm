"""Decoder construction and forwarding into the real generic routed-only layer."""

from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3Next, Qwen35Moe
from rtp_llm.models_py.model_desc import generic_moe, qwen3_next
from rtp_llm.models_py.model_desc.fast_afd_qwen35 import (
    Qwen35AFDAttentionModel,
    Qwen35AFDExpertModel,
)
from rtp_llm.ops import (
    DeviceResourceConfig,
    HybridAttentionType,
    MoeConfig,
    ParallelismConfig,
)
from rtp_llm.utils.model_weight import W


class _Attention(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states


class _Norm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states, residual


class _RoutedBackend(nn.Module):
    includes_shared_expert = False
    topk_ids_dtype = torch.int32
    router = SimpleNamespace(tp_collective_size=1, supports_skip_tp_allreduce=False)

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, topk_ids, topk_weights, **kwargs):
        self.calls += 1
        return hidden_states + topk_weights.sum(dim=-1, keepdim=True)


class _RemoteExpertClient:
    topk_ids_dtype = torch.int32

    def __init__(self):
        self.calls = []

    def forward(self, layer_idx, hidden_states, topk_ids, topk_weights):
        self.calls.append((layer_idx, topk_ids.clone(), topk_weights.clone()))
        return hidden_states + 1


class _SharedExpert(nn.Module):
    def forward(self, hidden_states, skip_allreduce=False):
        return hidden_states * 2


class _StepClient:
    def __init__(self):
        self.started = 0
        self.finished = 0
        self.aborted = 0
        self.stopped = 0
        self.global_idle = False

    def begin_step(self):
        self.started += 1

    def finish(self):
        self.finished += 1

    def abort(self):
        self.aborted += 1

    def stop(self):
        self.stopped += 1


def _select_topk(logits, ids, weights):
    values, selected = logits.softmax(dim=-1).topk(ids.shape[-1])
    ids.copy_(selected)
    weights.copy_(values / values.sum(dim=-1, keepdim=True))


class Qwen3NextRoutedOnlyTest(TestCase):
    def test_fastafd_model_creation_allows_both_split_settings(self):
        for split in (0, 1):
            for expert in (False, True):
                with self.subTest(split=split, expert=expert):
                    model = Qwen35Moe.__new__(Qwen35Moe)
                    model.model_config = ModelConfig()
                    model.model_config.num_layers = 1
                    model.model_config.moe_layer_index = [0]
                    model.parallelism_config = ParallelismConfig()
                    ffn = model.parallelism_config.ffn_disaggregate_config
                    ffn.enable_ffn_disaggregate = True
                    ffn.is_ffn_rank = expert
                    model.device_resource_config = DeviceResourceConfig()
                    model.device_resource_config.enable_layer_micro_batch = split
                    model.fmha_config = None
                    model.hw_kernel_config = None
                    model.moe_config = MoeConfig()
                    model.max_generate_batch_size = 1
                    model.weight = None
                    with (
                        patch(
                            "rtp_llm.models_py.utils.arch.is_cuda", return_value=True
                        ),
                        patch(
                            "rtp_llm.models_py.model_desc.fast_afd_qwen35.Qwen35AFDAttentionModel"
                        ) as attention_cls,
                        patch(
                            "rtp_llm.models_py.model_desc.fast_afd_qwen35.Qwen35AFDExpertModel"
                        ) as expert_cls,
                    ):
                        selected, other = (
                            (expert_cls, attention_cls)
                            if expert
                            else (attention_cls, expert_cls)
                        )
                        self.assertIs(
                            model._create_python_model(), selected.return_value
                        )
                        self.assertIs(model.py_model, selected.return_value)
                        selected.assert_called_once()
                        other.assert_not_called()
                        self.assertIs(
                            selected.call_args.kwargs["device_resource_config"],
                            model.device_resource_config,
                        )

    def test_expert_rank_uses_local_moe_topology(self):
        config = ModelConfig()
        config.num_layers = 1
        config.hidden_size = 8
        config.max_seq_len = 16
        config.activation_type = "SiGLU"
        Qwen35Moe._parse_moe_config(
            {
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 4,
                "shared_expert_intermediate_size": 4,
            },
            config,
        )
        union_parallelism = ParallelismConfig()
        union_parallelism.world_size = 3
        union_parallelism.world_rank = 2
        union_parallelism.tp_size = 1
        union_parallelism.dp_size = 3
        union_parallelism.dp_rank = 2
        union_parallelism.ep_size = 3
        union_parallelism.ep_rank = 2
        union_parallelism.ffn_disaggregate_config.attention_dp_size = 2
        union_parallelism.ffn_disaggregate_config.attention_tp_size = 1
        weights = ModelWeights(1, "cpu", torch.bfloat16)
        weights.set_layer_weight(0, W.moe_w1, torch.zeros(1))
        weights.set_layer_weight(0, W.moe_w2, torch.zeros(1))
        moe_config = MoeConfig()
        moe_config.use_all_gather = True
        seen_adapters = []

        def create_fused_moe(adapter, _weights):
            seen_adapters.append(adapter)
            return _RoutedBackend()

        with (
            patch.object(
                generic_moe.FusedMoeFactory,
                "create_fused_moe",
                side_effect=create_fused_moe,
            ),
            patch(
                "rtp_llm.models_py.distributed.fast_afd._GlooControlTransport"
            ) as control_transport,
        ):
            model = Qwen35AFDExpertModel(
                config, union_parallelism, weights, moe_config, 1
            )

        control_transport.assert_called_once_with()
        self.assertIs(
            model.fast_afd_service._control_transport, control_transport.return_value
        )
        self.assertEqual(len(seen_adapters), 1)
        adapter = seen_adapters[0]
        self.assertEqual((adapter.tp_size, adapter.dp_size, adapter.ep_size), (1, 1, 1))
        self.assertEqual((adapter.world_size, adapter.world_rank), (1, 0))
        self.assertEqual(adapter.n_shared_experts, 0)
        self.assertEqual(set(model.fast_afd_service.fused_moe_by_layer), {0})
        self.assertTrue(model.requires_micro_batch_forward)

    def test_expert_parallel_ranks_have_disjoint_local_expert_topology(self):
        for rank in (2, 3):
            with self.subTest(rank=rank):
                config = ModelConfig()
                config.num_layers = 1
                config.hidden_size = 8
                config.max_seq_len = 16
                config.activation_type = "SiGLU"
                Qwen35Moe._parse_moe_config(
                    {
                        "num_experts": 4,
                        "num_experts_per_tok": 2,
                        "moe_intermediate_size": 4,
                        "shared_expert_intermediate_size": 4,
                    },
                    config,
                )
                parallelism = ParallelismConfig()
                parallelism.world_size = 4
                parallelism.world_rank = rank
                parallelism.local_rank = rank
                parallelism.local_world_size = 4
                parallelism.dp_size = 4
                parallelism.ep_size = 4
                ffn = parallelism.ffn_disaggregate_config
                ffn.attention_dp_size = 2
                ffn.ffn_tp_size = 2
                weights = ModelWeights(1, "cpu", torch.bfloat16)
                weights.set_layer_weight(0, W.moe_w1, torch.zeros(1))
                weights.set_layer_weight(0, W.moe_w2, torch.zeros(1))
                with (
                    patch.object(
                        generic_moe.FusedMoeFactory,
                        "create_fused_moe",
                        return_value=_RoutedBackend(),
                    ) as create_moe,
                    patch(
                        "rtp_llm.models_py.model_desc.fast_afd_qwen35.FastAFDExpertService"
                    ) as service,
                ):
                    model = Qwen35AFDExpertModel(
                        config, parallelism, weights, MoeConfig(), 1
                    )
                adapter = create_moe.call_args.args[0]
                self.assertEqual(
                    (adapter.tp_size, adapter.ep_size, adapter.dp_size), (2, 2, 1)
                )
                self.assertEqual(
                    (adapter.world_size, adapter.world_rank), (2, rank - 2)
                )
                self.assertEqual(
                    (adapter.ep_rank, adapter.n_local_experts), (rank - 2, 2)
                )
                self.assertEqual(adapter.local_expert_start, (rank - 2) * 2)
                self.assertEqual(adapter.local_rank, rank)
                self.assertEqual(adapter.n_shared_experts, 0)
                self.assertEqual(service.call_args.kwargs["expert_ranks"], (2, 3))
                self.assertEqual(service.call_args.kwargs["attention_ranks"], [0, 1])
                self.assertEqual(parallelism.tp_size, 1)
                self.assertIs(model.fast_afd_service, service.return_value)

    def test_attention_client_uses_first_expert_rank_and_full_expert_group(self):
        parallelism = ParallelismConfig()
        parallelism.world_size = 4
        parallelism.ffn_disaggregate_config.attention_dp_size = 2
        parallelism.ffn_disaggregate_config.ffn_tp_size = 2
        config = ModelConfig()
        config.hidden_size = 8
        config.moe_k = 2
        config.expert_num = 4
        with (
            patch.object(
                qwen3_next.Qwen35Model,
                "__init__",
                lambda instance, *args, **kwargs: nn.Module.__init__(instance),
            ),
            patch(
                "rtp_llm.models_py.model_desc.fast_afd_qwen35.FastAFDClient"
            ) as client,
        ):
            Qwen35AFDAttentionModel(
                config,
                parallelism,
                ModelWeights(1, "cpu", torch.bfloat16),
                MoeConfig(),
                1,
            )
        self.assertEqual(client.call_args.kwargs["service_rank"], 2)
        self.assertEqual(client.call_args.kwargs["expert_ranks"], (2, 3))

    def test_attention_step_aborts_after_forward_error(self):
        model = Qwen35AFDAttentionModel.__new__(Qwen35AFDAttentionModel)
        nn.Module.__init__(model)
        client = _StepClient()
        model.fast_afd_client = client

        def fail_forward(_):
            raise RuntimeError("decoder failed")

        model.forward = fail_forward
        with self.assertRaisesRegex(RuntimeError, "decoder failed"):
            model.forward_micro_batch([object()])
        self.assertEqual((client.started, client.finished, client.aborted), (1, 0, 1))

    def test_attention_step_preserves_decoder_error_if_abort_fails(self):
        model = Qwen35AFDAttentionModel.__new__(Qwen35AFDAttentionModel)
        nn.Module.__init__(model)
        client = _StepClient()
        model.fast_afd_client = client

        def fail_forward(_):
            raise RuntimeError("decoder failed")

        def fail_abort():
            client.aborted += 1
            raise RuntimeError("abort failed")

        model.forward = fail_forward
        client.abort = fail_abort
        with patch("logging.exception") as log_error:
            with self.assertRaisesRegex(RuntimeError, "decoder failed"):
                model.forward_micro_batch([object()])
        log_error.assert_called_once()
        self.assertEqual((client.started, client.finished, client.aborted), (1, 0, 1))

    def test_attention_step_finishes_after_real_micro_batches(self):
        for inputs in (["whole_batch"], ["first", "second"]):
            with self.subTest(micro_batch_count=len(inputs)):
                model = Qwen35AFDAttentionModel.__new__(Qwen35AFDAttentionModel)
                nn.Module.__init__(model)
                client = _StepClient()
                model.fast_afd_client = client
                self.assertTrue(model.requires_micro_batch_forward)
                self.assertTrue(model.micro_batch_outputs_are_normalized)
                with patch.object(
                    model, "forward", side_effect=lambda value: value
                ) as run:
                    self.assertEqual(model.forward_micro_batch(inputs), inputs)
                self.assertEqual(run.call_count, len(inputs))
                self.assertEqual(
                    (client.started, client.finished, client.aborted), (1, 1, 0)
                )

    def test_idle_attention_step_only_sends_finish(self):
        model = Qwen35AFDAttentionModel.__new__(Qwen35AFDAttentionModel)
        nn.Module.__init__(model)
        client = _StepClient()
        client.global_idle = True
        model.fast_afd_client = client
        with patch.object(model, "forward") as decoder_forward:
            self.assertEqual(model.forward_micro_batch([]), [])
        decoder_forward.assert_not_called()
        self.assertEqual((client.started, client.finished, client.aborted), (1, 1, 0))
        self.assertTrue(model.fast_afd_global_idle)

    def test_attention_model_shutdown_notifies_expert(self):
        model = Qwen35AFDAttentionModel.__new__(Qwen35AFDAttentionModel)
        nn.Module.__init__(model)
        client = _StepClient()
        model.fast_afd_client = client
        model.stop_fast_afd()
        self.assertEqual(client.stopped, 1)

    def test_expert_model_marks_service_finished(self):
        model = Qwen35AFDExpertModel.__new__(Qwen35AFDExpertModel)
        nn.Module.__init__(model)
        model.fast_afd_service = SimpleNamespace(
            serve_until_done=lambda: True, global_idle=True
        )
        model.fast_afd_service_finished = False
        self.assertEqual(model.forward_micro_batch([]), [])
        self.assertTrue(model.fast_afd_service_finished)
        self.assertTrue(model.fast_afd_global_idle)

    def test_remote_routed_experts_keep_gate_and_shared_expert_local(self):
        config = ModelConfig()
        config.num_layers = 1
        config.hidden_size = 8
        config.max_seq_len = 16
        config.activation_type = "SiGLU"
        config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.LINEAR
        ]
        Qwen35Moe._parse_moe_config(
            {
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 4,
                "shared_expert_intermediate_size": 4,
            },
            config,
        )
        weights = {
            W.pre_ln_gamma: torch.ones(8),
            W.post_ln_gamma: torch.ones(8),
            W.moe_gate: torch.ones(4, 8),
        }
        client = _RemoteExpertClient()
        with (
            patch.object(
                qwen3_next, "Qwen3NextGatedDeltaNet", return_value=_Attention()
            ),
            patch.object(
                qwen3_next, "RMSResNorm", side_effect=lambda *a, **kw: _Norm()
            ),
            patch.object(
                generic_moe.LinearFactory,
                "create_linear_from_weights",
                return_value=nn.Linear(8, 4, bias=False),
            ),
            patch.object(generic_moe, "SelectTopk", return_value=_select_topk),
            patch.object(generic_moe, "DenseMLP", return_value=_SharedExpert()),
            patch.object(
                generic_moe.FusedMoeFactory, "create_fused_moe"
            ) as local_factory,
        ):
            layer = qwen3_next.Qwen3NextDecoderLayer(
                config,
                ParallelismConfig(),
                weights,
                0,
                MoeConfig(),
                remote_expert_client=client,
            )
            hidden = torch.randn(3, 8)
            residual = torch.randn_like(hidden)
            output, result_residual = layer(hidden, residual, None)

        local_factory.assert_not_called()
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][0], 0)
        self.assertEqual(client.calls[0][1].dtype, torch.int32)
        torch.testing.assert_close(output, hidden * 3 + 1)
        self.assertIs(result_residual, residual)

    def test_routed_only_decoder_constructs_and_forwards_generic_moe(self):
        for model_cls in (Qwen3Next, Qwen35Moe):
            for shared_width in (None, 0):
                with self.subTest(model=model_cls.__name__, shared_width=shared_width):
                    config = ModelConfig()
                    config.num_layers = 1
                    config.hidden_size = 8
                    config.max_seq_len = 16
                    config.activation_type = "SiGLU"
                    config.hybrid_attention_config.hybrid_attention_types = [
                        HybridAttentionType.LINEAR
                    ]
                    hf = {
                        "num_experts": 4,
                        "num_experts_per_tok": 2,
                        "moe_intermediate_size": 4,
                    }
                    if shared_width is not None:
                        hf["shared_expert_intermediate_size"] = shared_width
                    model_cls._parse_moe_config(hf, config)
                    weights = {
                        W.pre_ln_gamma: torch.ones(8),
                        W.post_ln_gamma: torch.ones(8),
                        W.moe_gate: torch.ones(4, 8),
                        W.moe_w1: torch.ones(4, 8, 8),
                        W.moe_w2: torch.ones(4, 8, 4),
                    }
                    backend = _RoutedBackend()
                    with (
                        patch.object(
                            qwen3_next,
                            "Qwen3NextGatedDeltaNet",
                            return_value=_Attention(),
                        ),
                        patch.object(
                            qwen3_next,
                            "RMSResNorm",
                            side_effect=lambda *a, **kw: _Norm(),
                        ),
                        patch.object(
                            generic_moe.LinearFactory,
                            "create_linear_from_weights",
                            return_value=nn.Linear(8, 4, bias=False),
                        ),
                        patch.object(
                            generic_moe, "SelectTopk", return_value=_select_topk
                        ),
                        patch.object(
                            generic_moe.FusedMoeFactory,
                            "create_fused_moe",
                            return_value=backend,
                        ),
                        patch.object(generic_moe, "DenseMLP") as shared_mlp,
                    ):
                        layer = qwen3_next.Qwen3NextDecoderLayer(
                            config, ParallelismConfig(), weights, 0, MoeConfig()
                        )
                        self.assertIsInstance(layer.mlp, generic_moe.GenericMoeLayer)
                        self.assertIsNone(layer.mlp.shared_expert)
                        hidden = torch.randn(3, 8)
                        residual = torch.randn_like(hidden)
                        output, result_residual = layer(hidden, residual, None)
                    shared_mlp.assert_not_called()
                    self.assertEqual(backend.calls, 1)
                    torch.testing.assert_close(output, hidden + 1)
                    self.assertIs(result_residual, residual)


if __name__ == "__main__":
    main()
