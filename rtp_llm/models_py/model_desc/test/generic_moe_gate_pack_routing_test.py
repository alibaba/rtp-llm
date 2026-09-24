"""MegaMoE routing opt-out, 4096-token boundary and CUDA output equivalence."""

import os
from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import Mock, patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_desc import generic_moe
from rtp_llm.models_py.modules.base.cuda.select_topk import SelectTopk


class _Backend(torch.nn.Module):
    includes_shared_expert = False
    supports_gate_pack = True
    topk_ids_dtype = torch.int64
    router = SimpleNamespace(tp_collective_size=1, supports_skip_tp_allreduce=False)
    fused_experts = SimpleNamespace(
        gated_shared_expert_requested=False, uses_shared_expert_gates=False
    )

    def __init__(self, real_pack=False):
        super().__init__()
        self.real_pack = real_pack
        self.calls = []
        self.buffer = None

    def _buffer(self, x):
        n, dim = x.shape
        self.buffer = (
            torch.empty((n, dim), device=x.device, dtype=torch.float8_e4m3fn),
            torch.empty((n, dim // 128), device=x.device, dtype=torch.int32),
            torch.empty((n, 10), device=x.device, dtype=torch.int64),
            torch.empty((n, 10), device=x.device, dtype=torch.float32),
        )
        return self.buffer

    def forward_gate_pack(self, hidden_states, gate_payload, **kwargs):
        self.calls.append("fused")
        if self.real_pack:
            from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
                fused_pack_mega_moe_gate_inputs,
            )

            fused_pack_mega_moe_gate_inputs(
                hidden_states,
                gate_payload.scores,
                *self._buffer(hidden_states),
                topk=10,
                score_func="softmax",
                route_scale=1.0,
            )
        return hidden_states

    def forward(self, hidden_states, topk_ids, topk_weights, **kwargs):
        self.calls.append("separate")
        if self.real_pack:
            from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
                fused_pack_mega_moe_inputs,
            )

            fused_pack_mega_moe_inputs(
                hidden_states, topk_weights, topk_ids, *self._buffer(hidden_states)
            )
        return hidden_states


def make_layer(
    strategy="mega_moe_fp8", *, real_pack=False, chunk_tokens=0, **overrides
):
    config = ModelConfig()
    config.hidden_size = 4096
    config.inter_size = 128
    config.expert_num = 512
    config.moe_k = 10
    config.moe_style = 1
    config.scoring_func = 0
    config.has_moe_norm = True
    config.routed_scaling_factor = 1.0
    for key, value in overrides.items():
        setattr(config, key, value)
    parallel = SimpleNamespace(ep_size=4, get_ffn_tp_size=lambda: 1)
    backend = _Backend(real_pack)
    backend.fused_experts = SimpleNamespace(
        gated_shared_expert_requested=False,
        uses_shared_expert_gates=False,
        _mega_group=object(),
    )
    prefix = "rtp_llm.models_py.model_desc.generic_moe."
    with patch(
        prefix + "LinearFactory.create_linear_from_weights", return_value=Mock()
    ), patch(prefix + "MoEConfigAdapter"), patch(prefix + "FusedMoeFactory") as factory:
        generic_moe.MoEConfigAdapter.return_value.mega_moe_chunk_tokens = chunk_tokens
        generic_moe.MoEConfigAdapter.return_value.max_tokens_per_rank = chunk_tokens
        factory.return_value.create_fused_moe.return_value = backend
        layer = generic_moe.GenericMoeLayer(
            config,
            parallel,
            {},
            SimpleNamespace(moe_strategy=strategy, fake_balance_expert=False),
        )
    return layer, backend


class _ChunkBackend(_Backend):
    def __init__(self, capacity, shared):
        super().__init__()
        self.scratch = torch.empty(capacity, 2)
        self.fused_experts = SimpleNamespace(uses_shared_expert_gates=shared)
        self.inputs = []
        self.routing = []

    def _compute(self, x, extra_expert_args):
        n = x.shape[0]
        self.inputs.append(x.clone())
        self.scratch[:n].copy_(x * 2)
        if extra_expert_args is not None:
            gates = extra_expert_args["shared_expert_gates"]
            torch.testing.assert_close(gates, torch.sigmoid(x[:, 0]))
            self.scratch[:n].add_(gates[:, None])
        return self.scratch[:n]

    def forward_gate_pack(self, hidden_states, gate_payload, **kwargs):
        self.calls.append("fused")
        assert gate_payload.scores.shape[0] == hidden_states.shape[0]
        return self._compute(hidden_states, kwargs.get("extra_expert_args"))

    def forward(self, hidden_states, topk_ids, topk_weights, **kwargs):
        self.calls.append("separate")
        assert topk_ids.shape[0] == topk_weights.shape[0] == hidden_states.shape[0]
        self.routing.append((topk_weights.clone(), topk_ids.clone()))
        return self._compute(hidden_states, kwargs.get("extra_expert_args"))


class GenericMoeChunkIntegrationTest(TestCase):
    def test_single_launch_keeps_gate_pack_and_graph_path(self):
        with patch.object(generic_moe, "SelectTopk"):
            layer, backend = make_layer(chunk_tokens=4)
        layer.gate = Mock(return_value=torch.zeros(3, 512))
        x = torch.ones(3, 2)
        module = "rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.chunking"
        for capture in (False, True):
            with patch(module + "._capturing", return_value=capture), patch.object(
                layer._mega_moe_chunker, "max_tokens", return_value=3
            ), patch.object(layer, "_prepare_mega_moe_chunks") as prepare:
                torch.testing.assert_close(layer(x), x)
                prepare.assert_not_called()
        self.assertEqual(backend.calls, ["fused", "fused"])

    def test_standalone_shared_gate_is_precomputed_once(self):
        with patch.object(generic_moe, "SelectTopk"):
            layer, _ = make_layer(chunk_tokens=4)
        layer.fused_moe = _ChunkBackend(4, False)
        layer.gate = Mock(side_effect=lambda x: torch.zeros(x.shape[0], 512))
        layer.shared_expert = Mock(side_effect=lambda x, **kwargs: x + 3)
        layer.shared_expert_gate = Mock(side_effect=lambda x: x[:, :1])
        layer.sigmoid_gate_scale_add = Mock(
            side_effect=lambda gate, shared, routed: routed.add_(
                torch.sigmoid(gate) * shared
            )
        )
        x = torch.arange(18).reshape(9, 2).float() / 10
        with patch.object(layer._mega_moe_chunker, "max_tokens", return_value=17):
            y = layer(x)
        torch.testing.assert_close(y, x * 2 + torch.sigmoid(x[:, :1]) * (x + 3))
        layer.gate.assert_called_once()
        layer.shared_expert_gate.assert_called_once()
        self.assertEqual(layer.select_topk.call_count, 1)
        self.assertEqual(layer.shared_expert.call_count, 3)
        self.assertEqual(len(layer.fused_moe.calls), 5)

    def test_empty_rounds_skip_only_local_shared_expert(self):
        with patch.object(generic_moe, "SelectTopk"):
            layer, _ = make_layer(chunk_tokens=4)
        backend = _ChunkBackend(4, False)
        layer.fused_moe = backend
        layer.gate = Mock(side_effect=lambda x: torch.zeros(x.shape[0], 512))
        layer.shared_expert = Mock(side_effect=lambda x, **kwargs: x + 3)
        x = torch.ones(3, 2)
        with patch.object(layer._mega_moe_chunker, "max_tokens", return_value=9):
            y = layer(x)
        torch.testing.assert_close(y, x * 3 + 3)
        self.assertEqual(len(backend.calls), 3)
        layer.shared_expert.assert_called_once()

    def test_chunked_gate_paths_shared_gates_and_workspace_ownership(self):
        for shared in (False, True):
            for tokens in (0, 3, 8201):
                with self.subTest(shared=shared, tokens=tokens), patch.object(
                    generic_moe, "SelectTopk"
                ):
                    layer, _ = make_layer(
                        "mega_moe_fp8_se" if shared else "mega_moe_fp8",
                        chunk_tokens=4097,
                    )

                    def select(scores, ids, weights):
                        rows = torch.arange(scores.shape[0])[:, None]
                        ids.copy_(rows.expand_as(ids) % 512)
                        weights.copy_(rows.expand_as(weights).float() / 10000)

                    layer.select_topk.side_effect = select
                    backend = _ChunkBackend(4097, shared)
                    layer.fused_moe = backend
                    layer.gate = Mock(
                        side_effect=lambda x: torch.zeros(x.shape[0], 512)
                    )
                    layer.shared_expert_gate = Mock(side_effect=lambda x: x[:, :1])
                    self.assertEqual(layer._mega_moe_chunker.chunk_tokens, 4097)
                    x = torch.linspace(-2, 2, tokens * 2).reshape(tokens, 2)
                    with patch.object(
                        layer._mega_moe_chunker, "max_tokens", return_value=8201
                    ), patch(
                        "rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_inputs.shared_expert_sigmoid",
                        side_effect=lambda logits: torch.sigmoid(logits[:, 0]),
                    ) as sigmoid:
                        y = layer(x)
                    expected = x * 2
                    if shared:
                        expected = expected + torch.sigmoid(x[:, :1])
                    torch.testing.assert_close(y, expected)
                    self.assertEqual(len(backend.calls), 3)
                    torch.testing.assert_close(torch.cat(backend.inputs), x)
                    self.assertEqual(backend.calls, ["separate"] * 3)
                    self.assertEqual(layer.gate.call_count, int(tokens > 0))
                    self.assertEqual(layer.select_topk.call_count, int(tokens > 0))
                    self.assertEqual(
                        layer.shared_expert_gate.call_count, int(shared and tokens > 0)
                    )
                    self.assertEqual(sigmoid.call_count, int(shared))
                    all_weights = torch.cat([r[0] for r in backend.routing])
                    all_ids = torch.cat([r[1] for r in backend.routing])
                    rows = torch.arange(tokens)[:, None]
                    torch.testing.assert_close(all_ids, rows.expand(tokens, 10) % 512)
                    torch.testing.assert_close(
                        all_weights, rows.expand(tokens, 10).float() / 10000
                    )


class GenericMoeGatePackRoutingTest(TestCase):
    def test_boundary_and_backend_scope(self):
        for env in (None, "1", "0"):
            for strategy in ("mega_moe_fp8", "mega_moe_fp8_se", "auto", "mega_moe"):
                with self.subTest(env=env, strategy=strategy), patch.dict(
                    os.environ
                ), patch.object(generic_moe, "SelectTopk") as select:
                    if env is None:
                        os.environ.pop("RTP_FUSED_TOPK_512", None)
                    else:
                        os.environ["RTP_FUSED_TOPK_512"] = env
                    layer, backend = make_layer(strategy)
                    split = strategy in ("mega_moe_fp8", "mega_moe_fp8_se")
                    enabled = env != "0"
                    self.assertEqual(
                        select.call_args.kwargs.get("use_fused_512"),
                        enabled if split else None,
                    )
                    self.assertEqual(
                        select.call_args.kwargs.get("fuse_bf16_cast"),
                        True if split else None,
                    )
                    # Both branches must keep their construction-time choice.
                    os.environ["RTP_FUSED_TOPK_512"] = "0" if enabled else "1"
                    for n in (0, 1, 4095, 4096, 4097, 8192):
                        x = torch.zeros(n, 1)
                        layer.gate = Mock(
                            return_value=torch.zeros(n, 512, dtype=torch.bfloat16)
                        )
                        select.return_value.reset_mock()
                        layer(x)
                        separate = split and (not enabled or n > 4096)
                        self.assertEqual(
                            backend.calls[-1], "separate" if separate else "fused"
                        )
                        # Empty ranks still launch experts but need no local top-k.
                        self.assertEqual(
                            select.return_value.call_count, int(separate and n > 0)
                        )

    def test_opt_out_disables_topk_and_bf16_cast_fusion(self):
        for strategy in ("mega_moe_fp8", "mega_moe_fp8_se"):
            with self.subTest(strategy=strategy), patch.dict(
                os.environ, {"RTP_FUSED_TOPK_512": "0"}
            ), patch(
                "rtp_llm.models_py.modules.base.cuda.select_topk.compute_ops.SelectTopkOp"
            ) as op:
                layer, _ = make_layer(strategy)
                self.assertFalse(op.call_args.kwargs["use_fused_512"])
                self.assertFalse(layer.select_topk.fuse_bf16_cast)

    def test_other_routing_semantics_keep_existing_path(self):
        for overrides in (
            dict(expert_num=256),
            dict(scoring_func=2),
            dict(has_moe_norm=False),
            dict(routed_scaling_factor=2.5),
        ):
            with self.subTest(overrides=overrides), patch.object(
                generic_moe, "SelectTopk"
            ) as select:
                layer, _ = make_layer(**overrides)
                self.assertFalse(layer._split_mega_moe_gate_pack)
                self.assertNotIn("use_fused_512", select.call_args.kwargs)
                self.assertNotIn("fuse_bf16_cast", select.call_args.kwargs)

    def test_explicit_topk_override_and_environment_default(self):
        config = ModelConfig()
        for env in ("0", "1"):
            with patch.dict(os.environ, {"RTP_FUSED_TOPK_512": env}), patch(
                "rtp_llm.models_py.modules.base.cuda.select_topk.compute_ops.SelectTopkOp"
            ) as op:
                SelectTopk(config)
                self.assertEqual(op.call_args.kwargs["use_fused_512"], env == "1")
                SelectTopk(config, use_fused_512=True)
                self.assertTrue(op.call_args.kwargs["use_fused_512"])
                SelectTopk(config, use_fused_512=False)
                self.assertFalse(op.call_args.kwargs["use_fused_512"])

    @skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_boundary_matches_fused_reference(self):
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
            fused_pack_mega_moe_gate_inputs,
        )

        torch.manual_seed(20260920)
        for strategy in ("mega_moe_fp8", "mega_moe_fp8_se"):
            for env in ("0", "1"):
                with patch.dict(os.environ, {"RTP_FUSED_TOPK_512": env}):
                    layer, backend = make_layer(strategy, real_pack=True)
                for impl in ("legacy", "optimized"):
                    for n in (0, 1, 4095, 4096, 4097, 8192):
                        with self.subTest(
                            strategy=strategy, env=env, impl=impl, tokens=n
                        ), patch.dict(os.environ, {"MEGA_MOE_INPUT_PACKER_IMPL": impl}):
                            x = (
                                torch.randn(
                                    n, 4096, device="cuda", dtype=torch.bfloat16
                                )
                                * 0.3
                            )
                            scores = torch.randn(
                                n, 512, device="cuda", dtype=torch.bfloat16
                            )
                            layer.gate = Mock(return_value=scores)
                            separate = env == "0" or n > 4096
                            with patch.object(
                                layer.select_topk,
                                "forward",
                                wraps=layer.select_topk.forward,
                            ) as topk:
                                output = layer(x)
                                self.assertEqual(
                                    topk.call_count, int(separate and n > 0)
                                )
                            self.assertIs(output, x)
                            self.assertEqual(
                                backend.calls[-1],
                                "separate" if separate else "fused",
                            )
                            got = backend.buffer
                            ref = tuple(torch.empty_like(t) for t in got)
                            fused_pack_mega_moe_gate_inputs(
                                x,
                                scores,
                                *ref,
                                topk=10,
                                score_func="softmax",
                                route_scale=1.0,
                            )
                            self.assertTrue(
                                torch.equal(
                                    got[0].view(torch.uint8), ref[0].view(torch.uint8)
                                )
                            )
                            self.assertTrue(torch.equal(got[1], ref[1]))
                            self.assertTrue(torch.equal(got[2], ref[2]))
                            torch.testing.assert_close(
                                got[3], ref[3], rtol=1e-4, atol=1e-6
                            )


if __name__ == "__main__":
    main()
