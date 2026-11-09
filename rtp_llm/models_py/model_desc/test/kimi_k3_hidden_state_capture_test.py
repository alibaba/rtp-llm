"""K3 capture contracts using the existing cache-routing and CP fixtures."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc import kimi_k3
from rtp_llm.models_py.model_desc.kimi_k3_mtp import KimiK3MtpModel
from rtp_llm.models_py.model_desc.test import (
    kimi_k3_mtp_cp_hidden_test as cp_fixtures,
    kimi_k31_cache_routing_test as cache_fixtures,
)
from rtp_llm.ops import RoleType


class _Layer(cache_fixtures.RecordingLayer):
    def __init__(self, sparse, increment):
        super().__init__(sparse)
        self.register_buffer("increment", torch.tensor(increment, dtype=torch.bfloat16))

    def forward(self, hidden, *args):
        super().forward(hidden, *args)
        return hidden + self.increment


class _FinalResidual(nn.Module):
    def forward(self, hidden, anchors):
        return hidden + hidden.new_tensor([2, -1, 3, 1])


class KimiK3HiddenStateCaptureTest(unittest.TestCase):
    def model(self, layer_ids=(1, 0), *, tp_size=1, tp_rank=0):
        model = cp_fixtures.KimiK3MtpCpHiddenTest().model()
        model.tp_size, model.tp_rank = tp_size, tp_rank
        model._outer_layout = None
        model.parallelism_config = SimpleNamespace(role_type=RoleType.PREFILL)
        model.config.hidden_state_capture_layer_ids = list(layer_ids)
        model.layers = nn.ModuleList(
            [
                _Layer(False, [1, 2, 3, 4]),
                _Layer(True, [5, 1, -2, 7]),
            ]
        )
        model.kv_cache = cache_fixtures.TaggedCache()
        model.num_blocks = 0
        model.use_paged_conv_prefill = False
        model._init_capture_context(
            model._capture_canonical_layer, model._capture_canonical_final
        )
        return model

    def inputs(self, rows, *, requested=True, valid_mask=None, **flags):
        values = dict(
            is_prefill=True,
            is_warmup=False,
            is_target_verify=False,
            is_mtp_draft_update=False,
            is_cuda_graph=False,
            valid_token_mask=valid_mask,
        )
        values.update(flags)
        return SimpleNamespace(
            capture_hidden_states=requested,
            input_ids=torch.arange(rows),
            attention_inputs={
                "full": SimpleNamespace(**values),
                "indexer_kv": object(),
            },
        )

    def run_model(self, model, hidden, inputs):
        with patch.object(model, "_embed_inputs", side_effect=lambda _: hidden.clone()):
            return model(inputs, {"full": object(), "indexer_kv": object()})

    def boundaries(self, hidden):
        first = hidden + hidden.new_tensor([1, 2, 3, 4])
        second = first + hidden.new_tensor([5, 1, -2, 7])
        return first, second

    def test_constructor_initializes_the_existing_configured_capture_plan(self):
        config = SimpleNamespace(
            num_layers=2,
            vocab_size=8,
            hidden_size=4,
            is_mtp=False,
            capture_aux_hidden_layer_ids=[],
            hidden_state_capture_layer_ids=[1, 0],
            k3_runtime_config=SimpleNamespace(attn_res_block_size=0, dsa=None),
            layernorm_eps=1e-5,
        )
        parallel = SimpleNamespace(
            ep_size=1,
            tp_size=1,
            dp_size=1,
            pp_size=1,
            tp_rank=0,
            get_attn_tp_size=lambda: 1,
            role_type=RoleType.PREFILL,
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: False),
        )
        weights = SimpleNamespace(
            weights=[{}, {}],
            get_global_weight=lambda _: torch.ones(4),
        )
        with (
            patch.object(kimi_k3, "Embedding", return_value=nn.Identity()),
            patch.object(
                kimi_k3,
                "KimiK3DecoderLayer",
                side_effect=[
                    _Layer(False, [1, 2, 3, 4]),
                    _Layer(True, [5, 1, -2, 7]),
                ],
            ),
            patch.object(kimi_k3, "RMSNorm", return_value=nn.LayerNorm(4)),
        ):
            model = kimi_k3.KimiK3Model(config, parallel, weights, 2, max_moe_tokens=8)
        self.assertTrue(model.supports_hidden_state_capture)
        self.assertEqual(model._capture_context.layer_ids, (1, 0))
        self.assertFalse(model._capture_context.enabled)

    def test_canonical_ab_order_and_pre_post_norm_mtp_abi(self):
        for layer_ids in ((1, 0), (0,), (1,)):
            for use_final_residual in (False, True):
                with self.subTest(
                    layer_ids=layer_ids, final_residual=use_final_residual
                ):
                    model = self.model(layer_ids)
                    if use_final_residual:
                        model.num_blocks = 1
                        model.final_residual = _FinalResidual()
                    hidden = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
                    expected = self.boundaries(hidden)
                    recurrent = expected[-1]
                    if use_final_residual:
                        recurrent = model.final_residual(recurrent, None)
                    with patch.object(
                        model.norm, "forward", wraps=model.norm.forward
                    ) as norm:
                        baseline = self.run_model(
                            model, hidden, self.inputs(3, requested=False)
                        )
                        packed = self.run_model(model, hidden, self.inputs(3))
                    self.assertEqual(norm.call_count, 2)
                    torch.testing.assert_close(
                        packed.hidden_states, baseline.hidden_states, rtol=0, atol=0
                    )
                    self.assertIsNone(baseline.captured_hidden_states)
                    self.assertEqual(packed.capture_error, "")
                    self.assertTrue(model.supports_hidden_state_capture)
                    blocks = packed.captured_hidden_states.split(4, dim=-1)
                    self.assertEqual(len(blocks), len(layer_ids) + 1)
                    for actual, layer_id in zip(blocks, layer_ids):
                        torch.testing.assert_close(
                            actual, expected[layer_id], rtol=0, atol=0
                        )
                    torch.testing.assert_close(
                        blocks[-1], baseline.hidden_states, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        packed.mtp_target_hidden_states, recurrent, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        packed.mtp_target_hidden_states,
                        baseline.mtp_target_hidden_states,
                        rtol=0,
                        atol=0,
                    )
                    self.assertFalse(torch.equal(blocks[-1], recurrent))
                    dense, sparse = model.layers
                    self.assertIs(dense.received[1], model.kv_cache.full)
                    self.assertEqual(
                        sparse.received[1],
                        {
                            "full": model.kv_cache.full,
                            "indexer_kv": model.kv_cache.indexer,
                        },
                    )

    def test_capture_off_and_empty_plan_do_not_pack_clone_or_gather_aux(self):
        for layer_ids, requested in (((1, 0), False), ((), True)):
            with self.subTest(layer_ids=layer_ids, requested=requested):
                model = self.model(layer_ids)
                model._capture_context._canonical_layer = Mock(
                    side_effect=AssertionError("aux")
                )
                hidden = torch.ones(2, 4, dtype=torch.bfloat16)
                with (
                    patch.object(
                        kimi_k3, "all_gather", side_effect=AssertionError("gather")
                    ),
                    patch.object(torch, "cat", side_effect=AssertionError("pack")),
                    patch.object(
                        model,
                        "_capture_canonical_final",
                        side_effect=AssertionError("final"),
                    ),
                ):
                    output = self.run_model(
                        model, hidden, self.inputs(2, requested=requested)
                    )
                self.assertEqual(tuple(output.hidden_states.shape), (2, 4))
                self.assertFalse(model._capture_context.enabled)
                model._capture_context._canonical_layer.assert_not_called()

    def test_capture_storage_is_forward_local_and_missing_layers_fail_open(self):
        model = self.model()
        first = self.run_model(
            model, torch.ones(2, 4, dtype=torch.bfloat16), self.inputs(2)
        )
        expected = first.captured_hidden_states.clone()
        for requested in (False, True, False):
            self.run_model(
                model,
                torch.full((2, 4), 10, dtype=torch.bfloat16),
                self.inputs(2, requested=requested),
            )
            torch.testing.assert_close(
                first.captured_hidden_states, expected, rtol=0, atol=0
            )
        self.assertIsNone(model._capture_context._captured_hidden_states)
        model = self.model((2,))
        hidden = torch.ones(2, 4, dtype=torch.bfloat16)
        baseline = self.run_model(model, hidden, self.inputs(2, requested=False))
        output = self.run_model(model, hidden, self.inputs(2))
        self.assertIn("did not visit configured layers", output.capture_error)
        self.assertIsNone(output.captured_hidden_states)
        torch.testing.assert_close(
            output.hidden_states, baseline.hidden_states, rtol=0, atol=0
        )
        torch.testing.assert_close(
            output.mtp_target_hidden_states,
            baseline.mtp_target_hidden_states,
            rtol=0,
            atol=0,
        )

    def test_aux_snapshot_survives_later_workspace_mutation(self):
        model = self.model()
        hidden = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        expected = hidden.clone()
        captured = model._capture_canonical_layer(hidden, None)
        hidden.add_(100)
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        self.assertNotEqual(captured.data_ptr(), hidden.data_ptr())

    def test_sp_aux_and_final_share_global_physical_row_order(self):
        hidden = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
        boundaries = self.boundaries(hidden)
        valid = torch.tensor([True, True, True, True, False, False])
        for layer_ids in ((1, 0), (1,)):
            for rank in (0, 1):
                with self.subTest(rank=rank, layer_ids=layer_ids):
                    model = self.model(layer_ids, tp_size=2, tp_rank=rank)
                    seen = []
                    stages = [boundaries[index] for index in sorted(layer_ids)] + [
                        boundaries[1]
                    ]

                    def gather(local, group):
                        expected = stages[len(seen)]
                        torch.testing.assert_close(
                            local, expected.chunk(2)[rank], rtol=0, atol=0
                        )
                        self.assertIs(group, kimi_k3.Group.TP)
                        seen.append(local.clone())
                        return expected.clone()

                    with patch.object(kimi_k3, "all_gather", side_effect=gather):
                        output = self.run_model(
                            model, hidden, self.inputs(6, valid_mask=valid)
                        )
                    self.assertEqual(len(seen), len(layer_ids) + 1)
                    expected = torch.cat(
                        [boundaries[index] for index in layer_ids]
                        + [model.norm(boundaries[1])],
                        -1,
                    )
                    torch.testing.assert_close(
                        output.captured_hidden_states, expected, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        output.mtp_target_hidden_states, boundaries[1], rtol=0, atol=0
                    )

                    with patch.object(
                        kimi_k3, "all_gather", return_value=boundaries[1]
                    ) as gather_off:
                        baseline = self.run_model(
                            model,
                            hidden,
                            self.inputs(6, requested=False, valid_mask=valid),
                        )
                    gather_off.assert_called_once()
                    torch.testing.assert_close(
                        output.hidden_states,
                        baseline.hidden_states,
                        rtol=0,
                        atol=0,
                    )

    def test_cp_packed_rows_restore_once_with_ragged_requests_and_mtp_padding(self):
        layouts, shards, _ = cp_fixtures.KimiK3MtpCpHiddenTest().ragged_layouts()
        packed_shards = []
        for layout, hidden in zip(layouts, shards):
            model = self.model(tp_size=8, tp_rank=layout.rank)
            model._query_layout = layout
            model._initialize_mtp_hidden_handoff(
                SimpleNamespace(
                    is_speculative=True,
                    max_context_batch_size=2,
                )
            )
            inputs = self.inputs(layout.local_rows)
            with patch.object(
                kimi_k3, "all_gather", side_effect=AssertionError("CP restores in C++")
            ):
                baseline = self.run_model(
                    model, hidden, self.inputs(layout.local_rows, requested=False)
                )
                expected_mtp = baseline.mtp_target_hidden_states.clone()
                output = self.run_model(model, hidden, inputs)
            self.assertEqual(
                tuple(output.captured_hidden_states.shape), (layout.local_rows, 12)
            )
            self.assertEqual(tuple(output.hidden_states.shape), (layout.local_rows, 4))
            torch.testing.assert_close(
                output.hidden_states, baseline.hidden_states, rtol=0, atol=0
            )
            torch.testing.assert_close(
                output.mtp_target_hidden_states, expected_mtp, rtol=0, atol=0
            )
            torch.testing.assert_close(
                model.get_mtp_target_hidden_states(-1), expected_mtp, rtol=0, atol=0
            )
            self.assertEqual(
                output.mtp_target_hidden_states.data_ptr(),
                model._mtp_prefill_hidden_buffer.data_ptr(),
            )
            torch.testing.assert_close(
                output.captured_hidden_states[:, -4:],
                baseline.hidden_states,
                rtol=0,
                atol=0,
            )
            packed_shards.append(output.captured_hidden_states)

        # Model the existing CP owner's single restore over all feature blocks;
        # poisoned per-request padding must not leak into natural token rows.
        restored = layouts[0].gather_natural(torch.cat(packed_shards))
        natural = torch.arange(88, dtype=torch.bfloat16).reshape(22, 4)
        first, second = self.boundaries(natural)
        expected = torch.cat((second, first, model.norm(second)), -1)
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(restored).all())

    def test_natural_outer_capture_restores_snapshot_rows_without_changing_outputs(self):
        model = self.model(tp_size=2)
        hidden = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        valid = torch.ones(2, dtype=torch.bool)
        model._query_layout = SimpleNamespace(
            physical_rows=4, local_valid=valid, cu_host=torch.tensor([0, 2])
        )
        model._outer_layout = SimpleNamespace(local_valid=valid)
        model._routing_plan = object()
        # A reversible local permutation checks the hook boundary, not transport.
        exchange = Mock(side_effect=lambda values, plan, direction: values.flip(0))
        model._row_exchange = SimpleNamespace(exchange=exchange)
        with patch.object(
            kimi_k3, "all_gather", side_effect=AssertionError("CP restores in C++")
        ):
            baseline = self.run_model(model, hidden, self.inputs(4, requested=False))
            self.assertEqual(exchange.call_count, 2)
            exchange.reset_mock()
            output = self.run_model(model, hidden, self.inputs(4))
        self.assertEqual(
            [call.args[2] for call in exchange.call_args_list],
            ["to_natural", "to_zigzag", "to_zigzag", "to_zigzag"],
        )
        first, second = self.boundaries(hidden)
        torch.testing.assert_close(
            output.captured_hidden_states,
            torch.cat((second, first, model.norm(second)), dim=-1),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            output.hidden_states, baseline.hidden_states, rtol=0, atol=0
        )
        torch.testing.assert_close(
            output.mtp_target_hidden_states,
            baseline.mtp_target_hidden_states,
            rtol=0,
            atol=0,
        )
        self.assertEqual(output.capture_error, "")

    def test_non_target_prefill_modes_never_capture_even_if_requested(self):
        for flags in (
            {"is_prefill": False},
            {"is_warmup": True},
            {"is_target_verify": True},
            {"is_mtp_draft_update": True},
            {"is_cuda_graph": True},
        ):
            with self.subTest(flags=flags):
                model = self.model()
                with patch.object(
                    torch, "cat", side_effect=AssertionError("capture pack")
                ):
                    output = self.run_model(
                        model,
                        torch.ones(2, 4, dtype=torch.bfloat16),
                        self.inputs(2, **flags),
                    )
                self.assertEqual(tuple(output.hidden_states.shape), (2, 4))
        model.parallelism_config.role_type = RoleType.DECODE
        with patch.object(torch, "cat", side_effect=AssertionError("decode pack")):
            output = self.run_model(
                model, torch.ones(2, 4, dtype=torch.bfloat16), self.inputs(2)
            )
        self.assertEqual(tuple(output.hidden_states.shape), (2, 4))

    def test_native_draft_keeps_pre_norm_recurrence_and_ignores_capture_flag(self):
        model = cp_fixtures.KimiK3MtpCpHiddenTest().model(KimiK3MtpModel)
        model.tp_size = 1
        hidden = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        with (
            patch.object(model, "_project_local_mtp_input", return_value=hidden),
            patch.object(model, "_forward_layers", return_value=hidden) as layers,
            patch.object(
                model, "capture_context", side_effect=AssertionError("draft capture")
            ),
        ):
            output = model._forward_single(self.inputs(2))
        self.assertNotIn("capture", layers.call_args.kwargs)
        torch.testing.assert_close(
            output.mtp_target_hidden_states, hidden, rtol=0, atol=0
        )
        torch.testing.assert_close(
            output.hidden_states, model.norm(hidden), rtol=0, atol=0
        )

    def test_native_draft_terminal_outputs_keep_selected_row_abi_without_capture(self):
        for indexes in ((2, 0), ()):
            with self.subTest(indexes=indexes):
                model = cp_fixtures.KimiK3MtpCpHiddenTest().model(KimiK3MtpModel)
                model.tp_size, model.tp_rank = 1, 0
                hidden = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
                inputs = self.inputs(4)
                inputs.mtp_prefill_output_indexes = torch.tensor(
                    indexes, dtype=torch.int64
                )
                with (
                    patch.object(model, "_project_local_mtp_input", return_value=hidden),
                    patch.object(model, "_forward_layers", return_value=hidden),
                    patch.object(
                        model,
                        "_gather_modeling_boundary",
                        side_effect=AssertionError("terminal rows are already selected"),
                    ),
                    patch.object(
                        model, "capture_context", side_effect=AssertionError("draft capture")
                    ),
                ):
                    output = model._forward_single(inputs)
                selected = hidden.index_select(0, inputs.mtp_prefill_output_indexes)
                self.assertTrue(output.lm_output_already_selected)
                self.assertIsNone(output.captured_hidden_states)
                self.assertEqual(output.capture_error, "")
                torch.testing.assert_close(
                    output.mtp_target_hidden_states, selected, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    output.hidden_states,
                    model.norm(selected) if indexes else selected,
                    rtol=0,
                    atol=0,
                )

    def test_capture_failure_keeps_later_sp_gathers_and_canonical_output(self):
        hidden = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
        first, second = self.boundaries(hidden)
        for failed_rank in (0, 1):
            for rank in (0, 1):
                for failure in ("snapshot", "pack"):
                    with self.subTest(
                        rank=rank, failed_rank=failed_rank, failure=failure
                    ):
                        model = self.model(tp_size=2, tp_rank=rank)
                        snapshot = Mock(side_effect=ValueError("snapshot failure"))
                        if failure == "snapshot" and rank == failed_rank:
                            model._capture_context._canonical_layer = snapshot
                        with patch.object(
                            kimi_k3, "all_gather", side_effect=[first, second, second]
                        ) as gather:
                            if failure == "pack" and rank == failed_rank:
                                with patch.object(
                                    torch, "cat", side_effect=ValueError("pack failure")
                                ):
                                    output = self.run_model(
                                        model, hidden, self.inputs(4)
                                    )
                            else:
                                output = self.run_model(model, hidden, self.inputs(4))
                        # A failed rank and its healthy peer retain identical gather order.
                        self.assertEqual(gather.call_count, 3)
                        for call, expected in zip(
                            gather.call_args_list, (first, second, second)
                        ):
                            torch.testing.assert_close(
                                call.args[0], expected.chunk(2)[rank], rtol=0, atol=0
                            )
                        if rank == failed_rank:
                            if failure == "snapshot":
                                snapshot.assert_called_once()
                            self.assertIn(failure + " failure", output.capture_error)
                            self.assertIsNone(output.captured_hidden_states)
                        else:
                            self.assertEqual(output.capture_error, "")
                            torch.testing.assert_close(
                                output.captured_hidden_states,
                                torch.cat((second, first, model.norm(second)), dim=-1),
                                rtol=0,
                                atol=0,
                            )
                        torch.testing.assert_close(
                            output.hidden_states, model.norm(second), rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            output.mtp_target_hidden_states, second, rtol=0, atol=0
                        )

    def test_zero_rows_keep_canonical_and_separate_packed_width(self):
        model = self.model()
        hidden = torch.empty(0, 4, dtype=torch.bfloat16)
        output = self.run_model(model, hidden, self.inputs(0))
        baseline = self.run_model(model, hidden, self.inputs(0, requested=False))
        self.assertEqual(tuple(output.hidden_states.shape), (0, 4))
        self.assertEqual(tuple(output.captured_hidden_states.shape), (0, 12))
        self.assertEqual(tuple(output.mtp_target_hidden_states.shape), (0, 4))
        torch.testing.assert_close(
            output.hidden_states, baseline.hidden_states, rtol=0, atol=0
        )
        self.assertEqual(output.capture_error, "")

    def test_model_collective_and_oom_errors_propagate(self):
        model = self.model(tp_size=2)
        hidden = torch.ones(4, 4, dtype=torch.bfloat16)
        with patch.object(
            kimi_k3, "all_gather", side_effect=RuntimeError("collective failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "collective failure"):
                self.run_model(model, hidden, self.inputs(4))
        model = self.model()
        model.num_blocks = 1
        model.final_residual = _FinalResidual()
        for owner, name in (
            (model.layers[0], "forward"),
            (model.norm, "forward"),
            (model.final_residual, "forward"),
            (model, "_publish_mtp_prefill_hidden"),
        ):
            with patch.object(owner, name, side_effect=RuntimeError("model failure")):
                with self.assertRaisesRegex(RuntimeError, "model failure"):
                    self.run_model(model, hidden, self.inputs(4))
        with patch.object(
            torch, "cat", side_effect=torch.OutOfMemoryError("allocation failure")
        ):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "allocation failure"):
                self.run_model(model, hidden, self.inputs(4))


if __name__ == "__main__":
    unittest.main()
