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
    def forward(self, hidden, anchors, **kwargs):
        return hidden + hidden.new_tensor([2, -1, 3, 1])


class _Attention(nn.Module):
    _fp8_collective = None

    def __init__(self, sparse):
        super().__init__()
        self.indexer = object() if sparse else None

    def forward(self, hidden, *args):
        return hidden * 0.25


class _Mlp(nn.Module):
    def forward(self, hidden, valid_mask):
        return hidden * 0.125


class _AttnResLayer(kimi_k3.KimiK3DecoderLayer):
    """Real decoder residual math with lightweight attention/MLP on CPU."""

    def __init__(self, index, block_size, dtype):
        nn.Module.__init__(self)
        self.index = index
        self.block_size = block_size
        self.layer_type = "dense"
        self.attention = _Attention(sparse=index % 2 == 1)
        self.mlp = _Mlp()
        self.apply_mlp = self._apply_regular_mlp
        norm_weight = torch.tensor([1, 0.5, 1.5, 2], dtype=dtype)
        projection = torch.tensor([0.5, -1, 0.25, 1], dtype=dtype)
        self.attention_residual = kimi_k3.KimiK3AttentionResidual(norm_weight, projection.roll(index % 4), 1e-5)
        self.mlp_residual = kimi_k3.KimiK3AttentionResidual(norm_weight, -projection, 1e-5)
        self.attention_norm = SimpleNamespace(weight=norm_weight, variance_epsilon=1e-5)
        self.mlp_norm = SimpleNamespace(weight=norm_weight.flip(0), variance_epsilon=1e-5)


class KimiK3HiddenStateCaptureTest(unittest.TestCase):
    def model(self, layer_ids=(1, 0), *, tp_size=1, tp_rank=0):
        model = cp_fixtures.KimiK3MtpCpHiddenTest().model()
        model.tp_size, model.tp_rank = tp_size, tp_rank
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

    def test_binding_preserves_two_tensor_constructor_and_optional_side_fields(self):
        canonical = torch.ones(2, 4)
        prenorm = torch.zeros(2, 4)
        for output in (kimi_k3.PyModelOutputs(), kimi_k3.PyModelOutputs(canonical)):
            self.assertIsNone(output.captured_hidden_states)
            self.assertIsNone(output.capture_error)
        output = kimi_k3.PyModelOutputs(canonical, prenorm)
        self.assertEqual(output.hidden_states.data_ptr(), canonical.data_ptr())
        self.assertEqual(output.mtp_target_hidden_states.data_ptr(), prenorm.data_ptr())
        output.captured_hidden_states = canonical
        output.capture_error = "ValueError: capture"
        self.assertEqual(output.captured_hidden_states.data_ptr(), canonical.data_ptr())
        self.assertEqual(output.capture_error, "ValueError: capture")
        output.captured_hidden_states = None
        output.capture_error = None
        self.assertIsNone(output.captured_hidden_states)
        self.assertIsNone(output.capture_error)

    def test_recoverable_capture_error_keeps_sp_collective_schedule(self):
        hidden = torch.ones(4, 4, dtype=torch.bfloat16)
        for rank in (0, 1):
            model = self.model(tp_size=2, tp_rank=rank)
            model._capture_context._canonical_layer = Mock(
                side_effect=ValueError("snapshot failure")
            )
            with patch.object(
                kimi_k3, "all_gather", side_effect=lambda h, _: torch.cat((h, h))
            ) as gather:
                output = self.run_model(model, hidden, self.inputs(4))
            self.assertEqual(gather.call_count, 3)
            self.assertIsNone(output.captured_hidden_states)
            self.assertEqual(output.capture_error, "ValueError: snapshot failure")
            self.assertEqual(tuple(output.hidden_states.shape), (4, 4))
            model._capture_context._canonical_layer.assert_called_once()

    def test_model_norm_and_capture_oom_remain_fatal(self):
        hidden = torch.ones(2, 4, dtype=torch.bfloat16)
        for boundary in ("layer", "norm", "capture"):
            with self.subTest(boundary=boundary):
                model = self.model()
                error = (
                    torch.OutOfMemoryError("allocation")
                    if boundary == "capture"
                    else ValueError(boundary)
                )
                if boundary == "capture":
                    model._capture_context._canonical_layer = Mock(side_effect=error)
                    with self.assertRaises(type(error)):
                        self.run_model(model, hidden, self.inputs(2))
                else:
                    module = model.layers[0] if boundary == "layer" else model.norm
                    with patch.object(module, "forward", side_effect=error):
                        with self.assertRaisesRegex(ValueError, boundary):
                            self.run_model(model, hidden, self.inputs(2))

    def test_constructor_initializes_the_existing_configured_capture_plan(self):
        config = SimpleNamespace(
            num_layers=2,
            vocab_size=8,
            hidden_size=4,
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
                        model.layers[1].block_size = 12
                        model.layers[1].attention_residual = _FinalResidual()
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
                    self.assertTrue(model.supports_hidden_state_capture)
                    blocks = packed.captured_hidden_states.split(4, dim=-1)
                    torch.testing.assert_close(
                        packed.hidden_states, baseline.hidden_states, rtol=0, atol=0
                    )
                    self.assertIsNone(packed.capture_error)
                    self.assertEqual(len(blocks), len(layer_ids) + 1)
                    for actual, layer_id in zip(blocks, layer_ids):
                        feature = expected[layer_id]
                        if use_final_residual:
                            feature = model.final_residual(feature, None)
                        torch.testing.assert_close(actual, feature, rtol=0, atol=0)
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

    def attn_res_model(self, block_size, layer_ids, dtype=torch.float32):
        model = self.model(layer_ids)
        model.norm = nn.LayerNorm(4, dtype=dtype)
        cache = model.kv_cache
        cache.get_layer_cache_groups = lambda index: ([cache.indexer, cache.full] if index % 2 else [cache.full])
        cache.get_layer_cache = lambda index: cache.full
        model.layers = nn.ModuleList(_AttnResLayer(i, block_size, dtype) for i in range(2 * block_size + 2))
        model.num_blocks = (len(model.layers) + block_size - 1) // block_size
        model.final_residual = kimi_k3.KimiK3AttentionResidual(
            torch.tensor([0.5, 2, 1, 1.5], dtype=dtype),
            torch.tensor([-1, 0.25, 2, -0.5], dtype=dtype),
            1e-5,
        )
        return model

    @staticmethod
    def mixture_reference(prefix, anchors, residual, num_blocks):
        # Independent FP64 oracle; no production AttnRes call or output RMSNorm.
        sources = torch.stack(
            [anchors[:, i].double() for i in range(num_blocks)] + [prefix.double()],
            dim=1,
        )
        rms = (sources.square().mean(-1) + residual.eps).sqrt()
        query = residual.norm_weight.double() * residual.projection_weight.double()
        scores = (sources * query).sum(-1) / rms
        probabilities = scores.softmax(dim=1)
        return (sources * probabilities.unsqueeze(-1)).sum(1).to(prefix.dtype)

    def test_real_attn_res_mixture_boundaries_order_purity_and_final_reuse(self):
        # Hand-written post-layer -> (consumer, valid anchors) contract for B12.
        b12_contract = {
            0: (1, 1),
            11: (12, 1),
            12: (13, 2),
            23: (24, 2),
            24: (25, 3),
            25: ("output", 3),
        }
        for block_size in (3, 12):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(block_size=block_size, dtype=dtype):
                    last = 2 * block_size + 1
                    ids = (last, block_size, block_size - 1, 2 * block_size, 0)
                    if block_size == 12:
                        ids = (25, 12, 11, 24, 23, 0)
                        self.assertEqual(set(ids), set(b12_contract))
                    model = self.attn_res_model(block_size, ids, dtype)
                    hidden = torch.tensor([[2, -1, 3, 0.5], [-2, 4, 1, -3]], dtype=dtype)
                    baseline = self.run_model(model, hidden, self.inputs(2, requested=False))
                    expected = {}
                    raw = {}
                    checked_taps = []
                    checked_final = []
                    tap = model._capture_layer_output
                    final = model.final_residual.forward
                    tolerance = 0.008 if dtype == torch.bfloat16 else 1e-6

                    def observe(capture, index, prefix, anchors=None):
                        if anchors is None:
                            # The last aux slot must reuse the final consumer, not
                            # the last decoder's raw prefix or private norm.
                            if index in ids:
                                if block_size == 12:
                                    self.assertEqual(b12_contract[index], ("output", 3))
                                    checked_final.append(index)
                                torch.testing.assert_close(prefix, expected[index], rtol=tolerance, atol=tolerance)
                            return tap(capture, index, prefix)
                        before_prefix, before_anchors = prefix.clone(), anchors.clone()
                        if index in ids:
                            raw[index] = prefix.clone()
                            if block_size == 12:
                                consumer_id, count = b12_contract[index]
                                consumer = (
                                    model.final_residual
                                    if consumer_id == "output"
                                    else model.layers[consumer_id].attention_residual
                                )
                            else:
                                consumer = (
                                    model.final_residual if index == last else model.layers[index + 1].attention_residual
                                )
                                count = model.num_blocks if index == last else (index + block_size) // block_size
                            expected[index] = self.mixture_reference(prefix, anchors, consumer, count)
                        if block_size == 12 and index in ids and consumer_id != "output":
                            with patch.object(consumer, "forward", wraps=consumer.forward) as actual_tap:
                                result = tap(capture, index, prefix, anchors)
                            # Outside the capture error boundary: wrong consumer or
                            # count must fail, even if feature values happen to agree.
                            actual_tap.assert_called_once_with(
                                prefix,
                                anchors,
                                num_blocks=count,
                                delta=None,
                                block_write_idx=-1,
                                output_norm_weight=None,
                            )
                            checked_taps.append(index)
                        else:
                            result = tap(capture, index, prefix, anchors)
                        torch.testing.assert_close(prefix, before_prefix, rtol=0, atol=0)
                        torch.testing.assert_close(anchors, before_anchors, rtol=0, atol=0, equal_nan=True)
                        return result

                    with (
                        patch.object(model, "_capture_layer_output", side_effect=observe),
                        patch.object(model.final_residual, "forward", wraps=final) as output_mix,
                    ):
                        output = self.run_model(model, hidden, self.inputs(2))
                    output_mix.assert_called_once()
                    if block_size == 12:
                        self.assertEqual(checked_taps, [0, 11, 12, 23, 24])
                        self.assertEqual(checked_final, [25])
                        # Output AttnRes defaults to all anchors when no count is
                        # passed; inspect the actual serving call reused by the tap.
                        output_anchors = output_mix.call_args.args[1]
                        self.assertEqual(output_anchors.shape[1], b12_contract[25][1])
                        self.assertEqual(
                            output_mix.call_args.kwargs.get("num_blocks", output_anchors.shape[1]),
                            b12_contract[25][1],
                        )
                    self.assertIsNone(output.capture_error)
                    parts = output.captured_hidden_states.split(4, dim=-1)
                    for actual, index in zip(parts[:-1], ids):
                        self.assertEqual(actual.dtype, dtype)
                        torch.testing.assert_close(actual, expected[index], rtol=tolerance, atol=tolerance)
                        self.assertFalse(torch.allclose(actual, raw[index]))
                    torch.testing.assert_close(parts[-1], baseline.hidden_states, rtol=0, atol=0)
                    torch.testing.assert_close(output.hidden_states, baseline.hidden_states, rtol=0, atol=0)
                    torch.testing.assert_close(
                        output.mtp_target_hidden_states,
                        baseline.mtp_target_hidden_states,
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(parts[0], output.mtp_target_hidden_states, rtol=0, atol=0)

    def test_mixture_failure_isolated_with_real_layers_and_sp_schedule(self):
        for tp_size in (1, 2):
            with self.subTest(tp_size=tp_size):
                model = self.attn_res_model(12, (25, 12, 0))
                model.tp_size = tp_size
                hidden = torch.tensor([[2, -1, 3, 0.5]] * (2 * tp_size))
                with patch.object(kimi_k3, "all_gather", side_effect=lambda h, _: torch.cat((h, h))):
                    baseline = self.run_model(model, hidden, self.inputs(len(hidden), requested=False))
                consumer = model.layers[1].attention_residual
                original = consumer.forward
                tap_calls = []

                def fail_tap(*args, **kwargs):
                    if kwargs.get("output_norm_weight") is None:
                        tap_calls.append(True)
                        raise ValueError("mixture failure")
                    return original(*args, **kwargs)

                later = model.layers[13].attention_residual
                with (
                    patch.object(consumer, "forward", side_effect=fail_tap),
                    patch.object(later, "forward", wraps=later.forward) as later_calls,
                    patch.object(kimi_k3, "all_gather", side_effect=lambda h, _: torch.cat((h, h))) as gather,
                ):
                    output = self.run_model(model, hidden, self.inputs(len(hidden)))
                self.assertEqual(len(tap_calls), 1)
                # The later consumer still runs normally, but no longer as a tap.
                self.assertEqual(later_calls.call_count, 1)
                self.assertIsNotNone(later_calls.call_args.kwargs["output_norm_weight"])
                self.assertEqual(gather.call_count, 4 if tp_size > 1 else 0)
                self.assertIsNone(output.captured_hidden_states)
                self.assertEqual(output.capture_error, "ValueError: mixture failure")
                torch.testing.assert_close(output.hidden_states, baseline.hidden_states, rtol=0, atol=0)
                torch.testing.assert_close(
                    output.mtp_target_hidden_states, baseline.mtp_target_hidden_states, rtol=0, atol=0
                )

    def test_mixture_fatal_error_and_final_serving_error_propagate(self):
        for final in (False, True):
            model = self.attn_res_model(12, (0, 25))
            module = model.final_residual if final else model.layers[1].attention_residual
            error = ValueError("final serving failure") if final else torch.OutOfMemoryError("allocation")
            with patch.object(module, "forward", side_effect=error):
                with self.assertRaises(type(error)):
                    self.run_model(model, torch.ones(2, 4), self.inputs(2))

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

    def test_capture_storage_is_forward_local_and_missing_layers_fail(self):
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
        output = self.run_model(
            model, torch.ones(2, 4, dtype=torch.bfloat16), self.inputs(2)
        )
        self.assertIsNone(output.captured_hidden_states)
        self.assertIn("did not visit configured layers", output.capture_error)
        self.assertEqual(tuple(output.hidden_states.shape), (2, 4))

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
            self.assertEqual(tuple(output.hidden_states.shape), (layout.local_rows, 4))
            self.assertEqual(
                tuple(output.captured_hidden_states.shape), (layout.local_rows, 12)
            )
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
            packed_shards.append(output.captured_hidden_states)

        # Model the existing CP owner's single restore over all feature blocks;
        # poisoned per-request padding must not leak into natural token rows.
        restored = layouts[0].gather_natural(torch.cat(packed_shards))
        natural = torch.arange(88, dtype=torch.bfloat16).reshape(22, 4)
        first, second = self.boundaries(natural)
        expected = torch.cat((second, first, model.norm(second)), -1)
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(restored).all())

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

    def test_collective_and_pack_errors_propagate(self):
        model = self.model(tp_size=2)
        hidden = torch.ones(4, 4, dtype=torch.bfloat16)
        with patch.object(
            kimi_k3, "all_gather", side_effect=RuntimeError("collective failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "collective failure"):
                self.run_model(model, hidden, self.inputs(4))
        model = self.model()
        with patch.object(torch, "cat", side_effect=RuntimeError("CUDA out of memory")):
            with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                self.run_model(model, hidden, self.inputs(4))


if __name__ == "__main__":
    unittest.main()
