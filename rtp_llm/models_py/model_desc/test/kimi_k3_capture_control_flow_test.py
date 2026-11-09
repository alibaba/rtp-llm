"""Execute K3's actual control flow without torch; not a tensor/CUDA test.

Only the selected production methods are compiled. Dependency mocks model
control-flow boundaries, never tensor arithmetic or distributed correctness.
"""

import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class KimiK3CaptureControlFlowTest(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "kimi_k3.py"
        tree = ast.parse(source.read_text(), filename=str(source))
        model = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "KimiK3Model"
        )
        methods = {
            "_capture_hidden_states_requested",
            "_capture_canonical_layer",
            "_capture_layer",
            "_capture_canonical_final",
            "_forward_layers",
            "_forward_single",
        }
        selected = ast.ClassDef(
            name="KimiK3Model",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[
                node
                for node in model.body
                if isinstance(node, ast.FunctionDef) and node.name in methods
            ],
        )
        self.assertEqual({node.name for node in selected.body}, methods)
        self.primary = SimpleNamespace(
            is_prefill=True,
            is_warmup=False,
            is_target_verify=False,
            is_mtp_draft_update=False,
            is_cuda_graph=False,
            valid_token_mask=None,
        )
        self.cuda_check = Mock(return_value=False)
        self.gather = Mock()
        self.env = dict(
            RoleType=SimpleNamespace(DECODE="decode"),
            HybridAttentionType=SimpleNamespace(LINEAR="linear"),
            torch=SimpleNamespace(
                cuda=SimpleNamespace(is_current_stream_capturing=self.cuda_check)
            ),
            get_primary_attention_inputs=Mock(return_value=self.primary),
            get_attention_inputs_value=lambda inputs: inputs.attention_inputs,
            select_attention_inputs_for_tag=lambda tagged, tag: tagged[tag],
            select_attention_inputs_for_layer=Mock(),
            select_fmha_impl_for_group=lambda tagged, tag: tagged[tag],
            select_fmha_impl_for_layer=Mock(),
            get_layer_caches_for_groups=Mock(),
            KdaMetadata=SimpleNamespace,
            Group=SimpleNamespace(TP="tp"),
            all_gather=self.gather,
            profile_scope=lambda _: nullcontext(),
            PyModelOutputs=lambda hidden, recurrent=None: SimpleNamespace(
                hidden_states=hidden,
                mtp_target_hidden_states=recurrent,
                captured_hidden_states=None,
                capture_error="",
            ),
        )
        module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
        exec(compile(module, str(source), "exec"), self.env)
        self.cls = self.env["KimiK3Model"]
        self.model = self.cls()
        self.model.parallelism_config = SimpleNamespace(role_type="prefill")
        self.model.kv_cache = None
        self.model.tp_size, self.model.tp_rank = 1, 0
        self.model._query_layout = None
        self.model._outer_layout = None
        self.inputs = SimpleNamespace(
            capture_hidden_states=True,
            input_ids=SimpleNamespace(is_cuda=False, shape=(2,)),
            attention_inputs={"full": self.primary, "indexer_kv": object()},
        )

    def test_only_real_target_prefill_requests_are_eligible(self):
        for role in ("prefill", "pdfusion"):
            self.model.parallelism_config.role_type = role
            self.assertTrue(self.model._capture_hidden_states_requested(self.inputs))
        for field in (
            "is_warmup",
            "is_target_verify",
            "is_mtp_draft_update",
            "is_cuda_graph",
        ):
            with self.subTest(field=field):
                setattr(self.primary, field, True)
                self.assertFalse(
                    self.model._capture_hidden_states_requested(self.inputs)
                )
                setattr(self.primary, field, False)
        self.primary.is_prefill = False
        self.assertFalse(self.model._capture_hidden_states_requested(self.inputs))
        self.primary.is_prefill = True
        self.model.parallelism_config.role_type = "decode"
        self.assertFalse(self.model._capture_hidden_states_requested(self.inputs))
        self.cuda_check.assert_not_called()

    def test_disabled_request_and_native_draft_short_circuit_metadata(self):
        self.inputs.capture_hidden_states = False
        self.assertFalse(self.model._capture_hidden_states_requested(self.inputs))
        self.inputs.capture_hidden_states = True
        draft = type("NativeDraft", (self.cls,), {})()
        self.assertFalse(draft._capture_hidden_states_requested(self.inputs))
        self.env["get_primary_attention_inputs"].assert_not_called()
        self.cuda_check.assert_not_called()

    def test_stream_capture_is_excluded_and_cuda_errors_propagate(self):
        self.inputs.input_ids.is_cuda = True
        self.cuda_check.return_value = True
        self.assertFalse(self.model._capture_hidden_states_requested(self.inputs))
        self.cuda_check.return_value = False
        self.assertTrue(self.model._capture_hidden_states_requested(self.inputs))
        self.cuda_check.side_effect = RuntimeError("CUDA failure")
        with self.assertRaisesRegex(RuntimeError, "CUDA failure"):
            self.model._capture_hidden_states_requested(self.inputs)

    def test_aux_layout_transition_uses_sp_gather_but_not_query_cp(self):
        hidden = Mock()
        self.model.tp_size = 8
        capture = SimpleNamespace(
            wants_layer=lambda index: index in (0, 1),
            capture_layer=Mock(),
            failed=False,
        )
        self.model._capture_layer(capture, 0, hidden)
        capture.capture_layer.assert_called_once_with(0, self.gather.return_value)
        self.assertIs(
            self.model._capture_canonical_layer(self.gather.return_value, None),
            self.gather.return_value,
        )
        self.gather.assert_called_once_with(hidden, "tp")
        # Snapshot health must not suppress the next rank-matched collective.
        capture.failed = True
        self.model._capture_layer(capture, 1, hidden)
        self.assertEqual(self.gather.call_count, 2)
        self.model._capture_layer(capture, 2, hidden)
        self.assertEqual(self.gather.call_count, 2)
        hidden.clone.assert_not_called()
        self.gather.reset_mock()
        self.model._query_layout = object()
        self.model._capture_layer(capture, 0, hidden)
        capture.capture_layer.assert_called_with(0, hidden)
        self.assertIs(
            self.model._capture_canonical_layer(hidden, None), hidden.clone.return_value
        )
        self.gather.assert_not_called()
        self.gather.side_effect = RuntimeError("collective failure")
        self.model._query_layout = None
        with self.assertRaisesRegex(RuntimeError, "collective failure"):
            self.model._capture_layer(capture, 0, hidden)

    def test_natural_outer_capture_transport_survives_snapshot_failure(self):
        hidden = Mock()
        self.model.tp_size = 8
        self.model._query_layout = object()
        self.model._outer_layout = object()
        self.model._routing_plan = object()
        self.model._row_exchange = SimpleNamespace(exchange=Mock())
        capture = SimpleNamespace(
            wants_layer=lambda index: index in (0, 1),
            capture_layer=Mock(),
            failed=True,
        )
        for index in (0, 1):
            self.model._capture_layer(capture, index, hidden)
            capture.capture_layer.assert_called_with(
                index, self.model._row_exchange.exchange.return_value
            )
        self.assertEqual(self.model._row_exchange.exchange.call_count, 2)
        for call in self.model._row_exchange.exchange.call_args_list:
            self.assertEqual(
                call.args, (hidden, self.model._routing_plan, "to_zigzag")
            )
        self.gather.assert_not_called()
        self.model._capture_layer(None, 0, hidden)
        self.model._capture_layer(capture, 2, hidden)
        self.assertEqual(self.model._row_exchange.exchange.call_count, 2)
        self.model._row_exchange.exchange.side_effect = RuntimeError(
            "row exchange failure"
        )
        with self.assertRaisesRegex(RuntimeError, "row exchange failure"):
            self.model._capture_layer(capture, 0, hidden)

    def test_dense_sparse_hooks_precede_final_residual_and_mtp_publish(self):
        events = []
        hidden, dense_hidden, sparse_hidden, final_hidden = [
            Mock(shape=(2, 4)) for _ in range(4)
        ]
        dense = Mock(
            attention=SimpleNamespace(indexer=None),
            layer_type="dense",
            return_value=dense_hidden,
        )
        sparse = Mock(
            attention=SimpleNamespace(indexer=object()),
            layer_type="dense",
            return_value=sparse_hidden,
        )
        self.model.layers = [dense, sparse]
        self.model.num_blocks = 1
        self.model.final_residual = Mock(
            side_effect=lambda *args: events.append("final_residual") or final_hidden
        )
        self.model._publish_mtp_prefill_hidden = Mock(
            side_effect=lambda value: events.append("mtp") or value
        )
        capture = SimpleNamespace(
            wants_layer=lambda index: True,
            capture_layer=lambda index, value: events.append((index, value)),
        )
        actual = self.model._forward_layers(
            hidden,
            self.inputs,
            {"full": object(), "indexer_kv": object()},
            sequence_parallel_input=True,
            capture=capture,
        )
        self.assertIs(actual, final_hidden)
        self.assertEqual(
            events, [(0, dense_hidden), (1, sparse_hidden), "final_residual", "mtp"]
        )
        self.assertIs(sparse.call_args.args[0], dense_hidden)
        self.gather.assert_not_called()

    def test_output_second_argument_stays_recurrent_tensor_on_and_off(self):
        hidden, normalized, packed = object(), object(), object()
        self.model._forward_layers = Mock(return_value=hidden)
        self.model.norm = Mock(return_value=normalized)
        for enabled, error in ((False, ""), (True, ""), (True, "snapshot failure")):
            with self.subTest(enabled=enabled, error=error):
                capture = SimpleNamespace(
                    enabled=enabled,
                    finalize=Mock(
                        side_effect=lambda value: SimpleNamespace(
                            hidden_states=self.model.norm(value),
                            captured_hidden_states=None if error else packed,
                            capture_error=error,
                        )
                    ),
                )
                self.model.capture_context = Mock(return_value=capture)
                self.model.norm.reset_mock()
                output = self.model._forward_single(self.inputs)
                self.assertIs(output.mtp_target_hidden_states, hidden)
                self.assertIs(output.hidden_states, normalized)
                self.assertIs(
                    output.captured_hidden_states,
                    packed if enabled and not error else None,
                )
                self.assertEqual(output.capture_error, error)
                self.assertIs(
                    self.model._forward_layers.call_args.kwargs["capture"],
                    capture if enabled else None,
                )
                if enabled:
                    capture.finalize.assert_called_once_with(hidden)
                else:
                    capture.finalize.assert_not_called()
                self.model.norm.assert_called_once_with(hidden)
        # A real forward failure is outside the capture-isolation boundary.
        self.model._forward_layers.side_effect = RuntimeError("model failure")
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            self.model._forward_single(self.inputs)
        self.model._forward_layers.side_effect = None
        self.model.norm.side_effect = RuntimeError("final norm failure")
        with self.assertRaisesRegex(RuntimeError, "final norm failure"):
            self.model._forward_single(self.inputs)


if __name__ == "__main__":
    unittest.main()
