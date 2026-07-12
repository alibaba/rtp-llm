import unittest
from types import SimpleNamespace
from unittest import mock

import torch


class TestQwen3NextGraphProbe(unittest.TestCase):
    def test_model_probe_can_be_disabled_for_normal_graph_capture(self):
        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextModel,
            _CudaGraphLayerProbe,
        )

        model = Qwen3NextModel.__new__(Qwen3NextModel)
        torch.nn.Module.__init__(model)
        model._cuda_graph_layer_probe = _CudaGraphLayerProbe(
            enabled=True, layers=(0,), layer_num=1
        )

        self.assertTrue(model.get_cuda_graph_probe_enabled())
        self.assertTrue(model.set_cuda_graph_probe_enabled(False))
        self.assertFalse(model.get_cuda_graph_probe_enabled())

        hidden = torch.ones((2, 4), dtype=torch.float32)
        model._cuda_graph_layer_probe.record(
            0, hidden, hidden, graph_bs=2, is_cuda_graph=True
        )
        self.assertIsNone(model.get_cuda_graph_probe_buffer(2))

        self.assertFalse(model.set_cuda_graph_probe_enabled(True))
        model._cuda_graph_layer_probe.record(
            0, hidden, hidden, graph_bs=2, is_cuda_graph=True
        )
        self.assertIsNotNone(model.get_cuda_graph_probe_buffer(2))

    def test_trace_context_is_only_exposed_to_selected_layer(self):
        from rtp_llm.models_py.model_desc import qwen3_next

        metadata = SimpleNamespace(trace_ctx=None)
        trace_ctx = {"trace_ids": ["target"]}

        with mock.patch.object(qwen3_next, "_Q3N_TRACE_ENABLED", True), mock.patch.object(
            qwen3_next, "_Q3N_TRACE_LAYERS", {1}
        ):
            selected = qwen3_next._set_layer_trace_context(
                metadata,
                trace_ctx,
                1,
                qwen3_next.HybridAttentionType.LINEAR,
            )
            self.assertTrue(selected)
            self.assertIs(metadata.trace_ctx, trace_ctx)

            selected = qwen3_next._set_layer_trace_context(
                metadata,
                trace_ctx,
                2,
                qwen3_next.HybridAttentionType.LINEAR,
            )
            self.assertFalse(selected)
            self.assertIsNone(metadata.trace_ctx)

    def test_prefill_sequence_slice_uses_lane_cumulative_lengths(self):
        from rtp_llm.models_py.model_desc import qwen3_next

        attention_inputs = SimpleNamespace(
            cu_seqlens_host=torch.tensor([0, 2, 5], dtype=torch.int32),
            cu_seqlens=None,
        )
        packed = torch.arange(20).view(5, 4)

        selected = qwen3_next._prefill_sequence_slice(
            attention_inputs, packed, lane=1
        )

        torch.testing.assert_close(selected, packed[2:5])

    def test_stats_mask_nonfinite_values(self):
        from rtp_llm.models_py.model_desc.qwen3_next import _graph_probe_stats

        tensor = torch.tensor([[1.0, -2.0, float("nan"), float("inf")]])

        stats = _graph_probe_stats(tensor)

        torch.testing.assert_close(
            stats,
            torch.tensor([[-1.0, 3.0, 5.0, -2.0, 1.0, 2.0]]),
        )

    def test_aggregates_multi_token_rows_by_explicit_graph_batch_size(self):
        from rtp_llm.models_py.model_desc.qwen3_next import (
            _CudaGraphLayerProbe,
            _graph_probe_stats,
        )

        probe = _CudaGraphLayerProbe(enabled=True, layers=(0, 1), layer_num=2)
        hidden = torch.arange(32 * 3, dtype=torch.float32).reshape(32, 3)
        residual = hidden + 1000

        probe.record(
            1,
            hidden,
            residual,
            graph_bs=8,
            is_cuda_graph=True,
        )

        bucket = probe.get_buffer(8)
        self.assertEqual((2, 8, 12), tuple(bucket.shape))
        self.assertIsNone(probe.get_buffer(32))
        torch.testing.assert_close(
            bucket[1, :, :6], _graph_probe_stats(hidden.reshape(8, -1))
        )
        torch.testing.assert_close(
            bucket[1, :, 6:], _graph_probe_stats(residual.reshape(8, -1))
        )

    def test_validates_layers_initializes_unexecuted_rows_and_returns_metadata(self):
        from rtp_llm.models_py.model_desc.qwen3_next import _CudaGraphLayerProbe

        probe = _CudaGraphLayerProbe(
            enabled=True,
            layers=(3, -1, 1, 3, 4, 0),
            layer_num=4,
        )
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        probe.record(
            1,
            hidden,
            hidden,
            graph_bs=2,
            is_cuda_graph=True,
        )

        buffer, layers = probe.get_capture(2)
        self.assertEqual((3, 1, 0), layers)
        self.assertTrue(torch.count_nonzero(buffer[0]).item() == 0)
        self.assertTrue(torch.count_nonzero(buffer[2]).item() == 0)

    def test_reuses_persistent_destination_and_preserves_one_token_decode(self):
        from rtp_llm.models_py.model_desc.qwen3_next import (
            _CudaGraphLayerProbe,
            _graph_probe_stats,
        )

        probe = _CudaGraphLayerProbe(enabled=True, layers=(0,), layer_num=1)
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        residual = hidden + 10
        probe.record(
            0,
            hidden,
            residual,
            graph_bs=2,
            is_cuda_graph=True,
        )
        buffer = probe.get_buffer(2)
        data_ptr = buffer.data_ptr()

        updated_hidden = hidden + 100
        updated_residual = residual + 100
        probe.record(
            0,
            updated_hidden,
            updated_residual,
            graph_bs=2,
            is_cuda_graph=True,
        )

        self.assertEqual(data_ptr, probe.get_buffer(2).data_ptr())
        torch.testing.assert_close(
            buffer[0, :, :6], _graph_probe_stats(updated_hidden)
        )
        torch.testing.assert_close(
            buffer[0, :, 6:], _graph_probe_stats(updated_residual)
        )

    def test_disabled_or_non_graph_probe_does_not_allocate(self):
        from rtp_llm.models_py.model_desc.qwen3_next import _CudaGraphLayerProbe

        hidden = torch.ones((2, 4))
        disabled = _CudaGraphLayerProbe(enabled=False, layers=(0,), layer_num=1)
        enabled = _CudaGraphLayerProbe(enabled=True, layers=(0,), layer_num=1)

        disabled.record(0, hidden, hidden, graph_bs=2, is_cuda_graph=True)
        enabled.record(0, hidden, hidden, graph_bs=2, is_cuda_graph=False)

        self.assertIsNone(disabled.get_buffer(2))
        self.assertIsNone(enabled.get_buffer(2))

    def test_probe_reports_record_guard_outcomes(self):
        from rtp_llm.models_py.model_desc.qwen3_next import _CudaGraphLayerProbe

        probe = _CudaGraphLayerProbe(enabled=True, layers=(0,), layer_num=1)
        hidden = torch.ones((4, 2), dtype=torch.float32)

        probe.record(0, hidden, hidden, graph_bs=2, is_cuda_graph=False)
        probe.record(
            0,
            torch.tensor(1.0),
            torch.tensor(1.0),
            graph_bs=1,
            is_cuda_graph=True,
        )
        probe.record(0, hidden[:3], hidden[:3], graph_bs=2, is_cuda_graph=True)
        probe.record(0, hidden, hidden, graph_bs=2, is_cuda_graph=True)

        self.assertEqual(
            {
                "attempts": 4,
                "recorded": 1,
                "skipped_not_cuda_graph": 1,
                "skipped_invalid_tensor": 1,
                "skipped_invalid_layout": 1,
                "last_layer_idx": 0,
                "last_graph_bs": 2,
                "last_token_rows": 4,
                "last_residual_rows": 4,
                "last_is_cuda_graph": 1,
            },
            probe.get_debug_status(),
        )

    def test_records_selected_attention_stage_slots(self):
        from rtp_llm.models_py.model_desc.qwen3_next import (
            _CudaGraphLayerProbe,
            _graph_probe_stats,
        )

        probe = _CudaGraphLayerProbe(
            enabled=True,
            layers=(6, 7),
            layer_num=8,
            stage_layer=7,
        )
        primary = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        secondary = primary + 10

        probe.record_stage(
            7,
            "attention_input",
            primary,
            secondary,
            graph_bs=2,
            is_cuda_graph=True,
        )
        probe.record_stage(
            6,
            "fmha_output",
            primary + 100,
            graph_bs=2,
            is_cuda_graph=True,
        )

        buffer, slots = probe.get_capture(2)
        slot = slots.index(probe.stage_slot_ids["attention_input"])
        torch.testing.assert_close(buffer[slot, :, :6], _graph_probe_stats(primary))
        torch.testing.assert_close(
            buffer[slot, :, 6:], _graph_probe_stats(secondary)
        )
        self.assertEqual(0, torch.count_nonzero(buffer[slots.index(7)]).item())

    def test_attention_stage_slot_ids_remain_stable_when_linear_stages_are_added(self):
        from rtp_llm.models_py.model_desc.qwen3_next import _CudaGraphLayerProbe

        probe = _CudaGraphLayerProbe(
            enabled=True,
            layers=(2,),
            layer_num=3,
            stage_layer=2,
        )

        self.assertEqual(
            {
                "attention_input": -301,
                "attention_gate": -302,
                "qkv_projected": -303,
                "qkv_normalized": -304,
                "fmha_output": -305,
                "gated_output": -306,
                "o_proj_output": -307,
                "attention_output": -308,
                "post_attention_norm": -309,
            },
            {
                stage: probe.stage_slot_ids[stage]
                for stage in (
                    "attention_input",
                    "attention_gate",
                    "qkv_projected",
                    "qkv_normalized",
                    "fmha_output",
                    "gated_output",
                    "o_proj_output",
                    "attention_output",
                    "post_attention_norm",
                )
            },
        )

        self.assertEqual(-316, probe.stage_slot_ids["linear_conv_state_input"])
        self.assertEqual(-317, probe.stage_slot_ids["linear_ssm_state_input"])

    def test_gathers_decode_cache_state_with_kernel_block_formula(self):
        from rtp_llm.models_py.model_desc.qwen3_next import (
            _gather_decode_cache_state_for_graph_probe,
        )

        states = torch.arange(30 * 2, dtype=torch.float32).reshape(30, 2)
        block_map = torch.tensor(
            [
                [10, 11, 12],
                [20, 21, 22],
                [-1, -1, -1],
                [0, 0, 0],
            ],
            dtype=torch.int32,
        )
        sequence_lengths_plus_1 = torch.tensor(
            [66, 130, 0, 66], dtype=torch.int32
        )

        gathered = _gather_decode_cache_state_for_graph_probe(
            states,
            block_map,
            sequence_lengths_plus_1,
            seq_size_per_block=64,
        )

        torch.testing.assert_close(gathered[0], states[11])
        torch.testing.assert_close(gathered[1], states[22])
        torch.testing.assert_close(gathered[2], torch.zeros(2))
        torch.testing.assert_close(gathered[3], torch.zeros(2))

    def test_qwen3_next_attention_records_internal_stage_probe(self):
        from types import SimpleNamespace

        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextAttention,
            _CudaGraphLayerProbe,
        )

        class FakeFmha:
            def forward(self, qkv, _kv_cache, _layer_idx):
                return qkv + 3

        attention = Qwen3NextAttention.__new__(Qwen3NextAttention)
        torch.nn.Module.__init__(attention)
        attention.qwen_layer_idx = 7
        attention.layer_idx = 0
        attention.tp_size = 1
        attention.gate = torch.nn.Identity()
        attention.qkv_proj = torch.nn.Identity()
        attention.qk_fuse_norm = None
        attention.o_proj = torch.nn.Identity()
        probe = _CudaGraphLayerProbe(
            enabled=True,
            layers=(7,),
            layer_num=8,
            stage_layer=7,
        )
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        output = attention(
            hidden_states=hidden,
            fmha_impl=FakeFmha(),
            kv_cache=None,
            attention_inputs=SimpleNamespace(is_cuda_graph=True),
            graph_probe=probe,
            graph_probe_bs=2,
        )

        self.assertEqual((2, 2), tuple(output.shape))
        buffer, slots = probe.get_capture(2)
        for stage in (
            "attention_gate",
            "qkv_projected",
            "qkv_normalized",
            "fmha_output",
            "gated_output",
            "o_proj_output",
            "attention_output",
        ):
            slot = slots.index(probe.stage_slot_ids[stage])
            self.assertGreater(torch.count_nonzero(buffer[slot, :, :6]).item(), 0)

    def test_qwen3_next_linear_attention_records_outer_stage_probe(self):
        from types import SimpleNamespace

        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextGatedDeltaNet,
            Qwen3NextMetadata,
            _CudaGraphLayerProbe,
        )

        class Projection(torch.nn.Module):
            def __init__(self, width):
                super().__init__()
                self.width = width

            def forward(self, hidden_states):
                base = hidden_states.sum(dim=-1, keepdim=True)
                return base + torch.arange(
                    1, self.width + 1, dtype=hidden_states.dtype
                )

        class Decode(torch.nn.Module):
            def forward_with_graph_probe(
                self,
                mixed_qkv,
                b,
                a,
                attention_inputs,
                kv_cache,
                attn_meta,
                *,
                graph_probe=None,
                graph_probe_bs=0,
            ):
                self.graph_probe = graph_probe
                self.graph_probe_bs = graph_probe_bs
                return mixed_qkv[:, :1].reshape(-1, 1, 1)

        class GatedNorm(torch.nn.Module):
            def forward(self, value, gate):
                return value + gate

        linear = Qwen3NextGatedDeltaNet.__new__(Qwen3NextGatedDeltaNet)
        torch.nn.Module.__init__(linear)
        linear._qkvz_ba_fused = False
        linear.in_proj_qkvz = Projection(4)
        linear.in_proj_ba = Projection(2)
        linear.head_k_dim = 1
        linear.head_v_dim = 1
        linear.local_num_k_heads = 1
        linear.local_num_v_heads = 1
        linear.decode_gdn = Decode()
        linear.norm = GatedNorm()
        linear.out_proj = torch.nn.Identity()
        linear.parallelism_config = SimpleNamespace(get_attn_tp_size=lambda: 1)

        probe = _CudaGraphLayerProbe(
            enabled=True,
            layers=(2,),
            layer_num=3,
            stage_layer=2,
        )
        attention_inputs = SimpleNamespace(
            is_prefill=False,
            is_target_verify=False,
            is_cuda_graph=True,
        )
        metadata = Qwen3NextMetadata()
        metadata.trace_layer_idx = 2
        metadata.trace_group_id = 1
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        output = linear(
            hidden_states=hidden,
            fmha_impl=None,
            kv_cache=SimpleNamespace(),
            attention_inputs=attention_inputs,
            attn_meta=metadata,
            graph_probe=probe,
            graph_probe_bs=2,
        )

        self.assertEqual((2, 1), tuple(output.shape))
        self.assertIs(probe, linear.decode_gdn.graph_probe)
        self.assertEqual(2, linear.decode_gdn.graph_probe_bs)
        buffer, slots = probe.get_capture(2)
        for stage in (
            "linear_projected",
            "linear_fla_output",
            "linear_norm_output",
            "linear_o_proj_output",
            "linear_attention_output",
        ):
            slot = slots.index(probe.stage_slot_ids[stage])
            self.assertGreater(torch.count_nonzero(buffer[slot, :, :6]).item(), 0)

    def test_qwen3_next_linear_attention_preserves_probe_disabled_fast_path(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextGatedDeltaNet,
            Qwen3NextMetadata,
        )

        class Projection(torch.nn.Module):
            def __init__(self, width):
                super().__init__()
                self.width = width

            def forward(self, hidden_states):
                return hidden_states.sum(dim=-1, keepdim=True).expand(
                    -1, self.width
                )

        class Decode(torch.nn.Module):
            def forward(
                self,
                mixed_qkv,
                b,
                a,
                attention_inputs,
                kv_cache,
                attn_meta,
            ):
                return mixed_qkv[:, :1].reshape(-1, 1, 1)

        class GatedNorm(torch.nn.Module):
            def forward(self, value, gate):
                return value + gate

        linear = Qwen3NextGatedDeltaNet.__new__(Qwen3NextGatedDeltaNet)
        torch.nn.Module.__init__(linear)
        linear._qkvz_ba_fused = False
        linear.in_proj_qkvz = Projection(4)
        linear.in_proj_ba = Projection(2)
        linear.head_k_dim = 1
        linear.head_v_dim = 1
        linear.local_num_k_heads = 1
        linear.local_num_v_heads = 1
        linear.decode_gdn = Decode()
        linear.norm = GatedNorm()
        linear.out_proj = torch.nn.Identity()
        linear.parallelism_config = SimpleNamespace(get_attn_tp_size=lambda: 1)
        attention_inputs = SimpleNamespace(
            is_prefill=False,
            is_target_verify=False,
            is_cuda_graph=True,
        )
        metadata = Qwen3NextMetadata()
        metadata.trace_layer_idx = 2
        metadata.trace_group_id = 1

        with patch(
            "rtp_llm.models_py.model_desc.qwen3_next._graph_probe_stats",
            side_effect=AssertionError("disabled path must not reduce tensors"),
        ):
            output = linear(
                hidden_states=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                fmha_impl=None,
                kv_cache=SimpleNamespace(),
                attention_inputs=attention_inputs,
                attn_meta=metadata,
            )

        self.assertEqual((2, 1), tuple(output.shape))

    def test_qwen3_next_linear_decode_records_conv_and_fla_stage_probe(self):
        from types import SimpleNamespace

        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextGatedDeltaNetDecode,
            Qwen3NextMetadata,
            _CudaGraphLayerProbe,
        )

        class Decode(Qwen3NextGatedDeltaNetDecode):
            def __init__(self):
                torch.nn.Module.__init__(self)

            def _get_conv_states(self, kv_cache_tensor):
                return torch.ones((2, 1, 1))

            def _get_ssm_states(self, kv_cache_tensor):
                return torch.ones((2, 1, 1, 1))

            def _conv1d(self, mixed_qkv, *args, **kwargs):
                return mixed_qkv + 1

            def _fla(self, mixed_qkv, *args, **kwargs):
                return (mixed_qkv[:, :1] + 1).reshape(-1, 1, 1)

        probe = _CudaGraphLayerProbe(
            enabled=True,
            layers=(2,),
            layer_num=3,
            stage_layer=2,
        )
        metadata = Qwen3NextMetadata()
        metadata.trace_layer_idx = 2
        metadata.trace_group_id = 1
        attention_inputs = SimpleNamespace(
            is_cuda_graph=True,
            sequence_lengths_plus_1_d=torch.tensor([65, 65], dtype=torch.int32),
            kv_cache_kernel_block_id_device=torch.tensor(
                [[1], [2]], dtype=torch.int32
            ),
        )
        kv_cache = SimpleNamespace(
            kv_cache_base=torch.ones((2, 8)),
            seq_size_per_block=1024,
        )

        output = Decode().forward_with_graph_probe(
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            torch.ones((2, 1)),
            torch.ones((2, 1)),
            attention_inputs,
            kv_cache,
            metadata,
            graph_probe=probe,
            graph_probe_bs=2,
        )

        self.assertEqual((2, 1, 1), tuple(output.shape))
        buffer, slots = probe.get_capture(2)
        for stage in (
            "linear_conv_state_input",
            "linear_ssm_state_input",
            "linear_conv_output",
        ):
            slot = slots.index(probe.stage_slot_ids[stage])
            self.assertGreater(torch.count_nonzero(buffer[slot, :, :6]).item(), 0)
        fla_slot = slots.index(probe.stage_slot_ids["linear_fla_output"])
        self.assertEqual(0, torch.count_nonzero(buffer[fla_slot]).item())
        self.assertEqual(3, probe.get_debug_status()["recorded"])

    def test_model_reports_probe_runtime_status(self):
        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextModel,
            _CudaGraphLayerProbe,
            _Q3N_GRAPH_PROBE_ENABLED,
        )

        model = Qwen3NextModel.__new__(Qwen3NextModel)
        object.__setattr__(model, "_cuda_graph_layer_probe", None)

        self.assertEqual(
            {
                "module_env_enabled": _Q3N_GRAPH_PROBE_ENABLED,
                "probe_created": False,
                "buffer_available": False,
                "layers": (),
                "buffer_bucket_bs": (),
                "record_debug": {
                    "attempts": 0,
                    "recorded": 0,
                    "skipped_not_cuda_graph": 0,
                    "skipped_invalid_tensor": 0,
                    "skipped_invalid_layout": 0,
                    "last_layer_idx": -1,
                    "last_graph_bs": -1,
                    "last_token_rows": -1,
                    "last_residual_rows": -1,
                    "last_is_cuda_graph": -1,
                },
            },
            model.get_cuda_graph_probe_debug_status(16),
        )

        probe = _CudaGraphLayerProbe(enabled=True, layers=(1, 0), layer_num=2)
        hidden = torch.ones((8, 2), dtype=torch.float32)
        probe.record(1, hidden, hidden, graph_bs=8, is_cuda_graph=True)
        object.__setattr__(model, "_cuda_graph_layer_probe", probe)

        self.assertEqual(
            {
                "module_env_enabled": _Q3N_GRAPH_PROBE_ENABLED,
                "probe_created": True,
                "buffer_available": False,
                "layers": (1, 0),
                "buffer_bucket_bs": (8,),
                "record_debug": {
                    "attempts": 1,
                    "recorded": 1,
                    "skipped_not_cuda_graph": 0,
                    "skipped_invalid_tensor": 0,
                    "skipped_invalid_layout": 0,
                    "last_layer_idx": 1,
                    "last_graph_bs": 8,
                    "last_token_rows": 8,
                    "last_residual_rows": 8,
                    "last_is_cuda_graph": 1,
                },
            },
            model.get_cuda_graph_probe_debug_status(16),
        )
        self.assertTrue(
            model.get_cuda_graph_probe_debug_status(8)["buffer_available"]
        )


if __name__ == "__main__":
    unittest.main()
