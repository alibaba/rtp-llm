import unittest

import torch


class TestQwen3NextGraphProbe(unittest.TestCase):
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
