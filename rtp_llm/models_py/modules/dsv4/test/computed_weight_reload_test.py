import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


class ComputedWeightReloadTest(unittest.TestCase):
    def test_mega_registration_fails_closed_when_reload_is_required(self):
        from rtp_llm.model_loader import weight_memory_saver
        from rtp_llm.models_py.modules.dsv4.moe.mega_buf import _register_mega_strategy

        class NotWeakReferenceable:
            __slots__ = ("_sleep_model_scope",)

        with patch.object(
            weight_memory_saver, "keep_loader_database_for_wake", return_value=True
        ):
            with self.assertRaises(TypeError):
                _register_mega_strategy(NotWeakReferenceable())

    def test_compressor_registration_stays_best_effort_without_reload(self):
        from rtp_llm.model_loader import weight_memory_saver
        from rtp_llm.models_py.modules.dsv4.fp8.compressor import _register_compressor

        class NotWeakReferenceable:
            __slots__ = ("_sleep_model_scope",)

        with patch.object(
            weight_memory_saver, "keep_loader_database_for_wake", return_value=False
        ):
            _register_compressor(NotWeakReferenceable())

    def test_compressor_rebuild_continues_when_cache_trim_fails(self):
        from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorFP8

        raw_wkv = object()
        raw_wgate = object()
        compressor = SimpleNamespace(
            _wkv_wgate_fused=object(),
            coff=17,
            _raw_wkv_src=raw_wkv,
            _raw_wgate_src=raw_wgate,
            _fuse_wkv_wgate=Mock(),
        )
        error = RuntimeError("CUDA error: invalid argument")

        with patch(
            "rtp_llm.models_py.modules.dsv4.fp8.compressor.torch.cuda.empty_cache",
            side_effect=error,
        ) as empty_cache, self.assertLogs(level="WARNING") as logs:
            CompressorFP8.reload_fused_weights(compressor)

        empty_cache.assert_called_once_with()
        self.assertIsNone(compressor._wkv_wgate_fused)
        compressor._fuse_wkv_wgate.assert_called_once_with(17, raw_wkv, raw_wgate)
        self.assertTrue(any("invalid argument" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
