import types
import unittest
from unittest.mock import patch

from rtp_llm.models_py.modules.factory.attention import attn_factory


class _StubAttentionImpl:
    accepts_fmha_config = False
    graph_safe = False

    @classmethod
    def support(cls, _attn_configs, _attn_inputs):
        return True

    @classmethod
    def support_parallelism_config(cls, _parallelism_config):
        return True

    def __init__(self, *_args, **_kwargs):
        pass

    def support_cuda_graph(self):
        return True

    def supports_prefill_cuda_graph(self):
        return self.graph_safe


class _UnsafeAttentionImpl(_StubAttentionImpl):
    pass


class _SafeAttentionImpl(_StubAttentionImpl):
    graph_safe = True


class _NoCudaGraphAttentionImpl(_SafeAttentionImpl):
    def support_cuda_graph(self):
        return False


class AttentionFactoryPrefillCudaGraphTest(unittest.TestCase):
    def setUp(self):
        self.attn_configs = types.SimpleNamespace(
            rope_config=types.SimpleNamespace(style=None)
        )
        self.attn_inputs = types.SimpleNamespace(is_prefill=True)

    def _select(self, implementations, mode=None, is_cuda_graph=True):
        with (
            patch.object(attn_factory, "PREFILL_MHA_IMPS", implementations),
            patch.object(attn_factory, "VALIDATE_FMHA_CONFIG", None),
        ):
            return attn_factory.get_fmha_impl(
                self.attn_configs,
                None,
                self.attn_inputs,
                is_cuda_graph=is_cuda_graph,
                cuda_graph_selection_mode=mode,
            )

    def test_prefill_graph_selects_declared_safe_backend(self):
        selected = self._select(
            [_SafeAttentionImpl],
            attn_factory.CudaGraphSelectionMode.PREFILL_GRAPH,
        )
        self.assertIsInstance(selected, _SafeAttentionImpl)

    def test_prefill_graph_rejects_semantic_backend_instead_of_skipping_it(self):
        with self.assertRaises(attn_factory.PrefillCudaGraphUnsupportedBackend):
            self._select(
                [_UnsafeAttentionImpl, _SafeAttentionImpl],
                attn_factory.CudaGraphSelectionMode.PREFILL_GRAPH,
            )

    def test_prefill_graph_rejects_when_no_backend_matches(self):
        for implementations in ([], [_NoCudaGraphAttentionImpl]):
            with (
                self.subTest(implementations=implementations),
                self.assertRaises(attn_factory.PrefillCudaGraphUnsupportedBackend),
            ):
                self._select(
                    implementations,
                    attn_factory.CudaGraphSelectionMode.PREFILL_GRAPH,
                )

    def test_default_modes_keep_existing_routing(self):
        self.assertIsInstance(
            self._select([_UnsafeAttentionImpl], mode=None, is_cuda_graph=False),
            _UnsafeAttentionImpl,
        )
        with self.assertRaises(ValueError):
            self._select([_SafeAttentionImpl], mode="invalid_graph_mode")


if __name__ == "__main__":
    unittest.main()
