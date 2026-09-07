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

    def supports_generation_prefill_cuda_graph(self):
        return self.graph_safe


class _UnsafeAttentionImpl(_StubAttentionImpl):
    pass


class _SafeAttentionImpl(_StubAttentionImpl):
    graph_safe = True


class _NoCudaGraphAttentionImpl(_SafeAttentionImpl):
    def support_cuda_graph(self):
        return False


class _ConstructorFailureAttentionImpl(_SafeAttentionImpl):
    def __init__(self, *_args, **_kwargs):
        raise RuntimeError("injected attention constructor failure")


class _GenerationOnlyAttentionImpl(_SafeAttentionImpl):
    cuda_graph_selection_modes = frozenset(
        {attn_factory.CudaGraphSelectionMode.GENERATION_PREFILL_GRAPH}
    )


class AttentionFactoryGenerationPrefillCudaGraphTest(unittest.TestCase):
    def setUp(self):
        self.attn_configs = types.SimpleNamespace(
            rope_config=types.SimpleNamespace(style=None),
            need_rope_kv_cache=False,
        )
        self.attn_inputs = types.SimpleNamespace(is_prefill=True)

    def _select(self, implementations, mode=None, is_cuda_graph=True):
        with patch.object(attn_factory, "PREFILL_MHA_IMPS", implementations):
            return attn_factory.get_fmha_impl(
                self.attn_configs,
                None,
                self.attn_inputs,
                is_cuda_graph=is_cuda_graph,
                cuda_graph_selection_mode=mode,
            )

    def _select_decode(self, implementations):
        self.attn_inputs.is_prefill = False
        with patch.object(attn_factory, "DECODE_MHA_IMPS", implementations):
            return attn_factory.get_fmha_impl(
                self.attn_configs,
                None,
                self.attn_inputs,
                is_cuda_graph=True,
            )

    def test_generation_prefill_graph_selects_declared_safe_backend(self):
        selected = self._select(
            [_SafeAttentionImpl],
            attn_factory.CudaGraphSelectionMode.GENERATION_PREFILL_GRAPH,
        )
        self.assertIsInstance(selected, _SafeAttentionImpl)

    def test_generation_prefill_graph_rejects_semantic_backend_instead_of_skipping_it(
        self,
    ):
        with self.assertRaises(
            attn_factory.GenerationPrefillCudaGraphUnsupportedBackend
        ):
            self._select(
                [_UnsafeAttentionImpl, _SafeAttentionImpl],
                attn_factory.CudaGraphSelectionMode.GENERATION_PREFILL_GRAPH,
            )

    def test_generation_prefill_graph_rejects_when_no_backend_matches(self):
        for implementations in ([], [_NoCudaGraphAttentionImpl]):
            with (
                self.subTest(implementations=implementations),
                self.assertRaises(
                    attn_factory.GenerationPrefillCudaGraphUnsupportedBackend
                ),
            ):
                self._select(
                    implementations,
                    attn_factory.CudaGraphSelectionMode.GENERATION_PREFILL_GRAPH,
                )

    def test_generation_prefill_graph_does_not_fallback_after_constructor_failure(
        self,
    ):
        with self.assertRaisesRegex(
            RuntimeError, "injected attention constructor failure"
        ):
            self._select(
                [_ConstructorFailureAttentionImpl, _SafeAttentionImpl],
                attn_factory.CudaGraphSelectionMode.GENERATION_PREFILL_GRAPH,
            )

    def test_role_specific_backend_is_filtered_before_decode_selection(self):
        selected = self._select_decode(
            [_GenerationOnlyAttentionImpl, _SafeAttentionImpl]
        )
        self.assertIsInstance(selected, _SafeAttentionImpl)

    def test_default_modes_keep_existing_routing(self):
        self.assertIsInstance(
            self._select([_UnsafeAttentionImpl], mode=None, is_cuda_graph=False),
            _UnsafeAttentionImpl,
        )
        with self.assertRaises(ValueError):
            self._select([_SafeAttentionImpl], mode="invalid_graph_mode")
        self.assertIsInstance(
            self._select(
                [_ConstructorFailureAttentionImpl, _SafeAttentionImpl],
                mode=None,
                is_cuda_graph=False,
            ),
            _SafeAttentionImpl,
        )

    def test_decode_graph_skips_backend_without_graph_support(self):
        selected = self._select_decode([_NoCudaGraphAttentionImpl, _SafeAttentionImpl])
        self.assertIsInstance(selected, _SafeAttentionImpl)

    def test_decode_graph_fails_when_no_backend_supports_graph(self):
        with self.assertRaisesRegex(Exception, "can not find mha type"):
            self._select_decode([_NoCudaGraphAttentionImpl])


if __name__ == "__main__":
    unittest.main()
