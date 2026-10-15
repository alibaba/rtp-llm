from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from rtp_llm.models_py.modules.factory.attention import attn_factory


class CpAttentionMetadataTest(TestCase):
    def test_cp_factory_does_not_read_device_scalar(self) -> None:
        candidate = MagicMock()
        candidate.__name__ = "CpCandidate"
        candidate.support.return_value = True
        candidate.support_parallelism_config.return_value = True
        candidate.is_sparse.return_value = True
        inputs = SimpleNamespace(is_prefill=True, cu_kv_seqlens=MagicMock())
        inputs.cu_kv_seqlens.max.side_effect = AssertionError("CP needs no scalar read")
        parallelism = SimpleNamespace(prefill_cp_config=MagicMock())
        parallelism.prefill_cp_config.is_enabled.return_value = True
        weights = MagicMock()
        config = SimpleNamespace(is_sparse=True, indexer_topk=128)

        with patch.object(attn_factory, "PREFILL_MLA_IMPS", [candidate]):
            actual = attn_factory.get_mla_impl(
                config, weights, inputs, parallelism_config=parallelism
            )

        self.assertIs(actual, candidate.return_value)
        inputs.cu_kv_seqlens.max.assert_not_called()


if __name__ == "__main__":
    main()
