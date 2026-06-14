import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention import attn_factory


class _FakeWeights:
    weights = []

    def get_global_weight(self, _name):
        return torch.empty(0)


class _DenseMlaImpl:
    @staticmethod
    def is_sparse() -> bool:
        return False

    @classmethod
    def support(cls, _attn_configs, _attn_inputs) -> bool:
        return True

    @classmethod
    def support_parallelism_config(cls, _parallelism_config) -> bool:
        return True

    def __init__(self, *_args, **_kwargs):
        pass

    def support_cuda_graph(self) -> bool:
        return True


class _SparseMlaImpl(_DenseMlaImpl):
    @staticmethod
    def is_sparse() -> bool:
        return True


def _attn_configs():
    return SimpleNamespace(
        is_sparse=True,
        use_mla=True,
        indexer_topk=8,
        kv_cache_dtype="fp8",
    )


def _attn_inputs(mtp_iteration_step: int):
    return SimpleNamespace(
        is_prefill=True,
        is_target_verify=False,
        cu_kv_seqlens=torch.tensor([0, 4], dtype=torch.int32),
        mtp_iteration_step=mtp_iteration_step,
    )


class MtpMlaImplSelectionTest(unittest.TestCase):
    def test_short_sparse_prefill_uses_dense_fast_path_by_default(self):
        with patch.object(
            attn_factory, "PREFILL_MLA_IMPS", [_DenseMlaImpl, _SparseMlaImpl]
        ):
            impl = attn_factory.get_mla_impl(
                _attn_configs(),
                _FakeWeights(),
                _attn_inputs(mtp_iteration_step=-1),
            )

        self.assertIsInstance(impl, _DenseMlaImpl)
        self.assertNotIsInstance(impl, _SparseMlaImpl)

    def test_mtp_step0_uses_sparse_prefill_to_produce_shareable_topk(self):
        with patch.object(
            attn_factory, "PREFILL_MLA_IMPS", [_DenseMlaImpl, _SparseMlaImpl]
        ):
            impl = attn_factory.get_mla_impl(
                _attn_configs(),
                _FakeWeights(),
                _attn_inputs(mtp_iteration_step=0),
            )

        self.assertIsInstance(impl, _SparseMlaImpl)


if __name__ == "__main__":
    unittest.main()
