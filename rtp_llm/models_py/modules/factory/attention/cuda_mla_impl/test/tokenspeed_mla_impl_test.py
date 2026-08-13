"""CPU contract tests for the TokenSpeed MLA framework adapter."""

from types import SimpleNamespace
from unittest import TestCase, main, mock

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.tokenspeed_mla_impl import (
    TokenSpeedMlaDecodeImpl,
    _TokenSpeedDecodeMetadata,
)


class TokenSpeedMlaGraphAdapterTest(TestCase):
    def test_prepare_cuda_graph_uses_fixed_capacity_plan(self) -> None:
        impl = object.__new__(TokenSpeedMlaDecodeImpl)
        impl.prepare = mock.Mock()
        inputs = SimpleNamespace()

        impl.prepare_cuda_graph(inputs)

        impl.prepare.assert_called_once_with(inputs, forbid_realloc=True)


class TokenSpeedMlaMetadataContractTest(TestCase):
    def test_graph_metadata_rejects_capacity_growth(self) -> None:
        metadata = _TokenSpeedDecodeMetadata(
            token_per_block=64,
            max_bs=1,
            max_context_len=64,
            use_cuda_graph=True,
            device=torch.device("cpu"),
        )
        too_many_rows = SimpleNamespace(
            qo_indptr_h=torch.arange(3, dtype=torch.int32),
            kvlen_h=torch.tensor([1, 1], dtype=torch.int32),
        )
        too_many_blocks = SimpleNamespace(
            qo_indptr_h=torch.arange(2, dtype=torch.int32),
            kvlen_h=torch.tensor([65], dtype=torch.int32),
        )

        with self.assertRaisesRegex(ValueError, "too small for batch 2"):
            metadata.plan(too_many_rows)
        with self.assertRaisesRegex(ValueError, "needs 2 blocks, has 1"):
            metadata.plan(too_many_blocks)


if __name__ == "__main__":
    main()
