import unittest
import torch

from rtp_llm.models_py.modules.kimi_k3.native_conv import causal_conv1d_fn
from rtp_llm.models_py.triton_kernels.causal_conv1d.causal_conv1d import (
    prepare_causal_conv1d_metadata,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d.paged_short_conv_prefill import (
    paged_short_conv_prefill,
    prepare_paged_short_conv_metadata,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class PagedShortConvPrefillTest(unittest.TestCase):
  def test_matches_native_with_prefix_and_beta_pack(self):
    self._compare(
        lengths=(96, 64), page_size=4096, prefix=4096,
        block_map=[[1, 0], [2, 3]],
    )

  def test_publishes_state_at_page_boundary(self):
    self._compare(
        lengths=(160, 96), page_size=128, prefix=64,
        block_map=[[1, 3], [2, 4]],
    )

  def _compare(self, lengths, page_size, prefix, block_map):
    torch.manual_seed(20260928)
    device = torch.device("cuda:0")
    channels = 384
    total = sum(lengths)
    x = torch.randn(total, channels, device=device, dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device=device, dtype=torch.float32) * 0.15
    cache_base = torch.zeros(5, 3, channels, device=device, dtype=torch.bfloat16)
    cache_base[2].normal_()
    native_cache_base = cache_base.clone()
    paged_cache_base = cache_base.clone()
    native_cache = native_cache_base.transpose(1, 2)
    block_map = torch.tensor(block_map, device=device, dtype=torch.int32)
    prefixes = torch.tensor([0, prefix], device=device, dtype=torch.int32)
    cu_host = torch.tensor([0, lengths[0], total], dtype=torch.int32)
    cu = cu_host.to(device)
    beta_storage = torch.randn(total, 8, device=device, dtype=torch.bfloat16)
    raw_beta = beta_storage[:, :3]
    assert not raw_beta.is_contiguous()

    native_metadata = prepare_causal_conv1d_metadata(cu_host, device)
    paged_metadata = prepare_paged_short_conv_metadata(cu_host, device)
    native_mixed = causal_conv1d_fn(
        x.T, weight, None, native_cache, cu, block_map, prefixes,
        page_size, native_metadata,
    ).T
    native_outputs = (
        *(part.contiguous() for part in native_mixed.chunk(3, dim=-1)),
        raw_beta.contiguous(),
    )
    paged = paged_short_conv_prefill(
        x, weight, paged_cache_base, block_map, prefixes, cu,
        page_size, paged_metadata, aux=raw_beta,
    )
    assert len(paged) == 5 and paged[3] is None and paged[4] is not None
    for actual, reference in zip((*paged[:3], paged[4]), native_outputs):
        assert actual.is_contiguous()
        torch.testing.assert_close(actual.float(), reference.float(), rtol=0.02, atol=0.05)
    torch.testing.assert_close(paged_cache_base, native_cache_base, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
