import torch

from rtp_llm.models_py.triton_kernels.common.fp8_vocab_gather import (
    dequant_hyper,
    raw_rows,
)
from rtp_llm.models_py.triton_kernels.common.fp8_vocab_lookup import (
    gather_hyper_rows,
    vocab_rows,
)

MODULE = "rtp_llm.models_py.modules.base.common.fp8_embedding"


class VocabByteGatherTestFixture:
    def setUp(self):
        torch.manual_seed(578)
        self.weights = [
            torch.randn(257, 1280, device="cuda").to(torch.float8_e4m3fn)
            for _ in range(2)
        ]
        self.scales = torch.tensor([0.119713, 0.27351], device="cuda")

    def reference(self, ids, mask=None, branches=4):
        local = [
            vocab_rows(ids, w, self.scales[i : i + 1], mask)
            for i, w in enumerate(self.weights)
        ]
        return gather_hyper_rows(torch.cat(local), ids.numel(), 1280, 2, branches)

    def bytes_consumer(self, ids, mask=None, branches=4):
        gathered = torch.cat([raw_rows(ids, w) for w in self.weights])
        return dequant_hyper(
            gathered, self.scales, ids.numel(), 1280, 2, branches, mask
        )
