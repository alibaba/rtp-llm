"""FP8 QSA cache layout, independent attention reference and graph replay."""

import torch


class Fp8MainKvTestFixture:
    def _fixture(self):
        torch.manual_seed(211)
        q = torch.randn(2, 6, 4, 256, device="cuda", dtype=torch.bfloat16)
        source = torch.randn(7, 2, 2, 128, 256, device="cuda")
        scale = torch.rand(7, 2, 2, 128, device="cuda") * 0.4 + 0.1
        quant = (source / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        table = torch.tensor([[3, 1], [4, 6]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor(
            [[127, 128, 129, 130], [129, 130, 131, 132]],
            device="cuda",
            dtype=torch.int32,
        )
        selected = torch.tensor(
            [0, 7, 126, 127, 128, 129, 130, 131, -1], device="cuda", dtype=torch.int32
        ).repeat(2, 4, 1)
        selected = torch.where(selected < lengths.unsqueeze(-1), selected, -1)
        return q, quant, scale, table, lengths, selected
