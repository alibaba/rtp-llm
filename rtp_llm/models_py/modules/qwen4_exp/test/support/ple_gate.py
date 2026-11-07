from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpPLELayer


class PLEProjectionFixture:
    def layer(self):
        def w(shape):
            return (torch.randn(shape, device="cuda") * 0.02).to(torch.float8_e4m3fn)

        return Qwen4ExpPLELayer(
            SimpleNamespace(ngram_size=3, context_len=2),
            w((10240, 2560)),
            w((2560, 2560)),
            torch.randn(10240, 1, 4, device="cuda", dtype=torch.bfloat16) * 0.1,
            *[
                torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.07
                for _ in range(3)
            ],
            hc_mult=4,
            hidden_size=2560,
            conv_kernel_size=4,
            norm_eps=1e-6,
            projection_scales=[
                torch.tensor([0.03], device="cuda"),
                torch.tensor([0.07], device="cuda"),
            ],
        )

    def serial(self, layer, hidden, tokens, embeddings, initial):
        state = initial
        outputs, snapshots = [], []
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.ple.ple_gate_triton.try_ple_gate",
            return_value=None,
        ):
            for start in range(0, hidden.shape[1], 128):
                gated, normed = layer._gated_values(
                    hidden[:, start : start + 128],
                    tokens,
                    embeddings=embeddings[:, start : start + 128].contiguous(),
                )
                if state is None:
                    output = gated + layer._short_conv(normed)
                    state = layer.prefill_conv_state(normed)
                else:
                    history = torch.cat((state, normed), 1)
                    output = gated + F.silu(
                        F.conv1d(
                            history.transpose(1, 2),
                            layer.conv_weight,
                            groups=10240,
                            dilation=3,
                        )
                    ).transpose(1, 2)
                    state = history[:, -9:]
                outputs.append(output)
                snapshots.append(state)
        return torch.cat(outputs, 1), snapshots
