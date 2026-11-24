"""Consume native PPU executor inputs from direct parameters or legacy weights."""

import torch


class PpuRoutedInputs:
    def __init__(self, weights):
        self.weights = weights
        self.direct = hasattr(weights, "take_executor_weight")
        if self.direct:
            self.keys = ("routed_w13", "routed_s13", "routed_w2", "routed_s2")
            self.storage = tuple(getattr(weights, k) for k in self.keys)
            w13, s13, w2, s2 = self.storage
            w1, w3 = w13.chunk(2, dim=1)
            s1, s3 = s13.chunk(2, dim=1)
            self.parts = (w1, s1, w2, s2, w3, s3)
        else:
            from rtp_llm.utils.model_weight import W

            self.keys = (
                W.v4_routed_w1_w,
                W.v4_routed_w1_s,
                W.v4_routed_w2_w,
                W.v4_routed_w2_s,
                W.v4_routed_w3_w,
                W.v4_routed_w3_s,
            )
            self.storage = self.parts = tuple(weights[k] for k in self.keys)

    def packed(self):
        if self.direct:
            w13, s13, w2, s2 = self.storage
        else:
            w1, s1, w2, s2, w3, s3 = self.parts
            w13 = torch.cat((w1.view(torch.uint8), w3.view(torch.uint8)), dim=1)
            s13 = torch.cat((s1, s3), dim=1)
        return (
            w13.view(torch.uint8).contiguous(),
            s13.contiguous(),
            w2.view(torch.uint8).contiguous(),
            s2.contiguous(),
        )

    def consume(self):
        take = self.weights.take_executor_weight if self.direct else self.weights.pop
        for key in self.keys:
            take(key)


def shared_expert_inputs(weights):
    if hasattr(weights, "take_executor_weight"):
        return {
            key: weights.take_executor_weight(name)
            for key, name in (
                ("w13_w", "shared_w13"),
                ("w13_s", "shared_s13"),
                ("w2_w", "shared_w2"),
                ("w2_s", "shared_s2"),
            )
        }
    from rtp_llm.utils.model_weight import W

    return {
        "w13_w": weights[W.v4_shared_w13_w],
        "w13_s": weights[W.v4_shared_w13_s],
        "w2_w": weights[W.v4_shared_w2_w],
        "w2_s": weights[W.v4_shared_w2_s],
    }
