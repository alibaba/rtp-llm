"""Compare RTP routing and merged MoE outputs on either side of 64 tokens."""

import argparse

import torch
from gdn_test import bit_mismatches

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules import SelectTopk
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_executor import (
    TritonFusedMoeExecutor,
)
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy", action="store_true", help="Report baseline differences"
    )
    args = parser.parse_args()
    torch.manual_seed(48)
    config = ModelConfig()
    config.model_type = "qwen35_moe"
    config.data_type = "bf16"
    config.hidden_size = 2048
    config.expert_num = 256
    config.moe_inter_size = 512
    config.moe_k = 8
    config.has_moe_norm = True
    k, e, i, top_k = (
        config.hidden_size,
        config.expert_num,
        config.moe_inter_size,
        config.moe_k,
    )

    def weight(*shape):
        return torch.empty(shape, device="cuda", dtype=torch.bfloat16).normal_(std=0.02)

    x = weight(65, k)
    adapter = MoEConfigAdapter(
        config, ParallelismConfig(), MoeConfig(), batch_invariant=not args.legacy
    )
    executor = TritonFusedMoeExecutor(
        adapter,
        FusedMoEQuantConfig(quant_dtype=None),
        {W.moe_w1: weight(e, 2 * i, k), W.moe_w2: weight(e, k, i)},
    )
    hw = HWKernelConfig()
    hw.batch_invariant = not args.legacy
    gate = LinearFactory.create_linear(weight(k, e), None, None, None, hw)
    topk = SelectTopk(config)

    def run(hidden, row):
        logits = gate(hidden)
        ids = torch.empty((len(hidden), top_k), device="cuda", dtype=torch.int32)
        weights = torch.empty_like(ids, dtype=torch.float32)
        topk(logits, ids, weights)
        payload = ExpertForwardPayload(
            expert_x=hidden, expert_topk_ids=ids, expert_topk_weights=weights
        )
        output = executor.execute(payload, "SiGLU", None, None, False, None)
        return ids[row], weights[row], output.fused_expert_output[row]

    expected = run(x[:64], 0)
    for hidden, row in ((x, 0), (x.roll(-1, 0), 64), (x[:1], 0)):
        actual = run(hidden, row)
        differences = [
            bit_mismatches(left, right) for left, right in zip(expected, actual)
        ]
        print(
            f"64 vs {len(hidden)} tokens, target row {row}: ids, weights, output {differences}"
        )
        if not args.legacy:
            assert differences == [0, 0, 0]


if __name__ == "__main__":
    main()
