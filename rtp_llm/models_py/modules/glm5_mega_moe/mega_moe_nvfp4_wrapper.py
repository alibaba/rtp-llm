"""FusedMoe-compatible wrapper for NVFP4xNVFP4 MegaMoE."""

import torch

from .mega_moe_nvfp4 import GLM5MegaMoENVFP4
from .mega_moe_wrapper import MegaMoeWrapper


class MegaMoeNvfp4Wrapper(MegaMoeWrapper):
    """Route experts through DeepGEMM ``nvfp4_nvfp4_mega_moe``."""

    def _get_mega_moe_cls(self):
        return GLM5MegaMoENVFP4

    def forward(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        inplace=False,
        activation="silu",
        expert_map=None,
        a1_scale=None,
        a2_scale=None,
        apply_router_weight_on_input=False,
        extra_expert_args=None,
        extra_finalize_args=None,
    ):
        kwargs = dict(
            inplace=inplace,
            activation=activation,
            expert_map=expert_map,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            apply_router_weight_on_input=apply_router_weight_on_input,
            extra_expert_args=extra_expert_args,
            extra_finalize_args=extra_finalize_args,
        )
        extra = dict(extra_expert_args or {})
        plan = extra.pop("prefill_chunk_plan", None)
        if plan is None:
            return super().forward(hidden_states, topk_weights, topk_ids, **kwargs)

        from .prefill_chunk_plan import run_prefill_chunks

        if plan.capacity != self.mega_moe._mega_buf.num_max_tokens_per_rank:
            raise ValueError("prefill chunk plan does not match NVFP4 buffer capacity")
        if topk_ids.shape[1] > self.expert_num:
            raise ValueError("dummy top-k routes exceed expert count")
        kwargs["extra_expert_args"] = extra
        return run_prefill_chunks(
            hidden_states,
            topk_weights,
            topk_ids,
            lambda h, w, i: super(MegaMoeNvfp4Wrapper, self).forward(h, w, i, **kwargs),
            plan,
            forward_into_fn=lambda h, w, i, out: self.mega_moe(
                h, w, i, activation=activation, extra_expert_args=extra, out=out
            ),
        )

    def clone_for_cuda_graph(self) -> "MegaMoeNvfp4Wrapper":
        clone = object.__new__(type(self))
        torch.nn.Module.__init__(clone)
        clone.mega_moe = self.mega_moe.clone_for_cuda_graph()
        clone.expert_num = self.expert_num
        return clone
