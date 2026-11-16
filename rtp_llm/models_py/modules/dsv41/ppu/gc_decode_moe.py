"""PPU GC MoE workspace and weight bindings for V4.1 decode."""

import torch

from .input_packer import pack_ppu_gc_moe_inputs


class DecodeGCMoE:
    def __init__(self, engine, num_topk, token_bound, *, mode, graph_safe=False):
        import gc_moe

        if engine.ep_size != 8 or engine.group is None:
            raise ValueError("V4.1 PPU GC MoE requires a single-node EP8 group")
        if engine.precision != "w4a4":
            raise ValueError("V4.1 PPU GC MoE requires W4A4 routed experts")
        if token_bound <= 0:
            raise ValueError("GC MoE requires a common positive decode token bound")
        self.engine = engine
        self.token_bound = token_bound
        self.api = gc_moe
        self.num_topk = num_topk
        self.mode = mode

        self.fuses_shared = mode in ("w4a4", "w8a8")
        self.graph_safe = graph_safe
        self.weights = {}
        self.workspace = None
        self.calls = 0
        self.capture_calls = 0

    def bind_weights(self, layer_id, w13, s13, w2, s2):
        from .moe import _ue8m0_scale_bytes

        self.weights[layer_id] = self.api.transform_weights_for_gc_moe_ppu(
            (w13, _ue8m0_scale_bytes(s13).view(torch.uint8)),
            (w2, _ue8m0_scale_bytes(s2).view(torch.uint8)),
        )

    def prepare_shared_weights(self, shared):
        from .linear import PPUPTPCLinear
        from .moe import downcast_to_mxfp4

        if shared is None or not all(
            isinstance(op, PPUPTPCLinear) for op in (shared.w13, shared.w2)
        ):
            raise ValueError("PPU GC MoE SE requires the resident RTP PTPC shared expert")
        pairs = tuple((op.weight, op.scale) for op in (shared.w13, shared.w2))
        if self.mode == "w8a8":
            return pairs

        # Requantize once during load; prefill keeps the checkpoint's original PTPC weights.
        mx_weights = []
        for weight, scale in pairs:
            packed, scales = downcast_to_mxfp4(weight.float() * scale)
            mx_weights.append((packed, scales.contiguous().view(torch.uint8)))
        return self.api.transform_weights_for_gc_moe_ppu(*mx_weights)

    def prepare_workspace(self):
        engine = self.engine
        if self.workspace is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("initialize the PPU GC MoE workspace before capture")
            self.workspace = self.api.get_symm_buffer_for_gc_moe_ppu(
                engine.group,
                engine.num_physical_experts,
                self.token_bound,
                self.num_topk,
                engine.hidden_size,
                engine.inter_dim,
                num_shared_experts=int(self.fuses_shared),
                shared_fp8=self.mode == "w8a8",
                **({"graph_safe": True} if self.graph_safe else {}),
            )
        return self.workspace

    def forward(
        self,
        x,
        ids,
        weights,
        *,
        layer_id,
        expert_load=None,
        valid=None,
        shared_weights=()
    ):
        engine = self.engine
        rows = x.shape[0]
        if rows > self.token_bound:
            raise ValueError("decode rows exceed the PPU GC MoE token bound")
        capturing = self.graph_safe and torch.cuda.is_current_stream_capturing()
        self.prepare_workspace()
        pack_ppu_gc_moe_inputs(x, ids, weights, self.workspace, valid=valid)
        out = torch.empty_like(
            x, dtype=torch.bfloat16, memory_format=torch.contiguous_format
        )
        stats = (
            None
            if expert_load is None
            else engine._expert_load_counts(expert_load, x.device)
        )
        shared_options = {}
        if self.fuses_shared:
            shared_l1, shared_l2 = shared_weights
            shared_options = {
                "shared_l1_weights": shared_l1,
                "shared_l2_weights": shared_l2,
            }
        self.api.ppu_fp4_fp4_gc_moe(
            out,
            *self.weights[layer_id],
            self.workspace,
            stats,
            num_tokens_bound=self.token_bound,
            activation_clamp=engine.swiglu_limit if engine.swiglu_limit > 0 else None,
            **shared_options,
        )
        self.calls += 1
        self.capture_calls += int(capturing)
        return out

    def close(self):
        if self.workspace is not None:
            self.workspace.destroy()
            self.workspace = None
        self.weights.clear()
