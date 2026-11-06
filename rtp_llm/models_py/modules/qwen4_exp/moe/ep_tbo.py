"""Two-microbatch scheduling for the qualified Qwen DeepEP decode path."""

import copy
from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def _communication_stream(device_index):
    return torch.cuda.Stream(device=torch.device("cuda", device_index))


class QwenEpTwoBatch:
    def __init__(self, layer):
        self.tp_size = layer.parallelism_config.tp_size
        # Handles belong to individual in-flight calls. Both routers retain the
        # same DeepEP buffer, whose two slots are consumed in D0,D1,C0,C1 order.
        self.routers = tuple(copy.copy(layer.fused_moe.router) for _ in range(2))
        self.stream = _communication_stream(
            layer.shared_expert.up_proj.weight.device.index
        )
        self.dispatch_a_ready = torch.cuda.Event()
        self.dispatch_b_ready = torch.cuda.Event()
        self.compute_a_ready = torch.cuda.Event()
        self.compute_b_ready = torch.cuda.Event()

    def supports(self, x):
        # Both microbatches need equal nonempty slices across every TP rank.
        partition = 2 * self.tp_size
        return partition <= x.shape[0] <= 32 and x.shape[0] % partition == 0

    def forward(self, layer, x):
        main = torch.cuda.current_stream(x.device)
        logits = layer.gate(x)
        weights = torch.empty(
            (x.shape[0], layer.top_k), device=x.device, dtype=torch.float32
        )
        ids = torch.empty(
            weights.shape, device=x.device, dtype=layer.fused_moe.topk_ids_dtype
        )
        # Route the full batch before splitting: retain the original router
        # projection's dynamic per-tensor FP8 scale and top-k behavior.
        layer.select_topk(logits, ids, weights)
        half = x.shape[0] // 2
        xs = (x[:half], x[half:])
        ws = (weights[:half], weights[half:])
        ts = (ids[:half], ids[half:])
        router_a, router_b = self.routers
        assert router_a.handle is None and router_b.handle is None

        def prepare(router, index):
            return router.prepare(xs[index], None, None, ws[index], ts[index])

        def execute(payload):
            result = layer.fused_moe.fused_experts.execute(
                payload,
                activation="SiGLU",
                expert_map=None,
                a2_scale=None,
                apply_router_weight_on_input=False,
                extra_expert_args=None,
            )
            return result

        def finalize(router, index, expert_payload, result):
            return router.finalize(
                result,
                expert_payload.expert_topk_weights,
                expert_payload.expert_topk_ids,
                False,
                {"original_num_tokens": half, "a1_shape": xs[index].shape},
            )

        payload_a = prepare(router_a, 0)
        self.dispatch_a_ready.record(main)
        try:
            with torch.cuda.stream(self.stream):
                self.dispatch_a_ready.wait(self.stream)
                payload_b = prepare(router_b, 1)
                self.dispatch_b_ready.record(self.stream)
            result_a = execute(payload_a)
            self.compute_a_ready.record(main)
            with torch.cuda.stream(self.stream):
                # Avoid competing expert GEMMs; overlap B's compute with A's
                # combine instead. No more than two DeepEP results stay live.
                self.compute_a_ready.wait(self.stream)
                result_b = execute(payload_b)
                self.compute_b_ready.record(self.stream)
            self.dispatch_b_ready.wait(main)
            output_a = finalize(router_a, 0, payload_a, result_a)
            self.compute_b_ready.wait(main)
            if not torch.cuda.is_current_stream_capturing():
                result_b.fused_expert_output.record_stream(main)
            output_b = finalize(router_b, 1, payload_b, result_b)
        finally:
            # Join submitted auxiliary work even when a Python wrapper fails.
            main.wait_stream(self.stream)
        return torch.cat((output_a, output_b), dim=0)
