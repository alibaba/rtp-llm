"""RTP-side SM120 Decode adapter for the pinned FlashInfer 0.6.9 wheel."""

from functools import cache
from importlib.metadata import PackageNotFoundError, version
from weakref import WeakValueDictionary

import torch

_WORKSPACE_POOLS = WeakValueDictionary()


class _WorkspacePool:
    def __init__(self, model_config):
        self.model_config = model_config
        self.states = {}


@cache
def uses_locked_wheel():
    try:
        return version("flashinfer-python") == "0.6.9+8c4f4dcf"
    except PackageNotFoundError:
        return False


@cache
def micro_available():
    try:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        return version("nvidia-cutlass-dsl") == "4.4.2" and callable(
            moe_dispatch.launch_sm120_static_moe
        )
    except (ImportError, OSError, RuntimeError, PackageNotFoundError):
        return False


@cache
def cutlass_runner(device):
    from flashinfer.fused_moe.core import gen_cutlass_fused_moe_sm120_module

    with torch.cuda.device(device):
        module = gen_cutlass_fused_moe_sm120_module(False).build_and_load()
        # Match MoERunner in the locked wheel for BF16 x packed NVFP4.
        runner = module.init(
            torch.bfloat16, torch.int64, torch.bfloat16, False, False, False, False
        )
    return module, runner


class LockedDecodeMoE:
    # Stage IDs from the full SM120 12 + 24 tactic binary.
    profiles = {4: (8, 32), 8: (8, 32), 16: (0, 32)}

    def __init__(self, w1, w2, scales, rank, model_config, top_k, enable_micro):
        self.w1, self.w2, self.scales = w1, w2, scales
        self.rank, self.top_k = rank, top_k
        self.module, self.runner = cutlass_runner(w1.device)
        self.profile_table_matches = (
            self.runner.get_gemm1_tactic_count(),
            self.runner.get_gemm2_tactic_count(),
        ) == (12, 24)
        self.micro = enable_micro and micro_available()
        if not self.micro:
            return

        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        self.dispatch = moe_dispatch
        self.experts, self.hidden, packed = w2.shape
        self.intermediate = packed * 16

        def mma(scale, rows, columns):
            return convert_sf_to_mma_layout(
                scale.contiguous().view(torch.float8_e4m3fn),
                rows,
                columns,
                num_groups=self.experts,
                sf_vec_size=16,
            )

        self.views = moe_dispatch._get_weight_views(
            w1.view(torch.uint8),
            mma(scales[1], 2 * self.intermediate, self.hidden),
            w2.view(torch.uint8),
            mma(scales[4], self.hidden, self.intermediate),
            scales[2],
            scales[5],
            self.intermediate,
            self.hidden,
        )
        # 0.6.9's default launcher consumes dequantizers, independently of GEMM alpha.
        self.up_input = scales[0].reciprocal().expand(self.experts).contiguous()
        self.down_input = scales[3].reciprocal().expand(self.experts).contiguous()
        key = (
            "0.6.9",
            id(model_config),
            w1.device,
            self.experts,
            self.hidden,
            self.intermediate,
            top_k,
        )
        pool = _WORKSPACE_POOLS.get(key)
        if pool is None:
            pool = _WorkspacePool(model_config)
            _WORKSPACE_POOLS[key] = pool
        self.pool = pool
        if 2 not in pool.states:
            # Same graph workspace capacity as the tested B12xMoEWrapper.
            pool.states[2] = moe_dispatch.allocate_sm120_static_workspace(
                state_E=self.experts,
                weight_E=self.experts,
                max_rows=64 * top_k,
                k=self.hidden,
                n=self.intermediate,
                num_topk=top_k,
                device=w1.device,
            )
        self.slots = torch.arange(top_k, device=w1.device, dtype=torch.int32)

    def execute(self, x, ids, router):
        from flashinfer import ActivationType

        rows = x.shape[0]
        if rows == 2 and self.micro:
            local = ids - self.rank * self.experts
            owned = (local >= 0) & (local < self.experts)
            # The stock micro kernel indexes every ID, including zero-weight routes.
            local_ids = torch.where(owned, local, self.slots)
            scores = torch.where(owned, router, 0.0)
            return self.dispatch.launch_sm120_static_moe(
                workspace=self.pool.states[2],
                weights=self.views,
                a=x,
                topk_ids=local_ids,
                topk_weights=scores,
                input_gs=self.up_input,
                down_input_scale=self.down_input,
                scatter_output=torch.empty_like(x),
                num_experts=self.experts,
                num_tokens=rows,
                k=self.hidden,
                n=self.intermediate,
                top_k=self.top_k,
            )
        pair = self.profiles.get(rows)
        if pair is None or not self.profile_table_matches:
            # Unprofiled M and alternate native builds keep FlashInfer's autotuner.
            return None
        output = torch.empty_like(x)
        self.runner.run_moe(
            output,
            x,
            ids,
            router,
            self.w1,
            None,
            self.w2,
            None,
            self.scales,
            None,
            None,
            None,
            None,
            True,  # no biases/limits, swizzled input scales
            1,
            0,
            2,
            self.rank,
            1,
            0,
            False,
            False,
            list(pair),
            False,
            ActivationType.Swiglu,
        )
        return output
