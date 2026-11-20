# Adapted from FlashInfer's fused_moe/core.py, Apache-2.0.
# Copyright (c) 2025 by FlashInfer team.
"""SM120 MXFP8 x MXFP4 MoE with a separate finalize and caller workspace."""
from bisect import bisect_left
from functools import cache

import torch
from flashinfer.autotuner import (
    AutoTuner,
    DynamicTensorSpec,
    TunableRunner,
    TuningConfig,
)
from flashinfer.tllm_enums import ActivationType
from flashinfer.utils import check_shape_dtype_device

from .jit import MODULE_NAME, native_module

OP_NAME = f"rtp_llm::{MODULE_NAME}"
DTYPES = (torch.float8_e4m3fn, torch.long, torch.bfloat16)
SWIGLU = ActivationType.Swiglu.value


def _mode(
    x_dtype,
    weight_dtype,
    output_dtype,
    activation_type,
    use_mxfp8_act_scaling,
    use_fused_finalize,
):
    if (x_dtype, weight_dtype, output_dtype) != DTYPES:
        raise ValueError(
            "SM120 graph MoE requires MXFP8 activations, packed FP4 weights and BF16 output"
        )
    if (
        activation_type != ActivationType.Swiglu
        or not use_mxfp8_act_scaling
        or use_fused_finalize
    ):
        raise ValueError("SM120 graph MoE requires MXFP8 SwiGLU and separate finalize")


def _device_index(device):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("SM120 graph MoE requires a CUDA device")
    index = torch.cuda.current_device() if device.index is None else device.index
    if torch.cuda.get_device_capability(index) != (12, 0):
        raise ValueError("SM120 graph MoE requires SM120")
    return index


@cache
def _native_runner(device_index):
    with torch.cuda.device(device_index):
        return native_module().init(*DTYPES, False, False, True, False, False)


def _hybrid_num_tokens_buckets(max_tokens):
    """Keep the qualified FlashInfer bucket phases for separate finalize."""
    buckets = []
    tokens = 1
    while tokens <= min(max_tokens, 256):
        buckets.append(tokens)
        tokens *= 2
    buckets.extend(range(512, min(max_tokens, 2048) + 1, 256))
    buckets.extend(range(2560, min(max_tokens, 4096) + 1, 512))
    tokens = 8192
    while tokens <= max_tokens:
        buckets.append(tokens)
        tokens *= 2
    if not buckets or buckets[-1] != max_tokens:
        buckets.append(max_tokens)
    return tuple(buckets)


@cache
def _tuning_config(max_tokens):
    buckets = _hybrid_num_tokens_buckets(max_tokens)

    def round_up(tokens):
        if tokens <= 0:
            return 1
        return buckets[bisect_left(buckets, min(tokens, max_tokens))]

    return TuningConfig(
        dynamic_tensor_specs=(DynamicTensorSpec((0,), (0,), buckets, round_up),)
    )


class Dsv4Sm120SeparateMoeRunner(TunableRunner):
    def __init__(self, device_index, top_k, gemm_idx):
        self.device_index = device_index
        self.top_k = top_k
        self.gemm_idx = gemm_idx
        self.native = _native_runner(device_index)

    def tactic_range(self):
        gemm1_count = self.native.get_gemm1_tactic_count()
        if self.gemm_idx == 1:
            return range(gemm1_count)
        return range(gemm1_count, gemm1_count + self.native.get_gemm2_tactic_count())

    def get_valid_tactics(self, inputs, profile):
        tactics = self.tactic_range()
        return [i for i in tactics if self.native.get_tactic_occupancy(i) > 0] or list(
            tactics
        )

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        x, w1, b1, w2, b2 = inputs
        self.native.run_gemm_profile(
            x,
            w1,
            b1,
            w2,
            b2,
            self.top_k,
            1,
            0,
            1,
            0,
            1,
            0,
            False,
            False,
            self.gemm_idx,
            tactic,
            do_preparation,
            False,
            SWIGLU,
        )


def _tactic_name(top_k, stage):
    # Saved default-tuner keys omit the runner hash and extra fields. Encode
    # static identity here so the provider's ordinary names cannot collide.
    return f"{OP_NAME}::top{top_k}::gemm{stage}"


def _choose(input, w1, w2, top_k, max_tokens):
    tuner = AutoTuner.get()
    cfg = _tuning_config(max_tokens)
    device_index = _device_index(input.device)
    native = _native_runner(device_index)
    tactics = []
    for stage in (1, 2):
        runner = Dsv4Sm120SeparateMoeRunner(device_index, top_k, stage)
        _, tactic = tuner.choose_one(
            _tactic_name(top_k, stage),
            [runner],
            cfg,
            [input, w1, None, w2, None],
        )
        if tactic != -1 and tactic not in runner.tactic_range():
            raise ValueError(
                f"Cached tactic {tactic} is outside SM120 graph MoE GEMM{stage} range"
            )
        tactics.append(tactic)
    return native, tactics


@torch.library.custom_op(
    OP_NAME, mutates_args=("output", "workspace_buffer"), device_types="cuda"
)
def _run(
    output: torch.Tensor,
    workspace_buffer: torch.Tensor,
    input: torch.Tensor,
    selected: torch.Tensor,
    router: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    quant_scales: list[torch.Tensor],
    input_sf: torch.Tensor,
    swiglu_limit: torch.Tensor | None,
    tune_max_num_tokens: int,
) -> None:
    with torch.cuda.device(input.device):
        native, tactics = _choose(input, w1, w2, selected.size(1), tune_max_num_tokens)
        native.run_moe(
            output,
            input,
            selected,
            router,
            w1,
            None,
            w2,
            None,
            quant_scales,
            input_sf,
            None,
            None,
            swiglu_limit,
            True,
            1,
            0,
            1,
            0,
            1,
            0,
            False,
            False,
            tactics,
            False,
            SWIGLU,
            workspace_buffer,
        )


@_run.register_fake
def _fake_run(
    output,
    workspace_buffer,
    input,
    selected,
    router,
    w1,
    w2,
    quant_scales,
    input_sf,
    swiglu_limit,
    tune_max_num_tokens,
):
    return None


def cutlass_fused_moe(
    *,
    input,
    token_selected_experts,
    token_final_scales,
    fc1_expert_weights,
    fc2_expert_weights,
    output_dtype,
    quant_scales,
    input_sf,
    swiglu_limit,
    output,
    use_mxfp8_act_scaling,
    use_fused_finalize,
    enable_pdl,
    workspace_buffer,
    tune_max_num_tokens,
    activation_type,
) -> torch.Tensor:
    """Write into and return the caller-owned BF16 output."""
    _mode(
        input.dtype,
        fc1_expert_weights.dtype,
        output_dtype,
        activation_type,
        use_mxfp8_act_scaling,
        use_fused_finalize,
    )
    if enable_pdl:
        raise ValueError("SM120 graph MoE requires enable_pdl=False")
    check_shape_dtype_device(
        output,
        (input.size(0), fc2_expert_weights.size(1)),
        output_dtype,
        input.device,
        "output",
    )
    _run(
        output,
        workspace_buffer,
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc2_expert_weights,
        quant_scales,
        input_sf,
        swiglu_limit,
        tune_max_num_tokens,
    )
    return output


def cutlass_fused_moe_workspace_size(
    max_num_tokens,
    hidden_size,
    intermediate_size,
    num_experts_total,
    top_k,
    *,
    x_dtype,
    weight_dtype,
    output_dtype,
    activation_type,
    use_mxfp8_act_scaling,
    use_fused_finalize,
    device,
):
    _mode(
        x_dtype,
        weight_dtype,
        output_dtype,
        activation_type,
        use_mxfp8_act_scaling,
        use_fused_finalize,
    )
    index = _device_index(device)
    with torch.cuda.device(index):
        return int(
            _native_runner(index).get_workspace_size(
                max_num_tokens,
                hidden_size,
                intermediate_size,
                num_experts_total,
                top_k,
                1,
                0,
                1,
                0,
                False,
                SWIGLU,
            )
        )
