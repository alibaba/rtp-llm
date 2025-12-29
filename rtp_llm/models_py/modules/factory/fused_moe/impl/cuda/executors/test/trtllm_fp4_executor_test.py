from pathlib import Path
from dataclasses import dataclass
import torch
from torch.nn import functional as F

from flashinfer import (
    RoutingMethodType,
    GatedActType,
    e2m1_and_ufp8sf_scale_to_float,
    fp4_quantize,
)
from flashinfer.fp4_quantization import block_scale_interleave
from flashinfer.fused_moe import (
    WeightLayout,
    trtllm_fp4_block_scale_moe,
)
from flashinfer.fused_moe.core import (
    get_w2_permute_indices_with_cache,
    _maybe_get_cached_w3_w1_permute_indices,
)

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.ops import ParallelismConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import MoEConfigAdapter
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.trtllm_fp4_executor import (
    TrtllmFp4Executor,
)
from rtp_llm.utils.model_weight import W

@dataclass(frozen=False, slots=True)
class moe_args:
    num_tokens: int = None
    num_experts: int = None
    hidden_size: int = None
    intermediate_size: int = None
    top_k: int = None
    padding: int = None
    n_groups : int = None
    top_k_groups : int = None
    routed_scaling: float = None
    routing_method_type: RoutingMethodType = None
    permute_info: torch.Tensor = None
    use_routing_scales_on_input: bool = None
    gated_act_type: GatedActType = None
    routing_bias: torch.Tensor = None
    topk_ids: torch.Tensor = None
    topk_weights: torch.Tensor = None
    hidden_states: torch.Tensor = None
    hidden_states_scale: torch.Tensor = None
    hidden_states_scale_global: torch.Tensor = None
    hidden_states_dequant: torch.Tensor = None
    hidden_states_orig: torch.Tensor = None
    expert_logits: torch.Tensor = None
    gemm1_weights: torch.Tensor = None
    gemm1_scales: torch.Tensor = None
    gemm1_scales_global: torch.Tensor = None
    gemm1_scales_linear: torch.Tensor = None
    gemm1_weights_dequant: torch.Tensor = None
    gemm1_weights_orig: torch.Tensor = None
    gemm1_weights_fp4_shuffled: torch.Tensor = None
    gemm1_scales_fp4_shuffled: torch.Tensor = None
    gemm2_weights: torch.Tensor = None
    gemm2_scales: torch.Tensor = None
    gemm2_scales_global: torch.Tensor = None
    gemm2_scales_linear: torch.Tensor = None
    gemm2_weights_dequant: torch.Tensor = None
    gemm2_weights_orig: torch.Tensor = None
    gemm2_weights_fp4_shuffled: torch.Tensor = None
    gemm2_scales_fp4_shuffled: torch.Tensor = None
    c_global_sf: torch.Tensor = None
    scale_c_fc1: torch.Tensor = None
    scale_gate_fc1: torch.Tensor = None
    scale_c_fc2: torch.Tensor = None

cache_permute_indices = dict()

class FP4Moe:
    def __init__(self):
        self.sf_vec_size = 16
        global cache_permute_indices
        self._cache_permute_indices = cache_permute_indices

    def quantize_weights(self, gemm1_weights, gemm2_weights, hidden_states_sample):
        num_experts = gemm1_weights.shape[0]
        hidden_states_scale_global = calculate_fp4_global_scale_factor(
            hidden_states_sample,
            False,
        )

        gemm1_weights_fp4_bytes, gemm1_scales_fp4_bytes, gemm1_scales_global = (
            quant_nvfp4_batches(gemm1_weights, num_experts, False)
        )
        gemm2_weights_fp4_bytes, gemm2_scales_fp4_bytes, gemm2_scales_global = (
            quant_nvfp4_batches(gemm2_weights, num_experts, False)
        )

        return {
            "hidden_states_scale_global": hidden_states_scale_global,
            "gemm1_weights": gemm1_weights_fp4_bytes,
            "gemm1_scales": gemm1_scales_fp4_bytes,
            "gemm1_scales_global": gemm1_scales_global,
            "gemm1_weights_orig": gemm1_weights,
            "gemm2_weights": gemm2_weights_fp4_bytes,
            "gemm2_scales": gemm2_scales_fp4_bytes,
            "gemm2_scales_global": gemm2_scales_global,
            "gemm2_weights_orig": gemm2_weights,
        }

    def quantize_inputs(
        self, hidden_states, hidden_states_scale_global, is_swizzling=True
    ):
        (
            hidden_states_fp4_bytes,
            hidden_states_scale_fp4_bytes,
        ) = fp4_quantize(hidden_states.cuda(), hidden_states_scale_global.cuda(), 16, False, is_swizzling)
        hidden_states_scale_fp4_bytes = hidden_states_scale_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(*hidden_states.shape[:-1], -1)

        return {
            "hidden_states": hidden_states_fp4_bytes,
            "hidden_states_scale": hidden_states_scale_fp4_bytes,
        }

    def prepare_static_weights_for_kernel(self, args):
        hidden_size = args.hidden_size
        intermediate_size = args.intermediate_size
        num_experts = args.num_experts
        epilogue_tile_m = 128

        gemm1_weights_fp4 = args.gemm1_weights.view(torch.float8_e4m3fn)
        gemm1_scales_linear_fp4 = args.gemm1_scales.view(torch.float8_e4m3fn)
        gemm2_weights_fp4 = args.gemm2_weights.view(torch.float8_e4m3fn)
        gemm2_scales_linear_fp4 = args.gemm2_scales.view(torch.float8_e4m3fn)

        gemm1_weights_fp4_shuffled = []
        gemm1_scales_fp4_shuffled = []
        gemm2_weights_fp4_shuffled = []
        gemm2_scales_fp4_shuffled = []
        for i in range(num_experts):
            permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                self._cache_permute_indices,
                gemm1_weights_fp4[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm1_weights_fp4_shuffled.append(
                gemm1_weights_fp4[i]
                .view(torch.uint8)[permute_indices.to(gemm1_weights_fp4.device)]
                .contiguous()
            )

            permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
                self._cache_permute_indices,
                gemm1_scales_linear_fp4[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm1_scales_fp4_shuffled.append(
                block_scale_interleave(
                    gemm1_scales_linear_fp4[i]
                    .view(torch.uint8)[
                        permute_sf_indices.to(gemm1_scales_linear_fp4.device)
                    ]
                    .contiguous()
                )
            )

            permute_indices = get_w2_permute_indices_with_cache(
                self._cache_permute_indices,
                gemm2_weights_fp4[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm2_weights_fp4_shuffled.append(
                gemm2_weights_fp4[i]
                .view(torch.uint8)[permute_indices.to(gemm2_weights_fp4.device)]
                .contiguous()
            )

            permute_sf_indices = get_w2_permute_indices_with_cache(
                self._cache_permute_indices,
                gemm2_scales_linear_fp4[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm2_scales_fp4_shuffled.append(
                block_scale_interleave(
                    gemm2_scales_linear_fp4[i]
                    .view(torch.uint8)[
                        permute_sf_indices.to(gemm2_scales_linear_fp4.device)
                    ]
                    .contiguous()
                )
            )

        gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
        gemm1_scales_fp4_shuffled = (
            torch.stack(gemm1_scales_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(
                num_experts, 2 * intermediate_size, hidden_size // self.sf_vec_size
            )
        )

        gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
        gemm2_scales_fp4_shuffled = (
            torch.stack(gemm2_scales_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, hidden_size, intermediate_size // self.sf_vec_size)
        )

        scale_c_fc1 = (
            args.c_global_sf
            * (1.0 / args.gemm1_scales_global)
            * (1.0 / args.hidden_states_scale_global)
        )
        scale_gate_fc1 = (1.0 / args.gemm1_scales_global) * (
            1.0 / args.hidden_states_scale_global
        )
        scale_c_fc2 = (1.0 / args.c_global_sf) * (
            1.0 / args.gemm2_scales_global
        )

        args.gemm1_weights_fp4_shuffled = gemm1_weights_fp4_shuffled
        args.gemm1_scales_fp4_shuffled = gemm1_scales_fp4_shuffled
        args.gemm2_weights_fp4_shuffled = gemm2_weights_fp4_shuffled
        args.gemm2_scales_fp4_shuffled = gemm2_scales_fp4_shuffled
        args.scale_c_fc1 = scale_c_fc1
        args.scale_gate_fc1 = scale_gate_fc1
        args.scale_c_fc2 = scale_c_fc2

    def call_moe(self, args):
        input_quantized = self.quantize_inputs(
            args.hidden_states_orig,
            args.hidden_states_scale_global,
            is_swizzling=False,
        )

        args.hidden_states = input_quantized["hidden_states"]
        args.hidden_states_scale = input_quantized["hidden_states_scale"]
        output = trtllm_fp4_block_scale_moe(
            routing_logits=args.expert_logits,
            routing_bias=args.routing_bias,
            hidden_states=args.hidden_states,
            hidden_states_scale=args.hidden_states_scale,
            gemm1_weights=args.gemm1_weights_fp4_shuffled,
            gemm1_weights_scale=args.gemm1_scales_fp4_shuffled,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=args.gemm2_weights_fp4_shuffled,
            gemm2_weights_scale=args.gemm2_scales_fp4_shuffled,
            gemm2_bias=None,
            output1_scale_scalar=args.scale_c_fc1,
            output1_scale_gate_scalar=args.scale_gate_fc1,
            output2_scale_scalar=args.scale_c_fc2,
            num_experts=args.num_experts,
            top_k=args.top_k,
            n_group=args.n_groups,
            topk_group=args.top_k_groups,
            intermediate_size=args.intermediate_size,
            local_expert_offset=0,
            local_num_experts=args.num_experts,
            routed_scaling_factor=args.routed_scaling,
            tile_tokens_dim=None,  # tile_tokens_dim
            routing_method_type=args.routing_method_type,
            gated_act_type=args.gated_act_type,
            do_finalize=True,
            tune_max_num_tokens=4096,
        )
        return output[0].to(torch.float)

    def compute_reference(self, args):
        sf_vec_size = 16
        ufp8_type_weights = 1

        inputs_data = self.quantize_inputs(
            args.hidden_states_orig, args.hidden_states_scale_global, is_swizzling=False
        )
        args.hidden_states = inputs_data["hidden_states"]
        args.hidden_states_scale = inputs_data["hidden_states_scale"]
        args.hidden_states_dequant = e2m1_and_ufp8sf_scale_to_float(
            args.hidden_states.cpu(),
            args.hidden_states_scale.cpu().view(torch.uint8).reshape(-1),
            (1 / args.hidden_states_scale_global).cpu(),
            sf_vec_size,
            ufp8_type_weights,
            False,  # is_sf_swizzled_layout
        ).cuda()

        args.gemm1_weights_dequant = e2m1_and_ufp8_scale_batches(
            args.gemm1_weights,
            args.gemm1_scales,
            1 / args.gemm1_scales_global,
            sf_vec_size,
            ufp8_type_weights,
        ).cuda()

        args.gemm2_weights_dequant = e2m1_and_ufp8_scale_batches(
            args.gemm2_weights,
            args.gemm2_scales,
            1 / args.gemm2_scales_global,
            sf_vec_size,
            ufp8_type_weights,
        ).cuda()

        return run_moe_dequant(args)

    def get_tolerances(self):
        """Get FP4-specific accuracy tolerances."""
        return {"atol": 0.1, "rtol": 0.15, "percent": 0.91}

class FP4MoeExecutor(FP4Moe):
    def prepare_static_weights_for_kernel(self, args):
        pass

    def call_moe(self, args):
        model_config = ModelConfig()
        model_config.expert_num = args.num_experts
        model_config.hidden_size = args.hidden_size
        model_config.moe_inter_size = args.intermediate_size
        model_config.moe_k = args.top_k
        parallelism_config = ParallelismConfig()
        parallelism_config.dp_size = 1
        parallelism_config.tp_size = 1
        parallelism_config.ep_size = 1
        config = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
        )
        payload = ExpertForwardPayload(
            expert_x=args.hidden_states_orig,
            expert_x_origin_dtype=torch.bfloat16,
            expert_topk_ids=args.topk_ids,
            expert_topk_weights=args.topk_weights,
        )
        weights = {
            W.moe_w1: args.gemm1_weights.view(torch.float8_e4m3fn),
            W.moe_w2: args.gemm2_weights.view(torch.float8_e4m3fn),
            W.moe_s1: args.gemm1_scales.view(torch.float8_e4m3fn),
            W.moe_s2: args.gemm2_scales.view(torch.float8_e4m3fn),
            "w13_input_scale": 1.0 / args.hidden_states_scale_global,
            "w13_weight_scale_2": 1.0 / args.gemm1_scales_global,
            "w2_input_scale": 1.0 / args.c_global_sf,
            "w2_weight_scale_2": 1.0 / args.gemm2_scales_global,
        }

        executor = TrtllmFp4Executor(config, weights, FusedMoEQuantConfig())
        output = executor.execute(payload, "silu", None, None, False, None)
        return output.to(torch.float)



def routing_reference(expertLogits, topK, padding):
    """Reference routing implementation for permutation calculation."""
    originalDevice = expertLogits.device
    expertLogits = expertLogits.cpu()
    numTokens, numExperts = expertLogits.shape
    assert topK <= numExperts

    numTokensPerExpert = torch.zeros(numExperts, dtype=torch.int64)
    expandedTokenIdxToExpert = -torch.ones(numTokens * topK, dtype=torch.int64)
    expandedTokenIdxToIdxInExpert = -torch.ones(numTokens * topK, dtype=torch.int64)

    topKLogits, topKIndices = torch.topk(expertLogits, topK, dim=1)
    for tokenIdx in range(numTokens):
        for k in range(topK):
            expandedIdx = tokenIdx * topK + k
            expertIndex = topKIndices[tokenIdx, k]
            expandedTokenIdxToExpert[expandedIdx] = expertIndex
            expandedTokenIdxToIdxInExpert[expandedIdx] = numTokensPerExpert[expertIndex]
            numTokensPerExpert[expertIndex] += 1

    paddedTokensPerExpertPrefixSum = torch.zeros(numExperts + 1, dtype=torch.int64)
    for ii in range(numExperts):

        def divUpMul(a, b):
            return (a + b - 1) // b * b

        paddedTokensPerExpertPrefixSum[ii + 1] = paddedTokensPerExpertPrefixSum[
            ii
        ] + divUpMul(numTokensPerExpert[ii], padding)
    permutedBufferSize = paddedTokensPerExpertPrefixSum[numExperts]

    expandedTokenIdxToPermutedIdx = -torch.ones(numTokens * topK, dtype=torch.int64)
    permutedIdxToExpandedIdx = -torch.ones(permutedBufferSize, dtype=torch.int64)
    permutedIdxToTokenIdx = -torch.ones(permutedBufferSize, dtype=torch.int64)
    for tokenIdx in range(numTokens):
        for k in range(topK):
            expandedIdx = tokenIdx * topK + k
            expert = expandedTokenIdxToExpert[expandedIdx]
            offsetWithinExpert = expandedTokenIdxToIdxInExpert[expandedIdx]
            offsetForExpert = paddedTokensPerExpertPrefixSum[expert]
            permutedIdx = offsetForExpert + offsetWithinExpert

            expandedTokenIdxToPermutedIdx[expandedIdx] = permutedIdx
            permutedIdxToExpandedIdx[permutedIdx] = expandedIdx
            permutedIdxToTokenIdx[permutedIdx] = tokenIdx
    return {
        "paddedTokensPerExpertPrefixSum": paddedTokensPerExpertPrefixSum.to(
            originalDevice
        ),
        "permutedBufferSize": permutedBufferSize.item(),
        "expandedTokenIdxToPermutedIdx": expandedTokenIdxToPermutedIdx.to(
            originalDevice
        ),
        "permutedIdxToExpandedIdx": permutedIdxToExpandedIdx.to(originalDevice),
        "numTokensPerExpert": numTokensPerExpert.to(originalDevice),
        "expandedTokenIdxToExpert": expandedTokenIdxToExpert.to(originalDevice),
        "topKLogits": topKLogits.to(originalDevice),
        "permutedIdxToTokenIdx": permutedIdxToTokenIdx.to(originalDevice),
        "topKIndices": topKIndices.to(originalDevice),
    }


def noaux_tc_ref(logits, bias, n_group, topk_group, top_k, routed_scaling_factor):
    """DeepSeek-style no-aux routing reference implementation."""
    scores = F.sigmoid(logits)
    scores_with_bias = scores + bias
    if n_group > 1:
        scores_shape = list(scores_with_bias.shape)
        group_scores = torch.sum(
            torch.topk(
                scores_with_bias.view(
                    scores_shape[:-1] + [n_group, scores_shape[-1] // n_group]
                ),
                k=2,
                dim=-1,
                largest=True,
                sorted=True,
            )[0],
            dim=-1,
        )
        _, group_idx = torch.topk(
            group_scores, k=topk_group, dim=-1, largest=True, sorted=True
        )
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(-1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(scores_shape[:-1] + [n_group, scores_shape[-1] // n_group])
            .reshape(scores_shape)
        )
        scores_with_bias = scores_with_bias * score_mask

    _, topk_idx = torch.topk(
        scores_with_bias, k=top_k, dim=-1, largest=True, sorted=True
    )
    new_mask = torch.zeros_like(scores)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = scores * new_mask
    score_sum = torch.sum(scores, dim=-1, keepdim=True) + 1e-20
    scores = scores / score_sum * routed_scaling_factor
    return scores


def routing_reference_no_aux(
    expert_logits,
    routing_bias,
    top_k,
    n_groups,
    top_k_groups,
    routed_scaling,
    padding,
    use_routing_scales_on_input=False,
):
    """Tiered TopK routing used by DeepSeek."""
    routing_logits = expert_logits.to(dtype=torch.float, device="cuda")
    if use_routing_scales_on_input:
        # if using routing scales on input, topK == 1 and the score is a plain sigmoid
        scores = F.sigmoid(routing_logits)
    else:
        scores = noaux_tc_ref(
            routing_logits, routing_bias, n_groups, top_k_groups, top_k, routed_scaling
        )
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


def routing_reference_renormalize(expert_logits, top_k, num_experts, padding):
    """TopK -> Softmax routing reference."""
    topk_values, topk_idx = torch.topk(expert_logits, k=top_k, dim=-1)
    topk_values = torch.nn.functional.softmax(topk_values.float(), dim=-1)

    new_mask = torch.zeros_like(expert_logits)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = expert_logits * new_mask

    for i in range(topk_idx.shape[0]):
        for j in range(topk_idx.shape[1]):
            scores[i, topk_idx[i, j]] = topk_values[i, j]
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


def routing_reference_renormalize_naive(expert_logits, top_k, num_experts, padding):
    """Softmax->TopK -> Normalize routing reference."""
    norm_topk_prob = True
    scores = torch.nn.functional.softmax(expert_logits.float(), dim=-1)
    topk_values, topk_idx = torch.topk(scores, k=top_k, dim=-1)

    if norm_topk_prob:  # only diff with mixtral sparse moe block!
        topk_values /= topk_values.sum(dim=-1, keepdim=True)
    topk_values = topk_values.to(expert_logits.dtype)
    scores = scores.to(expert_logits.dtype)

    new_mask = torch.zeros_like(expert_logits)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = expert_logits * new_mask

    for i in range(topk_idx.shape[0]):
        for j in range(topk_idx.shape[1]):
            scores[i, topk_idx[i, j]] = topk_values[i, j]
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


def routing_reference_topk(expert_logits, top_k, num_experts, padding):
    """TopK only (no softmax) routing reference."""
    topk_values, topk_idx = torch.topk(expert_logits, k=top_k, dim=-1)

    new_mask = torch.zeros_like(expert_logits)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = expert_logits * new_mask

    for i in range(topk_idx.shape[0]):
        for j in range(topk_idx.shape[1]):
            scores[i, topk_idx[i, j]] = topk_values[i, j]
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


def check_accuracy(a, b, atol, rtol, percent):
    """Unified accuracy checking function with detailed error reporting."""
    if not torch.isfinite(a).all():
        raise Exception("Non-finite values in reference output")
    if not torch.isfinite(b).all():
        raise Exception("Non-finite values in actual output")
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    close = torch.isclose(a, b, atol=atol, rtol=rtol)
    match_ratio = close.float().mean()
    if match_ratio >= percent:
        return

    mismatch_percent = 1.0 - match_ratio.item()
    if mismatch_percent > 1 - percent:
        raise Exception(
            f"Mismatch percentage is {mismatch_percent:.4f} for rtol {rtol} "
            f"(threshold: {1 - percent:.4f})"
        )


def calculate_fp4_global_scale_factor(tensor, use_ue8m0=False):
    if use_ue8m0:
        return torch.tensor(1.0, dtype=torch.float32)
    else:
        return (448 * 6) / tensor.float().abs().nan_to_num().max()


def e2m1_and_ufp8_scale_batches(
    mat_fp4: torch.Tensor,
    scale_tensor: torch.Tensor,
    global_scale_tensor: torch.Tensor,
    sf_vec_size: int,
    ufp8_type: int = 1,
):
    """Batch FP4 dequantization helper."""
    num_batches = mat_fp4.size(0)
    scale_tensor = scale_tensor.view(num_batches, -1)

    tensors = [
        e2m1_and_ufp8sf_scale_to_float(
            mat_fp4[b, :, :].cpu(),
            scale_tensor[b, :].cpu().reshape(-1),
            global_scale_tensor[b].cpu(),
            sf_vec_size,
            ufp8_type,
            False,  # is_sf_swizzled_layout
        )
        for b in range(num_batches)
    ]

    result = torch.stack(tensors)
    return result


def quant_nvfp4_batches(a, num_experts, is_sf_swizzled_layout=True):
    quant_a = []
    sfs = []
    global_sfs = []
    for i in range(num_experts):
        a_global_sf = calculate_fp4_global_scale_factor(a[i], False)
        a_fp4, a_sf = fp4_quantize(a[i].cuda(), a_global_sf.cuda(), 16, False, is_sf_swizzled_layout)
        quant_a.append(a_fp4)
        sfs.append(a_sf)
        global_sfs.append(a_global_sf)

    result_quant_a = torch.stack(quant_a)
    result_sfs = torch.stack(sfs)
    result_global_sfs = torch.stack(global_sfs)

    return result_quant_a, result_sfs, result_global_sfs


def quant_dequant_fp4(a, use_ue8m0=False, is_sf_swizzled_layout=True):
    """FP4 quantize-dequantize roundtrip function with centralized global scale factor calculation."""
    # Use centralized global scale factor calculation
    a_global_sf = calculate_fp4_global_scale_factor(a, use_ue8m0)
    sf_vec_size = 32 if use_ue8m0 else 16

    a_fp4, a_sf = fp4_quantize(
        a.cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0, is_sf_swizzled_layout
    )

    a_pt = e2m1_and_ufp8sf_scale_to_float(
        a_fp4.cpu(),
        a_sf.cpu().reshape(-1),
        (1 / a_global_sf).cpu(),
        sf_vec_size,
        1 if not use_ue8m0 else 0,  # ufp8_type
        is_sf_swizzled_layout,
    )

    return a_pt.cuda(), a_global_sf

def run_moe_dequant(args):
    """Common dequantized MoE reference implementation."""
    # Permute
    total_num_padded_tokens = args.permute_info["permutedBufferSize"]
    expanded_idx_to_permuted_idx = args.permute_info[
        "expandedTokenIdxToPermutedIdx"
    ].cpu()
    num_tokens_per_expert = args.permute_info["numTokensPerExpert"].cpu()
    permute_output = torch.full(
        (total_num_padded_tokens, args.hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    for i in range(args.num_tokens):
        for j in range(args.top_k):
            permuted_idx = expanded_idx_to_permuted_idx[i * args.top_k + j]
            permute_output[permuted_idx] = args.hidden_states_dequant[i]

    # Gemm1
    gemm1_output = torch.full(
        (total_num_padded_tokens, 2 * args.intermediate_size),
        float("nan"),
        device="cuda",
    ).to(torch.float)
    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = permute_output[i : i + my_num_tokens]
        my_b = args.gemm1_weights_dequant[expert_idx]
        my_c = my_a @ my_b.t()
        gemm1_output[i : i + my_num_tokens] = my_c
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding

    if args.use_routing_scales_on_input:
        assert args.top_k == 1
        # For each token and its top_k experts
        for token_idx in range(args.num_tokens):
            for k in range(args.top_k):
                # Get the permuted index for this token's k-th expert
                expanded_idx = token_idx * args.top_k + k
                permuted_idx = expanded_idx_to_permuted_idx[expanded_idx]
                expert_weight = args.permute_info["topKLogits"].to(torch.float)
                # Get the expert weight for this token and expert
                weight = expert_weight[token_idx, k]
                # Scale the corresponding row in gemm1_output
                gemm1_output[permuted_idx] *= weight

    # Activation
    activation_output = torch.full(
        (total_num_padded_tokens, args.intermediate_size), float("nan"), device="cuda"
    ).to(torch.float)

    gated_act_type = args.gated_act_type
    gated_act_type_to_func = {
        0: F.silu,
        1: F.gelu,
    }
    gated_act_func = gated_act_type_to_func[gated_act_type]

    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = gemm1_output[i : i + my_num_tokens]
        my_x1 = my_a[:, : args.intermediate_size]
        my_x2 = my_a[:, args.intermediate_size :]
        activation_output[i : i + my_num_tokens] = gated_act_func(my_x2) * my_x1
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding

    activation_output, c_global_sf = quant_dequant_fp4(
        activation_output.to(torch.bfloat16), False, True
    )
    activation_output = activation_output.to(torch.float)
    args.c_global_sf = c_global_sf

    # Gemm2
    gemm2_output = torch.full(
        (total_num_padded_tokens, args.hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = activation_output[i : i + my_num_tokens]
        my_b = args.gemm2_weights_dequant[expert_idx]
        my_c = my_a @ my_b.t()
        gemm2_output[i : i + my_num_tokens] = my_c
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding

    # Finalize
    expert_weight = args.permute_info["topKLogits"].to(torch.float)
    finalize_output = torch.full(
        (args.num_tokens, args.hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    for i in range(args.num_tokens):
        acc = torch.zeros(args.hidden_size, dtype=torch.float, device="cuda")
        for top_k_idx in range(args.top_k):
            expanded_idx = i * args.top_k + top_k_idx
            permuted_idx = expanded_idx_to_permuted_idx[expanded_idx]
            original_vector = gemm2_output[permuted_idx]
            weight = (
                expert_weight[i, top_k_idx]
                if not args.use_routing_scales_on_input
                else 1.0
            )
            acc += original_vector * weight
        finalize_output[i] = acc
    return finalize_output

def test_moe(
    num_tokens,
    hidden_size,
    intermediate_size,
    moe_impl,
    routing_config,
    weight_processing,
    gated_act_type,
):
    torch.cuda.synchronize()

    seed = 0
    torch.random.manual_seed(seed)

    top_k = routing_config["top_k"]
    padding = routing_config["padding"]
    n_groups = routing_config.get("n_groups")
    top_k_groups = routing_config.get("top_k_groups")
    routed_scaling = routing_config.get("routed_scaling")
    num_experts = routing_config["num_experts"]
    routing_method_type = routing_config["routing_method_type"]

    assert top_k <= num_experts
    assert top_k <= 10
    if (top_k_groups is not None) and (n_groups is not None) and (n_groups > 0):
        assert top_k_groups <= 4
        assert num_experts > n_groups
        assert num_experts % n_groups == 0
        assert num_experts % 4 == 0
        assert top_k < (top_k_groups * num_experts / n_groups)

    expert_logits = torch.randn((num_tokens, num_experts), device="cuda")
    if routing_method_type == RoutingMethodType.DeepSeekV3:
        expert_logits = expert_logits.to(torch.float)
    else:
        expert_logits = expert_logits.to(torch.bfloat16)

    if routing_config.get("has_routing_bias"):
        routing_bias = torch.randn(num_experts, device="cuda", dtype=torch.bfloat16)
    else:
        routing_bias = None

    hidden_states = 2 * torch.randn(
        (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    gemm1_weights = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    gemm2_weights = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device="cuda",
        dtype=torch.bfloat16,
    )

    if routing_method_type == RoutingMethodType.DeepSeekV3:
        permute_info, scores = routing_reference_no_aux(
            expert_logits,
            routing_bias,
            top_k,
            n_groups,
            top_k_groups,
            routed_scaling,
            padding,
            False,
        )
    elif routing_method_type == RoutingMethodType.Renormalize:
        permute_info, scores = routing_reference_renormalize(
            expert_logits, top_k, num_experts, padding
        )
    elif routing_method_type == RoutingMethodType.RenormalizeNaive:
        permute_info, scores = routing_reference_renormalize_naive(
            expert_logits, top_k, num_experts, padding
        )
    elif routing_method_type == RoutingMethodType.TopK:
        permute_info, scores = routing_reference_topk(
            expert_logits, top_k, num_experts, padding
        )
    elif routing_method_type == RoutingMethodType.Llama4:
        permute_info, scores = routing_reference_no_aux(
            expert_logits,
            routing_bias,
            top_k,
            n_groups,
            top_k_groups,
            routed_scaling,
            padding,
            use_routing_scales_on_input=True,
        )
    else:
        raise NotImplementedError(
            f"Routing method {routing_method_type} not implemented"
        )

    weights_data = moe_impl.quantize_weights(gemm1_weights, gemm2_weights, hidden_states)

    topk_ids = permute_info["topKIndices"].to(torch.int32)
    moe_info = {
        "num_tokens": num_tokens,
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "top_k": top_k,
        "padding": padding,
        "expert_logits": expert_logits,
        "topk_ids": topk_ids,
        "topk_weights": scores.view(num_tokens, num_experts)[
            torch.arange(num_tokens).unsqueeze(1), topk_ids
        ].to(torch.bfloat16),
        "permute_info": permute_info,
        "use_routing_scales_on_input": routing_method_type == RoutingMethodType.Llama4,
        "gated_act_type": gated_act_type,
        "hidden_states_orig": hidden_states,
        "routing_method_type": routing_method_type.value,
    }
    args = moe_args(**moe_info, **weights_data)

    output_dequant_reference = moe_impl.compute_reference(args)
    moe_impl.prepare_static_weights_for_kernel(args)
    output_dequant_actual = moe_impl.call_moe(args)

    # Compare outputs
    tolerances = moe_impl.get_tolerances()
    print(moe_impl)
    print("ref", output_dequant_reference)
    print("actual", output_dequant_actual)
    check_accuracy(
        output_dequant_reference,
        output_dequant_actual,
        atol=tolerances["atol"],
        rtol=tolerances["rtol"],
        percent=tolerances["percent"],
    )


if __name__ == "__main__":
    for cls in [FP4Moe, FP4MoeExecutor]:
        test_moe(
            num_tokens=3072,
            hidden_size=1024,
            intermediate_size=768,
            moe_impl=cls(),
            routing_config={
                "num_experts": 128,
                "top_k": 8,
                "padding": 8,
                "routing_method_type": RoutingMethodType.Renormalize,
            },
            weight_processing={
                "layout": WeightLayout.MajorK,
            },
            gated_act_type=GatedActType.SwiGlu,
        )