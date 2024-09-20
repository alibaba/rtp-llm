#include "src/fastertransformer/devices/CudaDevice.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/kernels/mla_kernels/mla_merge_transpose_kernel.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"

namespace fastertransformer {
BufferPtr CudaDevice::mlaQKVGemm(const AttentionLayerParams& params) {
    auto        datatype = params.input.type();
    const auto& input    = params.input;

    auto token_num     = params.input.shape()[0];
    auto head_num      = params.configs.head_num;
    auto nope_head_dim = params.configs.nope_head_dim;
    auto rope_head_dim = params.configs.rope_head_dim;
    auto v_head_dim    = params.configs.v_head_dim;
    auto nope_rope_dim = nope_head_dim + rope_head_dim;

    auto qkv =
        allocateBuffer({datatype, {token_num, head_num * nope_rope_dim * 3}, AllocationType::DEVICE}, {"mla_qkv"});

    // Q_a = input * W_qa
    // Q_a = normalize(Q_a)
    // Q = Q_a * W_qb
    BufferPtr q;
    if (params.weights.q_a_weight) {
        auto q_a = gemm(GemmParams(input, *(params.weights.q_a_weight->kernel)));

        allGather({{q_a}});

        layernorm(LayernormParams(q_a,
                                q_a,
                                mayGetRef(params.weights.q_a_norm_weight),
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                1.0f,
                                params.ln_params.eps,
                                true,
                                false,
                                params.ln_params.norm_type));
        q = gemm(GemmParams(*q_a, *(params.weights.q_b_weight->kernel)));
    } else {
        q = gemm(GemmParams(input, *(params.weights.q_weight->kernel)));
    }

    // kv_a = input * W_kva
    // kv_a = normalize(kv_a)
    // knope = kv_a * W_knope
    // v = kv_a * W_v
    auto kv_a = gemm(GemmParams(input, *(params.weights.kv_a_weight->kernel)));

    allGather({{kv_a}});

    layernorm(LayernormParams(kv_a,
                              kv_a,
                              mayGetRef(params.weights.kv_a_norm_weight),
                              std::nullopt,
                              std::nullopt,
                              std::nullopt,
                              1.0f,
                              params.ln_params.eps,
                              true,
                              false,
                              params.ln_params.norm_type));
    auto k_nope = gemm(GemmParams(*kv_a, *(params.weights.k_nope_weight->kernel)));

    auto v      = gemm(GemmParams(*kv_a, *(params.weights.v_weight->kernel)));

    // k_rope = input * W_krope
    auto k_rope = gemm(GemmParams(input, *(params.weights.k_rope_weight->kernel)));

    // invoke mla merge transpose
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                     invokeMlaMergeTranspose,
                                     q->data(),
                                     k_nope->data(),
                                     k_rope->data(),
                                     v->data(),
                                     qkv->data(),
                                     token_num,
                                     head_num,
                                     nope_head_dim,
                                     rope_head_dim,
                                     v_head_dim,
                                     stream_);
    printBufferData(*qkv, "MLA QKV Gemm output");
    return qkv;
}
}  // namespace fastertransformer