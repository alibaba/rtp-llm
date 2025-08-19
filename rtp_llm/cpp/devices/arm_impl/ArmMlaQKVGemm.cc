#include "rtp_llm/cpp/devices/arm_impl/ArmDevice.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"

namespace rtp_llm {

template <typename T>
void mla_merge_transpose_cpu(T* q,
                            T* k_nope,
                            T* k_rope,
                            T* v,
                            float* qkv,
                            int token_num,
                            int head_num,
                            int nope_head_dim,
                            int rope_head_dim,
                            int v_head_dim) {
    int nope_rope_dim = nope_head_dim + rope_head_dim;
    int hidden_size = head_num * nope_rope_dim;

    for (int bs_idx = 0; bs_idx < token_num; ++bs_idx) {
        for (int head_idx = 0; head_idx < head_num; ++head_idx) {
            for (int tidx = 0; tidx < nope_rope_dim; ++tidx) {
                int q_offset = bs_idx * head_num * nope_rope_dim + head_idx * nope_rope_dim + tidx;
                int k_nope_offset = bs_idx * head_num * nope_head_dim + head_idx * nope_head_dim + tidx;
                int rope_idx = tidx - nope_head_dim;
                int k_rope_offset = bs_idx * rope_head_dim + rope_idx;  // broadcast to head_num
                int v_offset = bs_idx * head_num * v_head_dim + head_idx * v_head_dim + tidx;
                int dst_base_offset = bs_idx * 3 * hidden_size + head_idx * nope_rope_dim + tidx;

                if (tidx < nope_head_dim) {
                    qkv[dst_base_offset] = q[q_offset];
                    qkv[dst_base_offset + hidden_size] = k_nope[k_nope_offset];
                } else {
                    int trans_idx = rope_idx / 2;
                    int trans_offset = trans_idx + (rope_idx % 2 ? 1 : 0) * (rope_head_dim / 2) - tidx + nope_head_dim;
                    int q_dst = dst_base_offset + trans_offset;
                    int k_dst = q_dst + hidden_size;
                    qkv[q_dst] = q[q_offset];
                    qkv[k_dst] = k_rope[k_rope_offset];
                }

                if (tidx < v_head_dim) {
                    qkv[dst_base_offset + 2 * hidden_size] = v[v_offset];
                } else {
                    qkv[dst_base_offset + 2 * hidden_size] = 0;
                }
            }
        }
    }
}
BufferPtr ArmCpuDevice::mlaQKVGemm(const AttentionLayerParams& params) {
    auto        datatype = params.input.type();
    const auto& input    = params.input;

    auto token_num     = params.input.shape()[0];
    auto head_num      = params.configs.head_num;
    auto nope_head_dim = params.configs.nope_head_dim;
    auto rope_head_dim = params.configs.rope_head_dim;
    auto v_head_dim    = params.configs.v_head_dim;
    auto nope_rope_dim = nope_head_dim + rope_head_dim;

    auto qkv =
        allocateBuffer({DataType::TYPE_FP32, {token_num, head_num * nope_rope_dim * 3}, AllocationType::DEVICE}, {"mla_qkv"});

    // Q_a = input * W_qa
    // Q_a = normalize(Q_a)
    // Q = Q_a * W_qb
    BufferPtr fused_qkv = nullptr;
    BufferPtr q = nullptr;
    int64_t kv_offset = 0;
    if (params.weights.fusedqkrope_weight != nullptr) {
        fused_qkv = gemm(GemmParams(input, *(params.weights.fusedqkrope_weight->kernel)));
        kv_offset = params.configs.q_lora_rank;
        layernorm(LayernormParams(fused_qkv,
                                fused_qkv,
                                mayGetRef(params.weights.q_a_norm_weight),
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                1.0f,
                                params.ln_params.eps,
                                true,
                                false,
                                params.ln_params.norm_type));
        q = gemm(GemmParams(*fused_qkv, *(params.weights.q_b_weight->kernel)));
    } else {
        fused_qkv = gemm(GemmParams(input, *(params.weights.fusedqkrope_no_lora_weight->kernel)));
        kv_offset = params.configs.head_num * params.configs.size_per_head;
        printf("@mlaQKVGemm kv_offset %ld\n", kv_offset);
        q = slice(SliceParams({*fused_qkv, -1, 0, (int64_t)(params.configs.head_num * params.configs.size_per_head)}));
    }

    // kv_a = input * W_kva
    // kv_a = normalize(kv_a)
    // knope = kv_a * W_knope
    // v = kv_a * W_v
    auto kv_a = gemm(GemmParams(input, *(params.weights.kv_a_weight->kernel)));
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

    if (datatype == DataType::TYPE_FP16) {
        mla_merge_transpose_cpu<__fp16>(
                                    (__fp16 *)q->data(),
                                    (__fp16 *)k_nope->data(),
                                    (__fp16 *)k_rope->data(),
                                    (__fp16 *)v->data(),
                                    (float *)qkv->data(),
                                    token_num,
                                    head_num,
                                    nope_head_dim,
                                    rope_head_dim,
                                    v_head_dim);
    } else if (datatype == DataType::TYPE_FP16) {
        mla_merge_transpose_cpu<float>(
                                    (float *)q->data(),
                                    (float *)k_nope->data(),
                                    (float *)k_rope->data(),
                                    (float *)v->data(),
                                    (float *)qkv->data(),
                                    token_num,
                                    head_num,
                                    nope_head_dim,
                                    rope_head_dim,
                                    v_head_dim);
    } else {
        throw std::runtime_error("mla_merge_transpose_cpu type is not supported");
    }
    printBufferData(*qkv, "MLA QKV Gemm output");
    return qkv;
}
}  // namespace rtp_llm