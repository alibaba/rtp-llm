#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/arm_impl/ArmDevice.h"
#include "rtp_llm/cpp/devices/arm_impl/ArmDispatch.h"
#include "rtp_llm/cpp/devices/utils/DevicePerfWrapper.h"
#include <cstdint>

using namespace std;
using namespace rtp_llm;

namespace rtp_llm {

extern void batch_matmul_kai_bf16(const float* input, size_t input_batch_stride, size_t input_row_stride, const bfloat16_t* weight,
    float* output, size_t output_batch_stride, size_t output_row_stride, int m, int k, int n, int bs);

void InputGemmWrapper(const Buffer& q, Buffer& fused_q_input, const MlaAttentionModuleParams& params) {
    if (params.weights.kc_weight->kernel->type() != DataType::TYPE_BF16) {
        throw std::runtime_error("InputGemmWrapper weight data type is not BF16");
    }

    // q: [token_num, head_num, nope_head_dim + rope_head_dim]
    // kc_weight: [head_num, nope_head_dim, ckv_dim]
    // fused_q_input: [token_num, head_num, ckv_dim + rope_dim]

    int token_num = q.shape()[0];
    int head_num = params.configs.head_num;
    int nope_head_dim = params.configs.nope_head_dim;
    int rope_head_dim = params.configs.rope_head_dim;
    int ckv_dim = params.configs.kv_lora_rank;

    batch_matmul_kai_bf16(
        (float*)q.data(), nope_head_dim + rope_head_dim, head_num * (nope_head_dim + rope_head_dim),
        (bfloat16_t*)params.weights.kc_weight->kernel->data(),
        (float*)fused_q_input.data(), ckv_dim + rope_head_dim, head_num * (ckv_dim + rope_head_dim),
        token_num, nope_head_dim, ckv_dim, head_num);
}


void OutputGemmWrapper(const Buffer& attn_out, Buffer& qkv_output, const MlaAttentionModuleParams& params) {
    if (params.weights.vc_weight->kernel->type() != DataType::TYPE_BF16) {
        throw std::runtime_error("OutputGemmWrapper weight data type is not BF16");
    }

    // attn_out: [token_num, head_num, ckv_dim]
    // vc_weight: [head_num, ckv_dim, v_head_dim]
    // qkv_output_t: [token_num, head_num, v_head_dim]

    int token_num = params.qkv_output->shape()[0];
    int head_num = params.configs.head_num;
    int ckv_dim = params.configs.kv_lora_rank;
    int v_head_dim = params.configs.v_head_dim;

    batch_matmul_kai_bf16(
        (float*)attn_out.data(), ckv_dim, head_num * ckv_dim,
        (bfloat16_t*)params.weights.vc_weight->kernel->data(),
        (float*)qkv_output.data(), v_head_dim, head_num * v_head_dim,
        token_num, ckv_dim, v_head_dim, head_num);
}

void ArmCpuDevice::mlaAbsorbAttention(const MlaAttentionModuleParams& params) {
    DevicePerfWrapper wrapper(this, "mlaDecoder_layer_%d", params.layer_id);

    auto fused_q_input = allocateBuffer({params.q.type(), {params.q.shape()[0], params.configs.head_num, params.configs.kv_lora_rank + params.configs.rope_head_dim}, AllocationType::DEVICE});

    mlaRotaryWriteKVCache({params.q,
                           fused_q_input,
                           params.fused_qkv,
                           params.kv_offset,
                           params.is_prefill ? params.common.prefill_flash_infer_attn : params.common.decode_flash_infer_attn,
                           params.common,
                           params.weights,
                           params.configs,
                           params.qscheme,
                           params.is_prefill});

    if (params.is_prefill) {
        writeCacheStore(params);
    }

    computeInsertedMoE();

    InputGemmWrapper(params.q, *fused_q_input, params);
    printBufferData(*fused_q_input, "fused_q_input");

    auto datatype = params.q.type();
    BufferPtr attn_out;

    attn_out = allocateBuffer({datatype, {params.q.shape()[0], params.configs.head_num, params.configs.kv_lora_rank}, AllocationType::DEVICE});

    if (params.q.type() != DataType::TYPE_FP32) {
        throw std::runtime_error("mla absorb attention data type is not supported");
    }

    AttentionModuleParams attn_params = AttentionModuleParams(
        {params.layer_id, *fused_q_input, *attn_out, params.common, params.weights, params.configs, params.qscheme});
    decoderSelfAttention(attn_params);

    OutputGemmWrapper(*attn_out, *params.qkv_output, params);
}

template <typename T>
void invokeMlaQKVMerge(T* q,
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

    parallel_for(token_num, [&](int bs_idx) {
        for (int head_idx = 0; head_idx < head_num; ++head_idx) {
            int q_offset = bs_idx * head_num * nope_rope_dim + head_idx * nope_rope_dim;
            int k_nope_offset = bs_idx * head_num * nope_head_dim + head_idx * nope_head_dim;
            int k_rope_offset = bs_idx * rope_head_dim;  // broadcast to head_num
            int v_offset = bs_idx * head_num * v_head_dim + head_idx * v_head_dim;
            int dst_base_offset = bs_idx * 3 * hidden_size + head_idx * nope_rope_dim;

            memcpy(qkv + dst_base_offset, q + q_offset, nope_rope_dim * sizeof(T));

            memcpy(qkv + dst_base_offset + hidden_size, k_nope + k_nope_offset, nope_head_dim * sizeof(T));
            memcpy(qkv + dst_base_offset + hidden_size + nope_head_dim, k_rope + k_rope_offset, rope_head_dim * sizeof(T));

            memcpy(qkv + dst_base_offset + 2 * hidden_size, v + v_offset, v_head_dim * sizeof(T));
            memset(qkv + dst_base_offset + 2 * hidden_size + v_head_dim, 0, (nope_rope_dim - v_head_dim) * sizeof(T));
        }
    });
}

AttentionModuleOutput ArmCpuDevice::mlaContextAttention(const MlaAttentionModuleParams& params) {
    DevicePerfWrapper wrapper(this, "mlaContext_layer_%d", params.layer_id);
    auto& q = params.q;
    auto& fused_qkv = params.fused_qkv;

    auto const token_num     = q.shape()[0];
    auto const head_num      = params.configs.head_num;
    auto const nope_head_dim = params.configs.nope_head_dim;
    auto const rope_head_dim = params.configs.rope_head_dim;
    auto const v_head_dim    = params.configs.v_head_dim;
    auto const nope_rope_dim = nope_head_dim + rope_head_dim;
    auto const size_per_head = params.configs.size_per_head;
    mlaRotaryWriteKVCache({q,
                           nullptr,
                           fused_qkv,
                           params.kv_offset,
                           params.common.prefill_flash_infer_attn,
                           params.common,
                           params.weights,
                           params.configs,
                           params.qscheme,
                           params.is_prefill});
    writeCacheStore(params);

    computeInsertedMoE();

    auto split_result = split({fused_qkv, {(size_t)params.kv_offset, (size_t)params.configs.kv_lora_rank, (size_t)params.configs.rope_head_dim}, 1});
    auto kv_a = split_result.outputs[1];
    auto k_rope   = split_result.outputs[2];
    printBufferData(q, "q_after_rope");
    printBufferData(*kv_a, "kv_a");
    printBufferData(*k_rope, "k_rope");

    auto datatype = fused_qkv.type();
    auto qkv =
        allocateBuffer({datatype, {token_num, head_num * nope_rope_dim * 3}, AllocationType::DEVICE}, {"mla_qkv"});

    auto k_nope = gemm(GemmParams(*kv_a, *(params.weights.k_nope_weight->kernel)));
    auto v      = gemm(GemmParams(*kv_a, *(params.weights.v_weight->kernel)));

    printBufferData(*k_nope, "k_nope");
    printBufferData(*v, "v");

    DISPATCH_ARM_FUNCTION_DATA_TYPE(datatype,
                                    invokeMlaQKVMerge,
                                    q.data(),
                                    k_nope->data(),
                                    k_rope->data(),
                                    v->data(),
                                    qkv->data(),
                                    token_num,
                                    head_num,
                                    nope_head_dim,
                                    rope_head_dim,
                                    v_head_dim);

    printBufferData(*qkv, "mla_qkv");
    auto padded_qkv_output_t = allocateBuffer({datatype, {token_num, head_num * size_per_head}, AllocationType::DEVICE}, {"padded_qkv_output"});
    AttentionModuleParams attn_params = AttentionModuleParams(
        {params.layer_id, *qkv, *padded_qkv_output_t, params.common, params.weights, params.configs, params.qscheme});
    // only paged fmha use kv_block_array, mld not use paged fmha
    contextAttention(attn_params);

    auto qkv_output_reshaped = padded_qkv_output_t->reshape({token_num, params.configs.head_num, size_per_head});
    auto sliced_buffer = slice({qkv_output_reshaped, -1, 0, (int64_t)v_head_dim});
    copy({*params.qkv_output, *sliced_buffer});
}

template<typename T>
void ApplyRope(int token_num, int *position_ids, int num_heads, T *src, int src_stride, int src_head_size, int src_head_offset, T *dst, int dst_stride, int dst_head_size, int dst_head_offset, int rope_dim, T* rope_cos_sin_cache) {
    if (src_head_size != src_head_offset + rope_dim) {
        throw std::runtime_error("Rope src wrong dim");
    }

    if (dst_head_size != dst_head_offset + rope_dim) {
        throw std::runtime_error("Rope dst wrong dim");
    }

    size_t inv_freq_size = (rope_dim + 1) / 2;

    auto rope = [&](int i) {
        T* input = src + i * src_stride + src_head_offset;
        T* output = dst + i * dst_stride + dst_head_offset;

        int pos = position_ids[i];

        for (int h = 0; h < num_heads; h++) {
            for (int d = 0; d < inv_freq_size; d++) {
                float fcr = rope_cos_sin_cache[pos * rope_dim + d];
                float fci = rope_cos_sin_cache[pos * rope_dim + inv_freq_size + d];
                float x = input[h * src_head_size + d];
                float y = input[h * src_head_size + d + inv_freq_size];
                output[h * dst_head_size + d] = x * fcr - y * fci;
                output[h * dst_head_size + d + inv_freq_size] = x * fci + y * fcr;
            }
        }
    };

    if (token_num == 1) { // fast path for decode
        rope(0);
    } else {
        parallel_for(token_num, rope);
    }
}

// apply rope to a [token_num, num_head, head_size] buffer
template<typename T>
void ApplyRope(int token_num, int *position_ids, int num_heads, T *buffer, int stride, int head_size, int head_offset, int rope_dim, T* rope_cos_sin_cache) {
    ApplyRope(token_num, position_ids, num_heads, buffer, stride, head_size, head_offset, buffer, stride, head_size, head_offset, rope_dim, rope_cos_sin_cache);
}

void getCacheAddrFromIndex(const KvCacheInfo& kv_cache, size_t batch, size_t block_idx, void **k_addr) {
    const auto& kv_blocks_offset = *(kv_cache.kv_cache_block_id);
    const auto& k_cache = *(kv_cache.k_cache_buffer);
    const auto  max_blocks_per_batch = kv_blocks_offset.shape()[1];
    size_t block_size = k_cache[0].sizeBytes();
    int    *index = (int *)kv_blocks_offset.data();

    *k_addr = (char*)k_cache.data() + index[batch * max_blocks_per_batch + block_idx] * block_size;
}

// update KV cache, from fused_qkv to k_cache_buffer, [token_num, dim]
// dim = ckv_dim + rope_dim
template<typename T>
void updateKVCacheStride(const MlaRotaryWriteKVCacheParams& params, T* fused_qkv, int stride, int dim, int batch, size_t seq_len, size_t step) {
    auto block_tokens  = params.configs.tokens_per_block;
    size_t block_offset = step / block_tokens;
    void *k_block_addr;

    // fast path for decode
    if (seq_len == 1) {
        getCacheAddrFromIndex(params.common.kv_cache.value(), batch, block_offset, &k_block_addr);
        memcpy((T*)k_block_addr + step % block_tokens * dim, fused_qkv, dim * sizeof(T));
        return;
    }

    size_t block_num = (seq_len + block_tokens - 1) / block_tokens;
    size_t copied_len = 0;

    for (int i = 0; i < block_num; i++) {
        size_t len = std::min(block_tokens, seq_len - copied_len);
        getCacheAddrFromIndex(params.common.kv_cache.value(), batch, i + block_offset, &k_block_addr);

        T* input = fused_qkv + (i * block_tokens) * stride;
        parallel_for(len, [&](int tid) {
            memcpy((T*)k_block_addr + (step % block_tokens + tid) * dim, input + tid * stride, dim * sizeof(T));
        });

        copied_len += len;
    }
}

void ArmCpuDevice::mlaRotaryWriteKVCache(const MlaRotaryWriteKVCacheParams& params) {
    DevicePerfWrapper wrapper(this, "mlaRotaryWriteKVCache");
    float* q = (float*)params.q.data();
    float* fused_qkv = (float*)params.fused_qkv.data();
    float* rope_cos_sin_cache = (float*)params.weights.rope_cos_sin_cache->data();

    // fused_qkv:   q/cq + ckv + k_rope
    //              [token_num, kv_offset + ckv_dim + rope_dim]
    // q:           [token_num, head_num, nope_dim    +  rope_dim]
    // fuse_dest_q: [token_num, head_num, ckv_dim     +  rope_dim]

    // rope_cos_sin_cache: [position_id, rope_dim]

    int batch = params.is_prefill ? params.common.context_batch_size : params.common.decoder_batch_size;
    int token_num = params.q.shape()[0];

    auto position_buf = allocateBuffer({DataType::TYPE_INT32, {(size_t)token_num}});
    int* position_ids = (int*)position_buf->data();
    if (params.is_prefill) {
        int *inp_len = (int*)(params.common.input_lengths->dataWithOffset(params.common.decoder_batch_size));
        int offset = 0;
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < inp_len[b]; i++) {
                position_ids[offset + i] = i;
            }
            offset += inp_len[b];
        }
    } else {
        int *seq_len = (int*)(params.common.sequence_lengths->data());
        for (int b = 0; b < batch; b++) {
            position_ids[b] = seq_len[b];
        }
    }

    int head_num  = params.configs.head_num;
    int nope_dim  = params.configs.nope_head_dim;
    int rope_dim  = params.configs.rope_head_dim;
    int ckv_dim   = params.configs.kv_lora_rank;
    int kv_offset = params.kv_offset;
    // Q rope
    if (params.fused_dest_q) {
    float* dst_q = (float*)params.fused_dest_q->data();
        ApplyRope(token_num, position_ids, head_num,
                  q, head_num * (nope_dim + rope_dim), nope_dim + rope_dim, nope_dim,
                  dst_q, head_num * (ckv_dim + rope_dim), ckv_dim + rope_dim, ckv_dim,
                  rope_dim, rope_cos_sin_cache);
    } else {
        ApplyRope(token_num, position_ids, head_num,
                  q, head_num * (nope_dim + rope_dim), nope_dim + rope_dim, nope_dim,
                  rope_dim, rope_cos_sin_cache);
    }
    // K rope
    ApplyRope(token_num, position_ids, 1,
              fused_qkv + kv_offset + ckv_dim, kv_offset + ckv_dim + rope_dim, rope_dim, 0,
              rope_dim, rope_cos_sin_cache);

    if (params.common.kv_cache.has_value()) {
        int offset = 0;
        for (int b = 0; b < batch; b++) {
            int seq_len, step;
            if (params.is_prefill) {
                int *inp_len = (int*)(params.common.input_lengths->dataWithOffset(params.common.decoder_batch_size));
                step = 0;
                seq_len = inp_len[b];
            } else {
                int *steps = (int*)(params.common.sequence_lengths->data());
                seq_len = 1;
                step = steps[b];
            }
            updateKVCacheStride(params, fused_qkv + (kv_offset + ckv_dim + rope_dim) * offset + kv_offset, kv_offset + ckv_dim + rope_dim, ckv_dim + rope_dim, b, seq_len, step);
            offset += seq_len;
        }
    }
}

}  // namespace rtp_llm
