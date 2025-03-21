#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/kernels/mla_kernels/mla_merge_transpose_kernel.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include "src/fastertransformer/devices/utils/DevicePerfWrapper.h"
#include "flashmla/flashmla.h"

using namespace std;
using namespace rtp_llm;

namespace fastertransformer {

// q_input_shape [batch_size, head_num, nope_head_dim + rope_head_dim]

torch::Tensor QInputBatchMatmulWrapper(const MlaDecoderAttentionParams& params) {
    // from [token, head_num * (nope_head_dim + rope_head_dim)] to [batch_size, head_num, nope_head_dim + rope_head_dim]
    auto q_input_reshape = params.q.reshape(
        {params.q.shape()[0], params.configs.head_num, params.configs.nope_head_dim + params.configs.rope_head_dim});
    // from [batch_size, head_num, nope_head_dim + rope_head_dim] to [batch_size, head_num, nope_head_dim] with stride
    auto q_nope_t = Buffer2torchTensorWithStride(
        q_input_reshape,
        {(int64_t)params.q.shape()[0], (int64_t)params.configs.head_num, (int64_t)params.configs.nope_head_dim});
    // shape: [head_num, nope_head_dim, kv_lora_rank]
    auto w_kc_t = Buffer2torchTensor(params.weights.kc_weight->kernel, false);
    auto q_nope_out = torch::bmm(q_nope_t.transpose(0, 1), w_kc_t);
    auto q_input_reshap_t = Buffer2torchTensor(q_input_reshape, false);
    return q_nope_out.transpose(0, 1).contiguous();
}

torch::Tensor DecoderOutputGemmWrapper(const torch::Tensor& attn_out_t, const MlaDecoderAttentionParams&
params) {
    auto w_vc_t = Buffer2torchTensor(params.weights.vc_weight->kernel, false);
    return torch::bmm(attn_out_t.transpose(0, 1), w_vc_t).transpose_(0, 1).contiguous();
}

void CudaDevice::mlaDecoderSelfAttention(const MlaDecoderAttentionParams& params) {
    DevicePerfWrapper wrapper(this, "mlaDecoder_layer_%d", params.layer_id);

    const auto generate_batch_size = params.common.decoder_batch_size;

    auto absorb_q_input = QInputBatchMatmulWrapper(params);
    printBufferData(*torchTensor2Buffer(absorb_q_input), "mla_absorb_q_input");
    auto ckv = Buffer2torchTensor(params.common.kv_cache->k_cache_buffer, false);
    auto flash_infer_attn_params = (FlashInferAttnParams*)params.common.flash_infer_attn_params.get();
    if (!flash_infer_attn_params) {
        throw std::runtime_error("flash_infer_attn_params must be setting when using mla");
    }
    const auto &ckv_cache_shape = params.common.kv_cache->k_cache_buffer->shape();
    auto datatype = params.q.type();
    at::Tensor attn_out_t;
    const auto &flashinfer = *flash_infer_attn_params;
    // maybe some shape check ?

    auto q_reshape = params.q.reshape({params.q.shape()[0], params.configs.head_num, params.configs.nope_head_dim + params.configs.rope_head_dim});
    auto q_rope = Buffer2torchTensorWithStride(q_reshape, {(int64_t)params.q.shape()[0], (int64_t)params.configs.head_num, (int64_t)params.configs.rope_head_dim}, params.configs.nope_head_dim);

    if (mla_ops_type == MlaOpsType::FLASH_MLA) {
        // [batch_size, 1, num_heads, kv_lora_rank + rope_head_dim]
        auto q_with_rope = torch::cat({absorb_q_input, q_rope}, -1).reshape({
            (int64_t)params.q.shape()[0], (int64_t)1, (int64_t)params.configs.head_num, (int64_t)(params.configs.kv_lora_rank + params.configs.rope_head_dim)
        });
        printBufferData(*torchTensor2Buffer(q_with_rope), "q_with_rope");

        auto ckv_cache_reshape = params.common.kv_cache->k_cache_buffer->reshape({
            ckv_cache_shape[0], ckv_cache_shape[1], 1, ckv_cache_shape[2],
        });
        auto ckv_cache_reshape_t = Buffer2torchTensor(ckv_cache_reshape, false);
        printBufferData(ckv_cache_reshape, "ckv_cache_reshape");

        FT_LOG_TRACE("kv_lora_rank = %zu", params.configs.kv_lora_rank);

        printBufferData(*torchTensor2Buffer(flashinfer.kvlen_t), "kvlen_t");
        
        const auto decode_kv_cache_block_id_t = flashinfer.kv_cache_block_id_t.slice(0, c10::nullopt, c10::make_optional((int64_t)generate_batch_size));
        printBufferData(*torchTensor2Buffer(decode_kv_cache_block_id_t), "decode_kv_cache_block_id_t");

        const float softmax_scale = params.configs.softmax_extra_scale / sqrtf(params.configs.size_per_head * 1.0f);
        FT_LOG_TRACE("softmax_scale = %f", softmax_scale);

        const auto &tile_scheduler_metadata = flashinfer.flash_mla_plan[0];
        printBufferData(*torchTensor2Buffer(tile_scheduler_metadata), "tile_scheduler_metadata");
        
        const auto &num_splits = flashinfer.flash_mla_plan[1];
        printBufferData(*torchTensor2Buffer(num_splits), "num_splits");

        attn_out_t = mha_fwd_unified_kvcache_mla(
            q_with_rope,
            ckv_cache_reshape_t,
            params.configs.kv_lora_rank,
            flashinfer.kvlen_t,
            decode_kv_cache_block_id_t,
            softmax_scale,
            /* is_causal = */true,
            tile_scheduler_metadata,
            num_splits
        )[0].reshape({(int64_t)params.q.shape()[0], (int64_t)params.configs.head_num, (int64_t)params.configs.kv_lora_rank});
    } else {
        auto attn_out = allocateBuffer({datatype, {params.q.shape()[0], params.configs.head_num, params.configs.kv_lora_rank}, AllocationType::DEVICE});
        attn_out_t = Buffer2torchTensor(attn_out, false);

        auto ckv_nope_cache = Buffer2torchTensorWithStride(
            *params.common.kv_cache->k_cache_buffer, 
            {
                (int64_t)ckv_cache_shape[0],
                (int64_t)ckv_cache_shape[1],
                (int64_t)params.configs.kv_lora_rank
            },
            0);
        auto k_rope_cache = Buffer2torchTensorWithStride(
            *params.common.kv_cache->k_cache_buffer, 
            {
                (int64_t)ckv_cache_shape[0],
                (int64_t)ckv_cache_shape[1],
                (int64_t)params.configs.rope_head_dim
            },
            params.configs.kv_lora_rank);

        BatchMLAPagedAttentionRun(
            flashinfer.float_workspace_t,
            flashinfer.int_workspace_t,
            flashinfer.plan,
            absorb_q_input.contiguous(),
            q_rope.contiguous(),
            ckv_nope_cache,
            k_rope_cache,
            flashinfer.page_indice_t,
            attn_out_t,
            std::nullopt,
            1,
            params.configs.head_num,
            params.configs.tokens_per_block,
            (1.0f / sqrtf(params.configs.size_per_head * 1.0f)) * params.configs.softmax_extra_scale,
            (int64_t)stream_
        );
    }

    printBufferData(*torchTensor2Buffer(attn_out_t), "mla_attn_out_t");
    auto qkv_output_t = DecoderOutputGemmWrapper(attn_out_t, params);
    printBufferData(*torchTensor2Buffer(qkv_output_t), "mla_qkv_output_t");

    auto padding_tensor = torch::zeros({qkv_output_t.size(0), qkv_output_t.size(1), (int64_t)(params.configs.size_per_head - params.configs.v_head_dim)},
                                       qkv_output_t.options());

    auto padded_output = torch::cat({qkv_output_t, padding_tensor}, -1);

    auto qkv_tmp_buf = torchTensor2Buffer(padded_output);
    copy(CopyParams({*params.qkv_output, *qkv_tmp_buf}));
}

AttentionModuleOutput CudaDevice::mlaContextAttention(const MlaAttentionModuleParams& params) {
    DevicePerfWrapper wrapper(this, "mlaContext_layer_%d", params.layer_id);
    auto& q = params.q;
    auto& k_rope = params.k_rope;
    auto& kv_a   = params.kv_a;

    auto const token_num     = kv_a.shape()[0];
    auto const head_num      = params.configs.head_num;
    auto const nope_head_dim = params.configs.nope_head_dim;
    auto const rope_head_dim = params.configs.rope_head_dim;
    auto const v_head_dim    = params.configs.v_head_dim;
    auto const nope_rope_dim = nope_head_dim + rope_head_dim;

    writeCacheStore(params);

    auto datatype = kv_a.type();
    auto qkv =
        allocateBuffer({datatype, {token_num, head_num * nope_rope_dim * 3}, AllocationType::DEVICE}, {"mla_qkv"});

    auto k_nope = gemm(GemmParams(kv_a, *(params.weights.k_nope_weight->kernel)));
    auto v      = gemm(GemmParams(kv_a, *(params.weights.v_weight->kernel)));

    printBufferData(*k_nope, "k_nope");
    printBufferData(*v, "v");

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                     invokeMlaQKVMerge,
                                     q.data(),
                                     k_nope->data(),
                                     k_rope.data(),
                                     v->data(),
                                     qkv->data(),
                                     token_num,
                                     head_num,
                                     nope_head_dim,
                                     rope_head_dim,
                                     v_head_dim,
                                     stream_);

    printBufferData(*qkv, "mla_qkv");

    AttentionModuleParams attn_params = AttentionModuleParams(
        {params.layer_id, *qkv, *params.qkv_output, params.common, params.weights, params.configs, params.qscheme});
    // only paged fmha use kv_block_array, mld not use paged fmha
    prefillAttention(
        attn_params,
        KVBlockArray(),
        nullptr,
        nullptr,
        nullptr,
        nullptr);
}

void transpose_qk_inplace(torch::Tensor& q, torch::Tensor& k) {
    auto token_num = q.size(0);
    auto head_num_q = q.size(1);
    auto rope_size = q.size(2);
    auto q_transpose = q.reshape({token_num, head_num_q, rope_size / 2, 2}).transpose(2,3).reshape({token_num, head_num_q, rope_size}).contiguous();
    auto k_transpose = k.reshape({token_num, rope_size / 2, 2}).transpose(1,2).reshape({token_num, rope_size}).contiguous();
    q.copy_(q_transpose, true);
    k.copy_(k_transpose, true);
}

void CudaDevice::mlaRotaryWriteKVCache(const MlaRotaryWriteKVCacheParams& params) {    
    DevicePerfWrapper wrapper(this, "mlaRotaryWriteKVCache");
    auto flash_infer_attn_params = (FlashInferAttnParams*)params.common.flash_infer_attn_params.get();
    if (!flash_infer_attn_params) {
        throw std::runtime_error("flash_infer_attn_params must be setting when using mla");
    }
    const auto &flashinfer = *flash_infer_attn_params;
    // apply rotary embedding to qk
    auto& q = params.q;

    auto q_reshaped = q.reshape({q.shape()[0], params.configs.head_num, params.configs.nope_head_dim + params.configs.rope_head_dim});
    auto q_rope_t = Buffer2torchTensorWithStride(
        q_reshaped,
        {(int64_t)q_reshaped.shape()[0], (int64_t)params.configs.head_num, (int64_t)params.configs.rope_head_dim}, params.configs.nope_head_dim);

    auto k_rope_t = Buffer2torchTensor(params.k_rope, false);
    transpose_qk_inplace(q_rope_t, k_rope_t);
    auto cos_sin_cache_t = Buffer2torchTensor(params.weights.rope_cos_sin_cache, false);
    apply_rope_pos_ids_cos_sin_cache(q_rope_t, k_rope_t.unsqueeze(1), q_rope_t, k_rope_t.unsqueeze(1), cos_sin_cache_t, flashinfer.total_positions_t, false, (int64_t)stream_);
    auto append_ckv_t = Buffer2torchTensor(params.ckv, false);

    if (params.common.kv_cache.has_value()) {
        const auto &k_cache_shape = params.common.kv_cache->k_cache_buffer->shape();
        auto k_cache = Buffer2torchTensorWithStride(
            *params.common.kv_cache->k_cache_buffer, 
            {
                (int64_t)k_cache_shape[0],
                (int64_t)k_cache_shape[1],
                (int64_t)params.configs.kv_lora_rank
            },
            0);
        auto v_cache = Buffer2torchTensorWithStride(
            *params.common.kv_cache->k_cache_buffer, 
            {
                (int64_t)k_cache_shape[0],
                (int64_t)k_cache_shape[1],
                (int64_t)params.configs.rope_head_dim
            },
            params.configs.kv_lora_rank);
        append_paged_mla_kv_cache(append_ckv_t,
                                  k_rope_t,
                                  flashinfer.total_batch_indices_t,
                                  flashinfer.total_positions_t,
                                  k_cache,
                                  v_cache,
                                  flashinfer.total_page_indices_t,
                                  flashinfer.total_page_indptr_t,
                                  flashinfer.total_kv_last_page_len_1_t,
                                  (int64_t)stream_);
    }
}

}  // namespace fastertransformer
