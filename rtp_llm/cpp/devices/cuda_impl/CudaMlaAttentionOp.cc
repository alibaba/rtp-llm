#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/kernels/mla_kernels/mla_merge_transpose_kernel.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include "rtp_llm/cpp/devices/utils/DevicePerfWrapper.h"
#include "flashmla/flashmla.h"
#include <cstdint>

using namespace std;
using namespace rtp_llm;

namespace rtp_llm {

// q_input_shape [batch_size, head_num, nope_head_dim + rope_head_dim]

void CudaDevice::QInputBatchMatmulWrapper(torch::Tensor& fused_q_input_t, const MlaAttentionModuleParams& params) {
    auto              absorb_q = fused_q_input_t.slice(-1, 0, params.configs.kv_lora_rank);
    DevicePerfWrapper wrapper(this, "QInputBatchMatmulWrapper");
    // shape: [head_num, nope_head_dim, kv_lora_rank]
    auto q_input_reshape = params.q.reshape(
        {params.q.shape()[0], params.configs.head_num, params.configs.nope_head_dim + params.configs.rope_head_dim});
    auto q_nope_t = Buffer2torchTensorWithStride(
        q_input_reshape,
        {(int64_t)params.q.shape()[0], (int64_t)params.configs.head_num, (int64_t)params.configs.nope_head_dim});
    auto w_kc_t = Buffer2torchTensor(params.weights.kc_weight->kernel, false);
    torch::bmm_out(absorb_q.transpose_(0, 1), q_nope_t.transpose(0, 1), w_kc_t);
}

void CudaDevice::DecoderOutputGemmWrapper(torch::Tensor&                  qkv_output_t,
                                          const torch::Tensor&            mla_out_t,
                                          const MlaAttentionModuleParams& params) {
    DevicePerfWrapper wrapper(this, "DecoderOutputGemmWrapper");
    auto              w_vc_t = Buffer2torchTensor(params.weights.vc_weight->kernel, false);
    torch::bmm_out(qkv_output_t.transpose_(0, 1), mla_out_t.transpose(0, 1), w_vc_t);
}

void CudaDevice::mlaAbsorbAttention(const MlaAttentionModuleParams& params) {
    DevicePerfWrapper wrapper(this, "mlaDecoder_layer_%d", params.layer_id);

    auto fused_q_input = allocateBuffer(
        {params.q.type(),
         {params.q.shape()[0], params.configs.head_num, params.configs.kv_lora_rank + params.configs.rope_head_dim},
         AllocationType::DEVICE});

    mlaRotaryWriteKVCache(
        {params.q,
         fused_q_input,
         params.fused_qkv,
         params.kv_offset,
         params.is_prefill ? params.common.prefill_flash_infer_attn : params.common.decode_flash_infer_attn,
         params.common,
         params.weights,
         params.configs,
         params.qscheme});

    if (params.is_prefill) {
        writeCacheStore(params);
    }

    computeInsertedMoE();

    auto fused_q_input_t = Buffer2torchTensor(fused_q_input, false);
    QInputBatchMatmulWrapper(fused_q_input_t, params);
    printBufferData(*fused_q_input, "fused_q_input");

    auto ckv              = Buffer2torchTensor(params.common.kv_cache->kv_cache_buffer, false);
    auto flash_infer_attn = params.is_prefill ? (FlashInferAttnParams*)params.common.prefill_flash_infer_attn.get() :
                                                (FlashInferAttnParams*)params.common.decode_flash_infer_attn.get();
    if (!flash_infer_attn) {
        throw std::runtime_error("flash_infer_attn must be setting when using mla");
    }
    const auto& ckv_cache_shape = params.common.kv_cache->kv_cache_buffer->shape();
    auto        datatype        = params.q.type();
    at::Tensor  attn_out_t;
    BufferPtr   attn_out;
    const auto& flashinfer = *flash_infer_attn;
    // maybe some shape check ?

    RTP_LLM_LOG_DEBUG("mla_ops_type %d", mla_ops_type);
    if (flashinfer.mla_ops_type == MlaOpsType::FLASH_MLA) {
        if (params.is_prefill) {
            fused_q_input_t =
                fused_q_input_t.reshape({(int64_t)params.common.context_batch_size,
                                         (int64_t)(params.q.shape()[0] / params.common.context_batch_size),
                                         (int64_t)params.configs.head_num,
                                         (int64_t)(params.configs.kv_lora_rank + params.configs.rope_head_dim)});
        } else {
            fused_q_input_t =
                fused_q_input_t.reshape({(int64_t)params.q.shape()[0],
                                         1,
                                         (int64_t)params.configs.head_num,
                                         (int64_t)(params.configs.kv_lora_rank + params.configs.rope_head_dim)});
        }
        // [batch_size, 1, num_heads, kv_lora_rank + rope_head_dim]
        printBufferData(*torchTensor2Buffer(fused_q_input_t), "fused_q_input_t");

        auto ckv_cache_reshape   = params.common.kv_cache->kv_cache_buffer->reshape({
            ckv_cache_shape[0],
            ckv_cache_shape[1],
            1,
            ckv_cache_shape[2],
        });
        auto ckv_cache_reshape_t = Buffer2torchTensor(ckv_cache_reshape, false);
        printBufferData(ckv_cache_reshape, "ckv_cache_reshape");

        RTP_LLM_LOG_TRACE("kv_lora_rank = %zu", params.configs.kv_lora_rank);

        printBufferData(*torchTensor2Buffer(flashinfer.kvlen_d), "kvlen");
        printBufferData(*torchTensor2Buffer(flashinfer.kv_cache_block_id_d), "kv_cache_block_id");

        const float softmax_scale = params.configs.softmax_extra_scale / sqrtf(params.configs.size_per_head * 1.0f);
        RTP_LLM_LOG_TRACE("softmax_scale = %f", softmax_scale);

        const auto& tile_scheduler_metadata = flashinfer.flash_mla_plan[0];
        printBufferData(*torchTensor2Buffer(tile_scheduler_metadata), "tile_scheduler_metadata");

        const auto& num_splits = flashinfer.flash_mla_plan[1];
        printBufferData(*torchTensor2Buffer(num_splits), "num_splits");
        attn_out_t = mha_fwd_unified_kvcache_mla(fused_q_input_t,
                                                 ckv_cache_reshape_t,
                                                 params.configs.kv_lora_rank,
                                                 flashinfer.kvlen_d,
                                                 flashinfer.kv_cache_block_id_d,
                                                 softmax_scale,
                                                 /* is_causal = */ true,
                                                 tile_scheduler_metadata,
                                                 num_splits)[0]
                         .reshape({(int64_t)params.q.shape()[0],
                                   (int64_t)params.configs.head_num,
                                   (int64_t)params.configs.kv_lora_rank});
    } else {
        auto q_rope = fused_q_input_t.slice(
            -1, params.configs.kv_lora_rank, params.configs.kv_lora_rank + params.configs.rope_head_dim);

        attn_out   = allocateBuffer({datatype,
                                     {params.q.shape()[0], params.configs.head_num, params.configs.kv_lora_rank},
                                     AllocationType::DEVICE});
        attn_out_t = Buffer2torchTensor(attn_out, false);

        auto ckv_nope_cache = Buffer2torchTensorWithStride(
            *params.common.kv_cache->kv_cache_buffer,
            {(int64_t)ckv_cache_shape[0], (int64_t)ckv_cache_shape[1], (int64_t)params.configs.kv_lora_rank},
            0);
        auto k_rope_cache = Buffer2torchTensorWithStride(
            *params.common.kv_cache->kv_cache_buffer,
            {(int64_t)ckv_cache_shape[0], (int64_t)ckv_cache_shape[1], (int64_t)params.configs.rope_head_dim},
            params.configs.kv_lora_rank);

        BatchMLAPagedAttentionRun(flashinfer.float_workspace_d,
                                  flashinfer.int_workspace_d,
                                  flashinfer.plan,
                                  fused_q_input_t.slice(-1, 0, params.configs.kv_lora_rank),
                                  q_rope,
                                  ckv_nope_cache,
                                  k_rope_cache,
                                  flashinfer.page_indice_d,
                                  attn_out_t,
                                  std::nullopt,
                                  1,
                                  params.configs.head_num,
                                  params.configs.tokens_per_block,
                                  (1.0f / sqrtf(params.configs.size_per_head * 1.0f))
                                      * params.configs.softmax_extra_scale,
                                  (int64_t)stream_);
    }
    printBufferData(*torchTensor2Buffer(attn_out_t), "mla_attn_out_t");
    auto qkv_output_t = Buffer2torchTensor(
        params.qkv_output->reshape({params.qkv_output->shape()[0], params.configs.head_num, params.configs.v_head_dim}),
        false);
    DecoderOutputGemmWrapper(qkv_output_t, attn_out_t, params);
}

AttentionModuleOutput CudaDevice::mlaContextAttention(const MlaAttentionModuleParams& params) {
    DevicePerfWrapper wrapper(this, "mlaContext_layer_%d", params.layer_id);
    auto&             q         = params.q;
    auto&             fused_qkv = params.fused_qkv;

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
                           params.qscheme});
    writeCacheStore(params);

    computeInsertedMoE();

    auto split_result =
        split({fused_qkv,
               {(size_t)params.kv_offset, (size_t)params.configs.kv_lora_rank, (size_t)params.configs.rope_head_dim},
               1});
    auto kv_a   = split_result.outputs[1];
    auto k_rope = split_result.outputs[2];
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

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
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
                                     v_head_dim,
                                     stream_);
    k_nope.reset();
    v.reset();

    printBufferData(*qkv, "mla_qkv");
    auto padded_qkv_output_t = allocateBuffer({datatype, {token_num, head_num * size_per_head}, AllocationType::DEVICE},
                                              {"padded_qkv_output"});
    AttentionModuleParams attn_params = AttentionModuleParams(
        {params.layer_id, *qkv, *padded_qkv_output_t, params.common, params.weights, params.configs, params.qscheme});
    // only paged fmha use kv_block_array, mld not use paged fmha
    prefillAttention(attn_params, KVBlockArray(), nullptr, nullptr, nullptr, nullptr, nullptr);
    auto qkv_output_reshaped = padded_qkv_output_t->reshape({token_num, params.configs.head_num, size_per_head});
    auto sliced_buffer       = slice({qkv_output_reshaped, -1, 0, (int64_t)v_head_dim});
    copy({*params.qkv_output, *sliced_buffer});
}

void CudaDevice::mlaRotaryWriteKVCache(const MlaRotaryWriteKVCacheParams& params) {
    DevicePerfWrapper wrapper(this, "mlaRotaryWriteKVCache");
    auto              flash_infer_attn = (FlashInferAttnParams*)params.flash_infer.get();
    if (!flash_infer_attn) {
        throw std::runtime_error("flash_infer_attn must be setting when using mlaRotaryWriteKVCache");
    }
    const auto& flashinfer = *flash_infer_attn;
    // apply rotary embedding to qk
    auto& q = params.q;

    auto q_reshaped =
        q.reshape({q.shape()[0], params.configs.head_num, params.configs.nope_head_dim + params.configs.rope_head_dim});
    auto q_rope_t = Buffer2torchTensorWithStride(
        q_reshaped,
        {(int64_t)q_reshaped.shape()[0], (int64_t)params.configs.head_num, (int64_t)params.configs.rope_head_dim},
        params.configs.nope_head_dim);
    torch::Tensor dest_q;
    if (params.fused_dest_q) {
        dest_q = Buffer2torchTensorWithStride(*params.fused_dest_q,
                                              {(int64_t)params.fused_dest_q->shape()[0],
                                               (int64_t)params.configs.head_num,
                                               (int64_t)params.configs.rope_head_dim},
                                              params.configs.kv_lora_rank);
    } else {
        dest_q = q_rope_t;
    }
    auto k_rope_t =
        Buffer2torchTensorWithStride(params.fused_qkv,
                                     {(int64_t)params.fused_qkv.shape()[0], (int64_t)params.configs.rope_head_dim},
                                     params.kv_offset + params.configs.kv_lora_rank);
    auto cos_sin_cache_t = Buffer2torchTensor(params.weights.rope_cos_sin_cache, false);
    apply_rope_pos_ids_cos_sin_cache(q_rope_t,
                                     k_rope_t.unsqueeze(1),
                                     dest_q,
                                     k_rope_t.unsqueeze(1),
                                     cos_sin_cache_t,
                                     flashinfer.positions_d,
                                     true,
                                     (int64_t)stream_);
    auto append_ckv_t =
        Buffer2torchTensorWithStride(params.fused_qkv,
                                     {(int64_t)params.fused_qkv.shape()[0], (int64_t)params.configs.kv_lora_rank},
                                     params.kv_offset);

    if (params.common.kv_cache.has_value()) {
        const auto& k_cache_shape = params.common.kv_cache->kv_cache_buffer->shape();
        auto        k_cache       = Buffer2torchTensorWithStride(
            *params.common.kv_cache->kv_cache_buffer,
            {(int64_t)k_cache_shape[0], (int64_t)k_cache_shape[1], (int64_t)params.configs.kv_lora_rank},
            0);
        auto v_cache = Buffer2torchTensorWithStride(
            *params.common.kv_cache->kv_cache_buffer,
            {(int64_t)k_cache_shape[0], (int64_t)k_cache_shape[1], (int64_t)params.configs.rope_head_dim},
            params.configs.kv_lora_rank);
        append_paged_mla_kv_cache(append_ckv_t,
                                  k_rope_t,
                                  flashinfer.batch_indice_d,
                                  flashinfer.positions_d,
                                  k_cache,
                                  v_cache,
                                  flashinfer.page_indice_d,
                                  flashinfer.page_indptr_d,
                                  flashinfer.paged_kv_last_page_len_d,
                                  (int64_t)stream_);
    }
}

}  // namespace rtp_llm
