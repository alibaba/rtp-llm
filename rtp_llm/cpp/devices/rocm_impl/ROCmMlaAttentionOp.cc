#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/kernels/mla_kernels/mla_merge_transpose_kernel.h"
#include <iostream>
// #include "mla/asm_mla_decode_fwd_torch.h"
#include <vector>
#include <fstream>
#include <hip/hip_runtime.h>
#include <cstring>
#include <cmath>
#include "rtp_llm/cpp/devices/utils/DevicePerfWrapper.h"
#include "rtp_llm/cpp/kernels/mla_kernels_rocm/page_hip.h"
#include "rtp_llm/cpp/kernels/mla_kernels_rocm/rope_hip.h"

using namespace std;
using namespace rtp_llm;

namespace rtp_llm {

void ROCmDevice::mlaRotaryWriteKVCache(const MlaRotaryWriteKVCacheParams& params) {
    DevicePerfWrapper wrapper(this, "mlaRotaryWriteKVCache");
    auto              flash_infer_attn = (FlashInferAttnParams*)params.flash_infer.get();
    if (!flash_infer_attn) {
        throw std::runtime_error("flash_infer_attn must be setting when using mlaRotaryWriteKVCachela");
    }
    const auto& f = *flash_infer_attn;
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
                                     f.positions_t,
                                     false,
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
        if (params.is_decode) {
            k_cache = k_cache.view({-1, 1, (int64_t)params.configs.kv_lora_rank});
            v_cache = v_cache.view({-1, 1, (int64_t)params.configs.rope_head_dim});
        }
        append_paged_mla_kv_cache(append_ckv_t,
                                  k_rope_t,
                                  f.batch_indice_t,
                                  f.positions_t,
                                  k_cache,
                                  v_cache,
                                  f.page_indice_t,
                                  f.page_indptr_t,
                                  f.paged_kv_last_page_len_1_t,
                                  (int64_t)stream_);
    }
}

void ROCmDevice::QInputBatchMatmulWrapper(torch::Tensor& fused_q_input_t, const MlaAttentionModuleParams& params) {
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

void ROCmDevice::DecoderOutputGemmWrapper(torch::Tensor&                  qkv_output_t,
                                          const torch::Tensor&            mla_out_t,
                                          const MlaAttentionModuleParams& params) {
    DevicePerfWrapper wrapper(this, "DecoderOutputGemmWrapper");
    auto              w_vc_t = Buffer2torchTensor(params.weights.vc_weight->kernel, false);
    torch::bmm_out(qkv_output_t.transpose_(0, 1), mla_out_t.transpose(0, 1), w_vc_t);
}

void ROCmDevice::mlaAbsorbAttention(const MlaAttentionModuleParams& params) {
    auto fused_q_input = allocateBuffer(
        {params.q.type(),
         {params.q.shape()[0], params.configs.head_num, params.configs.kv_lora_rank + params.configs.rope_head_dim},
         AllocationType::DEVICE});
    auto fused_q_input_t = Buffer2torchTensor(fused_q_input, false);
    mlaRotaryWriteKVCache({params.q,
                           fused_q_input,
                           params.fused_qkv,
                           params.kv_offset,
                           params.common.decode_flash_infer_attn,
                           params.common,
                           params.weights,
                           params.configs,
                           params.qscheme,
                           true});

    QInputBatchMatmulWrapper(fused_q_input_t, params);
    printBufferData(*fused_q_input, "fused_q_input");

    auto flash_infer_attn = (FlashInferAttnParams*)params.common.decode_flash_infer_attn.get();
    if (!flash_infer_attn) {
        throw std::runtime_error("flash_infer_attn must be setting when using mla");
    }
    const auto& ckv_cache_shape = params.common.kv_cache->kv_cache_buffer->shape();
    auto        datatype        = params.q.type();
    at::Tensor  attn_out_t;
    BufferPtr   attn_out;
    const auto& flashinfer = *flash_infer_attn;
    // maybe some shape check ?

    auto q_reshape = params.q.reshape(
        {params.q.shape()[0], params.configs.head_num, params.configs.nope_head_dim + params.configs.rope_head_dim});
    auto q_rope = fused_q_input_t.slice(
        -1, params.configs.kv_lora_rank, params.configs.kv_lora_rank + params.configs.rope_head_dim);

    const int64_t batch_size        = fused_q_input_t.size(0);
    const int64_t num_heads         = fused_q_input_t.size(1);
    const float   softmax_scale     = params.configs.softmax_extra_scale / sqrtf(params.configs.size_per_head * 1.0f);
    const int64_t kv_lora_rank      = params.configs.kv_lora_rank;
    auto          ckv_cache_reshape = params.common.kv_cache->kv_cache_buffer->reshape({
        ckv_cache_shape[0],
        ckv_cache_shape[1],
        1,
        ckv_cache_shape[2],
    });
    auto ckv_cache_reshape_t = Buffer2torchTensor(ckv_cache_reshape, false).view({-1, 1, 1, fused_q_input_t.size(-1)});

    // #ifdef TORCH_MLA_DECODE
    auto kv_indptr  = Buffer2torchTensor(flashinfer.page_indptr_host, false);
    auto kv_indices = Buffer2torchTensor(flashinfer.page_indice_host, false);
    int  max_seq_len =
        *((kv_indptr.slice(0, 1, batch_size + 1, 1) - kv_indptr.slice(0, 0, batch_size, 1)).max().data_ptr<int>());
    std::vector<torch::Tensor> k_vector;
    std::vector<torch::Tensor> mask_vector;
    for (int i = 0; i < batch_size; i++) {
        std::vector<torch::Tensor> tmp;
        for (int j = *(kv_indptr[i].data_ptr<int>()); j < *(kv_indptr[i + 1].data_ptr<int>()); j++) {
            tmp.push_back(ckv_cache_reshape_t[*kv_indices[j].data_ptr<int>()]);
        }
        auto tmp_t  = torch::cat(tmp, 1);
        namespace F = torch::nn::functional;
        k_vector.push_back(
            F::pad(tmp_t, F::PadFuncOptions({0, 0, 0, max_seq_len - tmp_t.size(1)}).mode(torch::kConstant)));
        auto mask_item = torch::ones({1, 1, max_seq_len}, torch::dtype(torch::kBool));
        std::memset(mask_item.data_ptr(), 0, tmp_t.size(1));
        mask_vector.push_back(mask_item);
    }

    // [batch_size, 1, max_seq_len]
    auto mask = torch::cat(mask_vector, 0).repeat_interleave(num_heads, 1).cuda();
    auto k    = torch::cat(k_vector, 0);
    auto v    = k.slice(-1, 0, params.configs.kv_lora_rank, 1);

    // [batch_size, head_num, max_seq_len]
    auto before_softmax =
        torch::bmm(fused_q_input_t.to(torch::kFloat), k.transpose(-1, -2).to(torch::kFloat)) * softmax_scale;
    before_softmax.masked_fill_(mask, -INFINITY);
    auto attn_weights = torch::softmax(before_softmax, -1);
    attn_out_t        = torch::bmm(attn_weights, v.to(torch::kFloat)).to(fused_q_input_t.dtype());
    // #else
    //     attn_out_t = torch::empty({batch_size, num_heads, kv_lora_rank},
    //         torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    //
    //     auto kv_indptr = Buffer2torchTensor(flashinfer.page_indptr_host, false).cuda();
    //     auto kv_indices = Buffer2torchTensor(flashinfer.page_indice_host, false).cuda();
    //     auto kv_last_page_lens = torch::ones({batch_size},
    //         torch::dtype(torch::kInt32).device(torch::kCUDA));
    //
    //     const int num_kv_splits = 16;
    //     auto logits = torch::zeros({batch_size, num_kv_splits, num_heads, kv_lora_rank},
    //         torch::dtype(torch::kFloat32).device(torch::kCUDA));
    //
    //     auto attn_lse = torch::zeros({batch_size, num_kv_splits, num_heads, 1},
    //         torch::dtype(torch::kFloat32).device(torch::kCUDA));
    //
    //     aiter::mla_decode_fwd(
    //         fused_q_input_t, ckv_cache_reshape_t, attn_out_t, kv_indptr, kv_indices,
    //         kv_last_page_lens, softmax_scale, 0.0, num_kv_splits, logits, attn_lse
    //     );
    // #endif
    auto qkv_output_t = Buffer2torchTensor(
        params.qkv_output->reshape({params.qkv_output->shape()[0], params.configs.head_num, params.configs.v_head_dim}),
        false);
    DecoderOutputGemmWrapper(qkv_output_t, attn_out_t, params);
}

AttentionModuleOutput ROCmDevice::mlaContextAttention(const MlaAttentionModuleParams& params) {
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

    auto const batch_size = params.common.context_batch_size;
    auto const seq_len    = params.common.context_max_seq_len;

    auto softmax_extra_scale = params.configs.softmax_extra_scale;

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

    printBufferData(*qkv, "mla_qkv");
    const size_t hidden_units = head_num * nope_rope_dim;

    fmha_runner_->setup(
        datatype, params.configs.is_causal, head_num, head_num, nope_rope_dim, params.configs.q_scaling);

    auto lse_acc_buf = allocateBuffer({DataType::TYPE_FP32, {1, 1, 1, 1}, AllocationType::DEVICE}, {"lse_acc_buf"});

    auto padded_qkv_output_t = allocateBuffer({datatype, {token_num, head_num * size_per_head}, AllocationType::DEVICE},
                                              {"padded_qkv_output"});

    fmha_runner_->runCKFmhaMLA(qkv->data(),
                               qkv->dataWithOffset(hidden_units),
                               qkv->dataWithOffset(hidden_units * 2),
                               padded_qkv_output_t->data(),
                               nullptr,  // buffer for store out softmax_lse, looks like not used by RTP
                               batch_size,
                               seq_len,
                               softmax_extra_scale,
                               // context_token_num,
                               params.common.cu_seqlens->data(),
                               params.common.cu_seqlens->data(),
                               lse_acc_buf->data(),
                               params.common.linear_bias_slopes ? params.common.linear_bias_slopes->data() : nullptr,
                               nullptr);
    auto qkv_output_reshaped = padded_qkv_output_t->reshape({token_num, params.configs.head_num, size_per_head});
    auto sliced_buffer       = slice({qkv_output_reshaped, -1, 0, (int64_t)v_head_dim});
    copy({*params.qkv_output, *sliced_buffer});
}

}  // namespace rtp_llm
