#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/kernels/mla_kernels/mla_merge_transpose_kernel.h"
#include "3rdparty/flashinfer/flashinfer.h"

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
    auto absorb_q_input = QInputBatchMatmulWrapper(params);
    printBufferData(*torchTensor2Buffer(absorb_q_input), "mla_absorb_q_input");
    auto q_reshape = params.q.reshape({params.q.shape()[0], params.configs.head_num, params.configs.nope_head_dim + params.configs.rope_head_dim});
    auto q_rope = Buffer2torchTensorWithStride(q_reshape, {(int64_t)params.q.shape()[0], (int64_t)params.configs.head_num, (int64_t)params.configs.rope_head_dim}, params.configs.nope_head_dim);
    auto ckv = Buffer2torchTensor(params.common.kv_cache->k_cache_buffer, false);
    auto flash_infer_attn_params = (FlashInferAttnParams*)params.common.flash_infer_attn_params.get();
    if (!flash_infer_attn_params) {
        throw std::runtime_error("flash_infer_attn_params must be setting when using mla");
    }
    auto ckv_cache = Buffer2torchTensor(params.common.kv_cache->k_cache_buffer, false);
    auto datatype = params.q.type();
    auto rope_k_cache = Buffer2torchTensor(params.common.kv_cache->v_cache_buffer, false);
    auto attn_out = allocateBuffer({datatype, {params.q.shape()[0], params.configs.head_num, params.configs.kv_lora_rank}, AllocationType::DEVICE});
    auto attn_out_t = Buffer2torchTensor(attn_out, false);
    const auto &flashinfer = *flash_infer_attn_params;
    // maybe some shape check ?
    BatchMLAPagedAttentionRun(
        flashinfer.float_workspace_t,
        flashinfer.int_workspace_t,
        flashinfer.plan,
        absorb_q_input.contiguous(),
        q_rope.contiguous(),
        ckv_cache,
        rope_k_cache,
        flashinfer.page_indice_t,
        attn_out_t,
        std::nullopt,
        1,
        params.configs.head_num,
        params.configs.tokens_per_block,
        (1.0f / sqrtf(params.configs.size_per_head * 1.0f)) * params.configs.softmax_extra_scale,
        (int64_t)stream_
    );
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
    auto& q = params.q;
    auto& k_rope = params.k_rope;
    auto& kv_a   = params.kv_a;

    auto const token_num     = kv_a.shape()[0];
    auto const head_num      = params.configs.head_num;
    auto const nope_head_dim = params.configs.nope_head_dim;
    auto const rope_head_dim = params.configs.rope_head_dim;
    auto const v_head_dim    = params.configs.v_head_dim;
    auto const nope_rope_dim = nope_head_dim + rope_head_dim;
    auto const batch_size    = params.common.context_batch_size;
    auto const seq_len       = params.common.context_max_seq_len;

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

    switch (fmha_type_) {
        case FMHAType::OPEN_SOURCE: {
            const auto ws_size = cufmha_runner_->getOpenSourceWorkSpaceSize(batch_size, seq_len);
            auto ws = allocateBuffer({DataType::TYPE_INT8, {ws_size}, AllocationType::DEVICE}, {"open_source_fmha_ws"});
            const size_t hidden_units = head_num * nope_rope_dim;

            cufmha_runner_->runOpenSourceFmha(
                qkv->data(),
                qkv->dataWithOffset(hidden_units),
                qkv->dataWithOffset(hidden_units * 2),
                params.qkv_output->data(),
                params.common.cu_seqlens->data<int>(),
                batch_size,
                seq_len,
                ws->data(),
                params.common.linear_bias_slopes ? params.common.linear_bias_slopes->data<float>() : nullptr,
                params.configs.softmax_extra_scale);
            break;
        }
        default: {
            throw std::runtime_error("Unsupported FMHA type");
            break;
        }
    }
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
        auto k_cache = Buffer2torchTensor(params.common.kv_cache->k_cache_buffer, false);
        auto v_cache = Buffer2torchTensor(params.common.kv_cache->v_cache_buffer, false);
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
