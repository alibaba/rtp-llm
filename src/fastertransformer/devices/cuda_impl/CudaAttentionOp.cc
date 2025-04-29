
#include <iostream>
#include <numeric>
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/cuda_impl/CudaFlashInfer.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/utils/compiler_config.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/kernels/kv_cache/kv_cache_utils.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "maga_transformer/cpp/utils/RpcErrorCode.h"

using namespace std;
using namespace rtp_llm;

namespace fastertransformer {

AttentionModuleOutput CudaDevice::contextAttention(const AttentionModuleParams& params) {
    auto datatype       = params.input.type();
    auto token_num      = params.input.shape()[0];
    auto batch_size     = params.common.context_batch_size;
    auto decoder_batch_size = params.common.decoder_batch_size;
    auto seq_len        = params.common.context_max_seq_len;
    auto seq_len_with_prefix = seq_len + params.common.max_prefix_length;
    auto head_num       = params.configs.head_num;
    auto kv_head_num    = params.configs.kv_head_num;
    auto size_per_head  = params.configs.size_per_head;

    auto q_output = allocateBuffer({params.input.type(),
                                    {batch_size, head_num, seq_len, size_per_head},
                                    AllocationType::DEVICE},
                                    {"q_output"});

    auto k_output = allocateBuffer({params.input.type(),
                                    {batch_size, kv_head_num, seq_len_with_prefix, size_per_head},
                                    AllocationType::DEVICE},
        {"k_output"});

    auto v_output = allocateBuffer({params.input.type(),
                                    {batch_size, kv_head_num, seq_len_with_prefix, size_per_head},
                                    AllocationType::DEVICE},
                                    {"v_output"});

    BufferPtr qkv_buf_fp8;
    if (use_fp8_fmha_) {
        qkv_buf_fp8 = allocateBuffer({DataType::TYPE_FP8_E4M3,
                                      {batch_size, (head_num + kv_head_num * 2), seq_len_with_prefix, size_per_head},
                                      AllocationType::DEVICE},
                                     {"qkv_fp8_output"});
        cudaMemsetAsync(qkv_buf_fp8->data(), 0, qkv_buf_fp8->sizeBytes(), stream_);
    }

    if (fmha_type_ == FMHAType::NONE) {
        cudaMemsetAsync(q_output->data(), 0, q_output->sizeBytes(), stream_);
        cudaMemsetAsync(k_output->data(), 0, k_output->sizeBytes(), stream_);
        cudaMemsetAsync(v_output->data(), 0, v_output->sizeBytes(), stream_);
    }

    BufferPtr kv_cache_block_id = nullptr;
    BufferPtr kv_cache_offset_host = nullptr;

    KVBlockArray                  kv_block_array;
    PrefixPromptBatchWeightsParam prefix_prompt_param;

    if (params.common.kv_cache) {
        const auto max_blocks_per_batch = params.common.kv_cache->kv_cache_block_id->shape()[1];
        kv_cache_block_id = allocateBuffer({DataType::TYPE_INT32, {batch_size, 1, 2, max_blocks_per_batch}, AllocationType::DEVICE},
                                         {"kv_cache_block_id"});

        kv_block_array = getKVBlockArray(params, *kv_cache_block_id, batch_size, use_fp8_fmha_);

        if (is_sm90() && fmha_type_ == FMHAType::PAGED_TRT_V2) {
            kv_cache_offset_host = allocateBuffer({DataType::TYPE_INT32, {batch_size, 1, 2, max_blocks_per_batch}, AllocationType::HOST},
                                            {"kv_cache_offset_host"});
            this->copy({*kv_cache_offset_host, *kv_cache_block_id});
            kv_block_array.pagedKVBlockOffsetsOnHost = kv_cache_offset_host->data();
        }

        prefix_prompt_param.kv_block_array = kv_block_array;

        if (params.common.prefix_prompt_lengths) {
            prefix_prompt_param.d_prefix_prompt_lengths  = params.common.prefix_prompt_lengths->data<int>();
            prefix_prompt_param.max_prefix_prompt_length = params.common.max_prefix_length;
            prefix_prompt_param.count_length             = 1;
        }
    }

    if (fmha_type_ == FMHAType::NONE && prefix_prompt_param.max_prefix_prompt_length > 0) {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                         invokeLoadPrefixKVCache,
                                         q_output->data(),
                                         k_output->data(),
                                         v_output->data(),
                                         &prefix_prompt_param,
                                         batch_size,
                                         seq_len,
                                         head_num,
                                         kv_head_num,
                                         size_per_head,
                                         nullptr, // scale_out_ptr,
                                         0, //int8_mode,
                                         stream_);
        sync_check_cuda_error();
    }

    // if all condition satisfy, no need to do invokeAddFusedQKVBiasTranspose
    bool skip_add_bias_transpose = (params.configs.rope_config.style == RopeStyle::No && !params.common.kv_cache
                                    && !params.configs.fuse_qkv_add_bias && fmha_type_ != FMHAType::NONE);
    FT_LOG_DEBUG("skip_add_bias_transpose: %d", skip_add_bias_transpose);
    if (!skip_add_bias_transpose) {
        bool store_qkv   = fmha_type_ != FMHAType::PAGED_TRT_V2 && fmha_type_ != FMHAType::NONE;
        bool store_q     = fmha_type_ == FMHAType::PAGED_TRT_V2 || fmha_type_ == FMHAType::NONE;
        bool store_kv    = fmha_type_ == FMHAType::NONE;
        // if use mla cache, no need to store cache
        bool store_cache = params.common.kv_cache.has_value();

        DISPATCH_CUDA_FUNCTION_DATA_TYPE(
            datatype,
            invokeAddFusedQKVBiasTranspose,
            q_output->data(),
            k_output->data(),
            v_output->data(),
            &prefix_prompt_param,
            params.input.data(),
            qkv_buf_fp8 != nullptr ? qkv_buf_fp8->data() : nullptr,
            params.common.position_ids ? params.common.position_ids->dataWithOffset<int>(decoder_batch_size * params.configs.rope_config.index_factor): nullptr,
            params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias ? params.weights.qkv_weight->bias->data() : nullptr,
            params.common.padding_offset->data<int>(),
            params.common.cu_seqlens->data<int>(),
            batch_size,
            seq_len,
            token_num,
            head_num,
            kv_head_num,
            size_per_head,
            params.configs.rope_config,
            params.configs.use_logn_attn,
            nullptr, // scale_out_ptr,
            0, // int8_mode,
            fmha_type_ == FMHAType::PAGED_TRT_V2,
            store_qkv,
            store_q,
            store_kv,
            store_cache,
            stream_);
        sync_check_cuda_error();

        if (!qkv_buf_fp8) {
            printBufferData(params.input, "after invoke transpse");
        } else {
            printBufferData(params.input, "after invoke transpse");
            FT_LOG_DEBUG("now print qkv_buf_fp8");
            printBufferData(*qkv_buf_fp8.get(), "after invoke transpse fp8");
        }
        sync_check_cuda_error();

        if (store_cache) {
            writeCacheStore(params);
        }

        // printBufferData(params.input, "after invoke transpse");
        printBufferData(*q_output, "Q after invoke transpose");
        printBufferData(*k_output, "K after invoke transpose");
        printBufferData(*v_output, "V after invoke transpose");
    }

    prefillAttention(params, kv_block_array, q_output, k_output, v_output, qkv_buf_fp8);
}

template<typename T>
void selfAttentionwrapper(const AttentionModuleParams params,
                          bool use_multi_block_mode,
                          bool use_fp8_fmha,
                          size_t max_seq_len_tile,
                          void* partial_out,
                          float* partial_sum,
                          float* partial_max,
                          int* block_counter,
                          KVBlockArray kv_block_array,
                          cudaStream_t stream,
                          CudaDevice *device)
{
    size_t batch_size           = params.common.decoder_batch_size;
    size_t step                 = params.common.decoder_max_seq_len + 1;
    size_t local_head_num       = params.configs.head_num;
    size_t local_head_num_kv    = params.configs.kv_head_num;
    size_t size_per_head        = params.configs.size_per_head;
    const auto& output = params.output;

    const T* qkv_buf_ptr = params.input.data<T>();
    void* attn_out_ptr = nullptr;
    BufferPtr f16_out;
    if (use_fp8_fmha) {
        f16_out = device->allocateBuffer({params.input.type(), output.shape(), AllocationType::DEVICE}, {"f16_out"});
        attn_out_ptr = f16_out->data();
    } else {
        attn_out_ptr = output.data();
    }

    const T* bias_ptr = (params.weights.qkv_weight->bias == nullptr || !params.configs.fuse_qkv_add_bias) ?
                        nullptr :
                        params.weights.qkv_weight->bias->data<T>();

    // TODO(lidongjin) support relative attention
    const T* relative_attention_bias_ptr = nullptr;

    // prefix prompt

    const auto* input_lengths = params.common.input_lengths->data<int>();
    const auto* sequence_lengths = params.common.sequence_lengths->data<int>();

    float q_scaling = params.configs.q_scaling;
    int relative_attention_bias_stride = 0;
    const float* linear_bias_slopes = params.common.linear_bias_slopes ? params.common.linear_bias_slopes->data<float>() : nullptr;
    const bool* masked_tokens = nullptr;

    tensorrt_llm::common::QuantMode kv_cache_quant_mode = trt_common::QuantMode::fromDescription(false, false, false, false, false, false, false, false);
    if (params.configs.kv_cache_dtype == KvCacheDataType::INT8) {
        kv_cache_quant_mode = trt_common::QuantMode::fromDescription(true, true, false, false, false, true, false, true);
    } else if (params.configs.kv_cache_dtype == KvCacheDataType::FP8 && use_fp8_fmha) {
        kv_cache_quant_mode = trt_common::QuantMode::fromDescription(true, true, false, false, false, false, true, true);
    }

    // TODO: refactor QBuffer to suppport view and return QBuffer
    if (params.output.isQBuffer()) {
        params.output.updateTypeAndShape(DataType::TYPE_FP8_E4M3, params.output.shape());
    }

    auto flash_infer_attn_params = (FlashInferAttnParams*)params.common.decode_flash_infer_attn_params.get();
    if (flash_infer_attn_params && flash_infer_attn_params->plan.numel() > 0) {
        flash_infer_attn_params->run(
                params, f16_out,
                reinterpret_cast<int64_t>(stream));
    } else {
        const float* attention_output_orig_quant_scale = nullptr;

        if (params.weights.static_scale_reciprocal_weight) {
            attention_output_orig_quant_scale = params.weights.static_scale_reciprocal_weight->kernel->data<float>();
        }
        fusedQKV_masked_attention_dispatch<T, KVBlockArray>(
                qkv_buf_ptr,
                bias_ptr,
                relative_attention_bias_ptr,
                nullptr, // cache_indir
                reinterpret_cast<T*>(attn_out_ptr),
                nullptr, // finished
                sequence_lengths,
                batch_size,
                1, // beam_width
                local_head_num,
                local_head_num_kv,
                size_per_head,
                params.configs.rope_config,
                params.configs.use_logn_attn,
                params.common.position_ids ? params.common.position_ids->data<int>() : nullptr,
                step,
                nullptr, // prefix_prompt_lengths
                0, // max_prefix_prompt_length
                true, // count_prefix_length
                input_lengths,
                step,
                q_scaling,
                relative_attention_bias_stride,
                linear_bias_slopes,
                masked_tokens,
                nullptr, // query_weight_scale_out
                attention_output_orig_quant_scale,
                0, // int8_mode,
                kv_cache_quant_mode,
                use_multi_block_mode,
                (int)max_seq_len_tile,
                reinterpret_cast<T*>(partial_out),
                partial_sum,
                partial_max,
                block_counter,
                params.configs.softmax_extra_scale,
                kv_block_array,
                stream);
        sync_check_cuda_error();
        if (f16_out) {
            cudaMemcpyAsync(output.data(), f16_out->data(), output.size(), cudaMemcpyDeviceToDevice, stream);
        }
        sync_check_cuda_error();
    }
}

AttentionModuleOutput CudaDevice::decoderSelfAttention(const AttentionModuleParams& params) {
    auto datatype = params.input.type();
    size_t max_seq_len_tile = 0;
    BufferPtr partial_out = nullptr;
    BufferPtr partial_sum = nullptr;
    BufferPtr partial_max = nullptr;
    BufferPtr block_counter = nullptr;

    size_t batch_size           = params.common.decoder_batch_size;
    size_t local_head_num       = params.configs.head_num;
    size_t size_per_head        = params.configs.size_per_head;

    if (use_multi_block_mode) {
        const int threads_per_value = pow2roundup(size_per_head) * getTypeSize(datatype) / 16;
        // for allocate partial output results memory. Regardless to THDS_PER_BLOCK
        max_seq_len_tile = 256 / threads_per_value;
        partial_out = allocateBuffer({datatype,
                                     {batch_size, max_seq_len_tile, local_head_num, size_per_head},
                                     AllocationType::DEVICE},
            {"partial_out"});
        partial_sum = allocateBuffer({DataType::TYPE_FP32,
                                     {batch_size, max_seq_len_tile, local_head_num},
                                     AllocationType::DEVICE},
            {"partial_sum"});
        partial_max = allocateBuffer({DataType::TYPE_FP32,
                                     {batch_size, max_seq_len_tile, local_head_num},
                                     AllocationType::DEVICE},
            {"partial_max"});
        block_counter = allocateBuffer({DataType::TYPE_INT32,
                                      {batch_size, local_head_num},
                                      AllocationType::DEVICE},
                                       {"block_counter"});
        // TODO(lidongjin) use fill op to set zeros.
        cudaMemsetAsync(block_counter->data(), 0, sizeof(int) * batch_size * local_head_num, stream_);
    }
    void* partial_out_data = (partial_out == nullptr) ? nullptr : partial_out->data();
    float* partial_sum_data = (partial_sum == nullptr) ? nullptr : partial_sum->data<float>();
    float* partial_max_data = (partial_max == nullptr) ? nullptr : partial_max->data<float>();
    int* block_counter_data = (block_counter == nullptr) ? nullptr : block_counter->data<int>();

    RUNTIME_ASSERT_OP_ARG(params.common.kv_cache, "kv cache can not be null for decoder self-attention");
    const auto max_blocks_per_batch = params.common.kv_cache->kv_cache_block_id->shape()[1];
    auto       kv_cache_block_id      = allocateBuffer(
        {DataType::TYPE_INT32, {batch_size, 1, 2, max_blocks_per_batch}, AllocationType::DEVICE}, {"kv_cache_block_id"});
    KVBlockArray kv_block_array = getKVBlockArray(params, *kv_cache_block_id, batch_size, use_fp8_fmha_);
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                     selfAttentionwrapper,
                                     params,
                                     use_multi_block_mode,
                                     use_fp8_fmha_,
                                     max_seq_len_tile,
                                     partial_out_data,
                                     partial_sum_data,
                                     partial_max_data,
                                     block_counter_data,
                                     kv_block_array,
                                     stream_,
                                     this);
}

}  // namespace fastertransformer
