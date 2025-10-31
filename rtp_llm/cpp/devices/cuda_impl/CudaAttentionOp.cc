#include <iostream>
#include <numeric>
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/devices/utils/RopeCache.h"
#include "rtp_llm/cpp/kernels/layernorm_kernels.h"
#include "rtp_llm/cpp/kernels/activation_kernels.h"
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/cpp/kernels/kv_cache_kernels.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

#ifdef USING_CUDA12
#include "rtp_llm/cpp/devices/cuda_impl/CudaXqa.h"
#endif

using namespace std;
using namespace rtp_llm;

namespace rtp_llm {

ParamsPtr CudaDevice::prepareTrtAttn(const AttentionConfigs& configs,
                                     const BufferPtr&        layer_cache,
                                     const BufferPtr&        kv_cache_block_id,
                                     int                     batch_size) {
    return prepareTrtAttn(configs, 1, kv_cache_block_id, batch_size);
}

ParamsPtr CudaDevice::prepareTrtAttn(const AttentionConfigs& configs,
                                     int                     kv_block_offset,
                                     const BufferPtr&        kv_cache_block_id,
                                     int                     batch_size) {
    if (!kv_block_offset || !kv_cache_block_id || 0 == batch_size) {
        return nullptr;
    }

    auto trt_attn = std::make_shared<TRTAttn>();

    int             ele_size   = 2;
    KvCacheDataType cache_type = KvCacheDataType::BASE;
#ifdef ENABLE_FP8
    if (use_fp8_fmha_) {
        cache_type = KvCacheDataType::FP8;
        ele_size   = 1;
    } else
#endif
        if (configs.kv_cache_dtype == KvCacheDataType::INT8) {
        cache_type = KvCacheDataType::INT8;
        ele_size   = 1;
    } else if (configs.kv_cache_dtype == KvCacheDataType::FP8) {
        cache_type = KvCacheDataType::FP8;
        ele_size   = 1;
    }

    RUNTIME_ASSERT_OP_ARG(kv_cache_block_id->shape()[0] == batch_size,
                          "context attention kv blocks batch size expected [%d] but buffer[%s]",
                          (int)batch_size,
                          kv_cache_block_id->debugString().c_str());
    const size_t max_blocks_per_batch = kv_cache_block_id->shape()[1];

    trt_attn->kv_cache_offset =
        allocateBuffer({DataType::TYPE_INT32, {size_t(batch_size), 1, 2, max_blocks_per_batch}, AllocationType::DEVICE},
                       {"kv_cache_offset"});
    trt_attn->kv_block_array                     = KVBlockArray(batch_size,
                                            max_blocks_per_batch,
                                            configs.tokens_per_block,
                                            configs.kv_head_num * configs.size_per_head * ele_size,
                                            0,
                                            0,
                                            nullptr,  // (uint64_t*)k_cache.data(),
                                            nullptr,
                                            (rtp_llm::KVCacheIndex*)trt_attn->kv_cache_offset->data<int>());
    trt_attn->kv_block_array.cache_type          = cache_type;
    trt_attn->kv_block_array.mScaleBytesPerBlock = configs.tokens_per_block * configs.kv_head_num * sizeof(float);

    invokeConvertOffsetToBlockArrayData(trt_attn->kv_cache_offset->data<int>(),
                                        kv_cache_block_id->data<int>(),
                                        batch_size,
                                        max_blocks_per_batch,
                                        stream_);
    if (is_sm90() && fmha_type_ == FMHAType::PAGED_TRT_V2) {
        trt_attn->kv_cache_offset_h = allocateBuffer(
            {DataType::TYPE_INT32, {size_t(batch_size), 1, 2, max_blocks_per_batch}, AllocationType::HOST},
            {"kv_cache_offset_h"});
        copy({*trt_attn->kv_cache_offset_h, *trt_attn->kv_cache_offset});
        trt_attn->kv_block_array.pagedKVBlockOffsetsOnHost = trt_attn->kv_cache_offset_h->data();
    }

    check_cuda_error();
    return trt_attn;
}

AttentionModuleOutput CudaDevice::contextAttention(const AttentionModuleParams& params) {
    RTP_LLM_LOG_DEBUG("FMHA Type use %s.", std::to_string((int)fmha_type_).c_str());
    KVBlockArray kv_block_array;

    if (params.common.kv_cache) {
        auto trt_attn  = ((TRTAttn*)params.common.prefill_trt_attn.get());
        kv_block_array = trt_attn->kv_block_array;
        TRTAttn::setKvCache(kv_block_array, *params.common.kv_cache);
    }

    auto datatype            = params.input.type();
    auto token_num           = params.input.shape()[0];
    auto batch_size          = params.common.context_batch_size;
    auto decoder_batch_size  = params.common.decoder_batch_size;
    auto seq_len             = params.common.context_max_seq_len;
    auto seq_len_with_prefix = seq_len + params.common.max_prefix_length;
    auto head_num            = params.configs.head_num;
    auto kv_head_num         = params.configs.kv_head_num;
    auto size_per_head       = params.configs.size_per_head;

    // for flashinfer/xqa
    auto q_no_transpose_output = allocateBuffer(
        {params.input.type(), {token_num, head_num, size_per_head}, AllocationType::DEVICE}, {"q_no_transpose_output"});

    auto q_output = allocateBuffer(
        {params.input.type(), {batch_size, head_num, seq_len, size_per_head}, AllocationType::DEVICE}, {"q_output"});

    auto k_output = allocateBuffer(
        {params.input.type(), {batch_size, kv_head_num, seq_len_with_prefix, size_per_head}, AllocationType::DEVICE},
        {"k_output"});

    auto v_output = allocateBuffer(
        {params.input.type(), {batch_size, kv_head_num, seq_len_with_prefix, size_per_head}, AllocationType::DEVICE},
        {"v_output"});

    BufferPtr qkv_buf_fp8;
    if (use_fp8_fmha_ && fmha_type_ != FMHAType::FLASH_INFER && fmha_type_ != FMHAType::XQA) {
        qkv_buf_fp8 = allocateBuffer({DataType::TYPE_FP8_E4M3,
                                      {batch_size, (head_num + kv_head_num * 2), seq_len_with_prefix, size_per_head},
                                      AllocationType::DEVICE},
                                     {"qkv_fp8_output"});
        check_cuda_value(cudaMemsetAsync(qkv_buf_fp8->data(), 0, qkv_buf_fp8->sizeBytes(), stream_));
    }

    if (fmha_type_ == FMHAType::NONE) {
        check_cuda_value(cudaMemsetAsync(q_output->data(), 0, q_output->sizeBytes(), stream_));
        check_cuda_value(cudaMemsetAsync(k_output->data(), 0, k_output->sizeBytes(), stream_));
        check_cuda_value(cudaMemsetAsync(v_output->data(), 0, v_output->sizeBytes(), stream_));
    }

    PrefixPromptBatchWeightsParam prefix_prompt_param;

    if (params.common.kv_cache) {
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
                                         nullptr,  // scale_out_ptr,
                                         0,        // int8_mode,
                                         stream_);
        check_cuda_error();
    }

    // if all condition satisfy, no need to do invokeAddFusedQKVBiasTranspose
    bool skip_add_bias_transpose = (params.configs.rope_config.style == RopeStyle::No && !params.common.kv_cache
                                    && !params.configs.fuse_qkv_add_bias && fmha_type_ != FMHAType::NONE);
    RTP_LLM_LOG_DEBUG("skip_add_bias_transpose: %d", skip_add_bias_transpose);
    if (!skip_add_bias_transpose) {
        bool store_qkv = fmha_type_ != FMHAType::PAGED_TRT_V2 && fmha_type_ != FMHAType::NONE
                         && fmha_type_ != FMHAType::FLASH_INFER && fmha_type_ != FMHAType::XQA;
        bool store_q_no_transpose = fmha_type_ == FMHAType::FLASH_INFER || fmha_type_ == FMHAType::XQA;
        bool store_q              = fmha_type_ == FMHAType::PAGED_TRT_V2 || fmha_type_ == FMHAType::NONE;
        bool store_kv             = fmha_type_ == FMHAType::NONE;
        // if use mla cache, no need to store cache
        bool store_cache = params.common.kv_cache.has_value();

        DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                         invokeAddFusedQKVBiasTranspose,
                                         q_no_transpose_output->data(),
                                         q_output->data(),
                                         k_output->data(),
                                         v_output->data(),
                                         &prefix_prompt_param,
                                         params.input.data(),
                                         qkv_buf_fp8 != nullptr ? qkv_buf_fp8->data() : nullptr,
                                         params.common.position_ids ?
                                             params.common.position_ids->dataWithOffset<int>(
                                                 decoder_batch_size * params.configs.rope_config.index_factor) :
                                             nullptr,
                                         params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias ?
                                             params.weights.qkv_weight->bias->data() :
                                             nullptr,
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
                                         nullptr,  // scale_out_ptr,
                                         0,        // int8_mode,
                                         fmha_type_ == FMHAType::PAGED_TRT_V2,
                                         store_qkv,
                                         store_q_no_transpose,
                                         store_q,
                                         store_kv,
                                         store_cache,
                                         stream_);
        check_cuda_error();

        if (!qkv_buf_fp8) {
            printBufferData(params.input, "after invoke transpse");
        } else {
            printBufferData(params.input, "after invoke transpse");
            RTP_LLM_LOG_DEBUG("now print qkv_buf_fp8");
            printBufferData(*qkv_buf_fp8.get(), "after invoke transpse fp8");
        }
        check_cuda_error();

        if (store_cache) {
            writeCacheStore(params);
        }

        // printBufferData(params.input, "after invoke transpse");
        printBufferData(*q_output, "Q after invoke transpose");
        printBufferData(*k_output, "K after invoke transpose");
        printBufferData(*v_output, "V after invoke transpose");
    }

    computeInsertedMoE();
    prefillAttention(params, kv_block_array, q_no_transpose_output, q_output, k_output, v_output, qkv_buf_fp8);
}

template<typename T>
void selfAttentionwrapper(const AttentionModuleParams params,
                          bool                        use_multi_block_mode,
                          bool                        use_fp8_fmha,
                          size_t                      max_seq_len_tile,
                          void*                       partial_out,
                          float*                      partial_sum,
                          float*                      partial_max,
                          int*                        block_counter,
                          KVBlockArray                kv_block_array,
                          cudaStream_t                stream,
                          CudaDevice*                 device) {
    size_t      batch_size        = params.common.decoder_batch_size;
    size_t      step              = params.common.decoder_max_seq_len + 1;
    size_t      local_head_num    = params.configs.head_num;
    size_t      local_head_num_kv = params.configs.kv_head_num;
    size_t      size_per_head     = params.configs.size_per_head;
    const auto& output            = params.output;

    const T*  qkv_buf_ptr  = params.input.data<T>();
    void*     attn_out_ptr = nullptr;
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

    const auto* input_lengths    = params.common.input_lengths->data<int>();
    const auto* sequence_lengths = params.common.sequence_lengths->data<int>();

    float        q_scaling = params.configs.q_scaling;
    const float* linear_bias_slopes =
        params.common.linear_bias_slopes ? params.common.linear_bias_slopes->data<float>() : nullptr;

    tensorrt_llm::common::QuantMode kv_cache_quant_mode =
        trt_common::QuantMode::fromDescription(false, false, false, false, false, false, false, false);
    if (params.configs.kv_cache_dtype == KvCacheDataType::INT8) {
        kv_cache_quant_mode =
            trt_common::QuantMode::fromDescription(true, true, false, false, false, true, false, true);
    } else if (params.configs.kv_cache_dtype == KvCacheDataType::FP8 && use_fp8_fmha) {
        kv_cache_quant_mode =
            trt_common::QuantMode::fromDescription(true, true, false, false, false, false, true, true);
    }

    const float* attention_output_orig_quant_scale = nullptr;
    if (params.weights.static_scale_reciprocal_weight) {
        attention_output_orig_quant_scale = params.weights.static_scale_reciprocal_weight->kernel->data<float>();
    }

    fusedQKV_masked_attention_dispatch<T, KVBlockArray>(
        qkv_buf_ptr,
        bias_ptr,
        nullptr,  // relative_attention_bias
        nullptr,  // cache_indir
        reinterpret_cast<T*>(attn_out_ptr),
        nullptr,  // finished
        sequence_lengths,
        batch_size,
        1,  // beam_width
        local_head_num,
        local_head_num_kv,
        size_per_head,
        params.configs.rope_config,
        params.configs.use_logn_attn,
        params.common.position_ids ? params.common.position_ids->data<int>() : nullptr,
        step,
        nullptr,  // prefix_prompt_lengths
        0,        // max_prefix_prompt_length
        true,     // count_prefix_length
        input_lengths,
        step,
        q_scaling,
        0,  // relative_attention_bias_stride,
        linear_bias_slopes,
        nullptr,  // masked_tokens,
        nullptr,  // query_weight_scale_out
        attention_output_orig_quant_scale,
        0,  // int8_mode,
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
    check_cuda_error();
    if (f16_out) {
        cudaMemcpyAsync(output.data(), f16_out->data(), output.size(), cudaMemcpyDeviceToDevice, stream);
    }
    check_cuda_error();
}

static std::once_flag rope_cache_flag;

AttentionModuleOutput CudaDevice::decoderSelfAttention(const AttentionModuleParams& params) {

    // TODO: refactor QBuffer to suppport view and return QBuffer
    if (params.output.isQBuffer()) {
        params.output.updateTypeAndShape(DataType::TYPE_FP8_E4M3, params.output.shape());
    }

    auto      datatype         = params.input.type();
    size_t    max_seq_len_tile = 0;
    BufferPtr partial_out      = nullptr;
    BufferPtr partial_sum      = nullptr;
    BufferPtr partial_max      = nullptr;
    BufferPtr block_counter    = nullptr;

    size_t batch_size        = params.common.decoder_batch_size;
    size_t local_head_num    = params.configs.head_num;
    size_t local_kv_head_num = params.configs.kv_head_num;
    size_t size_per_head     = params.configs.size_per_head;

    RUNTIME_ASSERT_OP_ARG(params.common.kv_cache, "kv cache can not be null for decoder self-attention");
    auto trt_attn       = ((TRTAttn*)params.common.decode_trt_attn.get());
    auto kv_block_array = trt_attn->kv_block_array;
    TRTAttn::setKvCache(kv_block_array, *params.common.kv_cache);
    printBufferData(*trt_attn->kv_cache_offset, "kv_cache_offset");

    BufferPtr q_output;
    auto      flash_infer    = (FlashInferAttnParams*)params.common.decode_flash_infer_attn.get();
    bool      use_flashinfer = flash_infer && flash_infer->plan.numel() > 0;
    if (use_xqa || use_flashinfer) {
        q_output = allocateBuffer(
            {params.input.type(), {batch_size, local_head_num, size_per_head}, AllocationType::DEVICE}, {"q_output"});

        bool use_rope_cache =
            params.configs.rope_config.style == RopeStyle::Base || params.configs.rope_config.style == RopeStyle::Yarn;
        static torch::Tensor rope_cache;
        std::call_once(rope_cache_flag, [&]() {
            if (use_rope_cache) {
                rope_cache = getRopeCache(params.configs.rope_config, init_params_.max_seq_len);
            }
        });

        DISPATCH_CUDA_FUNCTION_DATA_TYPE(params.input.type(),
                                         invokeDecodeAddFusedQKVBiasTranspose,
                                         q_output->data(),
                                         nullptr,  // k_buf
                                         nullptr,  // v_buf
                                         kv_block_array,
                                         params.input.data(),
                                         params.common.position_ids ? params.common.position_ids->data<int>() :
                                                                      params.common.sequence_lengths->data<int>(),
                                         params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias ?
                                             params.weights.qkv_weight->bias->data() :
                                             nullptr,
                                         use_rope_cache && rope_cache.defined() ? rope_cache.data_ptr<float>() :
                                                                                  nullptr,
                                         batch_size,
                                         local_head_num,
                                         local_kv_head_num,
                                         size_per_head,
                                         params.configs.rope_config,
                                         params.configs.use_logn_attn,
                                         true,   // store_q,
                                         false,  // store_kv,
                                         true,   // store_cache,
                                         stream_);

        check_cuda_error();
    }

    computeInsertedMoE();

#ifdef USING_CUDA12
    size_t local_tokens_per_block = params.configs.tokens_per_block;
    if (use_xqa
        && supportXqa(params.input.type(),
                      params.output.type(),
                      params.common.kv_cache->k_cache_buffer->type(),
                      local_head_num / local_kv_head_num,
                      size_per_head,
                      local_tokens_per_block)) {

        runXqa(q_output->data(),
               q_output->type() == DataType::TYPE_BF16,
               params.output.data(),
               local_head_num,
               local_kv_head_num,
               size_per_head,
               params.common.decoder_batch_size,
               static_cast<size_t>(kv_block_array.mMaxBlocksPerSeq),
               params.common.decoder_max_seq_len + 1,
               local_tokens_per_block,
               kv_block_array.mPrimaryPoolPtr,
               reinterpret_cast<int32_t*>(const_cast<KVCacheIndex*>(kv_block_array.data)),
               params.common.kv_cache->k_cache_buffer->type() == DataType::TYPE_FP8_E4M3,
               reinterpret_cast<uint32_t*>(params.common.sequence_lengths->data()),
               this,
               params.output.type() == DataType::TYPE_FP8_E4M3 ?
                   reinterpret_cast<float*>(params.weights.static_scale_reciprocal_weight->kernel->data()) :
                   nullptr,
               static_cast<size_t>(init_params_.sp_config.gen_num_per_cycle + 1));

        return;
    }
#endif

    if (use_flashinfer) {
        BufferPtr f16_out;
        if (use_fp8_fmha_) {
            f16_out = allocateBuffer({params.input.type(), params.output.shape(), AllocationType::DEVICE}, {"f16_out"});
        }
        flash_infer->run(params, q_output, f16_out, reinterpret_cast<int64_t>(stream_));
        return;
    }

    if (use_multi_block_mode) {
        const int threads_per_value = pow2roundup(size_per_head) * getTypeSize(datatype) / 16;
        // for allocate partial output results memory. Regardless to THDS_PER_BLOCK
        max_seq_len_tile = 256 / threads_per_value;
        partial_out      = allocateBuffer(
            {datatype, {batch_size, max_seq_len_tile, local_head_num, size_per_head}, AllocationType::DEVICE},
            {"partial_out"});
        partial_sum = allocateBuffer(
            {DataType::TYPE_FP32, {batch_size, max_seq_len_tile, local_head_num}, AllocationType::DEVICE},
            {"partial_sum"});
        partial_max = allocateBuffer(
            {DataType::TYPE_FP32, {batch_size, max_seq_len_tile, local_head_num}, AllocationType::DEVICE},
            {"partial_max"});
        block_counter = allocateBuffer({DataType::TYPE_INT32, {batch_size, local_head_num}, AllocationType::DEVICE},
                                       {"block_counter"});
        // TODO(lidongjin) use fill op to set zeros.
        cudaMemsetAsync(block_counter->data(), 0, sizeof(int) * batch_size * local_head_num, stream_);
    }
    void*  partial_out_data   = (partial_out == nullptr) ? nullptr : partial_out->data();
    float* partial_sum_data   = (partial_sum == nullptr) ? nullptr : partial_sum->data<float>();
    float* partial_max_data   = (partial_max == nullptr) ? nullptr : partial_max->data<float>();
    int*   block_counter_data = (block_counter == nullptr) ? nullptr : block_counter->data<int>();

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

}  // namespace rtp_llm
