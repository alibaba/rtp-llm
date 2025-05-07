#include "maga_transformer/cpp/kernels/gpt_kernels.h"
#include "maga_transformer/cpp/devices/OpData.h"
#include "maga_transformer/cpp/devices/cuda_impl/CudaDevice.h"
#include "maga_transformer/cpp/devices/CommonDefines.h"
#include "maga_transformer/cpp/cuda/Dispatch.h"
#include "maga_transformer/cpp/devices/utils/DebugUtils.h"
#include "maga_transformer/cpp/kernels/unfused_attention_kernels.h"
#include "maga_transformer/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "3rdparty/flashinfer/flashinfer.h"

namespace rtp_llm {

KVBlockArray CudaDevice::getKVBlockArray(const AttentionModuleParams& params,
                             const Buffer&                kv_cache_offset_pointers,
                             int                          batch_size,
                             bool                         use_fp8_fmha) {
    const auto& kv_cache         = params.common.kv_cache;
    const auto& kv_blocks_offset = *(kv_cache->kv_cache_block_id);
    const auto& kv_block_offset  = (kv_cache->k_cache_buffer)->shape()[0] * kv_cache->layer_num;
    RUNTIME_ASSERT_OP_ARG(kv_blocks_offset.shape()[0] == batch_size,
                          "context attention kv blocks batch size expected [%d] but buffer[%s]",
                          (int)batch_size,
                          kv_blocks_offset.debugString().c_str());
    const auto  max_blocks_per_batch = kv_blocks_offset.shape()[1];
    const auto& k_cache              = *(kv_cache->k_cache_buffer);
    const auto& v_cache              = *(kv_cache->v_cache_buffer);
    auto const  elemSize             = kv_cache->k_scale_buffer || use_fp8_fmha ? sizeof(int8_t) : 2;  // 2 for kv cache fp16
    // RTP_LLM_LOG_INFO("kv_cache[0].typeSize():%d", kv_cache[0].typeSize());
    RTP_LLM_LOG_DEBUG(
        "kv_blocks_offset size:%d, k_cache:%p, v_cache:%p, k_cache[0].sizeBytes():%d, params.configs.tokens_per_block:%d, kv_block_offset:%d",
        kv_blocks_offset.size(),
        (uint64_t*)k_cache.data(),
        (uint64_t)v_cache.data(),
        k_cache[0].sizeBytes(),
        params.configs.tokens_per_block,
        kv_block_offset);
    auto const   sizePerToken = params.configs.kv_head_num * params.configs.size_per_head * elemSize;
    KVBlockArray kv_cache_buffer =
        KVBlockArray(batch_size,
                     max_blocks_per_batch,
                     params.configs.tokens_per_block,
                     sizePerToken,
                     0,
                     0,
                     (uint64_t*)k_cache.data(),
                     nullptr,
                     (rtp_llm::KVBlockArrayForContextFMHA::DataType*)kv_cache_offset_pointers.data());
    invokeConvertOffsetToBlockArrayData((int32_t*)kv_cache_offset_pointers.data(),
                                        (int*)kv_blocks_offset.data(),
                                        batch_size,
                                        max_blocks_per_batch,
                                        kv_block_offset,
                                        stream_);
    sync_check_cuda_error();
    if (kv_cache->k_scale_buffer) {
        RUNTIME_ASSERT_OP_ARG(kv_cache->v_scale_buffer,
                              "v scale buffer should has value when use k scale buffer has value");
        const auto& k_scale = *(kv_cache->k_scale_buffer);
        kv_cache_buffer.scale = k_scale.data();
        kv_cache_buffer.mScaleBytesPerBlock = k_scale[0].sizeBytes();
    }
    KvCacheDataType cache_type = KvCacheDataType::BASE;
#ifdef ENABLE_FP8
    if (use_fp8_fmha) {
        cache_type = KvCacheDataType::FP8;
    } else
#endif
    if (kv_cache->k_scale_buffer && params.configs.kv_cache_dtype == KvCacheDataType::INT8) {
        RTP_LLM_LOG_DEBUG("now use kv_cache int8");
        cache_type = KvCacheDataType::INT8;
    }
    kv_cache_buffer.cache_type = cache_type;
    sync_check_cuda_error();
    return kv_cache_buffer;
}

void CudaDevice::prefillAttention(const AttentionModuleParams& params,
                            KVBlockArray                 kv_block_array,
                            const BufferPtr&             q_output,
                            const BufferPtr&             k_output,
                            const BufferPtr&             v_output,
                            const BufferPtr&             qkv_buf_fp8) {
    auto fmha_type = fmha_type_;
    auto stream  = stream_;
    auto cufmha_runner = cufmha_runner_;
    RTP_LLM_LOG_DEBUG("FMHA Type use %s.", std::to_string((int)fmha_type).c_str());
    auto      datatype            = params.input.type();
    auto      token_num           = params.input.shape()[0];
    auto      batch_size          = params.common.context_batch_size;
    auto      seq_len             = params.common.context_max_seq_len;
    auto      seq_len_with_prefix = seq_len + params.common.max_prefix_length;
    auto      head_num            = params.configs.head_num;
    auto      kv_head_num         = params.configs.kv_head_num;
    auto      size_per_head       = params.configs.size_per_head;
    bool      use_fp8_fmha        = qkv_buf_fp8 != nullptr;
    BufferPtr tiled_counter_ptr;
    if (FMHAType::PAGED_TRT_V2 == fmha_type || FMHAType::TRT_V2 == fmha_type) {
        tiled_counter_ptr =
            allocateBuffer({DataType::TYPE_UINT32, {1}, AllocationType::DEVICE}, {"tiled_counter_pointer"});
        cudaMemsetAsync(tiled_counter_ptr->data(), 0, sizeof(uint32_t), stream);
    }
    switch (fmha_type) {
        case FMHAType::PAGED_TRT_V2: {
            RTP_LLM_CHECK_WITH_INFO(q_output != nullptr, "q_output must be provided for paged trt v2 fmha");
            cufmha_runner->runTrtV2FmhaPaged(q_output->data(),
                                             params.common.cu_seqlens->data(),
                                             params.common.cu_kv_seqlens->data(),
                                             params.output.data(),
                                             reinterpret_cast<uint32_t*>(tiled_counter_ptr->data()),
                                             batch_size,
                                             seq_len,
                                             seq_len_with_prefix,
                                             token_num,
                                             kv_block_array,
                                             false,
                                             false,
                                             params.common.linear_bias_slopes != nullptr,
                                             false);
            break;
        }
        case FMHAType::TRT_V2: {
            void*  fmha_input_ptr                    = use_fp8_fmha ? qkv_buf_fp8->data() : params.input.data();
            void*  fmha_output_ptr                   = params.output.data();
            RTP_LLM_CHECK_WITH_INFO(fmha_input_ptr, "fmha_input_ptr must be provided for trt v2 fmha");
            float* attention_output_orig_quant_scale = nullptr;
            if (params.weights.static_scale_reciprocal_weight && use_fp8_fmha) {
                printBufferData(*(params.weights.static_scale_reciprocal_weight->kernel), "attn scale");
                attention_output_orig_quant_scale =
                    (params.weights.static_scale_reciprocal_weight->kernel->data<float>());
            }
            bool      need_quant_fmha_out = !use_fp8_fmha && params.output.isQBuffer();
            BufferPtr tmp_fmha_output;
            if (need_quant_fmha_out) {
                // for sm89 cannot use fp8_fmha, but attention output should be fp8
                tmp_fmha_output = allocateBuffer({DataType::TYPE_FP16,
                                                  {batch_size, head_num * seq_len_with_prefix * size_per_head},
                                                  AllocationType::DEVICE},
                                                 {"fmha_fp16_output"});
                cudaMemsetAsync(tmp_fmha_output->data(), 0, tmp_fmha_output->sizeBytes(), stream);
                fmha_output_ptr = tmp_fmha_output->data();
            }
            RTP_LLM_CHECK_WITH_INFO(fmha_output_ptr, "fmha_output_ptr must be provided for trt v2 fmha");
            cufmha_runner->runTrtV2Fmha(fmha_input_ptr,
                                        params.common.cu_seqlens->data(),
                                        fmha_output_ptr,
                                        reinterpret_cast<uint32_t*>(tiled_counter_ptr->data()),
                                        attention_output_orig_quant_scale,
                                        batch_size,
                                        seq_len,
                                        token_num,
                                        kv_block_array,
                                        false,
                                        false,
                                        params.common.linear_bias_slopes != nullptr,
                                        false);
            if (need_quant_fmha_out) {
                DataType quant_out_data_type = DataType::TYPE_FP8_E4M3;
                auto     quant_params =
                    QuantizeParams(*tmp_fmha_output,
                                   quant_out_data_type,
                                   1,
                                   QScheme::Qfp8PerTensor,
                                   std::nullopt,
                                   std::nullopt,
                                   (OptionalConstBufferRef)*params.weights.static_quant_weight->kernel,
                                   (OptionalConstBufferRef)*params.weights.static_scale_reciprocal_weight->kernel);
                auto quant_output = quantize(quant_params);
                cudaMemcpyAsync(
                    params.output.data(), quant_output->data(), params.output.size(), cudaMemcpyDeviceToDevice, stream);
            }
            break;
        }
        case FMHAType::PAGED_OPEN_SOURCE: {
            const size_t max_blocks_per_batch = params.common.kv_cache->kv_cache_block_id->shape()[1];
            const auto   ws_size              = cufmha_runner->getOpenSourceWorkSpaceSize(
                batch_size, seq_len, max_blocks_per_batch * params.configs.tokens_per_block, true);
            auto ws = allocateBuffer({DataType::TYPE_INT8, {ws_size}, AllocationType::DEVICE},
                                             {"open_source_paged_fmha_ws"});
            cufmha_runner->runOpenSourceFmhaPaged(
                params.input.data(),
                params.common.kv_cache->k_cache_buffer->data(),
                params.common.kv_cache->v_cache_buffer->data(),
                params.output.data(),
                params.common.cu_seqlens->data<int>(),
                params.common.cu_kv_seqlens->data<int>(),
                params.common.kv_cache->kv_cache_block_id->data<int>(),
                batch_size,
                max_blocks_per_batch,
                params.configs.tokens_per_block,
                seq_len,
                ws->data(),
                params.common.linear_bias_slopes ? params.common.linear_bias_slopes->data<float>() : nullptr,
                params.configs.softmax_extra_scale);
            break;
        }
        case FMHAType::OPEN_SOURCE: {
            const auto   ws_size      = cufmha_runner->getOpenSourceWorkSpaceSize(batch_size, seq_len);
            auto         ws           = allocateBuffer({DataType::TYPE_INT8, {ws_size}, AllocationType::DEVICE},
                                                               {"open_source_fmha_ws"});
            const size_t hidden_units = head_num * size_per_head;
            const size_t hidden_units_kv = kv_head_num * size_per_head;
            cufmha_runner->runOpenSourceFmha(
                params.input.data(),
                params.input.dataWithOffset(hidden_units),
                params.input.dataWithOffset(hidden_units + hidden_units_kv),
                params.output.data(),
                params.common.cu_seqlens->data<int>(),
                batch_size,
                seq_len,
                ws->data(),
                params.common.linear_bias_slopes ? params.common.linear_bias_slopes->data<float>() : nullptr,
                params.configs.softmax_extra_scale);
            break;
        }
        case FMHAType::TRT_V1: {
            auto qkv_buf_temp = allocateBuffer(
                {datatype, {token_num, head_num + 2 * kv_head_num, size_per_head}, AllocationType::DEVICE},
                {"qkv_buf_temp"});
            cufmha_runner->runTrtV1Fmha(params.input.data(),
                                        params.common.cu_seqlens->data(),
                                        params.output.data(),
                                        qkv_buf_temp->data(),
                                        batch_size,
                                        seq_len,
                                        token_num);
            break;
        }
        default: {
            RTP_LLM_CHECK_WITH_INFO(q_output && k_output && v_output, "q_output/k_output/v_output must be provided for default context attention implementation");
            q_output->updateShape({batch_size, kv_head_num, (head_num / kv_head_num) * seq_len, size_per_head});
            auto qk_output = gemm({*q_output,
                                           *k_output,
                                           std::nullopt,
                                           nullptr,
                                           DataType::TYPE_FP32,
                                           TransposeOperation::NONE,
                                           TransposeOperation::TRANSPOSE});
            qk_output->updateShape({batch_size, head_num, seq_len, seq_len_with_prefix});
            printBufferData(*qk_output, "qk_output: ");
            float scale = (1.0f / sqrtf(size_per_head * 1.0f)) * params.configs.softmax_extra_scale;
            // TODO(lidongjin): Only support float32(in)\float16(output).
            RUNTIME_ASSERT_OP_ARG(params.common.attention_mask,
                                  "attention_mask must be provided for default context attention implementation");
            auto softmax_qk_output = softmax({std::move(qk_output),
                                                      *params.common.attention_mask,
                                                      std::nullopt,
                                                      scale,
                                                      datatype,
                                                      params.common.linear_bias_slopes ?
                                                          (OptionalConstBufferRef)*params.common.linear_bias_slopes :
                                                          std::nullopt});
            softmax_qk_output->updateShape(
                {batch_size, kv_head_num, (head_num / kv_head_num) * seq_len, seq_len_with_prefix});
            printBufferData(*softmax_qk_output, "softmax_qk_output: ");
            auto qkv_output = gemm({*softmax_qk_output, *v_output});
            qkv_output->updateShape({batch_size, head_num, seq_len, size_per_head});
            printBufferData(*qkv_output, "qkv_output");
            auto& qkv_transpose_output = params.output;
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                             invokeTransposeAttentionOutRemovePadding,
                                             qkv_output->data(),
                                             qkv_transpose_output.data(),
                                             token_num,
                                             batch_size,
                                             seq_len,
                                             head_num,
                                             size_per_head,
                                             params.common.padding_offset->data<int>(),
                                             nullptr,
                                             0,
                                             stream);
        }
    }
}
}  // namespace rtp_llm
