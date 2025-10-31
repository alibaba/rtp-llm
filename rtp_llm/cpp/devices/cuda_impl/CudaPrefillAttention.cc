#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/cpp/devices/DeviceData.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"

#ifdef USING_CUDA12
#include "rtp_llm/cpp/devices/cuda_impl/CudaXqa.h"
#endif

namespace rtp_llm {

void CudaDevice::prefillAttention(const AttentionModuleParams& params,
                                  KVBlockArray                 kv_block_array,
                                  const BufferPtr&             q_no_transpose_output,
                                  const BufferPtr&             q_output,
                                  const BufferPtr&             k_output,
                                  const BufferPtr&             v_output,
                                  const BufferPtr&             qkv_buf_fp8) {
    auto      fmha_type           = fmha_type_;
    auto      stream              = stream_;
    auto      cufmha_runner       = cufmha_runner_;
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
        check_cuda_value(cudaMemsetAsync(tiled_counter_ptr->data(), 0, sizeof(uint32_t), stream));
    }
    switch (fmha_type) {
#ifdef USING_CUDA12
        case FMHAType::XQA: {
            RTP_LLM_CHECK_WITH_INFO(q_no_transpose_output != nullptr, "q_no_transpose_output must be provided for xqa");
            runXqa(q_no_transpose_output->data(),
                   q_no_transpose_output->type() == DataType::TYPE_BF16,
                   params.output.data(),
                   head_num,
                   kv_head_num,
                   size_per_head,
                   batch_size,
                   static_cast<size_t>(kv_block_array.mMaxBlocksPerSeq),
                   seq_len_with_prefix,
                   params.configs.tokens_per_block,
                   kv_block_array.mPrimaryPoolPtr,
                   reinterpret_cast<int32_t*>(const_cast<KVCacheIndex*>(kv_block_array.data)),
                   params.common.kv_cache->k_cache_buffer->type() == DataType::TYPE_FP8_E4M3,
                   params.common.kv_seqlens->data<uint32_t>(),
                   this,
                   params.output.type() == DataType::TYPE_FP8_E4M3 ?
                       reinterpret_cast<float*>(params.weights.static_scale_reciprocal_weight->kernel->data()) :
                       nullptr,
                   seq_len,
                   params.common.cu_seqlens->data());
            break;
        }
#endif
        case FMHAType::FLASH_INFER: {
            RTP_LLM_CHECK_WITH_INFO(q_no_transpose_output != nullptr,
                                    "q_no_transpose_output must be provided for flashinfer");
            FlashInferAttnParams* flash_infer = (FlashInferAttnParams*)params.common.prefill_flash_infer_attn.get();
            RTP_LLM_CHECK(flash_infer && flash_infer->plan.numel() > 0);
            BufferPtr f16_out;
            if (use_fp8_fmha_) {
                f16_out =
                    allocateBuffer({params.input.type(), params.output.shape(), AllocationType::DEVICE}, {"f16_out"});
            }
            flash_infer->run(params, q_no_transpose_output, f16_out, reinterpret_cast<int64_t>(stream));
            break;
        }
        case FMHAType::PAGED_TRT_V2: {
            RTP_LLM_CHECK_WITH_INFO(q_output != nullptr, "q_output must be provided for paged trt v2 fmha");
            float* attention_output_orig_quant_scale = nullptr;
            if (params.weights.static_scale_reciprocal_weight && use_fp8_fmha) {
                printBufferData(*(params.weights.static_scale_reciprocal_weight->kernel), "attn scale");
                attention_output_orig_quant_scale =
                    (params.weights.static_scale_reciprocal_weight->kernel->data<float>());
            }
            bool      need_quant_fmha_out = !use_fp8_fmha && params.output.isQBuffer();
            BufferPtr tmp_fmha_output;
            void*     fmha_output_ptr = params.output.data();
            if (need_quant_fmha_out) {
                // for sm89 cannot use fp8_fmha, but attention output should be fp8
                tmp_fmha_output = allocateBuffer(
                    {datatype, {batch_size, head_num * seq_len_with_prefix * size_per_head}, AllocationType::DEVICE},
                    {"fmha_fp16_output"});
                check_cuda_value(cudaMemsetAsync(tmp_fmha_output->data(), 0, tmp_fmha_output->sizeBytes(), stream));
                fmha_output_ptr = tmp_fmha_output->data();
            } else if (use_fp8_fmha && params.output.type() != DataType::TYPE_QFP8_E4M3) {
                tmp_fmha_output = allocateBuffer({DataType::TYPE_FP8_E4M3,
                                                  {batch_size, head_num * seq_len_with_prefix * size_per_head},
                                                  AllocationType::DEVICE},
                                                 {"fmha_fp8_output"});
                fmha_output_ptr = tmp_fmha_output->data();
            }
            cufmha_runner->runTrtV2FmhaPaged(q_output->data(),
                                             params.common.cu_seqlens->data(),
                                             params.common.cu_kv_seqlens->data(),
                                             fmha_output_ptr,
                                             reinterpret_cast<uint32_t*>(tiled_counter_ptr->data()),
                                             attention_output_orig_quant_scale,
                                             batch_size,
                                             seq_len,
                                             seq_len_with_prefix,
                                             token_num,
                                             params.common.context_total_kv_length,
                                             kv_block_array);
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
                check_cuda_value(cudaMemcpyAsync(params.output.data(),
                                                 quant_output->data(),
                                                 params.output.size(),
                                                 cudaMemcpyDeviceToDevice,
                                                 stream));
            } else if (use_fp8_fmha && params.output.type() != DataType::TYPE_QFP8_E4M3) {
                RTP_LLM_CHECK_WITH_INFO(tmp_fmha_output != nullptr, "tmp_fmha_output must be provided for fp8 fmha");
                printBufferData(*tmp_fmha_output, "tmp_fmha_output");
                auto quant_fmha_output_t   = Buffer2torchTensor(*tmp_fmha_output, false);
                auto dequant_fmha_output_t = quant_fmha_output_t.to(dataTypeToTorchType(datatype));
                check_cuda_value(cudaMemcpyAsync(params.output.data(),
                                                 dequant_fmha_output_t.data_ptr(),
                                                 params.output.sizeBytes(),
                                                 cudaMemcpyDeviceToDevice,
                                                 stream));
                printBufferData(params.output, "params.output");
            }
            break;
        }
        case FMHAType::TRT_V2: {
            void* fmha_input_ptr  = use_fp8_fmha ? qkv_buf_fp8->data() : params.input.data();
            void* fmha_output_ptr = params.output.data();
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
                tmp_fmha_output = allocateBuffer(
                    {datatype, {batch_size, head_num * seq_len_with_prefix * size_per_head}, AllocationType::DEVICE},
                    {"fmha_fp16_output"});
                check_cuda_value(cudaMemsetAsync(tmp_fmha_output->data(), 0, tmp_fmha_output->sizeBytes(), stream));
                fmha_output_ptr = tmp_fmha_output->data();
            } else if (use_fp8_fmha && params.output.type() != DataType::TYPE_QFP8_E4M3) {
                tmp_fmha_output = allocateBuffer({DataType::TYPE_FP8_E4M3,
                                                  {batch_size, head_num * seq_len_with_prefix * size_per_head},
                                                  AllocationType::DEVICE},
                                                 {"fmha_fp8_output"});
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
                                        kv_block_array);
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
                check_cuda_value(cudaMemcpyAsync(params.output.data(),
                                                 quant_output->data(),
                                                 params.output.size(),
                                                 cudaMemcpyDeviceToDevice,
                                                 stream));
            } else if (use_fp8_fmha && params.output.type() != DataType::TYPE_QFP8_E4M3) {
                RTP_LLM_CHECK_WITH_INFO(tmp_fmha_output != nullptr, "tmp_fmha_output must be provided for fp8 fmha");
                printBufferData(*tmp_fmha_output, "tmp_fmha_output");
                auto quant_fmha_output_t   = Buffer2torchTensor(*tmp_fmha_output, false);
                auto dequant_fmha_output_t = quant_fmha_output_t.to(dataTypeToTorchType(datatype));
                check_cuda_value(cudaMemcpyAsync(params.output.data(),
                                                 dequant_fmha_output_t.data_ptr(),
                                                 params.output.sizeBytes(),
                                                 cudaMemcpyDeviceToDevice,
                                                 stream));
                printBufferData(params.output, "params.output");
            }
            break;
        }
        case FMHAType::PAGED_OPEN_SOURCE: {
            const size_t max_blocks_per_batch = params.common.kv_cache->kv_cache_block_id->shape()[1];
            const auto   ws_size              = cufmha_runner->getOpenSourceWorkSpaceSize(
                batch_size, seq_len, max_blocks_per_batch * params.configs.tokens_per_block, true);
            auto ws =
                allocateBuffer({DataType::TYPE_INT8, {ws_size}, AllocationType::DEVICE}, {"open_source_paged_fmha_ws"});
            // head_num * seq_size_per_block * size_per_head
            auto kv_offset = params.common.kv_cache->k_cache_buffer->shape()[2]
                             * params.common.kv_cache->k_cache_buffer->shape()[3]
                             * params.common.kv_cache->k_cache_buffer->shape()[4];
            cufmha_runner->runOpenSourceFmhaPaged(
                params.input.data(),
                params.common.kv_cache->k_cache_buffer->data(),
                params.common.kv_cache->k_cache_buffer->dataWithOffset(kv_offset),
                params.output.data(),
                params.common.cu_seqlens->data<int>(),
                params.common.cu_kv_seqlens->data<int>(),
                params.common.kv_cache->kv_cache_block_id->data<int>(),
                batch_size,
                max_blocks_per_batch,
                params.configs.tokens_per_block,
                seq_len,
                ws->data(),
                init_params_,
                params.common.linear_bias_slopes ? params.common.linear_bias_slopes->data<float>() : nullptr,
                params.configs.softmax_extra_scale);
            break;
        }
        case FMHAType::OPEN_SOURCE: {
            const auto ws_size = cufmha_runner->getOpenSourceWorkSpaceSize(batch_size, seq_len);
            auto ws = allocateBuffer({DataType::TYPE_INT8, {ws_size}, AllocationType::DEVICE}, {"open_source_fmha_ws"});
            const size_t hidden_units    = head_num * size_per_head;
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
                init_params_,
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
            RTP_LLM_CHECK_WITH_INFO(
                q_output && k_output && v_output,
                "q_output/k_output/v_output must be provided for default context attention implementation");
            q_output->updateShape({batch_size, kv_head_num, (head_num / kv_head_num) * seq_len, size_per_head});
            auto qk_output = gemm({*q_output,
                                   *k_output,
                                   std::nullopt,
                                   nullptr,
                                   DataType::TYPE_FP32,
                                   params.compute_type,
                                   TransposeOperation::NONE,
                                   TransposeOperation::TRANSPOSE});
            qk_output->updateShape({batch_size, head_num, seq_len, seq_len_with_prefix});
            printBufferData(*qk_output, "qk_output: ");
            float scale = (1.0f / sqrtf(size_per_head * 1.0f)) * params.configs.softmax_extra_scale;
            // TODO(lidongjin): Only support float32(in)\float16(output).
            RUNTIME_ASSERT_OP_ARG(params.common.attention_mask,
                                  "attention_mask must be provided for default context attention implementation");
            auto softmax_qk_output =
                softmax({std::move(qk_output),
                         *params.common.attention_mask,
                         std::nullopt,
                         scale,
                         datatype,
                         params.common.linear_bias_slopes ? (OptionalConstBufferRef)*params.common.linear_bias_slopes :
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
