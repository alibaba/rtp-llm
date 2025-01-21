
#include <iostream>
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/utils/compiler_config.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/kernels/kv_cache_utils.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "maga_transformer/cpp/utils/RpcErrorCode.h"
#if USING_CUDA12 == 1
#include "3rdparty/flashinfer/flashinfer.h"
#endif

using namespace std;
using namespace rtp_llm;

namespace fastertransformer {

void flashInferAttnParamsDeleter(void* p) {
    delete (FlashInferAttnParams*)p;
}

FlashInferAttnParamsPtr FlashInferAttnParams::prepareFlashInferAttnParams(
        fastertransformer::DeviceBase *device,
        const fastertransformer::AttentionConfigs &attn_configs,
        const BufferPtr &sequence_lengths_host,
        const BufferPtr &kv_cache_block_id_host,
        fastertransformer::DataType dtype)
{
    auto ret = FlashInferAttnParamsPtr(new FlashInferAttnParams, flashInferAttnParamsDeleter);
    auto params = (FlashInferAttnParams*)ret.get();

#if USING_CUDA12 == 1
    auto cuda_device = dynamic_cast<CudaDevice*>(device);
    const size_t batch_size = sequence_lengths_host->shape()[0];
    const int local_head_num = attn_configs.head_num;
    const int local_head_num_kv = attn_configs.kv_head_num;
    const int size_per_head = attn_configs.size_per_head;
    const int group_size = local_head_num / local_head_num_kv;

    if (!cuda_device ||
        (dtype != DataType::TYPE_FP16 && dtype != DataType::TYPE_BF16) ||
        0 == batch_size ||
        !kv_cache_block_id_host ||
        attn_configs.kv_cache_dtype != KvCacheDataType::BASE ||
        attn_configs.rope_config.style != RopeStyle::Base ||
        attn_configs.mask_type != causalMask ||
        attn_configs.q_scaling != 1.0f ||
        attn_configs.softmax_extra_scale != 1.0f ||
        attn_configs.use_mla ||
        attn_configs.use_logn_attn ||
        (size_per_head != 64 && size_per_head != 128) ||
        group_size > 10)
    {
        return ret;
    }

    if (group_size > 5) {
        params->decode = false;
    } else {
        params->decode = true;
    }

    const int tokens_per_block = attn_configs.tokens_per_block;
    const int max_batch_blocks = kv_cache_block_id_host->shape()[1];

    params->float_workspace = device->allocateBuffer({DataType::TYPE_INT8, {128 * 1024 *1024}, AllocationType::DEVICE}, {"float_workspace"});
    params->int_workspace = device->allocateBuffer({DataType::TYPE_INT8, {8 * 1024 *1024}, AllocationType::DEVICE}, {"int_workspace"});
    params->int_host_workspace = device->allocateBuffer({DataType::TYPE_INT8, {8 *1024 *1024}, AllocationType::HOST}, {"int_host_workspace"});
    params->page_indptr_host = device->allocateBuffer({DataType::TYPE_INT32, {batch_size + 1}, AllocationType::HOST}, {"page_indptr_host"});
    params->qo_indptr_host = device->allocateBuffer({DataType::TYPE_INT32, {batch_size + 1}, AllocationType::HOST}, {"qo_indptr_host"});

    auto batch_indice_host = device->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"batch_indice_host"});
    auto positions_host = device->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"positions_host"});
    auto paged_kv_last_page_len = device->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"paged_kv_last_page_len"});
    auto paged_kv_last_page_len_1 = device->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"paged_kv_last_page_len_1"});

    vector<int> page_indice_vec;
    params->qo_indptr_host->data<int>()[0] = 0;
    params->page_indptr_host->data<int>()[0] = 0;
    for (int i = 0; i < int(batch_size); i++) {
        batch_indice_host->data<int>()[i] = i;
        paged_kv_last_page_len->data<int>()[i] = sequence_lengths_host->data<int>()[i] %  tokens_per_block;
        paged_kv_last_page_len_1->data<int>()[i] = paged_kv_last_page_len->data<int>()[i] + 1;
        positions_host->data<int>()[i] = sequence_lengths_host->data<int>()[i];

        int page_nums = (sequence_lengths_host->data<int>()[i] + tokens_per_block) / tokens_per_block;
        for (int j = 0; j < page_nums; j++) {
            auto page_idx = kv_cache_block_id_host->data<int>()[i * max_batch_blocks + j];
            page_indice_vec.push_back(page_idx);
        }
        params->page_indptr_host->data<int>()[i + 1] = int(page_indice_vec.size());
        params->qo_indptr_host->data<int>()[i + 1] = i + 1;
    }

    auto page_indice_host = device->allocateBuffer({DataType::TYPE_INT32, {size_t(page_indice_vec.size())}, AllocationType::HOST}, {"page_indice_host"});
    std::copy(page_indice_vec.begin(), page_indice_vec.end(), page_indice_host->data<int>());

    params->batch_indice = device->clone({*batch_indice_host, AllocationType::DEVICE});
    params->positions = device->clone({*positions_host, AllocationType::DEVICE});
    params->paged_kv_last_page_len = device->clone({*paged_kv_last_page_len, AllocationType::DEVICE});
    params->paged_kv_last_page_len_1 = device->clone({*paged_kv_last_page_len_1, AllocationType::DEVICE});
    params->page_indptr = device->clone({*params->page_indptr_host, AllocationType::DEVICE});
    params->qo_indptr = device->clone({*params->qo_indptr_host, AllocationType::DEVICE});
    params->page_indice = device->clone({*page_indice_host, AllocationType::DEVICE});

    params->float_workspace_t = Buffer2torchTensor(params->float_workspace, false);
    params->int_workspace_t = Buffer2torchTensor(params->int_workspace, false);
    params->int_host_workspace_t = Buffer2torchTensor(params->int_host_workspace, false);

    params->batch_indice_t = Buffer2torchTensor(params->batch_indice, false);
    params->positions_t = Buffer2torchTensor(params->positions, false);
    params->paged_kv_last_page_len_t = Buffer2torchTensor(params->paged_kv_last_page_len, false);
    params->paged_kv_last_page_len_1_t = Buffer2torchTensor(params->paged_kv_last_page_len_1, false);

    params->qo_indptr_t = Buffer2torchTensor(params->qo_indptr, false);
    params->qo_indptr_host_t = Buffer2torchTensor(params->qo_indptr_host, false);
    params->page_indptr_t = Buffer2torchTensor(params->page_indptr, false);
    params->page_indptr_host_t = Buffer2torchTensor(params->page_indptr_host, false);
    params->page_indice_t = Buffer2torchTensor(params->page_indice, false);

    if (params->decode) {
        params->plan = BatchDecodeWithPagedKVCachePlan(
                false,
                size_per_head,
                torch::empty(0, dataTypeToTorchType(dtype)),
                torch::empty(0, dataTypeToTorchType(dtype)),
                params->float_workspace_t,
                params->int_workspace_t,
                params->int_host_workspace_t,
                params->page_indptr_host_t,
                batch_size,
                local_head_num,
                local_head_num_kv,
                tokens_per_block,
                false,
                reinterpret_cast<int64_t>(cuda_device->getStream()));
    } else {
        params->plan = BatchPrefillWithKVCachePlan(
                size_per_head,
                params->float_workspace_t,
                params->int_workspace_t,
                params->int_host_workspace_t,
                params->qo_indptr_host_t,
                params->page_indptr_host_t,
                batch_size,
                batch_size,
                local_head_num,
                local_head_num_kv,
                tokens_per_block,
                false,
                reinterpret_cast<int64_t>(cuda_device->getStream()));
    }
#endif

    return ret;
}

KVBlockArray getKVBlockArray(const AttentionModuleParams& params,
                             const Buffer&                kv_cache_offset_pointers,
                             int                          batch_size,
                             bool                         use_fp8_fmha,
                             cudaStream_t                 stream) {
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
    // FT_LOG_INFO("kv_cache[0].typeSize():%d", kv_cache[0].typeSize());
    FT_LOG_DEBUG(
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
                     (fastertransformer::KVBlockArrayForContextFMHA::DataType*)kv_cache_offset_pointers.data());
    invokeConvertOffsetToBlockArrayData((int32_t*)kv_cache_offset_pointers.data(),
                                        (int*)kv_blocks_offset.data(),
                                        batch_size,
                                        max_blocks_per_batch,
                                        kv_block_offset,
                                        stream);
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
        FT_LOG_DEBUG("now use kv_cache int8");
        cache_type = KvCacheDataType::INT8;
    }
    kv_cache_buffer.cache_type = cache_type;
    sync_check_cuda_error();
    return kv_cache_buffer;
}

void MHA(const AttentionModuleParams& params,
         FMHAType                     fmha_type,
         cufmha*                      cufmha_runner,
         KVBlockArray                 kv_block_array,
         const BufferPtr&             q_output,
         const BufferPtr&             k_output,
         const BufferPtr&             v_output,
         const BufferPtr&             qkv_buf_fp8,
         cudaStream_t                 stream,
         CudaDevice*                  device) {
    FT_LOG_DEBUG("FMHA Type use %s.", std::to_string((int)fmha_type).c_str());
    auto datatype            = params.input.type();
    auto token_num           = params.input.shape()[0];
    auto batch_size          = params.common.context_batch_size;
    auto seq_len             = params.common.context_max_seq_len;
    auto seq_len_with_prefix = seq_len + params.common.max_prefix_length;
    auto head_num            = params.configs.head_num;
    auto kv_head_num         = params.configs.kv_head_num;
    auto size_per_head       = params.configs.size_per_head;
    bool use_fp8_fmha        = qkv_buf_fp8 != nullptr;
    BufferPtr tiled_counter_ptr;
    if (FMHAType::PAGED_TRT_V2 == fmha_type || FMHAType::TRT_V2 == fmha_type) {
        tiled_counter_ptr =
            device->allocateBuffer({DataType::TYPE_UINT32, {1}, AllocationType::DEVICE}, {"tiled_counter_pointer"});
        cudaMemsetAsync(tiled_counter_ptr->data(), 0, sizeof(uint32_t), stream);
    }
    switch (fmha_type) {
        case FMHAType::PAGED_TRT_V2: {
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
            void  *fmha_input_ptr = use_fp8_fmha ? qkv_buf_fp8->data() : params.input.data();
            void  *fmha_output_ptr = params.output.data();
            float *attention_output_orig_quant_scale = nullptr;
            if (params.weights.static_scale_reciprocal_weight && use_fp8_fmha) {
                printBufferData(*(params.weights.static_scale_reciprocal_weight->kernel), "attn scale");
                attention_output_orig_quant_scale = (params.weights.static_scale_reciprocal_weight->kernel->data<float>());
            }
            bool need_quant_fmha_out = !use_fp8_fmha && params.output.isQBuffer();
            BufferPtr tmp_fmha_output;
            if (need_quant_fmha_out) {
                // for sm89 cannot use fp8_fmha, but attention output should be fp8
                tmp_fmha_output = device->allocateBuffer({DataType::TYPE_FP16,
                                                          {batch_size, head_num * seq_len_with_prefix * size_per_head},
                                                          AllocationType::DEVICE},
                                                         {"fmha_fp16_output"});
                cudaMemsetAsync(tmp_fmha_output->data(), 0, tmp_fmha_output->sizeBytes(), stream);
                fmha_output_ptr = tmp_fmha_output->data();
            }
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
                auto quant_params = QuantizeParams(*tmp_fmha_output,
                                                   quant_out_data_type,
                                                   1,
                                                   QScheme::Qfp8PerTensor,
                                                   std::nullopt,
                                                   std::nullopt,
                                                   (OptionalConstBufferRef)*params.weights.static_quant_weight->kernel,
                                                   (OptionalConstBufferRef)*params.weights.static_scale_reciprocal_weight->kernel);
                auto quant_output = device->quantize(quant_params);
                cudaMemcpyAsync(params.output.data(), quant_output->data(), params.output.size(), cudaMemcpyDeviceToDevice, stream);
            }
            break;
        }
        case FMHAType::PAGED_OPEN_SOURCE: {
            const size_t max_blocks_per_batch = params.common.kv_cache->kv_cache_block_id->shape()[1];
            const auto   ws_size              = cufmha_runner->getOpenSourceWorkSpaceSize(
                batch_size, seq_len, max_blocks_per_batch * params.configs.tokens_per_block, true);
            auto ws = device->allocateBuffer({DataType::TYPE_INT8, {ws_size}, AllocationType::DEVICE},
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
            auto         ws           = device->allocateBuffer({DataType::TYPE_INT8, {ws_size}, AllocationType::DEVICE},
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
            auto qkv_buf_temp = device->allocateBuffer(
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
            q_output->updateShape({batch_size, kv_head_num, (head_num / kv_head_num) * seq_len, size_per_head});
            auto qk_output = device->gemm({*q_output,
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
            auto softmax_qk_output = device->softmax({std::move(qk_output),
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
            auto qkv_output = device->gemm({*softmax_qk_output, *v_output});
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

        kv_block_array = getKVBlockArray(params, *kv_cache_block_id, batch_size, use_fp8_fmha_, stream_);

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

    // int8
    float* scale_out_ptr = nullptr;
    int    int8_mode     = 0;

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
                                         scale_out_ptr,
                                         int8_mode,
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
            scale_out_ptr,
            int8_mode,
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

        writeCacheStore(params);

        // printBufferData(params.input, "after invoke transpse");
        printBufferData(*q_output, "Q after invoke transpose");
        printBufferData(*k_output, "K after invoke transpose");
        printBufferData(*v_output, "V after invoke transpose");
    }

    MHA(params, fmha_type_, cufmha_runner_.get(), kv_block_array, q_output, k_output, v_output, qkv_buf_fp8, stream_, this);
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
    void* qkv_buf_2_ = nullptr;
    BufferPtr fp8_qkv_buf;
    if (use_fp8_fmha) {
        fp8_qkv_buf = device->allocateBuffer({DataType::TYPE_FP16, output.shape(), AllocationType::DEVICE}, {"fp8_qkv_buf"});
        qkv_buf_2_ = fp8_qkv_buf->data();
    } else {
        qkv_buf_2_ = output.data();
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

    // TODO(lidongjin) support int8
    const float* query_weight_scale_out            = nullptr;
    const float* attention_output_orig_quant_scale = nullptr;

    if (params.weights.static_scale_reciprocal_weight) {
        attention_output_orig_quant_scale = reinterpret_cast<float*>(params.weights.static_scale_reciprocal_weight->kernel->data());
    }

    int int8_mode = 0;
    tensorrt_llm::common::QuantMode kv_cache_quant_mode = trt_common::QuantMode::fromDescription(false, false, false, false, false, false, false, false);
    if (params.configs.kv_cache_dtype == KvCacheDataType::INT8) {
        kv_cache_quant_mode = trt_common::QuantMode::fromDescription(true, true, false, false, false, true, false, true);
    } else if (params.configs.kv_cache_dtype == KvCacheDataType::FP8 && use_fp8_fmha) {
        kv_cache_quant_mode = trt_common::QuantMode::fromDescription(true, true, false, false, false, false, true, true);
    }

#if USING_CUDA12 == 1
    auto flash_infer_attn_params = (FlashInferAttnParams*)params.common.flash_infer_attn_params.get();
    if (flash_infer_attn_params && !flash_infer_attn_params->plan.empty()) {
        const auto &flashinfer = *flash_infer_attn_params;
        at::Tensor qkv_input = Buffer2torchTensor(params.input, false);
        if (params.weights.qkv_weight->bias) {
            qkv_input = at::add(Buffer2torchTensor(params.weights.qkv_weight->bias, false), qkv_input);
        }
        qkv_input = qkv_input.reshape({-1, int(local_head_num + local_head_num_kv * 2), int(size_per_head)});
        auto q = qkv_input.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None, local_head_num), torch::indexing::Slice(torch::indexing::None)});
        auto append_k = qkv_input.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(local_head_num, local_head_num + local_head_num_kv), torch::indexing::Slice(torch::indexing::None)});
        auto append_v = qkv_input.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(local_head_num + local_head_num_kv, torch::indexing::None), torch::indexing::Slice(torch::indexing::None)});
        auto k_cache = Buffer2torchTensor(params.common.kv_cache->k_cache_buffer, false);
        auto v_cache = Buffer2torchTensor(params.common.kv_cache->v_cache_buffer, false);

        apply_rope_pos_ids(q,
                           append_k,
                           q,
                           append_k,
                           Buffer2torchTensor(params.common.sequence_lengths, false),
                           params.configs.rope_config.dim,
                           false,
                           params.configs.rope_config.scale,
                           params.configs.rope_config.base,
                           reinterpret_cast<int64_t>(stream));
        sync_check_cuda_error();

        append_paged_kv_cache(append_k,
                              append_v,
                              flashinfer.batch_indice_t,
                              flashinfer.positions_t,
                              k_cache,
                              v_cache,
                              flashinfer.page_indice_t,
                              flashinfer.page_indptr_t,
                              flashinfer.paged_kv_last_page_len_t,
                              1, reinterpret_cast<int64_t>(stream));
        sync_check_cuda_error();

        if (flashinfer.decode) {
            BatchDecodeWithPagedKVCacheRun(
                    flashinfer.float_workspace_t,
                    flashinfer.int_workspace_t,
                    flashinfer.plan,
                    q,
                    k_cache,
                    v_cache,
                    flashinfer.page_indptr_t,
                    flashinfer.page_indice_t,
                    flashinfer.paged_kv_last_page_len_1_t,
                    std::nullopt,
                    Buffer2torchTensor(params.output, false),
                    1,
                    -1,
                    0,
                    (1.0f / sqrtf(size_per_head * 1.0f)),
                    0,
                    0,
                    std::nullopt,
                    reinterpret_cast<int64_t>(stream));
        } else {
            BatchPrefillWithPagedKVCacheRun(
                    1,
                    flashinfer.float_workspace_t,
                    flashinfer.int_workspace_t,
                    flashinfer.plan,
                    q,
                    k_cache,
                    v_cache,
                    std::nullopt,
                    std::nullopt,
                    flashinfer.qo_indptr_t,
                    flashinfer.page_indptr_t,
                    flashinfer.page_indice_t,
                    flashinfer.paged_kv_last_page_len_1_t,
                    std::nullopt,
                    Buffer2torchTensor(params.output, false),
                    1,
                    -1,
                    0,
                    (1.0f / sqrtf(size_per_head * 1.0f)),
                    params.configs.rope_config.scale,
                    params.configs.rope_config.base,
                    std::nullopt,
                    reinterpret_cast<int64_t>(stream));
        }

        sync_check_cuda_error();
    } else {
#else
    {
#endif
        fusedQKV_masked_attention_dispatch<T, KVBlockArray>(
                qkv_buf_ptr,
                bias_ptr,
                relative_attention_bias_ptr,
                nullptr, // cache_indir
                reinterpret_cast<T*>(qkv_buf_2_),
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
                query_weight_scale_out,
                attention_output_orig_quant_scale,
                int8_mode,
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
    }
    sync_check_cuda_error();

    if (fp8_qkv_buf) {
        cudaMemcpyAsync(output.data(), fp8_qkv_buf->data(), output.size(), cudaMemcpyDeviceToDevice, stream);
    }
    sync_check_cuda_error();
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
    KVBlockArray kv_block_array = getKVBlockArray(params, *kv_cache_block_id, batch_size, use_fp8_fmha_, stream_);
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
