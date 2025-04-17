
#include <iostream>
#include <numeric>
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
#include "src/fastertransformer/kernels/kv_cache/kv_cache_utils.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "maga_transformer/cpp/utils/RpcErrorCode.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include "flashmla/flashmla.h"

using namespace std;
using namespace rtp_llm;

namespace fastertransformer {

void flashInferAttnParamsDeleter(void* p) {
    delete (FlashInferAttnParams*)p;
}

void prepareDecodeFlashInferAttnParamsImpl(FlashInferAttnParams* params,
                                    fastertransformer::DeviceBase *device,
                                    const fastertransformer::AttentionConfigs &attn_configs,
                                    const BufferPtr &sequence_lengths_host,
                                    const BufferPtr &kv_cache_block_id_host,
                                    const uint64_t batch_size,
                                    const uint64_t tokens_per_block,
                                    const uint64_t max_batch_blocks){
    FT_CHECK_WITH_INFO(max_batch_blocks > 0 && kv_cache_block_id_host, "max_batch_blocks and kv_cache_block_id_host must be set for decode");
    params->mla_ops_type = device->mla_ops_type;
    params->float_workspace = device->allocateBuffer({DataType::TYPE_INT8, {128 * 1024 *1024}, AllocationType::DEVICE}, {"float_workspace"});
    params->int_workspace = device->allocateBuffer({DataType::TYPE_INT8, {8 * 1024 *1024}, AllocationType::DEVICE}, {"int_workspace"});
    params->int_host_workspace = device->allocateBuffer({DataType::TYPE_INT8, {8 *1024 *1024}, AllocationType::HOST}, {"int_host_workspace"});
    params->page_indptr_host = device->allocateBuffer({DataType::TYPE_INT32, {batch_size + 1}, AllocationType::HOST}, {"page_indptr_host"});
    params->qo_indptr_host = device->allocateBuffer({DataType::TYPE_INT32, {batch_size + 1}, AllocationType::HOST}, {"qo_indptr_host"});

    params->batch_indice_host = device->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"batch_indice_host"});
    params->positions_host = device->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"positions_host"});
    params->kvlen_host = device->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"kvlen_host"});
    params->paged_kv_last_page_len_host = device->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"paged_kv_last_page_len_host"});
    params->paged_kv_last_page_len_1_host = device->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"paged_kv_last_page_len_1_host"});

    vector<int> page_indice_vec;
    params->qo_indptr_host->data<int>()[0] = 0;
    params->page_indptr_host->data<int>()[0] = 0;
    for (int i = 0; i < int(batch_size); i++) {
        params->batch_indice_host->data<int>()[i] = i;
        params->paged_kv_last_page_len_host->data<int>()[i] = sequence_lengths_host->data<int>()[i] %  tokens_per_block;
        params->paged_kv_last_page_len_1_host->data<int>()[i] = params->paged_kv_last_page_len_host->data<int>()[i] + 1;
        params->positions_host->data<int>()[i] = sequence_lengths_host->data<int>()[i];
        params->kvlen_host->data<int>()[i] = sequence_lengths_host->data<int>()[i] + 1;
        // sequence_length_host here is the index of the last token in the sequence, equals to length - 1
        int page_nums = (sequence_lengths_host->data<int>()[i] + tokens_per_block) / tokens_per_block;
        for (int j = 0; j < page_nums; j++) {
            auto page_idx = kv_cache_block_id_host->data<int>()[i * max_batch_blocks + j];
            page_indice_vec.push_back(page_idx);
        }
        params->page_indptr_host->data<int>()[i + 1] = int(page_indice_vec.size());
        params->qo_indptr_host->data<int>()[i + 1] = i + 1;
    }

    params->page_indice_host = device->allocateBuffer({DataType::TYPE_INT32, {size_t(page_indice_vec.size())}, AllocationType::HOST}, {"page_indice_host"});
    std::copy(page_indice_vec.begin(), page_indice_vec.end(), params->page_indice_host->data<int>());

    params->kv_cache_block_id   = device->clone({*kv_cache_block_id_host, AllocationType::DEVICE});
    params->batch_indice = device->clone({*params->batch_indice_host, AllocationType::DEVICE});
    params->positions = device->clone({*params->positions_host, AllocationType::DEVICE});
    params->paged_kv_last_page_len = device->clone({*params->paged_kv_last_page_len_host, AllocationType::DEVICE});
    params->paged_kv_last_page_len_1 = device->clone({*params->paged_kv_last_page_len_1_host, AllocationType::DEVICE});
    params->page_indptr = device->clone({*params->page_indptr_host, AllocationType::DEVICE});
    params->qo_indptr = device->clone({*params->qo_indptr_host, AllocationType::DEVICE});
    params->page_indice = device->clone({*params->page_indice_host, AllocationType::DEVICE});
    params->kvlen   = device->clone({*params->kvlen_host, AllocationType::DEVICE});

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
    params->kvlen_host_t = Buffer2torchTensor(params->kvlen_host, false);
    params->kvlen_t = Buffer2torchTensor(params->kvlen, false);
    params->kv_cache_block_id_t = Buffer2torchTensor(params->kv_cache_block_id, false);
}

// for mla, we need to prepare additional params for write kvcache and de rotary embedding
void prepareContextMLAFlashInferAttnParamsImpl(FlashInferAttnParams*                      params,
                                    fastertransformer::DeviceBase*             device,
                                    const fastertransformer::AttentionConfigs& attn_configs,
                                    const BufferPtr&                           prefix_lengths_host,
                                    const BufferPtr&                           sequence_lengths_host,
                                    const BufferPtr&                           input_lengths_host,
                                    const BufferPtr&                           kv_cache_block_id_host,
                                    const uint64_t                             prefill_token_num,
                                    const uint64_t                             context_batch_size,
                                    const uint64_t                             tokens_per_block,
                                    const uint64_t                             max_batch_blocks,
                                    const uint64_t                             batch_size) {
    params->mla_ops_type = device->mla_ops_type;
    params->float_workspace = device->allocateBuffer({DataType::TYPE_INT8, {128 * 1024 *1024}, AllocationType::DEVICE}, {"float_workspace"});
    params->int_workspace = device->allocateBuffer({DataType::TYPE_INT8, {8 * 1024 *1024}, AllocationType::DEVICE}, {"int_workspace"});
    params->int_host_workspace = device->allocateBuffer({DataType::TYPE_INT8, {8 *1024 *1024}, AllocationType::HOST}, {"int_host_workspace"});
    params->page_indptr_host = device->allocateBuffer({DataType::TYPE_INT32, {context_batch_size + 1}, AllocationType::HOST}, {"prefill_page_indptr_host"});
    params->qo_indptr_host = device->allocateBuffer({DataType::TYPE_INT32, {context_batch_size + 1}, AllocationType::HOST}, {"qo_indptr_host"});
    
    params->batch_indice_host = device->allocateBuffer({DataType::TYPE_INT32, {prefill_token_num}, AllocationType::HOST}, {"prefill_batch_indices_host"});
    params->positions_host = device->allocateBuffer({DataType::TYPE_INT32, {prefill_token_num}, AllocationType::HOST}, {"prefill_positions_host"});
    params->kvlen_host = device->allocateBuffer({DataType::TYPE_INT32, {context_batch_size}, AllocationType::HOST}, {"kvlen_host"});
    params->paged_kv_last_page_len_1_host = device->allocateBuffer({DataType::TYPE_INT32, {context_batch_size}, AllocationType::HOST}, {"prefill_kv_last_page_len_1_host"});
    
    
    params->qo_indptr_host->data<int>()[0] = 0;
    params->page_indptr_host->data<int>()[0] = 0;
    std::vector<int> prefill_page_indices_vec;

    int offset = 0;
    int accu_q_length = 0;
    int last_q_length = -1;
    bool same_q_length = true;
    for (int i = 0; i < context_batch_size; i++) {
        int input_length = input_lengths_host->data<int>()[i + batch_size];
        if (last_q_length > 0 && last_q_length != input_length) {
            same_q_length = false;
        }
        last_q_length = input_length;
        int prefix_length = prefix_lengths_host->data<int>()[i];
        FT_LOG_DEBUG("[%d] input_length: %d, prefix_length: %d", i, input_length, prefix_length);
        for (int j = 0; j < input_length; j++) {
            params->batch_indice_host->data<int>()[offset] = i;
            params->positions_host->data<int>()[offset]    = j + prefix_length;
            offset += 1;
        }
        if (kv_cache_block_id_host) {
            params->paged_kv_last_page_len_1_host->data<int>()[i] = (input_length - 1) % tokens_per_block + 1;
            params->kvlen_host->data<int>()[i]                    = prefix_length + input_length;
  
            int page_nums = ((input_length + prefix_length) + tokens_per_block - 1) / tokens_per_block;
            for (int j = 0; j < page_nums; j++) {
                auto page_idx = kv_cache_block_id_host->data<int>()[(i + batch_size) * max_batch_blocks + j];
                prefill_page_indices_vec.push_back(page_idx);
            }
            params->page_indptr_host->data<int>()[i + 1]          = int(prefill_page_indices_vec.size());
            accu_q_length                                         += input_length;
            params->qo_indptr_host->data<int>()[i + 1]            = accu_q_length;
         
        }
    }
    if (!same_q_length && params->mla_ops_type == FLASH_MLA) {
        FT_LOG_DEBUG("[WARNING] FLASH MLA only suport same q length, fallback to flashinfer");
        // FLASH MLA only suport same q length, fallback to flashinfer
        params->mla_ops_type = FLASH_INFER;
    }

    if (kv_cache_block_id_host) {
        params->page_indice_host =
            device->allocateBuffer({DataType::TYPE_INT32, {size_t(prefill_page_indices_vec.size())}, AllocationType::HOST}, 
                                   {"prefill_page_indices_host"});
        std::copy(
            prefill_page_indices_vec.begin(), prefill_page_indices_vec.end(), params->page_indice_host->data<int>());

        params->kv_cache_block_id   = device->clone({*kv_cache_block_id_host, AllocationType::DEVICE});
        params->paged_kv_last_page_len_1 = device->clone({*params->paged_kv_last_page_len_1_host, AllocationType::DEVICE});
        params->qo_indptr = device->clone({*params->qo_indptr_host, AllocationType::DEVICE});
        params->page_indice       = device->clone({*params->page_indice_host, AllocationType::DEVICE});
        params->kvlen   = device->clone({*params->kvlen_host, AllocationType::DEVICE});
        params->page_indptr = device->clone({*params->page_indptr_host, AllocationType::DEVICE});

        // printBufferData(*params->paged_kv_last_page_len_1, "metadata paged_kv_last_page_len_1");
        // printBufferData(*params->qo_indptr, "metadata qo_indptr");
        // printBufferData(*params->page_indice, "metadata page_indice");
        // printBufferData(*params->kvlen, "metadata kvlen");
        // printBufferData(*params->page_indptr, "metadata page_indptr");
        // printBufferData(*kv_cache_block_id_host, "kv_cache_block_id_host");
    }


    params->batch_indice      = device->clone({*params->batch_indice_host, AllocationType::DEVICE});
    params->positions          = device->clone({*params->positions_host, AllocationType::DEVICE});

    params->float_workspace_t = Buffer2torchTensor(params->float_workspace, false);
    params->int_workspace_t = Buffer2torchTensor(params->int_workspace, false);
    params->int_host_workspace_t = Buffer2torchTensor(params->int_host_workspace, false);
   
    params->batch_indice_t      = Buffer2torchTensor(params->batch_indice, false);
    params->positions_t          = Buffer2torchTensor(params->positions, false);

    if (kv_cache_block_id_host) {
        params->page_indptr_t        = Buffer2torchTensor(params->page_indptr, false);
        params->paged_kv_last_page_len_1_t = Buffer2torchTensor(params->paged_kv_last_page_len_1, false);
        params->kvlen_t = Buffer2torchTensor(params->kvlen, false);
        params->kvlen_host_t = Buffer2torchTensor(params->kvlen_host, false);
        params->kv_cache_block_id_t = Buffer2torchTensor(params->kv_cache_block_id, false);
        params->qo_indptr_host_t = Buffer2torchTensor(params->qo_indptr_host, false);
        params->page_indptr_host_t = Buffer2torchTensor(params->page_indptr_host, false);
        params->page_indice_t       = Buffer2torchTensor(params->page_indice, false);
    }
}


FlashInferAttnParamsPtr FlashInferAttnParams::preparePrefillFlashInferAttnParams(
        fastertransformer::DeviceBase *device,
        const fastertransformer::AttentionConfigs &attn_configs,
        const BufferPtr &prefix_lengths_host,
        const BufferPtr &sequence_lengths_host,
        const BufferPtr &input_lengths_host,
        const BufferPtr &kv_cache_block_id_host,
        fastertransformer::DataType dtype)
{    
    auto cuda_device = dynamic_cast<CudaDevice*>(device);
    const size_t batch_size         = sequence_lengths_host->shape()[0];
    const int local_head_num        = attn_configs.head_num;
    const size_t context_batch_size = input_lengths_host->shape()[0] - batch_size;
    if (context_batch_size == 0) {
        return nullptr;
    }

    const int tokens_per_block = attn_configs.tokens_per_block;

    const int    max_batch_blocks   = kv_cache_block_id_host ? kv_cache_block_id_host->shape()[1] : -1;
    const size_t prefill_token_num    = std::accumulate(input_lengths_host->data<int>() + batch_size,
                                                     input_lengths_host->data<int>() + context_batch_size + batch_size,
                                                     0);
    auto ret = FlashInferAttnParamsPtr(new FlashInferAttnParams, flashInferAttnParamsDeleter);
    auto params = (FlashInferAttnParams*)ret.get();
    prepareContextMLAFlashInferAttnParamsImpl(params, device, attn_configs, prefix_lengths_host, sequence_lengths_host, input_lengths_host, kv_cache_block_id_host, prefill_token_num, context_batch_size, tokens_per_block, max_batch_blocks, batch_size);
    if (kv_cache_block_id_host) {
        if (attn_configs.use_mla && params->mla_ops_type == MlaOpsType::FLASH_INFER) {
            params->plan = BatchMLAPagedAttentionPlan(
                params->float_workspace_t,
                params->int_workspace_t,
                params->int_host_workspace_t,
                params->qo_indptr_host_t,
                params->page_indptr_host_t,
                params->kvlen_host_t,
                local_head_num,
                attn_configs.kv_lora_rank,
                true,
                reinterpret_cast<int64_t>(cuda_device->getStream())
            );
        } else if (attn_configs.use_mla && params->mla_ops_type == MlaOpsType::FLASH_MLA) {
            printBufferData(*torchTensor2Buffer(params->kvlen_t), "metadata kvlen_t");
            FT_LOG_TRACE("batch_size = %zu", batch_size);
            FT_LOG_TRACE("local_head_num = %zu", local_head_num);
            params->flash_mla_plan = get_mla_metadata(
                params->kvlen_t,
                local_head_num,
                1
            );
        }
    }
    return ret;
}


FlashInferAttnParamsPtr FlashInferAttnParams::prepareDecodeFlashInferAttnParams(
        fastertransformer::DeviceBase *device,
        const fastertransformer::AttentionConfigs &attn_configs,
        const BufferPtr &sequence_lengths_host,
        const BufferPtr &input_lengths_host,
        const BufferPtr &kv_cache_block_id_host,
        fastertransformer::DataType dtype)
{
    const char* disable_flash_infer_env = getenv("DISABLE_FLASH_INFER");
    if (fastertransformer::get_sm() < 80 || (disable_flash_infer_env && strcmp(disable_flash_infer_env, "1") == 0)) {
        return nullptr;
    }

    const size_t batch_size         = sequence_lengths_host->shape()[0];
    if (batch_size == 0) {
        return nullptr;
    }

    auto cuda_device = dynamic_cast<CudaDevice*>(device);
    const int local_head_num    = attn_configs.head_num;
    const int local_head_num_kv = attn_configs.kv_head_num;
    const int size_per_head = attn_configs.size_per_head;
    const int group_size = local_head_num / local_head_num_kv;
    const int tokens_per_block = attn_configs.tokens_per_block;

    if (!cuda_device ||
        (dtype != DataType::TYPE_FP16 && dtype != DataType::TYPE_BF16 && dtype != DataType::TYPE_FP8_E4M3) ||
        attn_configs.kv_cache_dtype != KvCacheDataType::BASE ||
        (attn_configs.rope_config.style != RopeStyle::Base && attn_configs.rope_config.style != RopeStyle::No)  ||
        attn_configs.mask_type != causalMask ||
        attn_configs.q_scaling != 1.0f ||
        attn_configs.use_logn_attn ||
        (size_per_head != 64 && size_per_head != 128 && size_per_head != 192) ||
        (group_size > 10 && group_size != 16))
    {
        return nullptr;
    }

    const int    max_batch_blocks   = kv_cache_block_id_host ? kv_cache_block_id_host->shape()[1] : -1;
    auto ret = FlashInferAttnParamsPtr(new FlashInferAttnParams, flashInferAttnParamsDeleter);
    auto params = (FlashInferAttnParams*)ret.get();
    if (group_size > 5) {
        params->decode = false;
    } else {
        params->decode = true;
    }

    // prepare flashinfer params for decode
    prepareDecodeFlashInferAttnParamsImpl(params, device, attn_configs, sequence_lengths_host, kv_cache_block_id_host, batch_size, tokens_per_block, max_batch_blocks);
    
    if (!attn_configs.use_mla || params->mla_ops_type == MlaOpsType::MHA) {
        if (params->decode) {
            params->plan = BatchDecodeWithPagedKVCachePlan(
                    params->float_workspace_t, // float_workspace_buffer
                    params->int_workspace_t, // int_workspace_buffer
                    params->int_host_workspace_t, // page_locked_int_workspace_buffer
                    params->page_indptr_host_t, // indptr
                    batch_size, // batch_size
                    local_head_num, // num_qo_heads
                    local_head_num_kv, // num_kv_heads
                    tokens_per_block, // page_size
                    false, // enable_cuda_graph,
                    -1, // window_left
                    -1, // logits_soft_cap
                    size_per_head, // head_dim_qk
                    size_per_head, // head_dim_vo
                    torch::empty(0, dataTypeToTorchType(dtype)), // empty_q_data
                    torch::empty(0, dataTypeToTorchType(dtype)), // empty_kv_data
                    reinterpret_cast<int64_t>(cuda_device->getStream()) // cuda_stream
                    );
        } else {
            params->plan = BatchPrefillWithKVCachePlan(
                    params->float_workspace_t, // float_workspace_buffer
                    params->int_workspace_t, // int_workspace_buffer
                    params->int_host_workspace_t, // page_locked_int_workspace_buffer
                    params->qo_indptr_host_t, // qo_indptr
                    params->page_indptr_host_t, // kv_indptr
                    torch::empty(0, dataTypeToTorchType(dtype)), // kv_len_arr, not in use yet
                    batch_size, // total_num_rows
                    batch_size, // batch_size
                    local_head_num, // num_qo_heads
                    local_head_num_kv, // num_kv_heads
                    tokens_per_block, // page_size
                    false, // enable_cuda_graph
                    size_per_head, // head_dim_qk
                    size_per_head, // head_dim_vo
                    true, // causal
                    reinterpret_cast<int64_t>(cuda_device->getStream()));
        }
    } else if (params->mla_ops_type == MlaOpsType::FLASH_INFER) {
        params->plan = BatchMLAPagedAttentionPlan(
            params->float_workspace_t,
            params->int_workspace_t,
            params->int_host_workspace_t,
            params->qo_indptr_host_t,
            params->page_indptr_host_t,
            params->kvlen_host_t,
            local_head_num,
            attn_configs.kv_lora_rank,
            true,
            reinterpret_cast<int64_t>(cuda_device->getStream())
        );
    } else if (params->mla_ops_type == MlaOpsType::FLASH_MLA) {
        printBufferData(*torchTensor2Buffer(params->kvlen_t), "metadata kvlen_t");
        FT_LOG_TRACE("batch_size = %zu", batch_size);
        FT_LOG_TRACE("local_head_num = %zu", local_head_num);
        params->flash_mla_plan = get_mla_metadata(
            params->kvlen_t,
            local_head_num,
            1
        );
    } else {
        FT_FAIL("unexpected mla ops type: %d", int(params->mla_ops_type));
    }

    return ret;
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

    auto flash_infer_attn_params = (FlashInferAttnParams*)params.common.decode_flash_infer_attn_params.get();
    if (flash_infer_attn_params && flash_infer_attn_params->plan.numel() > 0) {
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
                    flashinfer.float_workspace_t, // float_workspace_buffer
                    flashinfer.int_workspace_t, // int_workspace_buffer
                    flashinfer.plan, // plan_info_vec
                    q, // q
                    k_cache, // paged_k_cache
                    v_cache, // paged_v_cache
                    flashinfer.page_indptr_t, // paged_kv_indptr
                    flashinfer.page_indice_t, // paged_kv_indices
                    flashinfer.paged_kv_last_page_len_1_t, // paged_kv_last_page_len
                    Buffer2torchTensor(params.output, false), // o
                    std::nullopt, // maybe_lse
                    1, // kv_layout_code
                    -1, // window_left
                    std::nullopt, // maybe_alibi_slopes
                    0, // logits_soft_cap
                    (1.0f / sqrtf(size_per_head * 1.0f)) * params.configs.softmax_extra_scale,
                    0,
                    0,
                    reinterpret_cast<int64_t>(stream));
        } else {
            BatchPrefillWithPagedKVCacheRun(
                    flashinfer.float_workspace_t, // float_workspace_buffer
                    flashinfer.int_workspace_t,  // int_workspace_buffer
                    flashinfer.plan, // plan_info_vec
                    q, // q
                    k_cache, // paged_k_cache
                    v_cache, // paged_v_cache
                    flashinfer.qo_indptr_t, // qo_indptr
                    flashinfer.page_indptr_t, // paged_kv_indptr
                    flashinfer.page_indice_t, // paged_kv_indices
                    flashinfer.paged_kv_last_page_len_1_t, // paged_kv_last_page_len
                    Buffer2torchTensor(params.output, false), // o
                    std::nullopt, // maybe_lse
                    1, // mask_mode_code,
                    1, // layout
                    -1, // window_left
                    std::nullopt, // maybe_custom_mask
                    std::nullopt, // maybe_mask_indptr
                    std::nullopt, // maybe_alibi_slopes
                    0, // logits_soft_cap
                    (1.0f / sqrtf(size_per_head * 1.0f)) * params.configs.softmax_extra_scale,
                    params.configs.rope_config.scale,
                    params.configs.rope_config.base,
                    reinterpret_cast<int64_t>(stream));
        }

        sync_check_cuda_error();
    } else {
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
