#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/devices/utils/RopeCache.h"
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention.h"
#include "rtp_llm/cpp/kernels/kv_cache_kernels.h"
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/rocm_impl/aiterPA.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;
namespace rtp_llm {

// #define DEBUG_PRINT_PARAMS(...) printParams(__VA_ARGS__)
#define DEBUG_PRINT_PARAMS(...)                                                                                        \
    do {                                                                                                               \
    } while (0)

void printParams(const AttentionModuleParams& params,
                 ROCmDevice*                  device,
                 const std::string&           prefix,
                 BufferPtr                    sliceQ = nullptr) {
    if (params.common.kv_cache && params.common.kv_cache->kv_cache_block_id) {
        auto kv_cache_block_id_host = device->clone({*params.common.kv_cache->kv_cache_block_id, AllocationType::HOST});

        auto getUniqueDumpDir = [](const std::string& root, const std::string& prefix) -> std::string {
            int count = 0;
            while (true) {
                std::ostringstream oss;
                oss << root << "/" << prefix << "_" << count;
                std::string path = oss.str();
                if (!fs::exists(path)) {
                    fs::create_directories(path);
                    return path;
                }
                ++count;
            }
        };

        std::string dump_dir = getUniqueDumpDir("attn", prefix);

        auto saveOneKVBlock =
            [](const BufferPtr& buffer, const std::string& dump_dir, const std::string& tag, int32_t block_id) {
                std::ostringstream oss;
                oss << dump_dir << "/" << tag << "_block_" << block_id << ".pt";
                std::string file_path = oss.str();

                printf("📦 Saving %s block_id [%d] → %s\n", tag.c_str(), block_id, file_path.c_str());
                saveBufferDataToTorch(*buffer, nullptr, file_path.c_str());
                printf("✅ Done saving %s block_id [%d]\n", tag.c_str(), block_id);
            };

        auto saveKVCacheToFile = [&]() {
            auto kv_cache = params.common.kv_cache;

            for (int i = 0; i < kv_cache_block_id_host->size(); ++i) {
                int32_t   block_id = *kv_cache_block_id_host->dataWithOffset<int32_t>(i);
                BufferPtr k_block  = kv_cache->k_cache_buffer->index(block_id);
                BufferPtr v_block  = kv_cache->v_cache_buffer->index(block_id);
                saveOneKVBlock(k_block, dump_dir, "k", block_id);
                saveOneKVBlock(v_block, dump_dir, "v", block_id);
            }

            saveBufferDataToTorch(*params.common.sequence_lengths, nullptr, dump_dir + "/sequence_lengths.pt");
            saveBufferDataToTorch(params.input, nullptr, dump_dir + "/qkv.pt");

            if (sliceQ) {
                saveBufferDataToTorch(*sliceQ, nullptr, dump_dir + "/q.pt");
            }
        };

        saveKVCacheToFile();

        printf("🧾 kv_cache_block_id:\n%s\n", kv_cache_block_id_host->debugStringWithData<int32_t>().c_str());
    } else {
        printf("❌ params.common.kv_cache->kv_cache_block_id is nullptr\n");
    }

    // k_cache_buffer
    if (params.common.kv_cache && params.common.kv_cache->k_cache_buffer) {
        printf("params.common.k_cache_buffer\n%s\n", params.common.kv_cache->k_cache_buffer->debugString().c_str());
    } else {
        printf("params.common.k_cache_buffer is nullptr\n");
    }

    // input_lengths
    if (params.common.input_lengths) {
        auto input_lengths = device->clone({*params.common.input_lengths, AllocationType::HOST});
        printf("params.common.input_lengths\n%s\n", input_lengths->debugStringWithData<int32_t>().c_str());
    } else {
        printf("params.common.input_lengths is nullptr\n");
    }

    // sequence_lengths
    if (params.common.sequence_lengths) {
        auto sequence_lengths = device->clone({*params.common.sequence_lengths, AllocationType::HOST});
        printf("params.common.sequence_lengths\n%s\n", sequence_lengths->debugStringWithData<int32_t>().c_str());
    } else {
        printf("params.common.sequence_lengths is nullptr\n");
    }

    // cu_seqlens
    if (params.common.cu_seqlens) {
        auto cu_seqlens = device->clone({*params.common.cu_seqlens, AllocationType::HOST});
        printf("params.common.cu_seqlens\n%s\n", cu_seqlens->debugStringWithData<int32_t>().c_str());
    } else {
        printf("params.common.cu_seqlens is nullptr\n");
    }

    // cu_kv_seqlens
    if (params.common.cu_kv_seqlens) {
        auto cu_kv_seqlens = device->clone({*params.common.cu_kv_seqlens, AllocationType::HOST});
        printf("params.common.cu_kv_seqlens\n%s\n", cu_kv_seqlens->debugStringWithData<int32_t>().c_str());
    } else {
        printf("params.common.cu_kv_seqlens is nullptr\n");
    }

    // padding_offset
    if (params.common.padding_offset) {
        auto padding_offset = device->clone({*params.common.padding_offset, AllocationType::HOST});
        printf("params.common.padding_offset\n%s\n", padding_offset->debugStringWithData<int32_t>().c_str());
    } else {
        printf("params.common.padding_offset is nullptr\n");
    }

    // input
    if (params.input.data()) {
        printf("params.input\n%s\n", params.input.debugString().c_str());
    } else {
        printf("params.input is nullptr\n");
    }

    // prefix_prompt_lengths
    if (params.common.prefix_prompt_lengths) {
        auto prefix_prompt_lengths = device->clone({*params.common.prefix_prompt_lengths, AllocationType::HOST});
        printf("params.common.prefix_prompt_lengths\n%s\n",
               prefix_prompt_lengths->debugStringWithData<int32_t>().c_str());
    } else {
        printf("params.common.prefix_prompt_lengths is nullptr\n");
    }

    printf("Context Batch Size       : %d\n", params.common.context_batch_size);
    printf("Decoder Batch Size       : %d\n", params.common.decoder_batch_size);
    printf("Context Max Seq Length   : %d\n", params.common.context_max_seq_len);
    printf("Decode Max Seq Length   : %d\n", params.common.decoder_max_seq_len);
    printf("Prefix Length (Max)      : %d\n", params.common.max_prefix_length);
    printf("==================================\n");
}

void flashInferAttnParamsDeleter(void* p) {
    delete (FlashInferAttnParams*)p;
}

void aiterAttnParamsDeleter(void* p) {
    delete (AiterAttnParams*)p;
}

void prepareDecodeFlashInferAttnParamsImpl(FlashInferAttnParams*            params,
                                           rtp_llm::DeviceBase*             device,
                                           const rtp_llm::AttentionConfigs& attn_configs,
                                           const BufferPtr&                 sequence_lengths_host,
                                           const BufferPtr&                 kv_cache_block_id_host,
                                           const uint64_t                   batch_size,
                                           const uint64_t                   tokens_per_block,
                                           const uint64_t                   max_batch_blocks) {
    RTP_LLM_CHECK_WITH_INFO(max_batch_blocks > 0 && kv_cache_block_id_host,
                            "max_batch_blocks and kv_cache_block_id_host must be set for decode");
    params->float_workspace =
        device->allocateBuffer({DataType::TYPE_INT8, {128 * 1024 * 1024}, AllocationType::DEVICE}, {"float_workspace"});
    params->int_workspace =
        device->allocateBuffer({DataType::TYPE_INT8, {8 * 1024 * 1024}, AllocationType::DEVICE}, {"int_workspace"});
    params->int_host_workspace =
        device->allocateBuffer({DataType::TYPE_INT8, {8 * 1024 * 1024}, AllocationType::HOST}, {"int_host_workspace"});
    params->page_indptr_host =
        device->allocateBuffer({DataType::TYPE_INT32, {batch_size + 1}, AllocationType::HOST}, {"page_indptr_host"});
    params->qo_indptr_host =
        device->allocateBuffer({DataType::TYPE_INT32, {batch_size + 1}, AllocationType::HOST}, {"qo_indptr_host"});

    params->batch_indice_host =
        device->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"batch_indice_host"});
    params->positions_host =
        device->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"positions_host"});
    params->kvlen_host =
        device->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"kvlen_host"});
    params->paged_kv_last_page_len_host = device->allocateBuffer(
        {DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"paged_kv_last_page_len_host"});
    params->paged_kv_last_page_len_1_host = device->allocateBuffer(
        {DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"paged_kv_last_page_len_1_host"});

    vector<int> page_indice_vec;
    params->qo_indptr_host->data<int>()[0]   = 0;
    params->page_indptr_host->data<int>()[0] = 0;
    for (int i = 0; i < int(batch_size); i++) {
        params->batch_indice_host->data<int>()[i]           = i;
        params->paged_kv_last_page_len_host->data<int>()[i] = sequence_lengths_host->data<int>()[i] % tokens_per_block;
        params->paged_kv_last_page_len_1_host->data<int>()[i] = params->paged_kv_last_page_len_host->data<int>()[i] + 1;
        params->positions_host->data<int>()[i]                = sequence_lengths_host->data<int>()[i];
        params->kvlen_host->data<int>()[i]                    = sequence_lengths_host->data<int>()[i] + 1;
        // sequence_length_host here is the index of the last token in the sequence, equals to length - 1
        int page_nums = (sequence_lengths_host->data<int>()[i] + tokens_per_block) / tokens_per_block;
        for (int j = 0; j < page_nums - 1; j++) {
            auto page_idx = kv_cache_block_id_host->data<int>()[i * max_batch_blocks + j];
            for (int k = page_idx * tokens_per_block; k < (page_idx + 1) * tokens_per_block; k++) {
                page_indice_vec.push_back(k);
            }
        }
        auto page_idx = kv_cache_block_id_host->data<int>()[i * max_batch_blocks + page_nums - 1];
        for (int k = page_idx * tokens_per_block;
             k < page_idx * tokens_per_block + params->paged_kv_last_page_len_1_host->data<int>()[i];
             k++) {
            page_indice_vec.push_back(k);
        }
        params->page_indptr_host->data<int>()[i + 1] = int(page_indice_vec.size());
        params->qo_indptr_host->data<int>()[i + 1]   = i + 1;
    }

    params->page_indice_host = device->allocateBuffer(
        {DataType::TYPE_INT32, {size_t(page_indice_vec.size())}, AllocationType::HOST}, {"page_indice_host"});
    std::copy(page_indice_vec.begin(), page_indice_vec.end(), params->page_indice_host->data<int>());

    params->kv_cache_block_id        = device->clone({*kv_cache_block_id_host, AllocationType::DEVICE});
    params->batch_indice             = device->clone({*params->batch_indice_host, AllocationType::DEVICE});
    params->positions                = device->clone({*params->positions_host, AllocationType::DEVICE});
    params->paged_kv_last_page_len   = device->clone({*params->paged_kv_last_page_len_host, AllocationType::DEVICE});
    params->paged_kv_last_page_len_1 = device->clone({*params->paged_kv_last_page_len_1_host, AllocationType::DEVICE});
    params->page_indptr              = device->clone({*params->page_indptr_host, AllocationType::DEVICE});
    params->qo_indptr                = device->clone({*params->qo_indptr_host, AllocationType::DEVICE});
    params->page_indice              = device->clone({*params->page_indice_host, AllocationType::DEVICE});
    params->kvlen                    = device->clone({*params->kvlen_host, AllocationType::DEVICE});

    params->float_workspace_t    = Buffer2torchTensor(params->float_workspace, false);
    params->int_workspace_t      = Buffer2torchTensor(params->int_workspace, false);
    params->int_host_workspace_t = Buffer2torchTensor(params->int_host_workspace, false);

    params->batch_indice_t             = Buffer2torchTensor(params->batch_indice, false);
    params->positions_t                = Buffer2torchTensor(params->positions, false);
    params->paged_kv_last_page_len_t   = Buffer2torchTensor(params->paged_kv_last_page_len, false);
    params->paged_kv_last_page_len_1_t = Buffer2torchTensor(params->paged_kv_last_page_len_1, false);

    params->qo_indptr_t         = Buffer2torchTensor(params->qo_indptr, false);
    params->qo_indptr_host_t    = Buffer2torchTensor(params->qo_indptr_host, false);
    params->page_indptr_t       = Buffer2torchTensor(params->page_indptr, false);
    params->page_indptr_host_t  = Buffer2torchTensor(params->page_indptr_host, false);
    params->page_indice_t       = Buffer2torchTensor(params->page_indice, false);
    params->kvlen_host_t        = Buffer2torchTensor(params->kvlen_host, false);
    params->kvlen_t             = Buffer2torchTensor(params->kvlen, false);
    params->kv_cache_block_id_t = Buffer2torchTensor(params->kv_cache_block_id, false);
}

// for mla, we need to prepare additional params for write kvcache and de rotary embedding
void prepareContextMLAFlashInferAttnParamsImpl(FlashInferAttnParams*            params,
                                               rtp_llm::DeviceBase*             device,
                                               const rtp_llm::AttentionConfigs& attn_configs,
                                               const BufferPtr&                 sequence_lengths_host,
                                               const BufferPtr&                 input_lengths_host,
                                               const BufferPtr&                 kv_cache_block_id_host,
                                               const uint64_t                   prefill_token_num,
                                               const uint64_t                   context_batch_size,
                                               const uint64_t                   tokens_per_block,
                                               const uint64_t                   max_batch_blocks,
                                               const uint64_t                   batch_size) {
    params->batch_indice_host = device->allocateBuffer(
        {DataType::TYPE_INT32, {prefill_token_num}, AllocationType::HOST}, {"prefill_batch_indices_host"});
    params->positions_host = device->allocateBuffer({DataType::TYPE_INT32, {prefill_token_num}, AllocationType::HOST},
                                                    {"prefill_positions_host"});
    params->paged_kv_last_page_len_1_host = device->allocateBuffer(
        {DataType::TYPE_INT32, {context_batch_size}, AllocationType::HOST}, {"prefill_kv_last_page_len_1_host"});
    params->page_indptr_host = device->allocateBuffer(
        {DataType::TYPE_INT32, {context_batch_size + 1}, AllocationType::HOST}, {"prefill_page_indptr_host"});
    params->page_indptr_host->data<int>()[0] = 0;
    std::vector<int> prefill_page_indices_vec;

    int offset = 0;
    for (int i = 0; i < context_batch_size; i++) {
        auto input_length = input_lengths_host->data<int>()[i + batch_size];
        for (int j = 0; j < input_length; j++) {
            params->batch_indice_host->data<int>()[offset] = i;
            params->positions_host->data<int>()[offset]    = j;
            offset += 1;
        }
        if (kv_cache_block_id_host) {
            int page_nums = (input_length + tokens_per_block - 1) / tokens_per_block;
            for (int j = 0; j < page_nums; j++) {
                auto page_idx = kv_cache_block_id_host->data<int>()[(i + batch_size) * max_batch_blocks + j];
                prefill_page_indices_vec.push_back(page_idx);
            }
            params->paged_kv_last_page_len_1_host->data<int>()[i] = (input_length - 1) % tokens_per_block + 1;
            params->page_indptr_host->data<int>()[i + 1]          = prefill_page_indices_vec.size();
        }
    }
    if (kv_cache_block_id_host) {
        params->page_indice_host = device->allocateBuffer(
            {DataType::TYPE_INT32, {size_t(prefill_page_indices_vec.size())}, AllocationType::HOST},
            {"prefill_page_indices_host"});
        std::copy(
            prefill_page_indices_vec.begin(), prefill_page_indices_vec.end(), params->page_indice_host->data<int>());
        params->page_indice   = device->clone({*params->page_indice_host, AllocationType::DEVICE});
        params->page_indice_t = Buffer2torchTensor(params->page_indice, false);
    }

    params->batch_indice             = device->clone({*params->batch_indice_host, AllocationType::DEVICE});
    params->positions                = device->clone({*params->positions_host, AllocationType::DEVICE});
    params->paged_kv_last_page_len_1 = device->clone({*params->paged_kv_last_page_len_1_host, AllocationType::DEVICE});
    params->page_indptr              = device->clone({*params->page_indptr_host, AllocationType::DEVICE});

    params->batch_indice_t             = Buffer2torchTensor(params->batch_indice, false);
    params->positions_t                = Buffer2torchTensor(params->positions, false);
    params->paged_kv_last_page_len_1_t = Buffer2torchTensor(params->paged_kv_last_page_len_1, false);
    params->page_indptr_t              = Buffer2torchTensor(params->page_indptr, false);
}

void prepareDecodeAiterAttnParamsImpl(AiterAttnParams*     params,
                                      rtp_llm::DeviceBase* device,
                                      const BufferPtr&     sequence_lengths_host,
                                      const uint64_t       batch_size) {
    if (device->nativeGraphCapturing()) {
        params->sequence_lengths_host = nullptr;
        params->sequence_lengths      = device->clone({*sequence_lengths_host, AllocationType::DEVICE});
        params->sequence_lengths_t    = Buffer2torchTensor(params->sequence_lengths, false);
        params->sequence_lengths_t += 1;
        return;
    }
    params->sequence_lengths_host =
        device->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST}, {"sequence_lengths_host"});

    for (int i = 0; i < int(batch_size); i++) {
        params->sequence_lengths_host->data<int>()[i] = sequence_lengths_host->data<int>()[i] + 1;
    }

    params->sequence_lengths = device->clone({*params->sequence_lengths_host, AllocationType::DEVICE});

    params->sequence_lengths_t = Buffer2torchTensor(params->sequence_lengths, false);
}

ParamsPtr FlashInferAttnParams::preparePrefillFlashInferAttnParams(rtp_llm::DeviceBase*             device,
                                                                   const rtp_llm::AttentionConfigs& attn_configs,
                                                                   const BufferPtr&                 prefix_lengths_host,
                                                                   const BufferPtr&  sequence_lengths_host,
                                                                   const BufferPtr&  input_lengths_host,
                                                                   const BufferPtr&  kv_cache_block_id_host,
                                                                   rtp_llm::DataType dtype) {
    const size_t batch_size         = sequence_lengths_host->shape()[0];
    const size_t context_batch_size = input_lengths_host->shape()[0] - batch_size;
    if (context_batch_size == 0) {
        return nullptr;
    }

    const int tokens_per_block = attn_configs.tokens_per_block;

    const int    max_batch_blocks  = kv_cache_block_id_host ? kv_cache_block_id_host->shape()[1] : -1;
    const size_t prefill_token_num = std::accumulate(input_lengths_host->data<int>() + batch_size,
                                                     input_lengths_host->data<int>() + context_batch_size + batch_size,
                                                     0);
    auto         ret               = ParamsPtr(new FlashInferAttnParams, flashInferAttnParamsDeleter);
    auto         params            = (FlashInferAttnParams*)ret.get();
    prepareContextMLAFlashInferAttnParamsImpl(params,
                                              device,
                                              attn_configs,
                                              sequence_lengths_host,
                                              input_lengths_host,
                                              kv_cache_block_id_host,
                                              prefill_token_num,
                                              context_batch_size,
                                              tokens_per_block,
                                              max_batch_blocks,
                                              batch_size);
    return ret;
}

ParamsPtr FlashInferAttnParams::prepareDecodeFlashInferAttnParams(rtp_llm::DeviceBase*             device,
                                                                  const rtp_llm::AttentionConfigs& attn_configs,
                                                                  const BufferPtr&  sequence_lengths_host,
                                                                  const BufferPtr&  input_lengths_host,
                                                                  const BufferPtr&  kv_cache_block_id_host,
                                                                  rtp_llm::DataType dtype) {
    const char* disable_flash_infer_env = getenv("DISABLE_FLASH_INFER");
    if (rtp_llm::rocm::get_sm() < 80 || (disable_flash_infer_env && strcmp(disable_flash_infer_env, "1") == 0)) {
        return nullptr;
    }

    const size_t batch_size = sequence_lengths_host->shape()[0];
    if (batch_size == 0) {
        return nullptr;
    }

    auto      cuda_device       = dynamic_cast<ROCmDevice*>(device);
    const int local_head_num    = attn_configs.head_num;
    const int local_head_num_kv = attn_configs.kv_head_num;
    const int size_per_head     = attn_configs.size_per_head;
    const int group_size        = local_head_num / local_head_num_kv;
    const int tokens_per_block  = attn_configs.tokens_per_block;

    if (!cuda_device || (dtype != DataType::TYPE_FP16 && dtype != DataType::TYPE_BF16)
        || attn_configs.kv_cache_dtype != KvCacheDataType::BASE
        || (attn_configs.rope_config.style != RopeStyle::Base && attn_configs.rope_config.style != RopeStyle::No)
        || attn_configs.mask_type != causalMask || attn_configs.q_scaling != 1.0f || attn_configs.use_logn_attn
        || (size_per_head != 64 && size_per_head != 128 && size_per_head != 192)
        || (group_size > 10 && group_size != 16)) {
        return nullptr;
    }

    const int max_batch_blocks = kv_cache_block_id_host ? kv_cache_block_id_host->shape()[1] : -1;
    auto      ret              = ParamsPtr(new FlashInferAttnParams, flashInferAttnParamsDeleter);
    auto      params           = (FlashInferAttnParams*)ret.get();
    if (group_size > 5) {
        params->decode = false;
    } else {
        params->decode = true;
    }

    // prepare flashinfer params for decode
    prepareDecodeFlashInferAttnParamsImpl(params,
                                          device,
                                          attn_configs,
                                          sequence_lengths_host,
                                          kv_cache_block_id_host,
                                          batch_size,
                                          tokens_per_block,
                                          max_batch_blocks);
    return ret;
}

ParamsPtr AiterAttnParams::prepareDecodeAiterAttnParams(rtp_llm::DeviceBase* device,
                                                        const BufferPtr&     sequence_lengths_host) {

    if (!device->initParams().use_aiter_pa) {
        return nullptr;
    }

    const size_t batch_size = sequence_lengths_host->shape()[0];
    if (batch_size == 0) {
        return nullptr;
    }

    auto ret    = ParamsPtr(new AiterAttnParams, aiterAttnParamsDeleter);
    auto params = (AiterAttnParams*)ret.get();

    prepareDecodeAiterAttnParamsImpl(params, device, sequence_lengths_host, batch_size);
    return ret;
}

KVBlockArray ROCmDevice::getKVBlockArray(const AttentionModuleParams& params,
                                         const Buffer&                kv_cache_offset_pointers,
                                         int                          batch_size,
                                         bool                         use_fp8_fmha,
                                         bool                         use_offset_array) {
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
    auto const  elemSize = kv_cache->k_scale_buffer || use_fp8_fmha ? sizeof(int8_t) : 2;  // 2 for kv cache fp16
    // RTP_LLM_LOG_INFO("kv_cache[0].typeSize():%d", kv_cache[0].typeSize());
    RTP_LLM_LOG_DEBUG("kv_blocks_offset size:%d, k_cache:%p, v_cache:%p, "
                      "k_cache[0].sizeBytes():%d, params.configs.tokens_per_block:%d, "
                      "kv_block_offset:%d, k_cache (int): %lu, v_cache (int): %lu, "
                      "max_blocks_per_batch:%d",
                      kv_blocks_offset.size(),
                      static_cast<void*>(k_cache.data()),  // for %p
                      static_cast<void*>(v_cache.data()),  // for %p
                      k_cache[0].sizeBytes(),
                      params.configs.tokens_per_block,
                      kv_block_offset,
                      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(k_cache.data())),  // for %lu
                      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(v_cache.data())),
                      max_blocks_per_batch);
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

    if (!use_offset_array) {
        invokeConvertOffsetToBlockArrayData((int32_t*)kv_cache_offset_pointers.data(),
                                            (int*)kv_blocks_offset.data(),
                                            batch_size,
                                            max_blocks_per_batch,
                                            stream_);
    }
    check_cuda_error();
    if (kv_cache->k_scale_buffer) {
        RUNTIME_ASSERT_OP_ARG(kv_cache->v_scale_buffer,
                              "v scale buffer should has value when use k scale buffer has value");
        const auto& k_scale                 = *(kv_cache->k_scale_buffer);
        kv_cache_buffer.scale               = k_scale.data();
        kv_cache_buffer.mScaleBytesPerBlock = k_scale[0].sizeBytes();
    }
    KvCacheDataType cache_type = KvCacheDataType::BASE;
#if defined(ENABLE_FP8)
    if (use_fp8_fmha_) {
        cache_type = KvCacheDataType::FP8;
    } else
#endif
        if (use_fp8_fmha) {
        cache_type = KvCacheDataType::FP8;
    } else if (kv_cache->k_scale_buffer && params.configs.kv_cache_dtype == KvCacheDataType::INT8) {
        RTP_LLM_LOG_DEBUG("now use kv_cache int8");
        cache_type = KvCacheDataType::INT8;
    }
    kv_cache_buffer.cache_type = cache_type;
    check_cuda_error();
    return kv_cache_buffer;
}

ParamsPtr ROCmDevice::PrepareCKAttn(const AttentionConfigs& configs,
                                    int                     kv_block_offset,
                                    const BufferPtr&        kv_cache_block_id,
                                    int                     batch_size) {
    RTP_LLM_LOG_DEBUG("PrepareCKAttn: kv_block_offset: %d, batch_size: %d, kv_cache_block_id: %s",
                      kv_block_offset,
                      batch_size,
                      kv_cache_block_id ? kv_cache_block_id->debugString().c_str() : "nullptr");
    if (kv_block_offset <= 0 || batch_size <= 0 || !kv_cache_block_id) {
        return nullptr;
    }
    auto            ck_attn    = std::make_shared<CKAttn>();
    KvCacheDataType cache_type = KvCacheDataType::BASE;
#ifdef ENABLE_FP8
    if (use_fp8_fmha_) {
        cache_type = KvCacheDataType::FP8;
    } else
#endif
        if (configs.kv_cache_dtype == KvCacheDataType::INT8) {
        RTP_LLM_LOG_DEBUG("now use kv_cache int8");
        cache_type = KvCacheDataType::INT8;
    }
    const auto max_blocks_per_batch = kv_cache_block_id->shape()[1];
    auto const elemSize             = 2;  // 2 for kv cache fp16

    ck_attn->kv_cache_offset =
        allocateBuffer({DataType::TYPE_INT32, {size_t(batch_size), 1, 2, max_blocks_per_batch}, AllocationType::DEVICE},
                       {"kv_cache_offset"});
    ck_attn->kv_block_array                     = KVBlockArray(batch_size,
                                           max_blocks_per_batch,
                                           configs.tokens_per_block,
                                           configs.kv_head_num * configs.size_per_head * elemSize,
                                           0,
                                           0,
                                           nullptr,  // (uint64_t*)k_cache.data(),
                                           nullptr,
                                           (rtp_llm::KVCacheIndex*)ck_attn->kv_cache_offset->data<int>());
    ck_attn->kv_block_array.cache_type          = cache_type;
    ck_attn->kv_block_array.mScaleBytesPerBlock = configs.tokens_per_block * configs.kv_head_num * sizeof(float);
    invokeConvertOffsetToBlockArrayData(ck_attn->kv_cache_offset->data<int>(),
                                        kv_cache_block_id->data<int>(),
                                        batch_size,
                                        max_blocks_per_batch,
                                        stream_);
    check_cuda_error();
    return ck_attn;
}

static std::once_flag rope_cache_flag;

AttentionModuleOutput ROCmDevice::contextAttention(const AttentionModuleParams& params) {
    auto datatype            = params.input.type();
    auto token_num           = params.input.shape()[0];
    auto batch_size          = params.common.context_batch_size;
    auto decoder_batch_size  = params.common.decoder_batch_size;
    auto seq_len             = params.common.context_max_seq_len;
    auto seq_len_with_prefix = seq_len + params.common.max_prefix_length;
    // auto context_token_num   = params.common.context_token_num;
    auto head_num      = params.configs.head_num;
    auto kv_head_num   = params.configs.kv_head_num;
    auto size_per_head = params.configs.size_per_head;

    auto q_output = allocateBuffer(
        {params.input.type(), {batch_size, head_num, seq_len, size_per_head}, AllocationType::DEVICE}, {"q_output"});
    bufMemset(*q_output, 0);

    auto k_output = allocateBuffer(
        {params.input.type(), {batch_size, kv_head_num, seq_len_with_prefix, size_per_head}, AllocationType::DEVICE},
        {"k_output"});
    bufMemset(*k_output, 0);
    auto v_output = allocateBuffer(
        {params.input.type(), {batch_size, kv_head_num, seq_len_with_prefix, size_per_head}, AllocationType::DEVICE},
        {"v_output"});
    bufMemset(*v_output, 0);
    BufferPtr kv_cache_block_id = nullptr;

    KVBlockArray                  kv_block_array;
    PrefixPromptBatchWeightsParam prefix_prompt_param;

    if (params.common.kv_cache) {
        const auto max_blocks_per_batch = params.common.kv_cache->kv_cache_block_id->shape()[1];
        kv_cache_block_id =
            allocateBuffer({DataType::TYPE_INT32, {batch_size, 1, 2, max_blocks_per_batch}, AllocationType::DEVICE},
                           {"kv_cache_block_id"});
        kv_block_array                     = getKVBlockArray(params,
                                         *kv_cache_block_id,
                                         batch_size,
                                         params.common.kv_cache->k_cache_buffer->type() == DataType::TYPE_FP8_E4M3);
        prefix_prompt_param.kv_block_array = kv_block_array;

        if (params.common.prefix_prompt_lengths) {
            prefix_prompt_param.d_prefix_prompt_lengths  = params.common.prefix_prompt_lengths->data<int>();
            prefix_prompt_param.max_prefix_prompt_length = params.common.max_prefix_length;
            prefix_prompt_param.count_length             = 1;
        }
    }
    printBufferData(*params.common.input_lengths, "input_lengths");
    if (params.common.cu_seqlens) {
        printBufferData(*params.common.cu_seqlens, "cu_seqlens");
        printBufferData(*params.common.cu_kv_seqlens, "cu_kv_seqlens");
    }

    // int8
    float* scale_out_ptr = nullptr;
    int    int8_mode     = 0;

    if (prefix_prompt_param.max_prefix_prompt_length > 0) {
        if (init_params_.use_aiter_pa) {
            if (init_params_.use_asm_pa) {
                DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                                 invokeLoadPrefixKVCacheAiter,
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
            } else {
                DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                                 invokeLoadPrefixKVCacheAiterV1,
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
            }
        } else {
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
        }
    }

    bool store_qkv   = true;
    bool store_q     = true;
    bool store_kv    = true;
    bool store_cache = params.common.kv_cache.has_value();

    // if all condition satisfy, no need to do invokeAddFusedQKVBiasTranspose
    bool skip_add_bias_transpose = (params.configs.rope_config.style == RopeStyle::No && !params.common.kv_cache
                                    && !params.configs.fuse_qkv_add_bias);
    RTP_LLM_LOG_DEBUG("skip_add_bias_transpose: %d", skip_add_bias_transpose);
    if (!skip_add_bias_transpose) {

        bool                 use_rope_cache = params.configs.rope_config.style == RopeStyle::Base;
        static torch::Tensor rope_cache;
        std::call_once(rope_cache_flag, [&]() {
            if (use_rope_cache) {
                rope_cache = getRopeCache(params.configs.rope_config, init_params_.max_seq_len);
            }
        });

        if (init_params_.use_aiter_pa) {
            if (init_params_.use_asm_pa) {
                DISPATCH_CUDA_FUNCTION_DATA_TYPE(
                    datatype,
                    invokeAddFusedQKVBiasTransposePrefill,
                    q_output->data(),
                    k_output->data(),
                    v_output->data(),
                    &prefix_prompt_param,
                    params.input.data(),
                    nullptr,
                    params.common.position_ids ? params.common.position_ids->dataWithOffset<int>(
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
                    scale_out_ptr,
                    int8_mode,
                    false,
                    store_qkv,
                    store_q,
                    store_kv,
                    store_cache,
                    use_rope_cache && rope_cache.defined() ? static_cast<float2*>(rope_cache.data_ptr()) : nullptr,
                    stream_);
            } else {
                DISPATCH_CUDA_FUNCTION_DATA_TYPE(
                    datatype,
                    invokeAddFusedQKVBiasTransposePrefillV1,
                    q_output->data(),
                    k_output->data(),
                    v_output->data(),
                    &prefix_prompt_param,
                    params.input.data(),
                    nullptr,
                    params.common.position_ids ? params.common.position_ids->dataWithOffset<int>(
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
                    scale_out_ptr,
                    int8_mode,
                    false,
                    store_qkv,
                    store_q,
                    store_kv,
                    store_cache,
                    use_rope_cache && rope_cache.defined() ? static_cast<float2*>(rope_cache.data_ptr()) : nullptr,
                    stream_);
            }
            check_cuda_error();
        } else {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                             invokeAddFusedQKVBiasTranspose,
                                             nullptr,
                                             q_output->data(),
                                             k_output->data(),
                                             v_output->data(),
                                             &prefix_prompt_param,
                                             params.input.data(),
                                             nullptr,
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
                                             scale_out_ptr,
                                             int8_mode,
                                             false,
                                             store_qkv,
                                             false,
                                             store_q,
                                             store_kv,
                                             store_cache,
                                             stream_);
            check_cuda_error();
        }
        writeCacheStore(params);
    }

    fmha_runner_->setup(
        datatype, params.configs.mask_type, head_num, kv_head_num, size_per_head, params.configs.q_scaling);
    // auto seq_len_round_32 = (seq_len + 31) / 32 * 32;
    // auto softmax_lse_ = allocateBuffer({DataType::TYPE_FP32, // params.output.type(),
    //                                     {batch_size, head_num, seq_len_round_32},
    //                                     AllocationType::DEVICE},
    //                                     {"softmax_lse"});
    printBufferData(*q_output, "q_output");
    // printBufferData(*k_output, "k_output");
    // printBufferData(*v_output, "v_output");
    // if (v_output->shape()[0]>1) {
    //     printBufferData(*(v_output->index(1)), "v_output_batch1");
    // }

    const size_t hidden_units    = head_num * size_per_head;
    const size_t hidden_units_kv = kv_head_num * size_per_head;

    auto lse_acc_buf = allocateBuffer({DataType::TYPE_FP32, {1, 1, 1, 1}, AllocationType::DEVICE}, {"lse_acc_buf"});

    printBufferData(*q_output, "run_ck_q_output");
    printBufferData(*k_output, "run_ck_k_output");
    printBufferData(*v_output, "run_ck_v_output");
    printBufferData(params.input, "run_ck_input");
    if (skip_add_bias_transpose || prefix_prompt_param.max_prefix_prompt_length <= 0) {
        // not implemented reuse cache for this branch
        fmha_runner_->runCKFmha(params.input.data(),
                                params.input.dataWithOffset(hidden_units),
                                params.input.dataWithOffset(hidden_units + hidden_units_kv),
                                params.output.data(),
                                nullptr,  // buffer for store out softmax_lse, looks like not used by RTP
                                batch_size,
                                seq_len,
                                prefix_prompt_param.max_prefix_prompt_length,
                                // context_token_num,
                                params.common.cu_seqlens->data(),
                                params.common.cu_kv_seqlens->data(),
                                lse_acc_buf->data(),
                                params.common.linear_bias_slopes ? params.common.linear_bias_slopes->data() : nullptr,
                                nullptr,
                                false,
                                false);
        printBufferData(params.output, "run_ck_data_output");
    } else {
        // Processing continuous/variable-length sequences
        torch::Tensor q_output_tensor, k_output_tensor, v_output_tensor;
        auto          q_contiguous = allocateBuffer(
            {params.input.type(), {head_num, seq_len * batch_size, size_per_head}, AllocationType::DEVICE},
            {"q_contiguous"});
        bufMemset(*q_contiguous, 0);
        auto k_contiguous = allocateBuffer({params.input.type(),
                                            {kv_head_num, seq_len_with_prefix * batch_size, size_per_head},
                                            AllocationType::DEVICE},
                                           {"k_contiguous"});
        bufMemset(*k_contiguous, 0);
        auto v_contiguous = allocateBuffer({params.input.type(),
                                            {kv_head_num, seq_len_with_prefix * batch_size, size_per_head},
                                            AllocationType::DEVICE},
                                           {"v_contiguous"});
        bufMemset(*v_contiguous, 0);
        const int hidden_size_q  = head_num * size_per_head;
        const int hidden_size_kv = kv_head_num * size_per_head;
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                         invokeGatherSequencesCombined,
                                         q_contiguous->data(),
                                         k_contiguous->data(),
                                         v_contiguous->data(),
                                         q_output->data(),
                                         k_output->data(),
                                         v_output->data(),
                                         params.common.cu_seqlens->data<int>(),
                                         params.common.cu_kv_seqlens->data<int>(),
                                         batch_size,
                                         seq_len,
                                         seq_len_with_prefix,
                                         head_num,
                                         kv_head_num,
                                         size_per_head,
                                         stream_);
        printBufferData(*q_contiguous, "q_contiguous");
        printBufferData(*k_contiguous, "k_contiguous");
        printBufferData(*v_contiguous, "v_contiguous");

        fmha_runner_->setup(
            datatype, params.configs.mask_type, head_num, kv_head_num, size_per_head, params.configs.q_scaling);

        auto lse_acc_buf = allocateBuffer({DataType::TYPE_FP32, {1, 1, 1, 1}, AllocationType::DEVICE}, {"lse_acc_buf"});
        if (fmha_runner_->runCKFmhaV2(q_contiguous->data(),
                                      k_contiguous->data(),
                                      v_contiguous->data(),
                                      params.output.data(),
                                      nullptr,
                                      batch_size,
                                      seq_len,
                                      params.common.max_prefix_length,
                                      params.common.cu_seqlens->data(),
                                      params.common.cu_kv_seqlens->data(),
                                      lse_acc_buf->data(),
                                      params.common.linear_bias_slopes ? params.common.linear_bias_slopes->data() :
                                                                         nullptr,
                                      nullptr,
                                      token_num,
                                      true,
                                      false)) {
            printBufferData(params.output, "run_ck_data_output");
            return;
        } else {
            RTP_LLM_CHECK_WITH_INFO(
                q_output && k_output && v_output,
                "q_output/k_output/v_output must be provided for default context attention implementation");
            q_output->updateShape({batch_size, kv_head_num, (head_num / kv_head_num) * seq_len, size_per_head});
            auto qk_output = gemm({*q_output,
                                   *k_output,
                                   std::nullopt,
                                   nullptr,
                                   DataType::TYPE_FP32,
                                   DataType::TYPE_FP32,
                                   TransposeOperation::NONE,
                                   TransposeOperation::TRANSPOSE});
            qk_output->updateShape({batch_size, head_num, seq_len, seq_len_with_prefix});
            printBufferData(*qk_output, "qk_output: ");
            float scale = (1.0f / sqrtf(size_per_head * 1.0f));  // * params.configs.softmax_extra_scale;
            auto  lengths_host =
                clone({params.common.input_lengths->view(decoder_batch_size, batch_size), AllocationType::HOST});
            auto prefix_lengths_host =
                params.common.prefix_prompt_lengths ?
                    clone({*params.common.prefix_prompt_lengths, AllocationType::HOST}) :
                    BufferPtr(new Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_INVALID, {0}, nullptr));
            auto attention_mask    = attentionMask({*lengths_host,
                                                    *prefix_lengths_host,
                                                    q_output->type(),
                                                    params.configs.mask_type == AttentionMaskType::causalMask});
            auto softmax_qk_output = softmax({std::move(qk_output), *attention_mask, nullopt, scale, datatype});
            softmax_qk_output->updateShape(
                {batch_size, kv_head_num, (head_num / kv_head_num) * seq_len, seq_len_with_prefix});
            printBufferData(*softmax_qk_output, "softmax_qk_output: ");

            auto qkv_output = gemm(
                {*softmax_qk_output, *v_output, std::nullopt, nullptr, DataType::TYPE_INVALID, params.compute_type});
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
                                             stream_);
            printBufferData(params.output, "run_ck_data_output");
            return;
        }
    }
}

template<typename T>
void selfAttentionwrapper(const AttentionModuleParams params,
                          bool                        use_multi_block_mode,
                          size_t                      max_seq_len_tile,
                          void*                       partial_out,
                          float*                      partial_sum,
                          float*                      partial_max,
                          int*                        block_counter,
                          KVBlockArray                kv_block_array,
                          cudaStream_t                stream) {
    size_t      batch_size        = params.common.decoder_batch_size;
    size_t      step              = params.common.decoder_max_seq_len + 1;
    size_t      local_head_num    = params.configs.head_num;
    size_t      local_head_num_kv = params.configs.kv_head_num;
    size_t      size_per_head     = params.configs.size_per_head;
    const auto& output            = params.output;

    const T* qkv_buf_ptr  = params.input.data<T>();
    void*    attn_out_ptr = nullptr;
    attn_out_ptr          = output.data();

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
}

AttentionModuleOutput ROCmDevice::decoderSelfAttention(const AttentionModuleParams& params) {
    auto      datatype         = params.input.type();
    size_t    max_seq_len_tile = 0;
    BufferPtr partial_out      = nullptr;
    BufferPtr partial_sum      = nullptr;
    BufferPtr partial_max      = nullptr;
    BufferPtr block_counter    = nullptr;

    size_t batch_size     = params.common.decoder_batch_size;
    size_t local_head_num = params.configs.head_num;
    size_t size_per_head  = params.configs.size_per_head;

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

    RUNTIME_ASSERT_OP_ARG(params.common.kv_cache, "kv cache can not be null for decoder self-attention");
    const auto max_blocks_per_batch = params.common.kv_cache->kv_cache_block_id->shape()[1];
    auto       kv_cache_offset      = allocateBuffer(
        {DataType::TYPE_INT32, {batch_size, 1, 2, max_blocks_per_batch}, AllocationType::DEVICE}, {"kv_cache_offset"});

    if (init_params_.use_aiter_pa) {
        PrefixPromptBatchWeightsParam prefix_prompt_param;
        KVBlockArray                  kv_block_array =
            getKVBlockArray(params,
                            *kv_cache_offset,
                            batch_size,
                            params.common.kv_cache->k_cache_buffer->type() == DataType::TYPE_FP8_E4M3);
        prefix_prompt_param.kv_block_array = kv_block_array;

        auto   token_num          = params.input.shape()[0];
        auto   decoder_batch_size = params.common.decoder_batch_size;
        auto   head_num           = params.configs.head_num;
        auto   kv_head_num        = params.configs.kv_head_num;
        size_t seq_len            = 1;

        auto q_output = allocateBuffer(
            {params.input.type(), {batch_size, head_num, size_per_head}, AllocationType::DEVICE}, {"q_output"});

        bool        store_qkv        = false;
        bool        store_q          = true;
        bool        store_kv         = false;
        bool        store_cache      = params.common.kv_cache.has_value();
        const auto* sequence_lengths = params.common.sequence_lengths->data<int>();
        const auto* input_lengths    = params.common.input_lengths->data<int>();

        bool skip_add_bias_transpose = (params.configs.rope_config.style == RopeStyle::No && !params.common.kv_cache
                                        && !params.configs.fuse_qkv_add_bias);
        printBufferData(*params.common.input_lengths, "input_lengths");
        if (!skip_add_bias_transpose) {

            bool                 use_rope_cache = params.configs.rope_config.style == RopeStyle::Base;
            static torch::Tensor rope_cache;
            std::call_once(rope_cache_flag, [&]() {
                if (use_rope_cache) {
                    rope_cache = getRopeCache(params.configs.rope_config, init_params_.max_seq_len);
                }
            });

            if (init_params_.use_asm_pa) {
                DISPATCH_CUDA_FUNCTION_DATA_TYPE(
                    datatype,
                    invokeAddFusedQKVBiasTransposeDecode,
                    q_output->data(),
                    nullptr,
                    nullptr,
                    &prefix_prompt_param,
                    input_lengths,
                    params.input.data(),
                    nullptr,
                    params.common.position_ids ? params.common.position_ids->data<int>() : nullptr,
                    params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias ?
                        params.weights.qkv_weight->bias->data() :
                        nullptr,
                    /*params.common.padding_offset->data<int>(),*/ nullptr,
                    /*params.common.cu_seqlens->data<int>(),*/ nullptr,
                    params.common.sequence_lengths->data<int>(),
                    batch_size,
                    seq_len,
                    token_num,
                    head_num,
                    kv_head_num,
                    size_per_head,
                    params.configs.rope_config,
                    params.configs.use_logn_attn,
                    nullptr,
                    0,
                    false,
                    store_qkv,
                    store_q,
                    store_kv,
                    store_cache,
                    use_rope_cache && rope_cache.defined() ? static_cast<float2*>(rope_cache.data_ptr()) : nullptr,
                    stream_);
            } else {
                DISPATCH_CUDA_FUNCTION_DATA_TYPE(
                    datatype,
                    invokeAddFusedQKVBiasTransposeDecodeV1,
                    q_output->data(),
                    nullptr,
                    nullptr,
                    &prefix_prompt_param,
                    input_lengths,
                    params.input.data(),
                    nullptr,
                    params.common.position_ids ? params.common.position_ids->data<int>() : nullptr,
                    params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias ?
                        params.weights.qkv_weight->bias->data() :
                        nullptr,
                    /*params.common.padding_offset->data<int>(),*/ nullptr,
                    /*params.common.cu_seqlens->data<int>(),*/ nullptr,
                    params.common.sequence_lengths->data<int>(),
                    batch_size,
                    seq_len,
                    token_num,
                    head_num,
                    kv_head_num,
                    size_per_head,
                    params.configs.rope_config,
                    params.configs.use_logn_attn,
                    nullptr,
                    0,
                    false,
                    store_qkv,
                    store_q,
                    store_kv,
                    store_cache,
                    use_rope_cache && rope_cache.defined() ? static_cast<float2*>(rope_cache.data_ptr()) : nullptr,
                    stream_);
            }
            check_cuda_error();
            DEBUG_PRINT_PARAMS(params, this, "decode_writeKVCache", q_output);
            if (init_params_.use_asm_pa) {
                runAiterAsmPA(params, this, *q_output);
            } else {
                runAiterPA(params, this, *q_output);
            }
            check_cuda_error();
        }
    } else {
        KVBlockArray kv_block_array =
            getKVBlockArray(params,
                            *kv_cache_offset,
                            batch_size,
                            params.common.kv_cache->k_cache_buffer->type() == DataType::TYPE_FP8_E4M3);

        DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                         selfAttentionwrapper,
                                         params,
                                         use_multi_block_mode,
                                         max_seq_len_tile,
                                         partial_out_data,
                                         partial_sum_data,
                                         partial_max_data,
                                         block_counter_data,
                                         kv_block_array,
                                         stream_);
        check_cuda_error();
        DEBUG_PRINT_PARAMS(params, this, "decode_attn");
    }
}

}  // namespace rtp_llm
