#include "rtp_llm/cpp/cache/CacheStoreWriter.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ErrorCodeUtil.h"
#include "rtp_llm/cpp/runtime/CudaRuntime.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <algorithm>
#include <utility>

namespace rtp_llm {

void runtimeWriteCacheStore(const CacheStoreInputs&     cache_store_inputs,
                            const KvCacheInfo&          kv_cache,
                            bool                        mla_kvcache,
                            std::shared_ptr<CacheStore> cache_store) {
    if (cache_store_inputs.warmup) {
        RTP_LLM_LOG_DEBUG("is warmup, so ignore writeCacheStore");
        return;
    }
    if (!cache_store_inputs.pd_separation || cache_store_inputs.context_batch_size == 0) {
        RTP_LLM_LOG_DEBUG("pd_separation = %d, context_batch_size = %d, so ignore writeCacheStore",
                          cache_store_inputs.pd_separation,
                          cache_store_inputs.context_batch_size);
        return;
    }
    if (!cache_store) {
        RTP_LLM_LOG_DEBUG("cache_store is null, skip writeCacheStore");
        return;
    }

    // Wait for the CUDA event before reading pinned-host metadata.
    // The event was recorded on the main stream AFTER both the async D2H
    // copies (metadata) and KV cache writes were enqueued, so blocking
    // here guarantees all pinned buffers are populated.
    if (cache_store_inputs.pre_created_event) {
        cache_store_inputs.pre_created_event->synchronize();
    }

    const auto& param = cache_store_inputs;

    RTP_LLM_CHECK_WITH_INFO(param.host_kv_cache_offset.defined(), "failed to get host_kv_cache_offset");
    const int32_t* offset_addr          = nullptr;
    size_t         max_blocks_per_batch = 0;

    RTP_LLM_CHECK_WITH_INFO(!param.tag.empty(), "cache-store write requires a cache tag for layer=%d", param.layer_id);
    RTP_LLM_CHECK_WITH_INFO(param.host_kv_cache_offset.dim() == 2,
                            "cache-store block table for tag=%s must be group-local [batch, blocks], got dim=%ld",
                            param.tag.c_str(),
                            param.host_kv_cache_offset.dim());
    max_blocks_per_batch = param.host_kv_cache_offset.size(1);
    offset_addr          = param.host_kv_cache_offset.data_ptr<int32_t>();

    const auto policy_it = param.kv_cache_group_policies.find(param.tag);
    RTP_LLM_CHECK_WITH_INFO(policy_it != param.kv_cache_group_policies.end(),
                            "cache-store metadata has no group policy for tag=%s",
                            param.tag.c_str());
    const CacheGroupPolicy group_policy                    = policy_it->second;
    const bool             use_group_cache_transfer_policy = param.kv_cache_group_policies.size() > 1;

    const auto seq_it = param.tokens_per_block_by_tag.find(param.tag);
    const auto seq_size_per_block =
        seq_it != param.tokens_per_block_by_tag.end() ? seq_it->second : param.tokens_per_block;
    const auto kv_stride_it = param.kv_block_stride_bytes_by_tag.find(param.tag);
    const auto kv_block_stride_bytes =
        kv_stride_it != param.kv_block_stride_bytes_by_tag.end() ? kv_stride_it->second : param.kv_block_stride_bytes;
    const auto scale_stride_it       = param.kv_scale_stride_bytes_by_tag.find(param.tag);
    const auto kv_scale_stride_bytes = scale_stride_it != param.kv_scale_stride_bytes_by_tag.end() ?
                                           scale_stride_it->second :
                                           param.kv_scale_stride_bytes;
    const auto kv_transfer_it        = param.kv_block_transfer_bytes_by_tag.find(param.tag);
    const auto kv_block_transfer_bytes =
        kv_transfer_it != param.kv_block_transfer_bytes_by_tag.end() ? kv_transfer_it->second : kv_block_stride_bytes;
    const auto scale_transfer_it       = param.kv_scale_transfer_bytes_by_tag.find(param.tag);
    const auto kv_scale_transfer_bytes = scale_transfer_it != param.kv_scale_transfer_bytes_by_tag.end() ?
                                             scale_transfer_it->second :
                                             kv_scale_stride_bytes;
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0, "cache-store tag=%s has zero tokens_per_block", param.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(
        kv_block_stride_bytes > 0, "cache-store tag=%s has zero kv block stride", param.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(
        kv_block_transfer_bytes > 0, "cache-store tag=%s has zero kv transfer bytes", param.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(kv_block_transfer_bytes <= kv_block_stride_bytes,
                            "cache-store tag=%s transfer bytes=%zu exceed physical stride=%zu",
                            param.tag.c_str(),
                            kv_block_transfer_bytes,
                            kv_block_stride_bytes);
    RTP_LLM_CHECK_WITH_INFO(kv_scale_transfer_bytes <= kv_scale_stride_bytes,
                            "cache-store tag=%s scale transfer bytes=%zu exceed physical stride=%zu",
                            param.tag.c_str(),
                            kv_scale_transfer_bytes,
                            kv_scale_stride_bytes);
    auto       kv_cache_data  = (uint64_t*)kv_cache.kv_cache_buffer.data_ptr();
    auto       kv_cache_owner = std::make_shared<torch::Tensor>(kv_cache.kv_cache_buffer);
    const bool kv_gpu_mem     = kv_cache.kv_cache_buffer.is_cuda();
    const bool has_kv_scale   = kv_cache.kv_scale_buffer.defined() && kv_cache.kv_scale_buffer.numel() > 0
                              && kv_scale_stride_bytes > 0 && kv_scale_transfer_bytes > 0;
    uint64_t*                      kv_scale_data = nullptr;
    std::shared_ptr<torch::Tensor> kv_scale_owner;
    if (has_kv_scale) {
        kv_scale_data  = (uint64_t*)kv_cache.kv_scale_buffer.data_ptr();
        kv_scale_owner = std::make_shared<torch::Tensor>(kv_cache.kv_scale_buffer);
    }
    const bool kv_scale_gpu_mem = has_kv_scale && kv_cache.kv_scale_buffer.is_cuda();

    RTP_LLM_CHECK_WITH_INFO(param.context_batch_size == static_cast<size_t>(param.request_pd_separation.numel()),
                            "size not same");
    RTP_LLM_CHECK_WITH_INFO(param.context_batch_size == static_cast<size_t>(param.request_id.numel()),
                            "context batch size and request id size is not same");

    RTP_LLM_LOG_DEBUG("write cache store, context_batch_size is %ld", param.context_batch_size);

    // cache_keys is laid out [batch, global_max_blocks]; this stride is INDEPENDENT
    // of `max_blocks_per_batch` (which is per-group offset stride and may be smaller
    // for CP-sharded FULL groups whose offset is rank-local-compact).
    const size_t cache_keys_per_batch =
        param.context_batch_size > 0 ? (param.cache_keys.size() / param.context_batch_size) : 0;

    for (size_t batch_id = 0; batch_id < param.context_batch_size; batch_id++) {
        if (*(param.request_pd_separation.data_ptr<bool>() + batch_id) == false) {
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(param.prefix_lengths_host.defined() && param.input_lengths_host.defined(),
                                "failed to get prefix_length_host and input_length_host for cache store");
        const bool uses_cp_canonical_keys = param.cp_size > 1 && group_policy.cp_mapping != CpBlockMappingMode::NONE
                                            && seq_size_per_block % param.cp_size == 0;
        const size_t canonical_seq_size_per_block =
            uses_cp_canonical_keys ? seq_size_per_block / static_cast<size_t>(param.cp_size) : seq_size_per_block;
        const int prefix_length = param.prefix_lengths_host.data_ptr<int>()[batch_id];
        RTP_LLM_CHECK_WITH_INFO(prefix_length % static_cast<int>(canonical_seq_size_per_block) == 0,
                                "cache-store tag=%s prefix_length=%d is not aligned to canonical tokens_per_block=%zu "
                                "(physical tokens_per_block=%zu, cp_size=%d)",
                                param.tag.c_str(),
                                prefix_length,
                                canonical_seq_size_per_block,
                                seq_size_per_block,
                                param.cp_size);
        int reuse_block_num = prefix_length / seq_size_per_block;
        int block_num =
            (param.input_lengths_host.data_ptr<int>()[param.decoder_batch_size + batch_id] + seq_size_per_block - 1)
            / seq_size_per_block;
        int canonical_reuse_block_num = prefix_length / canonical_seq_size_per_block;
        int canonical_block_num       = (param.input_lengths_host.data_ptr<int>()[param.decoder_batch_size + batch_id]
                                   + canonical_seq_size_per_block - 1)
                                  / canonical_seq_size_per_block;
        auto request_id     = *(param.request_id.data_ptr<int64_t>() + batch_id);
        auto event          = param.pre_created_event ? param.pre_created_event : runtimeCreateEvent();
        auto request_blocks = std::make_shared<RequestBlockBuffer>(std::to_string(request_id), event);
        RTP_LLM_LOG_DEBUG(
            "write cache store, request id is %ld, blocks num is %ld", request_id, block_num + reuse_block_num);

        const int canonical_total_blocks = canonical_block_num + canonical_reuse_block_num;
        const int total_blocks = uses_cp_canonical_keys ? (canonical_total_blocks + param.cp_size - 1) / param.cp_size :
                                                          block_num + reuse_block_num;
        if (total_blocks <= 0) {
            continue;
        }

        auto addBlock = [&](int key_index, int offset_index) {
            RTP_LLM_CHECK_WITH_INFO(offset_index >= 0 && offset_index < static_cast<int>(max_blocks_per_batch),
                                    "invalid block offset_index=%d (max_blocks_per_batch=%zu)",
                                    offset_index,
                                    max_blocks_per_batch);
            RTP_LLM_CHECK_WITH_INFO(key_index >= 0 && key_index < static_cast<int>(cache_keys_per_batch),
                                    "invalid block key_index=%d (cache_keys_per_batch=%zu)",
                                    key_index,
                                    cache_keys_per_batch);
            std::string cache_key = makeCacheKey(param.model_id,
                                                 param.cache_keys[batch_id * cache_keys_per_batch + key_index],
                                                 param.layer_id,
                                                 param.tag);
            auto        block_id =
                *(offset_addr + (param.decoder_batch_size + batch_id) * max_blocks_per_batch + offset_index);
            // Host block-offset tables use -1 as the null block sentinel.
            if (block_id == -1) {
                RTP_LLM_LOG_DEBUG(
                    "PD_CACHE_KEY_WRITE_SKIP_NULL key=kv_%s request_id=%ld tag=%s layer=%d cp_rank=%d cp_size=%d "
                    "key_index=%d offset_index=%d block_id=%d",
                    cache_key.c_str(),
                    request_id,
                    param.tag.c_str(),
                    param.layer_id,
                    param.cp_rank,
                    param.cp_size,
                    key_index,
                    offset_index,
                    block_id);
                return;
            }
            const bool has_policy_cp_slice = param.cp_size > 1 && group_policy.cp_slice != CpBlockSliceMode::NONE;
            if (has_policy_cp_slice) {
                RTP_LLM_CHECK_WITH_INFO(param.cp_rank >= 0 && param.cp_rank < param.cp_size,
                                        "cache-store tag=%s invalid cp_rank=%d cp_size=%d",
                                        param.tag.c_str(),
                                        param.cp_rank,
                                        param.cp_size);
                // The prefill topology already materializes each rank's local
                // STATE/SWA row. Send that complete local row from offset zero;
                // decode applies the peer-rank offset in the corresponding
                // full row. Dividing here would slice an already-sliced row.
            }

            const bool use_opaque_key_prefix =
                param.use_opaque_kv_cache_store || use_group_cache_transfer_policy || mla_kvcache;
            void*                 kv_addr = (void*)((int8_t*)kv_cache_data + block_id * kv_block_stride_bytes);
            std::shared_ptr<void> kv_block_addr(kv_cache_owner, kv_addr);
            RTP_LLM_LOG_DEBUG("PD_CACHE_KEY_WRITE_BLOCK key=kv_%s request_id=%ld tag=%s layer=%d cp_rank=%d "
                              "cp_size=%d cp_slice=%d key_index=%d offset_index=%d block_id=%d addr=%p "
                              "physical_stride=%zu len=%zu",
                              cache_key.c_str(),
                              request_id,
                              param.tag.c_str(),
                              param.layer_id,
                              param.cp_rank,
                              param.cp_size,
                              static_cast<int>(group_policy.cp_slice),
                              key_index,
                              offset_index,
                              block_id,
                              kv_addr,
                              kv_block_stride_bytes,
                              kv_block_transfer_bytes);
            if (use_opaque_key_prefix) {
                request_blocks->addBlock("kv_" + cache_key, kv_block_addr, kv_block_transfer_bytes, kv_gpu_mem, true);
            } else {
                RTP_LLM_CHECK_WITH_INFO(kv_block_transfer_bytes % 2 == 0,
                                        "KV transfer bytes must split evenly into K/V");
                const uint32_t        kv_half = static_cast<uint32_t>(kv_block_transfer_bytes / 2);
                void*                 k_addr  = kv_addr;
                void*                 v_addr  = (void*)((int8_t*)kv_addr + kv_half);
                std::shared_ptr<void> k_block_addr(kv_cache_owner, k_addr);
                std::shared_ptr<void> v_block_addr(kv_cache_owner, v_addr);
                request_blocks->addBlock("k_" + cache_key, k_block_addr, kv_half, kv_gpu_mem, true);
                request_blocks->addBlock("v_" + cache_key, v_block_addr, kv_half, kv_gpu_mem, true);
            }

            if (kv_scale_data) {
                void* kv_scale_addr = (void*)((int8_t*)kv_scale_data + block_id * kv_scale_stride_bytes);

                std::shared_ptr<void> kv_scale_block_addr(kv_scale_owner, kv_scale_addr);
                if (use_opaque_key_prefix) {
                    request_blocks->addBlock(
                        "kv_scale_" + cache_key, kv_scale_block_addr, kv_scale_transfer_bytes, kv_scale_gpu_mem, true);
                } else {
                    RTP_LLM_CHECK_WITH_INFO(kv_scale_transfer_bytes % 2 == 0,
                                            "scale transfer bytes must split evenly into K/V");
                    const uint32_t        sc_half = static_cast<uint32_t>(kv_scale_transfer_bytes / 2);
                    void*                 k_sc    = kv_scale_addr;
                    void*                 v_sc    = (void*)((int8_t*)kv_scale_addr + sc_half);
                    std::shared_ptr<void> k_scale_block_addr(kv_scale_owner, k_sc);
                    std::shared_ptr<void> v_scale_block_addr(kv_scale_owner, v_sc);
                    request_blocks->addBlock(
                        "k_scale_" + cache_key, k_scale_block_addr, sc_half, kv_scale_gpu_mem, true);
                    request_blocks->addBlock(
                        "v_scale_" + cache_key, v_scale_block_addr, sc_half, kv_scale_gpu_mem, true);
                }
            }
        };

        // Under CP sharding, kv_cache_offset can be rank-local-compact while
        // cache_keys stays in the full logical namespace. The common cache
        // policy owns the key/offset projection for both legacy and sharded cases.
        // Clamp by cache_keys_per_batch (global stride) -- NOT max_blocks_per_batch,
        // which under CP shard is the local-compact stride for FULL groups.
        const auto block_plan = buildCacheStorePlan(
            group_policy,
            static_cast<size_t>(std::min<int>(canonical_total_blocks, static_cast<int>(cache_keys_per_batch))),
            /*reuse_block_size=*/0,
            use_group_cache_transfer_policy,
            param.cp_rank,
            param.cp_size);
        for (const auto& pair : block_plan) {
            addBlock(pair.key_index, pair.offset_index);
        }

        auto storeCallback = [layer_id = param.layer_id,
                              model_id = param.model_id,
                              tag      = param.tag,
                              request_id,
                              request_blocks](bool success, CacheStoreErrorCode ec) {
            if (!success) {
                RTP_LLM_LOG_WARNING("PD_CACHE_KEY_WRITE_FAILED request_id=%ld model_id=%zu local_layer_id=%d tag=%s "
                                    "error_code=%d error=%s buffer={%s}",
                                    static_cast<long>(request_id),
                                    model_id,
                                    layer_id,
                                    tag.c_str(),
                                    static_cast<int>(ec),
                                    ErrorCodeToString(transCacheStoreErrorCode(ec)).c_str(),
                                    request_blocks->debugInfo().c_str());
            }
        };
        if (request_blocks->getBlocksCount() > 0) {
            cache_store->store(request_blocks, storeCallback);
        } else {
            RTP_LLM_LOG_DEBUG("skip cache store because all selected blocks are null, request id [%ld], layer id [%d]",
                              request_id,
                              param.layer_id);
        }
    }
}

void execWriteCacheStore(const CacheStoreInputs&     inputs,
                         const KvCacheInfo&          kv_cache,
                         bool                        mla_kvcache,
                         std::shared_ptr<CacheStore> cache_store) {
    runtimeWriteCacheStore(inputs, kv_cache, mla_kvcache, std::move(cache_store));
}

}  // namespace rtp_llm
