#include "rtp_llm/cpp/cache/CacheStoreWriter.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ErrorCodeUtil.h"
#include "rtp_llm/cpp/runtime/CudaRuntime.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <utility>

namespace rtp_llm {

void runtimeWriteCacheStore(const torch_ext::PyCacheStoreInputs& cache_store_inputs,
                            const torch_ext::LayerKVCache&       layer_kv,
                            const CacheConfig&                   cache_config,
                            std::shared_ptr<CacheStore>          cache_store,
                            size_t                               cache_model_id,
                            int                                  cp_rank,
                            int                                  cp_size,
                            std::shared_ptr<torch::Event>        pre_created_event) {
    const auto& param = cache_store_inputs;
    const auto  requireHostTensor =
        [](const torch::Tensor& tensor, const char* name, int64_t expected_dim, c10::ScalarType expected_type) {
            RTP_LLM_CHECK_WITH_INFO(tensor.defined(), "cache-store %s must be defined", name);
            RTP_LLM_CHECK_WITH_INFO(tensor.dim() == expected_dim,
                                    "cache-store %s must be %ld-D, got dim=%ld",
                                    name,
                                    expected_dim,
                                    tensor.dim());
            RTP_LLM_CHECK_WITH_INFO(tensor.device().is_cpu(), "cache-store %s must be a CPU tensor", name);
            RTP_LLM_CHECK_WITH_INFO(tensor.scalar_type() == expected_type,
                                    "cache-store %s must use %s, got %s",
                                    name,
                                    c10::toString(expected_type),
                                    c10::toString(tensor.scalar_type()));
        };

    requireHostTensor(param.request_id, "request_id", 1, torch::kInt64);
    const size_t context_batch_size = static_cast<size_t>(param.request_id.numel());
    if (context_batch_size == 0) {
        return;
    }
    requireHostTensor(param.input_lengths_host, "input_lengths_host", 1, torch::kInt32);
    requireHostTensor(param.prefix_lengths_host, "prefix_lengths_host", 1, torch::kInt32);
    requireHostTensor(param.host_kv_cache_offset, "host_kv_cache_offset", 2, torch::kInt32);
    requireHostTensor(param.request_pd_separation, "request_pd_separation", 1, torch::kBool);
    requireHostTensor(param.cache_keys, "cache_keys", 2, torch::kInt64);

    if (!cache_store) {
        RTP_LLM_LOG_DEBUG("cache_store is null, skip writeCacheStore");
        return;
    }

    // Wait for the CUDA event before reading pinned-host metadata.
    // The event was recorded on the main stream AFTER both the async D2H
    // copies (metadata) and KV cache writes were enqueued, so blocking
    // here guarantees all pinned buffers are populated.
    if (pre_created_event) {
        while (!pre_created_event->query()) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

    RTP_LLM_CHECK_WITH_INFO(
        !layer_kv.tag.empty(), "cache-store write requires a cache tag for layer=%d", layer_kv.layer_id);

    const size_t max_blocks_per_batch = static_cast<size_t>(param.host_kv_cache_offset.size(1));

    const auto& group = cache_config.groupForLayer(layer_kv.layer_id, layer_kv.tag);
    RTP_LLM_CHECK_WITH_INFO(
        group.spec != nullptr, "cache-store tag=%s has no KVCacheSpec attached", layer_kv.tag.c_str());

    // Physical address stride and logical transfer length differ for a shared pool:
    // blocks use the allocation-wide stride, while each tag transfers only its group-local bytes.
    const bool use_group_local_storage_layout = cache_config.use_independent_block_pools;
    // LayerKVCache may expose kernel-page views; CacheStore keys and block IDs use physical pages.
    const size_t seq_size_per_block = group.seq_size_per_block;
    const size_t kv_block_stride_bytes =
        use_group_local_storage_layout ? group.kv_block_stride_bytes : cache_config.kv_block_stride_bytes;
    const size_t kv_scale_stride_bytes =
        use_group_local_storage_layout ? group.kv_scale_stride_bytes : cache_config.kv_scale_stride_bytes;
    const size_t kv_block_transfer_bytes         = group.kv_block_stride_bytes;
    const size_t kv_scale_transfer_bytes         = group.kv_scale_stride_bytes;
    const bool   use_group_cache_transfer_policy = cache_config.topology().groups().size() > 1;

    RTP_LLM_CHECK_WITH_INFO(
        seq_size_per_block > 0, "cache-store tag=%s has zero tokens_per_block", layer_kv.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(
        kv_block_stride_bytes > 0, "cache-store tag=%s has zero kv block stride", layer_kv.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(
        kv_block_transfer_bytes > 0, "cache-store tag=%s has zero kv transfer bytes", layer_kv.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(kv_block_transfer_bytes <= kv_block_stride_bytes,
                            "cache-store tag=%s transfer bytes=%zu exceed physical stride=%zu",
                            layer_kv.tag.c_str(),
                            kv_block_transfer_bytes,
                            kv_block_stride_bytes);
    RTP_LLM_CHECK_WITH_INFO(kv_scale_transfer_bytes <= kv_scale_stride_bytes,
                            "cache-store tag=%s scale transfer bytes=%zu exceed physical stride=%zu",
                            layer_kv.tag.c_str(),
                            kv_scale_transfer_bytes,
                            kv_scale_stride_bytes);

    auto       kv_cache_data  = static_cast<uint8_t*>(layer_kv.kv_cache_base.data_ptr());
    auto       kv_cache_owner = std::make_shared<torch::Tensor>(layer_kv.kv_cache_base);
    const bool kv_gpu_mem     = layer_kv.kv_cache_base.is_cuda();
    const bool has_kv_scale   = layer_kv.kv_scale_base.defined() && layer_kv.kv_scale_base.numel() > 0
                              && kv_scale_stride_bytes > 0 && kv_scale_transfer_bytes > 0;
    uint8_t*                       kv_scale_data = nullptr;
    std::shared_ptr<torch::Tensor> kv_scale_owner;
    if (has_kv_scale) {
        kv_scale_data  = static_cast<uint8_t*>(layer_kv.kv_scale_base.data_ptr());
        kv_scale_owner = std::make_shared<torch::Tensor>(layer_kv.kv_scale_base);
    }
    const bool kv_scale_gpu_mem = has_kv_scale && layer_kv.kv_scale_base.is_cuda();

    const size_t total_batch_size = static_cast<size_t>(param.input_lengths_host.numel());
    RTP_LLM_CHECK_WITH_INFO(param.prefix_lengths_host.numel() == static_cast<int64_t>(context_batch_size),
                            "cache-store tag=%s prefix_lengths numel=%ld != context batch=%zu",
                            layer_kv.tag.c_str(),
                            param.prefix_lengths_host.numel(),
                            context_batch_size);
    RTP_LLM_CHECK_WITH_INFO(param.request_pd_separation.numel() == static_cast<int64_t>(context_batch_size),
                            "cache-store tag=%s request_pd_separation numel=%ld != context batch=%zu",
                            layer_kv.tag.c_str(),
                            param.request_pd_separation.numel(),
                            context_batch_size);
    RTP_LLM_CHECK_WITH_INFO(total_batch_size >= context_batch_size,
                            "cache-store tag=%s input_lengths numel=%zu < context batch=%zu",
                            layer_kv.tag.c_str(),
                            total_batch_size,
                            context_batch_size);
    RTP_LLM_CHECK_WITH_INFO(param.host_kv_cache_offset.size(0) == static_cast<int64_t>(total_batch_size),
                            "cache-store tag=%s block table rows=%ld != total batch=%zu",
                            layer_kv.tag.c_str(),
                            param.host_kv_cache_offset.size(0),
                            total_batch_size);
    RTP_LLM_CHECK_WITH_INFO(param.cache_keys.size(0) == static_cast<int64_t>(context_batch_size),
                            "cache-store tag=%s cache_keys rows=%ld != context batch=%zu",
                            layer_kv.tag.c_str(),
                            param.cache_keys.size(0),
                            context_batch_size);

    const size_t decoder_batch_size = total_batch_size - context_batch_size;
    // cache_keys is laid out [batch, global_max_blocks]; this logical width is INDEPENDENT
    // of `max_blocks_per_batch` (which is per-group offset width and may be smaller
    // for CP-sharded FULL groups whose offset is rank-local-compact).
    const size_t cache_keys_per_batch  = static_cast<size_t>(param.cache_keys.size(1));
    const auto   host_kv_cache_offset  = param.host_kv_cache_offset.accessor<int32_t, 2>();
    const auto   input_lengths_host    = param.input_lengths_host.accessor<int32_t, 1>();
    const auto   prefix_lengths_host   = param.prefix_lengths_host.accessor<int32_t, 1>();
    const auto   request_ids           = param.request_id.accessor<int64_t, 1>();
    const auto   request_pd_separation = param.request_pd_separation.accessor<bool, 1>();
    const auto   cache_keys            = param.cache_keys.accessor<int64_t, 2>();

    RTP_LLM_LOG_DEBUG("write cache store, context_batch_size is %zu", context_batch_size);
    for (size_t batch_id = 0; batch_id < context_batch_size; ++batch_id) {
        const auto context_index = static_cast<int64_t>(batch_id);
        if (!request_pd_separation[context_index]) {
            continue;
        }

        const bool uses_cp_canonical_keys = cp_size > 1 && group.policy.cp_mapping != CpBlockMappingMode::NONE
                                            && seq_size_per_block % static_cast<size_t>(cp_size) == 0;
        const size_t canonical_seq_size_per_block =
            uses_cp_canonical_keys ? seq_size_per_block / static_cast<size_t>(cp_size) : seq_size_per_block;
        const int prefix_length = prefix_lengths_host[context_index];
        RTP_LLM_CHECK_WITH_INFO(prefix_length % static_cast<int>(canonical_seq_size_per_block) == 0,
                                "cache-store tag=%s prefix_length=%d is not aligned to canonical "
                                "tokens_per_block=%zu (physical tokens_per_block=%zu, cp_size=%d)",
                                layer_kv.tag.c_str(),
                                prefix_length,
                                canonical_seq_size_per_block,
                                seq_size_per_block,
                                cp_size);

        const auto input_index     = static_cast<int64_t>(decoder_batch_size + batch_id);
        const int  input_length    = input_lengths_host[input_index];
        const int  reuse_block_num = prefix_length / static_cast<int>(seq_size_per_block);
        const int  block_num =
            (input_length + static_cast<int>(seq_size_per_block) - 1) / static_cast<int>(seq_size_per_block);
        const int canonical_reuse_block_num = prefix_length / static_cast<int>(canonical_seq_size_per_block);
        const int canonical_block_num       = (input_length + static_cast<int>(canonical_seq_size_per_block) - 1)
                                        / static_cast<int>(canonical_seq_size_per_block);
        const int canonical_total_blocks = canonical_block_num + canonical_reuse_block_num;
        const int total_blocks =
            uses_cp_canonical_keys ? (canonical_total_blocks + cp_size - 1) / cp_size : block_num + reuse_block_num;
        if (total_blocks <= 0) {
            continue;
        }

        const int64_t request_id     = request_ids[context_index];
        auto          event          = pre_created_event ? pre_created_event : runtimeCreateEvent();
        auto          request_blocks = std::make_shared<RequestBlockBuffer>(std::to_string(request_id), event);
        RTP_LLM_LOG_DEBUG(
            "write cache store, request id is %ld, blocks num is %d", static_cast<long>(request_id), total_blocks);

        auto addBlock = [&](int key_index, int offset_index) {
            RTP_LLM_CHECK_WITH_INFO(offset_index >= 0 && offset_index < static_cast<int>(max_blocks_per_batch),
                                    "invalid block offset_index=%d (max_blocks_per_batch=%zu)",
                                    offset_index,
                                    max_blocks_per_batch);
            RTP_LLM_CHECK_WITH_INFO(key_index >= 0 && key_index < static_cast<int>(cache_keys_per_batch),
                                    "invalid block key_index=%d (cache_keys_per_batch=%zu)",
                                    key_index,
                                    cache_keys_per_batch);
            const std::string cache_key = makeCacheKey(
                cache_model_id,
                std::to_string(cache_keys[static_cast<int64_t>(batch_id)][static_cast<int64_t>(key_index)]),
                layer_kv.layer_id,
                layer_kv.tag);
            const int32_t block_id = host_kv_cache_offset[input_index][static_cast<int64_t>(offset_index)];
            // Host block-offset tables use -1 as the null block sentinel.
            if (block_id == -1) {
                RTP_LLM_LOG_DEBUG(
                    "PD_CACHE_KEY_WRITE_SKIP_NULL key=kv_%s request_id=%ld tag=%s layer=%d cp_rank=%d cp_size=%d "
                    "key_index=%d offset_index=%d block_id=%d",
                    cache_key.c_str(),
                    static_cast<long>(request_id),
                    layer_kv.tag.c_str(),
                    layer_kv.layer_id,
                    cp_rank,
                    cp_size,
                    key_index,
                    offset_index,
                    block_id);
                return;
            }

            if (cp_size > 1 && group.policy.cp_slice != CpBlockSliceMode::NONE) {
                RTP_LLM_CHECK_WITH_INFO(cp_rank >= 0 && cp_rank < cp_size,
                                        "cache-store tag=%s invalid cp_rank=%d cp_size=%d",
                                        layer_kv.tag.c_str(),
                                        cp_rank,
                                        cp_size);
                // The prefill topology already materializes each rank's local
                // STATE/SWA row. Send that complete local row from offset zero;
                // decode applies the peer-rank offset in the corresponding
                // full row. Dividing here would slice an already-sliced row.
            }

            const bool use_opaque_key_prefix = cache_config.use_opaque_kv_cache_store || use_group_cache_transfer_policy
                                               || group.spec->type == KVCacheSpecType::MultiHeadLatentAttention;
            void*                 kv_addr = kv_cache_data + static_cast<size_t>(block_id) * kv_block_stride_bytes;
            std::shared_ptr<void> kv_block_addr(kv_cache_owner, kv_addr);
            RTP_LLM_LOG_DEBUG("PD_CACHE_KEY_WRITE_BLOCK key=kv_%s request_id=%ld tag=%s layer=%d cp_rank=%d "
                              "cp_size=%d cp_slice=%d key_index=%d offset_index=%d block_id=%d addr=%p "
                              "physical_stride=%zu len=%zu",
                              cache_key.c_str(),
                              static_cast<long>(request_id),
                              layer_kv.tag.c_str(),
                              layer_kv.layer_id,
                              cp_rank,
                              cp_size,
                              static_cast<int>(group.policy.cp_slice),
                              key_index,
                              offset_index,
                              block_id,
                              kv_addr,
                              kv_block_stride_bytes,
                              kv_block_transfer_bytes);
            if (use_opaque_key_prefix) {
                request_blocks->addBlock(
                    "kv_" + cache_key, kv_block_addr, static_cast<uint32_t>(kv_block_transfer_bytes), kv_gpu_mem, true);
            } else {
                RTP_LLM_CHECK_WITH_INFO(kv_block_transfer_bytes % 2 == 0,
                                        "KV transfer bytes must split evenly into K/V");
                const auto            kv_half = static_cast<uint32_t>(kv_block_transfer_bytes / 2);
                std::shared_ptr<void> k_block_addr(kv_cache_owner, kv_addr);
                std::shared_ptr<void> v_block_addr(kv_cache_owner, static_cast<uint8_t*>(kv_addr) + kv_half);
                request_blocks->addBlock("k_" + cache_key, k_block_addr, kv_half, kv_gpu_mem, true);
                request_blocks->addBlock("v_" + cache_key, v_block_addr, kv_half, kv_gpu_mem, true);
            }

            if (kv_scale_data) {
                void* kv_scale_addr = kv_scale_data + static_cast<size_t>(block_id) * kv_scale_stride_bytes;
                std::shared_ptr<void> kv_scale_block_addr(kv_scale_owner, kv_scale_addr);
                if (use_opaque_key_prefix) {
                    request_blocks->addBlock("kv_scale_" + cache_key,
                                             kv_scale_block_addr,
                                             static_cast<uint32_t>(kv_scale_transfer_bytes),
                                             kv_scale_gpu_mem,
                                             true);
                } else {
                    RTP_LLM_CHECK_WITH_INFO(kv_scale_transfer_bytes % 2 == 0,
                                            "scale transfer bytes must split evenly into K/V");
                    const auto            sc_half = static_cast<uint32_t>(kv_scale_transfer_bytes / 2);
                    std::shared_ptr<void> k_scale_block_addr(kv_scale_owner, kv_scale_addr);
                    std::shared_ptr<void> v_scale_block_addr(kv_scale_owner,
                                                             static_cast<uint8_t*>(kv_scale_addr) + sc_half);
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
        // Clamp by cache_keys_per_batch (global width) -- NOT max_blocks_per_batch,
        // which under CP shard is the local-compact width for FULL groups.
        const auto block_plan = buildCacheStorePlan(
            group.policy,
            static_cast<size_t>(std::min<int>(canonical_total_blocks, static_cast<int>(cache_keys_per_batch))),
            /*reuse_block_size=*/0,
            use_group_cache_transfer_policy,
            cp_rank,
            cp_size);
        for (const auto& pair : block_plan) {
            addBlock(pair.key_index, pair.offset_index);
        }

        auto storeCallback = [layer_id = layer_kv.layer_id,
                              cache_model_id,
                              tag = layer_kv.tag,
                              request_id,
                              request_blocks](bool success, CacheStoreErrorCode ec) {
            if (!success) {
                RTP_LLM_LOG_WARNING("PD_CACHE_KEY_WRITE_FAILED request_id=%ld model_id=%zu local_layer_id=%d tag=%s "
                                    "error_code=%d error=%s buffer={%s}",
                                    static_cast<long>(request_id),
                                    cache_model_id,
                                    layer_id,
                                    tag.c_str(),
                                    static_cast<int>(ec),
                                    ErrorCodeToString(transCacheStoreErrorCode(ec)).c_str(),
                                    request_blocks->debugInfo().c_str());
            }
        };
        if (request_blocks->getBlocksCount() > 0) {
            cache_store->store(request_blocks, std::move(storeCallback));
        } else {
            RTP_LLM_LOG_DEBUG("skip cache store because all selected blocks are null, request id [%ld], layer id [%d]",
                              static_cast<long>(request_id),
                              layer_kv.layer_id);
        }
    }
}

}  // namespace rtp_llm
