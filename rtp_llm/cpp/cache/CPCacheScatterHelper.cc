#include <algorithm>
#include <cuda_runtime.h>

#include "rtp_llm/cpp/cache/CPCacheScatterHelper.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/kernels/cp_cache_scatter_kernel.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

bool inferPackedIndexerScaleLayout(const CacheConfig& cache_config,
                                   int                scale_bytes_per_token,
                                   int&               quant_bytes_per_token,
                                   int&               tail_scale_bytes_per_token) {
    if (!cache_config.use_mla || scale_bytes_per_token <= 0) {
        return false;
    }

    // Indexer K cache in kv_scale uses packed block layout:
    //   [all token fp8 K][all token fp32 scales]
    // For each quant group, 128 bytes FP8 data are paired with 4 bytes scale,
    // so total_bytes_per_token = quant_bytes_per_token + tail_scale_bytes_per_token
    // and quant : scale = 32 : 1.
    if (scale_bytes_per_token % 33 != 0) {
        return false;
    }

    tail_scale_bytes_per_token = scale_bytes_per_token / 33;
    quant_bytes_per_token      = scale_bytes_per_token - tail_scale_bytes_per_token;
    return quant_bytes_per_token > 0 && tail_scale_bytes_per_token > 0;
}

}  // namespace

// --- StagingPlan RAII destructor ---

CPCacheScatterHelper::StagingPlan::~StagingPlan() {
    if (!staging_block_ids.empty() && block_pool_) {
        block_pool_->requestFree(staging_block_ids);
        RTP_LLM_LOG_DEBUG("StagingPlan dtor: returned %zu staging blocks to pool", staging_block_ids.size());
    }
}

// --- CPCacheScatterHelper ---

CPCacheScatterHelper::CPCacheScatterHelper(KVCacheManager* cache_manager, DeviceBase* device):
    cache_manager_(cache_manager), device_(device) {}

CPCacheScatterHelper::~CPCacheScatterHelper() {
    if (scatter_stream_) {
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(scatter_stream_));
        scatter_stream_ = nullptr;
    }
}

void* CPCacheScatterHelper::getOrCreateScatterStream() {
    std::call_once(scatter_stream_init_, [this]() {
        cudaStream_t stream = nullptr;
        auto         err    = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        RTP_LLM_CHECK_WITH_INFO(
            err == cudaSuccess, "Failed to create scatter CUDA stream: %s", cudaGetErrorString(err));
        scatter_stream_ = reinterpret_cast<void*>(stream);
    });
    return scatter_stream_;
}

std::unique_ptr<CPCacheScatterHelper::StagingPlan>
CPCacheScatterHelper::prepareStagingPlan(int vblock_count, int cp_size, size_t layer_num) {
    const int staging_cnt = vblock_count * cp_size;

    auto block_pool = cache_manager_->getBlockPool();
    RTP_LLM_CHECK_WITH_INFO(block_pool != nullptr, "BlockPool is null");

    auto staging_ids = block_pool->malloc(staging_cnt);
    if (static_cast<int>(staging_ids.size()) < staging_cnt) {
        if (!staging_ids.empty()) {
            block_pool->requestFree(staging_ids);
        }
        RTP_LLM_CHECK_WITH_INFO(false, "CP staging: need %d blocks but only got %zu", staging_cnt, staging_ids.size());
    }

    auto plan               = std::make_unique<StagingPlan>();
    plan->staging_block_ids = std::move(staging_ids);
    plan->vblock_count      = vblock_count;
    plan->cp_size           = cp_size;
    plan->block_pool_       = block_pool.get();

    // Resolve per-layer GPU addresses for each staging block.
    plan->layer_infos.resize(layer_num);
    for (size_t layer_id = 0; layer_id < layer_num; ++layer_id) {
        auto& layer_info = plan->layer_infos[layer_id];
        layer_info.infos.resize(staging_cnt);
        for (int s = 0; s < staging_cnt; ++s) {
            auto parts          = cache_manager_->convertIndexToBuffer(plan->staging_block_ids[s], layer_id, 1, 0);
            layer_info.infos[s] = parts[0];
        }
    }

    RTP_LLM_LOG_DEBUG("CPCacheScatterHelper: prepared staging plan with %d blocks, vblock_count=%d, cp_size=%d",
                      staging_cnt,
                      vblock_count,
                      cp_size);
    return plan;
}

void CPCacheScatterHelper::scatterAndRelease(std::unique_ptr<StagingPlan> plan,
                                             const GroupBlockIds&         block_ids_by_group,
                                             const CacheConfig&           cache_config,
                                             size_t                       layer_num,
                                             int                          total_tokens) {
    RTP_LLM_CHECK_WITH_INFO(plan != nullptr, "StagingPlan is null");

    device_->preRun();
    auto       stream_opaque = getOrCreateScatterStream();
    auto       stream        = reinterpret_cast<cudaStream_t>(stream_opaque);
    const bool use_hybrid    = cache_config.groupNums() > 1;
    const int  block_size    = static_cast<int>(cache_config.seq_size_per_block);
    const int  vblock_count  = plan->vblock_count;
    const int  cp_size       = plan->cp_size;
    const int  staging_cnt   = vblock_count * cp_size;
    const int  max_tokens    = vblock_count * cp_size * block_size;

    RTP_LLM_CHECK_WITH_INFO(total_tokens >= 0, "[SCATTER-PAGED] invalid total_tokens=%d", total_tokens);
    RTP_LLM_CHECK_WITH_INFO(total_tokens <= max_tokens,
                            "[SCATTER-PAGED] total_tokens=%d exceeds staging capacity=%d",
                            total_tokens,
                            max_tokens);

    RTP_LLM_LOG_DEBUG(
        "[SCATTER-PAGED] begin: layers=%zu vblock_count=%d cp_size=%d block_size=%d staging_cnt=%d total_tokens=%d",
        layer_num,
        vblock_count,
        cp_size,
        block_size,
        staging_cnt,
        total_tokens);

    // Use cudaMalloc directly (not the device caching allocator) for temporary
    // GPU buffers.  The caching allocator is synchronized with the compute stream;
    // allocating through it and then using the memory on scatter_stream_ causes a
    // race: the allocator may hand the same memory to a compute-stream op before
    // scatter_stream_ finishes with it.  These buffers are tiny (a few KB of
    // address tables / block-ID arrays), so the cudaMalloc overhead is negligible.
    std::vector<void*> raw_gpu_ptrs;

    auto cudaRawAlloc = [&](size_t bytes) -> void* {
        void* ptr = nullptr;
        auto  err = cudaMalloc(&ptr, bytes);
        RTP_LLM_CHECK_WITH_INFO(err == cudaSuccess, "cudaMalloc(%zu) failed: %s", bytes, cudaGetErrorString(err));
        raw_gpu_ptrs.push_back(ptr);
        return ptr;
    };

    for (size_t layer_id = 0; layer_id < layer_num; ++layer_id) {
        size_t gid = 0;
        if (use_hybrid && layer_id < cache_config.layer_to_group_id.size()) {
            const int mapped_gid = cache_config.layer_to_group_id[layer_id];
            if (mapped_gid >= 0)
                gid = static_cast<size_t>(mapped_gid);
        }
        const auto& block_ids             = block_ids_by_group[gid]->blocks();
        const int   decode_blocks         = static_cast<int>(block_ids.size());
        const int   layer_capacity_tokens = decode_blocks * block_size;
        const int   layer_total_tokens    = std::min(total_tokens, layer_capacity_tokens);

        if (layer_total_tokens <= 0 || decode_blocks <= 0) {
            continue;
        }

        if (layer_capacity_tokens < total_tokens) {
            // TODO(xinfei.sxf): support offset/window-based CP scatter for LINEAR
            // groups or other partial-layer cache layouts instead of clamping to
            // the visible decode block window.
            RTP_LLM_LOG_WARNING("[SCATTER] layer %zu decode capacity (%d tokens, %d blocks) "
                                "is smaller than total_tokens(%d); clamp scatter range to avoid OOB",
                                layer_id,
                                layer_capacity_tokens,
                                decode_blocks,
                                total_tokens);
        }

        if (decode_blocks < staging_cnt) {
            RTP_LLM_LOG_WARNING("[SCATTER] decode_blocks(%d) < staging_cnt(%d) at layer %zu; "
                                "this can be valid when total_tokens(%d) is smaller than a full CP virtual block",
                                decode_blocks,
                                staging_cnt,
                                layer_id,
                                layer_total_tokens);
        }

        auto   sample_parts = cache_manager_->convertIndexToBuffer(block_ids[0], layer_id, 1, 0);
        size_t kv_stride    = sample_parts[0].size_bytes;
        int    elem_stride  = static_cast<int>(kv_stride / block_size);
        if (elem_stride <= 0) {
            continue;
        }

        int min_dst_bid = *std::min_element(block_ids.begin(), block_ids.end());
        int min_src_bid = *std::min_element(plan->staging_block_ids.begin(), plan->staging_block_ids.end());
        RTP_LLM_CHECK_WITH_INFO(min_dst_bid >= 0,
                                "[SCATTER] invalid dst block id %d at layer %zu (NULL_BLOCK_IDX?)",
                                min_dst_bid,
                                layer_id);
        RTP_LLM_CHECK_WITH_INFO(
            min_src_bid >= 0, "[SCATTER] invalid src staging block id %d at layer %zu", min_src_bid, layer_id);

        int max_dst_bid     = *std::max_element(block_ids.begin(), block_ids.end());
        int max_src_bid     = *std::max_element(plan->staging_block_ids.begin(), plan->staging_block_ids.end());
        int addr_table_size = std::max(max_dst_bid, max_src_bid) + 1;

        std::vector<void*> dst_addrs(addr_table_size, nullptr);
        std::vector<void*> src_addrs(addr_table_size, nullptr);

        for (int b = 0; b < decode_blocks; ++b) {
            auto parts              = cache_manager_->convertIndexToBuffer(block_ids[b], layer_id, 1, 0);
            dst_addrs[block_ids[b]] = parts[0].addr;
            RTP_LLM_CHECK_WITH_INFO(
                parts[0].addr != nullptr, "[SCATTER] null dst addr for block_id=%d layer=%zu", block_ids[b], layer_id);
        }
        for (int s = 0; s < staging_cnt; ++s) {
            auto parts = cache_manager_->convertIndexToBuffer(plan->staging_block_ids[s], layer_id, 1, 0);
            src_addrs[plan->staging_block_ids[s]] = parts[0].addr;
            RTP_LLM_CHECK_WITH_INFO(parts[0].addr != nullptr,
                                    "[SCATTER] null src addr for staging_block_id=%d layer=%zu",
                                    plan->staging_block_ids[s],
                                    layer_id);
        }

        void* addrs_gpu_ptr     = cudaRawAlloc(addr_table_size * sizeof(void*));
        void* dst_ids_gpu_ptr   = cudaRawAlloc(decode_blocks * sizeof(int));
        void* src_addrs_gpu_ptr = cudaRawAlloc(addr_table_size * sizeof(void*));
        void* src_ids_gpu_ptr   = cudaRawAlloc(staging_cnt * sizeof(int));

        cudaMemcpyAsync(
            addrs_gpu_ptr, dst_addrs.data(), addr_table_size * sizeof(void*), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dst_ids_gpu_ptr, block_ids.data(), decode_blocks * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(
            src_addrs_gpu_ptr, src_addrs.data(), addr_table_size * sizeof(void*), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(
            src_ids_gpu_ptr, plan->staging_block_ids.data(), staging_cnt * sizeof(int), cudaMemcpyHostToDevice, stream);

        invokeCPCacheScatterPaged(reinterpret_cast<void**>(addrs_gpu_ptr),
                                  reinterpret_cast<int*>(dst_ids_gpu_ptr),
                                  reinterpret_cast<void**>(src_addrs_gpu_ptr),
                                  reinterpret_cast<int*>(src_ids_gpu_ptr),
                                  vblock_count,
                                  cp_size,
                                  block_size,
                                  layer_total_tokens,
                                  elem_stride,
                                  addr_table_size,
                                  stream);

        // Scale scatter (same pattern)
        if (sample_parts.size() == 2 && sample_parts[1].size_bytes > 0) {
            size_t sc_stride = sample_parts[1].size_bytes;
            int    sc_elem   = static_cast<int>(sc_stride / block_size);
            if (sc_elem > 0) {
                int        quant_bytes_per_token      = 0;
                int        tail_scale_bytes_per_token = 0;
                const bool packed_indexer_scale       = inferPackedIndexerScaleLayout(
                    cache_config, sc_elem, quant_bytes_per_token, tail_scale_bytes_per_token);
                std::vector<void*> sc_dst_addrs(addr_table_size, nullptr);
                std::vector<void*> sc_src_addrs(addr_table_size, nullptr);
                for (int b = 0; b < decode_blocks; ++b) {
                    auto parts = cache_manager_->convertIndexToBuffer(block_ids[b], layer_id, 1, 0);
                    if (parts.size() == 2) {
                        sc_dst_addrs[block_ids[b]] = parts[1].addr;
                        RTP_LLM_CHECK_WITH_INFO(parts[1].addr != nullptr,
                                                "[SCATTER] null scale dst addr for block_id=%d layer=%zu",
                                                block_ids[b],
                                                layer_id);
                    }
                }
                for (int s = 0; s < staging_cnt; ++s) {
                    auto parts = cache_manager_->convertIndexToBuffer(plan->staging_block_ids[s], layer_id, 1, 0);
                    if (parts.size() == 2) {
                        sc_src_addrs[plan->staging_block_ids[s]] = parts[1].addr;
                        RTP_LLM_CHECK_WITH_INFO(parts[1].addr != nullptr,
                                                "[SCATTER] null scale src addr for staging_block_id=%d layer=%zu",
                                                plan->staging_block_ids[s],
                                                layer_id);
                    }
                }

                void* sc_dst_ptr = cudaRawAlloc(addr_table_size * sizeof(void*));
                void* sc_src_ptr = cudaRawAlloc(addr_table_size * sizeof(void*));

                cudaMemcpyAsync(
                    sc_dst_ptr, sc_dst_addrs.data(), addr_table_size * sizeof(void*), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(
                    sc_src_ptr, sc_src_addrs.data(), addr_table_size * sizeof(void*), cudaMemcpyHostToDevice, stream);

                if (packed_indexer_scale) {
                    invokeCPCacheScatterPagedPackedScale(reinterpret_cast<void**>(sc_dst_ptr),
                                                         reinterpret_cast<int*>(dst_ids_gpu_ptr),
                                                         reinterpret_cast<void**>(sc_src_ptr),
                                                         reinterpret_cast<int*>(src_ids_gpu_ptr),
                                                         vblock_count,
                                                         cp_size,
                                                         block_size,
                                                         layer_total_tokens,
                                                         quant_bytes_per_token,
                                                         tail_scale_bytes_per_token,
                                                         addr_table_size,
                                                         stream);
                } else {
                    invokeCPCacheScatterPaged(reinterpret_cast<void**>(sc_dst_ptr),
                                              reinterpret_cast<int*>(dst_ids_gpu_ptr),
                                              reinterpret_cast<void**>(sc_src_ptr),
                                              reinterpret_cast<int*>(src_ids_gpu_ptr),
                                              vblock_count,
                                              cp_size,
                                              block_size,
                                              layer_total_tokens,
                                              sc_elem,
                                              addr_table_size,
                                              stream);
                }
            }
        }
    }

    auto err = cudaStreamSynchronize(stream);
    RTP_LLM_CHECK_WITH_INFO(err == cudaSuccess, "scatter stream sync failed: %s", cudaGetErrorString(err));
    for (void* ptr : raw_gpu_ptrs) {
        cudaFree(ptr);
    }

    // Release staging blocks explicitly (StagingPlan dtor would also do this,
    // but explicit release lets us log at the right time).
    if (plan->block_pool_ && !plan->staging_block_ids.empty()) {
        plan->block_pool_->requestFree(plan->staging_block_ids);
        RTP_LLM_LOG_DEBUG("CPCacheScatterHelper: released %zu staging blocks after scatter",
                          plan->staging_block_ids.size());
        plan->staging_block_ids.clear();
    }

    RTP_LLM_LOG_DEBUG("[SCATTER-PAGED] all layers done");
}

}  // namespace rtp_llm
