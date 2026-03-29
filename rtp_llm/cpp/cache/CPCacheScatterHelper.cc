#include <algorithm>
#include <cuda_runtime.h>

#include "rtp_llm/cpp/cache/CPCacheScatterHelper.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/kernels/cp_cache_scatter_kernel.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// --- StagingPlan RAII destructor ---

CPCacheScatterHelper::StagingPlan::~StagingPlan() {
    if (!staging_block_ids.empty() && block_pool_) {
        block_pool_->requestFree(staging_block_ids);
        RTP_LLM_LOG_DEBUG("StagingPlan dtor: returned %zu staging blocks to pool", staging_block_ids.size());
    }
}

// --- CPCacheScatterHelper ---

CPCacheScatterHelper::CPCacheScatterHelper(KVCacheManager* cache_manager, DeviceBase* device)
    : cache_manager_(cache_manager), device_(device) {}

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
        RTP_LLM_CHECK_WITH_INFO(false,
                                "CP staging: need %d blocks but only got %zu",
                                staging_cnt,
                                staging_ids.size());
    }

    auto plan              = std::make_unique<StagingPlan>();
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
            auto parts                = cache_manager_->convertIndexToBuffer(plan->staging_block_ids[s], layer_id, 1, 0);
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
                                              const CacheConfig&          cache_config,
                                              size_t                      layer_num) {
    RTP_LLM_CHECK_WITH_INFO(plan != nullptr, "StagingPlan is null");

    auto   stream_opaque = getOrCreateScatterStream();
    auto   stream        = reinterpret_cast<cudaStream_t>(stream_opaque);
    const bool use_hybrid   = cache_config.groupNums() > 1;
    const int  block_size   = static_cast<int>(cache_config.seq_size_per_block);
    const int  vblock_count = plan->vblock_count;
    const int  cp_size      = plan->cp_size;
    const int  staging_cnt  = vblock_count * cp_size;

    RTP_LLM_LOG_DEBUG("[SCATTER-PAGED] begin: layers=%zu vblock_count=%d cp_size=%d block_size=%d",
                      layer_num,
                      vblock_count,
                      cp_size,
                      block_size);

    std::vector<BufferPtr> temp_gpu_buffers;

    for (size_t layer_id = 0; layer_id < layer_num; ++layer_id) {
        size_t gid = 0;
        if (use_hybrid && layer_id < cache_config.layer_to_group_id.size()) {
            const int mapped_gid = cache_config.layer_to_group_id[layer_id];
            if (mapped_gid >= 0)
                gid = static_cast<size_t>(mapped_gid);
        }
        const auto& block_ids     = block_ids_by_group[gid]->blocks();
        const int   decode_blocks = static_cast<int>(block_ids.size());
        const int   total_tokens  = decode_blocks * block_size;

        auto   sample_parts = cache_manager_->convertIndexToBuffer(block_ids[0], layer_id, 1, 0);
        size_t kv_stride    = sample_parts[0].size_bytes;
        int    elem_stride  = static_cast<int>(kv_stride / block_size);
        if (elem_stride <= 0 || elem_stride % 16 != 0) {
            continue;
        }

        int max_dst_bid     = *std::max_element(block_ids.begin(), block_ids.end());
        int max_src_bid     = *std::max_element(plan->staging_block_ids.begin(), plan->staging_block_ids.end());
        int addr_table_size = std::max(max_dst_bid, max_src_bid) + 1;

        std::vector<void*> dst_addrs(addr_table_size, nullptr);
        std::vector<void*> src_addrs(addr_table_size, nullptr);

        for (int b = 0; b < decode_blocks; ++b) {
            auto parts              = cache_manager_->convertIndexToBuffer(block_ids[b], layer_id, 1, 0);
            dst_addrs[block_ids[b]] = parts[0].addr;
        }
        for (int s = 0; s < staging_cnt; ++s) {
            auto parts = cache_manager_->convertIndexToBuffer(plan->staging_block_ids[s], layer_id, 1, 0);
            src_addrs[plan->staging_block_ids[s]] = parts[0].addr;
        }

        auto addrs_gpu =
            device_->allocateBuffer({DataType::TYPE_UINT64, {(size_t)addr_table_size}, AllocationType::DEVICE}, {});
        auto dst_ids_gpu =
            device_->allocateBuffer({DataType::TYPE_INT32, {(size_t)decode_blocks}, AllocationType::DEVICE}, {});
        auto src_addrs_gpu =
            device_->allocateBuffer({DataType::TYPE_UINT64, {(size_t)addr_table_size}, AllocationType::DEVICE}, {});
        auto src_ids_gpu =
            device_->allocateBuffer({DataType::TYPE_INT32, {(size_t)staging_cnt}, AllocationType::DEVICE}, {});
        temp_gpu_buffers.insert(temp_gpu_buffers.end(), {addrs_gpu, dst_ids_gpu, src_addrs_gpu, src_ids_gpu});

        cudaMemcpyAsync(
            addrs_gpu->data(), dst_addrs.data(), addr_table_size * sizeof(void*), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(
            dst_ids_gpu->data(), block_ids.data(), decode_blocks * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(src_addrs_gpu->data(),
                        src_addrs.data(),
                        addr_table_size * sizeof(void*),
                        cudaMemcpyHostToDevice,
                        stream);
        cudaMemcpyAsync(src_ids_gpu->data(),
                        plan->staging_block_ids.data(),
                        staging_cnt * sizeof(int),
                        cudaMemcpyHostToDevice,
                        stream);

        invokeCPCacheScatterPaged(reinterpret_cast<void**>(addrs_gpu->data()),
                                  dst_ids_gpu->data<int>(),
                                  reinterpret_cast<void**>(src_addrs_gpu->data()),
                                  src_ids_gpu->data<int>(),
                                  vblock_count,
                                  cp_size,
                                  block_size,
                                  total_tokens,
                                  elem_stride,
                                  stream);

        // Scale scatter (same pattern)
        if (sample_parts.size() == 2 && sample_parts[1].size_bytes > 0) {
            size_t sc_stride = sample_parts[1].size_bytes;
            int    sc_elem   = static_cast<int>(sc_stride / block_size);
            if (sc_elem > 0 && sc_elem % 16 == 0) {
                std::vector<void*> sc_dst_addrs(addr_table_size, nullptr);
                std::vector<void*> sc_src_addrs(addr_table_size, nullptr);
                for (int b = 0; b < decode_blocks; ++b) {
                    auto parts = cache_manager_->convertIndexToBuffer(block_ids[b], layer_id, 1, 0);
                    if (parts.size() == 2)
                        sc_dst_addrs[block_ids[b]] = parts[1].addr;
                }
                for (int s = 0; s < staging_cnt; ++s) {
                    auto parts = cache_manager_->convertIndexToBuffer(plan->staging_block_ids[s], layer_id, 1, 0);
                    if (parts.size() == 2)
                        sc_src_addrs[plan->staging_block_ids[s]] = parts[1].addr;
                }

                auto sc_dst_gpu = device_->allocateBuffer(
                    {DataType::TYPE_UINT64, {(size_t)addr_table_size}, AllocationType::DEVICE}, {});
                auto sc_src_gpu = device_->allocateBuffer(
                    {DataType::TYPE_UINT64, {(size_t)addr_table_size}, AllocationType::DEVICE}, {});
                temp_gpu_buffers.insert(temp_gpu_buffers.end(), {sc_dst_gpu, sc_src_gpu});

                cudaMemcpyAsync(sc_dst_gpu->data(),
                                sc_dst_addrs.data(),
                                addr_table_size * sizeof(void*),
                                cudaMemcpyHostToDevice,
                                stream);
                cudaMemcpyAsync(sc_src_gpu->data(),
                                sc_src_addrs.data(),
                                addr_table_size * sizeof(void*),
                                cudaMemcpyHostToDevice,
                                stream);

                invokeCPCacheScatterPaged(reinterpret_cast<void**>(sc_dst_gpu->data()),
                                          dst_ids_gpu->data<int>(),
                                          reinterpret_cast<void**>(sc_src_gpu->data()),
                                          src_ids_gpu->data<int>(),
                                          vblock_count,
                                          cp_size,
                                          block_size,
                                          total_tokens,
                                          sc_elem,
                                          stream);
            }
        }
    }

    auto err = cudaStreamSynchronize(stream);
    RTP_LLM_CHECK_WITH_INFO(err == cudaSuccess, "scatter stream sync failed: %s", cudaGetErrorString(err));
    temp_gpu_buffers.clear();

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
