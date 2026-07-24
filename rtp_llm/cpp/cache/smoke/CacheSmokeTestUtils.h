#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <c10/cuda/CUDAStream.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm::test {

inline void initCacheSmokeRuntime() {
    if (!isRuntimeInitialized()) {
        initRuntime(/*device_id=*/0,
                    /*trace_memory=*/false,
                    /*enable_comm_overlap=*/false,
                    MlaOpsType::AUTO);
    }
}

inline CacheConfig makeCacheSmokeConfig(int block_num = 8, DataType dtype = DataType::TYPE_INT8) {
    return makeSimpleMhaCacheConfig(/*layer_num=*/2,
                                    block_num,
                                    /*tokens_per_block=*/4,
                                    dtype,
                                    /*local_head_num_kv=*/2,
                                    /*size_per_head=*/8);
}

inline CacheConfig makeMultiGroupCacheSmokeConfig() {
    constexpr uint32_t kBlockNum       = 10;
    constexpr uint32_t kTokensPerBlock = 4;

    CacheConfig config;
    config.dtype                       = DataType::TYPE_FP16;
    config.layer_num                   = 3;
    config.layer_all_num               = 3;
    config.block_num                   = kBlockNum;
    config.seq_size_per_block          = kTokensPerBlock;
    config.kernel_seq_size_per_block   = kTokensPerBlock;
    config.linear_step                 = 1;
    config.use_independent_block_pools = true;

    const std::vector<std::string> tags = {
        "shared_full", "layer0_linear", "layer1_swa", "layer1_full", "layer2_linear"};
    std::vector<KVCacheSpecPtr> specs = {
        makeResolvedMhaSpec(config.dtype, 1, 2, kTokensPerBlock, tags[0]),
        makeResolvedLinearSpec(config.dtype, 1, 1, 2, 2, 2, kTokensPerBlock, config.dtype, config.dtype, tags[1]),
        makeResolvedMhaSpec(config.dtype, 1, 2, kTokensPerBlock, tags[2]),
        makeResolvedMhaSpec(config.dtype, 1, 2, kTokensPerBlock, tags[3]),
        makeResolvedLinearSpec(config.dtype, 1, 1, 2, 2, 2, kTokensPerBlock, config.dtype, config.dtype, tags[4]),
    };
    const std::vector<std::vector<int>> layers_by_group = {{0, 1, 2}, {0}, {1}, {1}, {2}};
    const std::vector<CacheGroupType>   group_types     = {
        CacheGroupType::FULL,
        CacheGroupType::LINEAR,
        CacheGroupType::SWA,
        CacheGroupType::FULL,
        CacheGroupType::LINEAR,
    };
    config.fromGroupedSpecs(specs, layers_by_group, group_types, tags);

    std::vector<uint32_t> block_nums(specs.size(), kBlockNum);
    std::vector<size_t>   kv_strides;
    std::vector<size_t>   scale_strides;
    kv_strides.reserve(specs.size());
    scale_strides.reserve(specs.size());
    for (const auto& spec : specs) {
        kv_strides.push_back(spec->block_size_bytes());
        scale_strides.push_back(spec->scale_block_size_bytes());
    }
    config.setGroupBlockLayout(block_nums, kv_strides, scale_strides);

    config.kv_block_stride_bytes = *std::max_element(kv_strides.begin(), kv_strides.end());
    config.kv_scale_stride_bytes = *std::max_element(scale_strides.begin(), scale_strides.end());
    config.kv_block_size_bytes   = config.layer_all_num * config.kv_block_stride_bytes;
    config.kv_scale_size_bytes   = config.layer_all_num * config.kv_scale_stride_bytes;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;
    config.layer_to_block_stride_bytes.assign(
        config.layer_all_num, static_cast<int>(config.kv_block_stride_bytes + config.kv_scale_stride_bytes));
    return config;
}

inline CompleteTokenIdsPtr makeCacheSmokeTokenIds(const std::vector<int32_t>& tokens) {
    auto complete_token_ids =
        std::make_shared<CompleteTokenIds>(/*batch_size=*/1, /*beam_width=*/1, tokens.size() + 32, /*block_size=*/4);
    auto input_ids = torch::from_blob(const_cast<int32_t*>(tokens.data()),
                                      {static_cast<int64_t>(tokens.size())},
                                      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU))
                         .clone();
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = std::move(input_ids);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    complete_token_ids->init(generate_input);
    complete_token_ids->setSeqLength(static_cast<int>(tokens.size()));
    return complete_token_ids;
}

inline std::vector<int32_t> makeTokenRange(int32_t begin, int count) {
    std::vector<int32_t> tokens(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        tokens[static_cast<size_t>(i)] = begin + i;
    }
    return tokens;
}

inline CacheKeysType makeCacheKeys(CacheKeyType begin, int count) {
    CacheKeysType keys(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        keys[static_cast<size_t>(i)] = begin + i;
    }
    return keys;
}

inline BatchKVCacheResourcePtr makeCacheSmokeResource(const CacheConfig& config, const CacheKeysType& cache_keys) {
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(1);
    resource->initGroups(config.topologyPtr());
    resource->setBatchCacheKeys(/*batch_id=*/0, cache_keys);
    return resource;
}

inline MallocResult allocateCacheSmokeResource(const KVCacheAllocatorPtr&     allocator,
                                               const BatchKVCacheResourcePtr& resource,
                                               const CompleteTokenIdsPtr&     token_ids,
                                               bool                           enable_device_cache = false) {
    MallocInfo malloc_info{resource, token_ids};
    malloc_info.reuse_cache         = enable_device_cache;
    malloc_info.enable_device_cache = enable_device_cache;
    return allocator->malloc(malloc_info);
}

inline torch::Tensor byteTensorForBlockInfo(const BlockInfo& info) {
    auto options = torch::TensorOptions().dtype(torch::kUInt8);
    if (info.is_cuda) {
        options = options.device(torch::Device(torch::kCUDA, info.device_index));
    } else {
        options = options.device(torch::kCPU);
    }
    return torch::from_blob(info.addr, {static_cast<int64_t>(info.size_bytes)}, options);
}

inline void synchronizeCacheSmokeDevice() {
    at::cuda::getCurrentCUDAStream().synchronize();
}

inline void fillBlockInfos(const std::vector<BlockInfo>& infos, uint8_t seed) {
    for (size_t i = 0; i < infos.size(); ++i) {
        ASSERT_NE(infos[i].addr, nullptr);
        ASSERT_GT(infos[i].size_bytes, 0u);
        byteTensorForBlockInfo(infos[i]).fill_(static_cast<uint8_t>(seed + i));
    }
    synchronizeCacheSmokeDevice();
}

inline std::vector<uint8_t> readBlockInfoBytes(const BlockInfo& info) {
    auto cpu = byteTensorForBlockInfo(info).cpu();
    synchronizeCacheSmokeDevice();
    std::vector<uint8_t> bytes(info.size_bytes);
    std::memcpy(bytes.data(), cpu.data_ptr(), info.size_bytes);
    return bytes;
}

inline std::vector<BlockInfo>
allocatorBlockInfos(const KVCacheAllocator& allocator, int layer_id, const std::string& tag, BlockIdxType block_id) {
    return allocator.convertIndexToBufferByTag(layer_id, tag, block_id, /*partition_count=*/1, /*partition_id=*/0);
}

inline std::vector<BlockInfo>
allocatorBlockInfos(const KVCacheAllocator& allocator, int layer_id, BlockIdxType block_id) {
    return allocatorBlockInfos(allocator, layer_id, "default", block_id);
}

inline std::vector<BlockInfo> managerBlockInfos(const KVCacheManager& manager, int layer_id, BlockIdxType block_id) {
    return manager.convertIndexToBufferByTag(
        block_id, layer_id, /*tag=*/"default", /*partition_count=*/1, /*partition_id=*/0);
}

inline void fillAllocatorResource(const KVCacheAllocator& allocator, const KVCacheResource& resource, uint8_t seed) {
    for (int layer_id = 0; layer_id < resource.layerNum(); ++layer_id) {
        const auto& tags = resource.groupTagsForLayer(layer_id);
        for (size_t group_pos = 0; group_pos < tags.size(); ++group_pos) {
            const auto& blocks = resource.blocksForLayer(layer_id, tags[group_pos]);
            for (size_t block_pos = 0; block_pos < blocks.size(); ++block_pos) {
                if (isNullBlockIdx(blocks[block_pos])) {
                    continue;
                }
                fillBlockInfos(allocatorBlockInfos(allocator, layer_id, tags[group_pos], blocks[block_pos]),
                               static_cast<uint8_t>(seed + layer_id * 31 + group_pos * 17 + block_pos * 7));
            }
        }
    }
}

inline void expectAllocatorResourcesEqual(const KVCacheAllocator& src_allocator,
                                          const KVCacheResource&  src_resource,
                                          const KVCacheAllocator& dst_allocator,
                                          const KVCacheResource&  dst_resource) {
    ASSERT_EQ(src_resource.layerNum(), dst_resource.layerNum());
    ASSERT_EQ(src_resource.cacheKeys(), dst_resource.cacheKeys());
    for (int layer_id = 0; layer_id < src_resource.layerNum(); ++layer_id) {
        ASSERT_EQ(src_resource.groupTagsForLayer(layer_id), dst_resource.groupTagsForLayer(layer_id));
        for (const auto& tag : src_resource.groupTagsForLayer(layer_id)) {
            const auto& src_blocks = src_resource.blocksForLayer(layer_id, tag);
            const auto& dst_blocks = dst_resource.blocksForLayer(layer_id, tag);
            ASSERT_EQ(src_blocks.size(), dst_blocks.size()) << "layer=" << layer_id << " tag=" << tag;
            for (size_t block_pos = 0; block_pos < src_blocks.size(); ++block_pos) {
                ASSERT_EQ(isNullBlockIdx(src_blocks[block_pos]), isNullBlockIdx(dst_blocks[block_pos]))
                    << "layer=" << layer_id << " tag=" << tag << " block_pos=" << block_pos;
                if (isNullBlockIdx(src_blocks[block_pos])) {
                    continue;
                }
                auto src_infos = allocatorBlockInfos(src_allocator, layer_id, tag, src_blocks[block_pos]);
                auto dst_infos = allocatorBlockInfos(dst_allocator, layer_id, tag, dst_blocks[block_pos]);
                ASSERT_EQ(src_infos.size(), dst_infos.size())
                    << "layer=" << layer_id << " tag=" << tag << " block_pos=" << block_pos;
                for (size_t buffer_id = 0; buffer_id < src_infos.size(); ++buffer_id) {
                    EXPECT_EQ(readBlockInfoBytes(src_infos[buffer_id]), readBlockInfoBytes(dst_infos[buffer_id]))
                        << "layer=" << layer_id << " tag=" << tag << " block_pos=" << block_pos
                        << " buffer_id=" << buffer_id;
                }
            }
        }
    }
}

inline void drainAllocatorCache(const KVCacheAllocatorPtr& allocator) {
    while (auto evicted = allocator->popBlocksFromCache(allocator->totalBlocksNum())) {
        allocator->blockCacheFree(evicted);
    }
}

inline SingleTypeKVCacheAllocatorPtr makeCacheSmokeAllocator(const CacheConfig& config, bool enable_prefix_cache) {
    auto allocator = std::make_shared<SingleTypeKVCacheAllocator>(config);
    if (enable_prefix_cache) {
        allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    }
    return allocator;
}

inline KVCacheAllocatorPtr makeCacheSmokeAllocatorForConfig(const CacheConfig& config, bool enable_prefix_cache) {
    KVCacheAllocatorPtr allocator;
    if (config.use_independent_block_pools) {
        allocator = std::make_shared<HybridPoolKVCacheAllocator>(config);
    } else {
        allocator = std::make_shared<SingleTypeKVCacheAllocator>(config);
    }
    if (enable_prefix_cache) {
        allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    }
    return allocator;
}

struct CacheSmokePoolCounters {
    size_t free_blocks;
    size_t available_blocks;
    size_t request_refs;
    size_t block_cache_refs;
    size_t connector_refs;
};

inline std::vector<CacheSmokePoolCounters> snapshotCacheSmokePools(const HybridPoolKVCacheAllocator& allocator) {
    std::vector<CacheSmokePoolCounters> counters;
    counters.reserve(allocator.groupBlockPools().size());
    for (const auto& pool : allocator.groupBlockPools()) {
        counters.push_back({pool->freeBlocksNum(),
                            pool->availableBlocksNum(),
                            pool->requestRefBlocksNum(),
                            pool->blockCacheRefBlocksNum(),
                            pool->connectorRefBlocksNum()});
    }
    return counters;
}

inline void expectCacheSmokePoolsEqual(const HybridPoolKVCacheAllocator&          allocator,
                                       const std::vector<CacheSmokePoolCounters>& expected) {
    ASSERT_EQ(allocator.groupBlockPools().size(), expected.size());
    for (size_t gid = 0; gid < expected.size(); ++gid) {
        const auto& pool = allocator.groupBlockPools()[gid];
        EXPECT_EQ(pool->freeBlocksNum(), expected[gid].free_blocks) << "gid=" << gid;
        EXPECT_EQ(pool->availableBlocksNum(), expected[gid].available_blocks) << "gid=" << gid;
        EXPECT_EQ(pool->requestRefBlocksNum(), expected[gid].request_refs) << "gid=" << gid;
        EXPECT_EQ(pool->blockCacheRefBlocksNum(), expected[gid].block_cache_refs) << "gid=" << gid;
        EXPECT_EQ(pool->connectorRefBlocksNum(), expected[gid].connector_refs) << "gid=" << gid;
    }
}

}  // namespace rtp_llm::test
