#pragma once

#include <gmock/gmock.h>

#include "rtp_llm/cpp/cache/KVCacheAllocator.h"

namespace rtp_llm {

class MockKVCacheAllocator: public KVCacheAllocator {
public:
    explicit MockKVCacheAllocator(const CacheConfig&   config,
                                  rtp_llm::DeviceBase* device,
                                  AllocationType       atype = AllocationType::DEVICE):
        KVCacheAllocator(config, device, atype) {}
    ~MockKVCacheAllocator() override = default;

public:
    MOCK_METHOD(bool, init, (), (override));
    MOCK_METHOD(void, free, (const FreeInfo&), (override));
    MOCK_METHOD(void, insertIntoCache, (const InsertInfo&), (override));
    MOCK_METHOD(BlockAddrInfo, convertIndexToAddr, (int layer_id, int block_id), (const, override));
    MOCK_METHOD(BlockBufferPtrInfo, convertIndexToBuffer, (int layer_id, int block_id), (const, override));
    MOCK_METHOD(std::vector<BufferPtr>,
                convertIndexToBuffer,
                (int layer_id, int block_id, int partition_count, int partition_id),
                (const, override));
    MOCK_METHOD(std::shared_ptr<KVCacheResourceV1>,
                incrKVCacheRef,
                (const KVCacheResourceV1& kvcache_resource, const CacheKeysType& cache_keys),
                (override));
    MOCK_METHOD(void, decrKVCacheRef, (const KVCacheResourceV1& kvcache_resource), (override));
    MOCK_METHOD(CacheLayerLayout, layerCacheBase, (), (const, override));
    MOCK_METHOD(bool,
                updateKVBlock,
                (const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                 const std::vector<int>&        block_src_batch,
                 bool                           copy_last_block,
                 std::vector<BlockIdPair>&      block_update_mapping),
                (override));
    MOCK_METHOD(int, seqSizePerBlock, (), (const, override));
    MOCK_METHOD((std::vector<std::pair<BufferPtr, size_t>>), getAllBuffers, (), (const, override));
    MOCK_METHOD(void, clearCache, (), (override));

protected:
    MOCK_METHOD(MallocResult, incrMalloc, (const MallocInfo&), (override));
    MOCK_METHOD(MallocResult, initMallocForCommonLen, (const MallocInfo&), (override));
};

}  // namespace rtp_llm
