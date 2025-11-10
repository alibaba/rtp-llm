#pragma once

#include <gmock/gmock.h>
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"

namespace rtp_llm {

class MockKVCacheAllocator: public KVCacheAllocator {
public:
    MockKVCacheAllocator(): KVCacheAllocator(CacheConfig(), nullptr) {}
    ~MockKVCacheAllocator() = default;

public:
    MOCK_METHOD(bool, init, (), (override));
    MOCK_METHOD(MallocResult, malloc, (const MallocInfo& malloc_info), (override));
    MOCK_METHOD(FreeResult, free, (const FreeInfo& free_info), (override));
    MOCK_METHOD(InsertResult, insertIntoCache, (const InsertInfo& insert_info), (override));
    MOCK_METHOD(BlockAddrInfo, convertIndexToAddr, (int layer_id, int block_id), (const override));
    MOCK_METHOD(BlockBufferInfo,
                convertIndexToBuffer,
                (int layer_id, int block_id, int partition_count, int partition_id),
                (const override));
    MOCK_METHOD(CacheLayerLayout, layerCacheBase, (), (const override));
    MOCK_METHOD(size_t, blockSize, (), (const override));
    MOCK_METHOD(std::vector<BufferPtr>, cacheBuffers, (), (const override));
};

}  // namespace rtp_llm