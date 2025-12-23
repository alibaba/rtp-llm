#pragma once

#include <gmock/gmock.h>

#include "rtp_llm/cpp/cache/KVCacheAllocator.h"

namespace rtp_llm {

class MockKVCacheAllocator final: public KVCacheAllocator {
public:
    explicit MockKVCacheAllocator(const CacheConfig&   config,
                                  rtp_llm::DeviceBase* device,
                                  AllocationType       atype = AllocationType::DEVICE):
        KVCacheAllocator(config, device, atype) {}
    ~MockKVCacheAllocator() override = default;

public:
    MOCK_METHOD(bool, init, (), (override));
    MOCK_METHOD(MallocResult, malloc, (const MallocInfo&), (override));
    MOCK_METHOD(FreeResult, free, (const FreeInfo&), (override));
    MOCK_METHOD(InsertResult, insertIntoCache, (const InsertInfo&), (override));
    MOCK_METHOD(BlockAddrInfo, convertIndexToAddr, (int layer_id, int block_id), (const, override));
    MOCK_METHOD(BlockBufferInfo, convertIndexToBuffer, (int layer_id, int block_id), (const, override));
    MOCK_METHOD(CacheLayerLayout, layerCacheBase, (), (const, override));
    MOCK_METHOD(void, regUserMr, (size_t model_id), (override));
    MOCK_METHOD(size_t, freeBlocksNums, (), (const, override));
    MOCK_METHOD(size_t, availableBlocksNums, (), (const, override));
    MOCK_METHOD(size_t, totalBlocksNums, (), (const, override));
    MOCK_METHOD(size_t, maxSeqLen, (), (const, override));
    MOCK_METHOD(KVCacheBuffer, kvCacheBuffer, (), (const, override));
    MOCK_METHOD(void, clearCache, (), (override));

protected:
    MOCK_METHOD(MallocResult, initMalloc, (const MallocInfo&), (override));
    MOCK_METHOD(MallocResult, incrMalloc, (const MallocInfo&), (override));
    MOCK_METHOD(MallocResult, initMallocForCommonLen, (const MallocInfo&), (override));
};

}  // namespace rtp_llm
