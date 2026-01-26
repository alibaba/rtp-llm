#pragma once

#include <gmock/gmock.h>

#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"

namespace rtp_llm {

class MockKVCacheConnector: public KVCacheConnector {
public:
    MockKVCacheConnector()           = default;
    ~MockKVCacheConnector() override = default;

public:
    MOCK_METHOD(std::shared_ptr<AsyncMatchContext>,
                asyncMatch,
                (const std::shared_ptr<KVCacheResource>& resource, const std::shared_ptr<Meta>& meta),
                (override));
    MOCK_METHOD(std::shared_ptr<AsyncContext>,
                asyncRead,
                (const std::shared_ptr<KVCacheResource>&   resource,
                 const std::shared_ptr<Meta>&              meta,
                 const std::shared_ptr<AsyncMatchContext>& match_context,
                 int                                       start_read_block_index,
                 int                                       read_block_num),
                (override));
    MOCK_METHOD(std::shared_ptr<AsyncContext>,
                asyncWrite,
                (const std::shared_ptr<KVCacheResource>& resource, const std::shared_ptr<Meta>& meta),
                (override));
    MOCK_METHOD(std::shared_ptr<AsyncContext>,
                asyncWriteByLayer,
                (int layer_id, const std::shared_ptr<KVCacheResource>& resource, const std::shared_ptr<Meta>& meta),
                (override));
};

class MockAsyncMatchContext: public KVCacheConnector::AsyncMatchContext {
public:
    MockAsyncMatchContext()           = default;
    ~MockAsyncMatchContext() override = default;

public:
    MOCK_METHOD(bool, done, (), (const, override));
    MOCK_METHOD(bool, success, (), (const, override));
    MOCK_METHOD(size_t, matchedBlockCount, (), (const, override));
};

}  // namespace rtp_llm
