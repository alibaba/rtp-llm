#pragma once

#include <gmock/gmock.h>

#include "rtp_llm/cpp/cache/connector/KVCacheConnectorReadWriteContext.h"

namespace rtp_llm {

class MockKVCacheConnector: public KVCacheConnector {
public:
    MockKVCacheConnector()           = default;
    ~MockKVCacheConnector() override = default;

public:
    MOCK_METHOD(std::shared_ptr<AsyncMatchContext>,
                asyncMatch,
                (const std::shared_ptr<KVCacheResourceV1>& resource, const std::shared_ptr<Meta>& meta),
                (override));
    MOCK_METHOD(std::shared_ptr<AsyncContext>,
                asyncRead,
                (const std::shared_ptr<KVCacheResourceV1>& resource,
                 const std::shared_ptr<Meta>&              meta,
                 const std::shared_ptr<AsyncMatchContext>& match_context),
                (override));
    MOCK_METHOD(std::shared_ptr<AsyncContext>,
                asyncWrite,
                (const std::shared_ptr<KVCacheResourceV1>& resource, const std::shared_ptr<Meta>& meta),
                (override));
    MOCK_METHOD(std::shared_ptr<AsyncContext>,
                asyncWriteByLayer,
                (int layer_id, const std::shared_ptr<KVCacheResourceV1>& resource, const std::shared_ptr<Meta>& meta),
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
    MOCK_METHOD(KVCacheConnector::ConnectorType, connectorType, (), (const, override));
};

}  // namespace rtp_llm
